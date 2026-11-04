"""整盤棋的「落子時間」盤面編碼＋小 ResNet：每盤一個樣本、直接預測棋力。

官方 baseline 把最後 250 手的每個盤面疊成 500 個通道，模型因此看得到每顆子是何時下的；
我們的板面 CNN 每盤只取 4 個快照，落子順序幾乎都丟了。這裡把同樣的資訊精簡成 16 個通道：
我方／對方的首次落子時間、開局／中盤／官子三段的落子位置、終局盤面、被提的子與被提時間。
每盤特徵約 1.3 毫秒。探針：寬 64、6 層、6000 步（10 分鐘），單盤 Spearman 0.667（RESULTS.md §2.36）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .goboard import BLACK, SIZE, WHITE, Board, normalise_color, parse_moves
from .nets import ResBlock, ResBlockGPB

C_IN = 16



def time_planes(sgf, color):
    mv = parse_moves(sgf); n = max(len(mv), 1)
    my = normalise_color(color); opp = WHITE if my == BLACK else BLACK
    P = np.zeros((C_IN, SIZE, SIZE), np.float32)
    b = Board()
    for t, (c, r, k) in enumerate(mv):
        before = b.stones.copy()
        b.play(c, r, k)
        own = c == my; tt = (t + 1) / n
        base = 0 if own else 1
        if P[base, r, k] == 0:
            P[base, r, k] = tt                                    # 首次落子時間
        ph = 0 if t < 50 else (1 if t < 150 else 2)
        P[2 + (0 if own else 3) + ph, r, k] = 1                   # 分段落子位置
        gone = np.flatnonzero((before != 0) & (b.stones == 0))    # 這手提掉的子
        for g in gone:
            gr, gk = divmod(int(g), SIZE)
            P[10 if before[g] == my else 11, gr, gk] = tt         # 被提時間
    s = b.stones.reshape(SIZE, SIZE)
    P[8] = s == my; P[9] = s == opp; P[12] = s == 0
    P[13] = n / 300.0; P[14] = 1.0; P[15] = float(my == BLACK)
    return P



class TimePlanesNet(nn.Module):
    def __init__(self, w=64, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(C_IN, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(True))
        self.blocks = nn.Sequential(*[ResBlockGPB(w) if (i + 1) % 2 == 0 else ResBlock(w) for i in range(blocks)])
        self.head = nn.Linear(w, 10)
    def forward(self, x):
        h = self.blocks(self.stem(x)); return self.head(h.mean((2, 3)))




def load_timeplanes(path: str, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = TimePlanesNet(ck["w"], ck["blocks"])
    net.load_state_dict(ck["model"])
    return net.eval().to(device)


@torch.no_grad()
def game_proba(net, games, device, batch: int = 512) -> np.ndarray:
    """[(sgf, color), ...] → (n_games, 10) 的棋力機率。"""
    out = []
    for a in range(0, len(games), batch):
        x = torch.from_numpy(np.stack([time_planes(s, c) for s, c in games[a:a + batch]])).to(device)
        out.append(torch.softmax(net(x), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 10), np.float32)


def question_log_posterior(game_proba_per_q) -> np.ndarray:
    """每題各盤機率 → 題目層級 log 後驗：各盤 log 機率平均後再正規化（與 eval_blend 相同）。"""
    out = []
    for P in game_proba_per_q:
        if len(P) == 0:
            out.append(np.full(10, -np.log(10.0)))
            continue
        L = np.log(np.clip(np.asarray(P), 1e-9, None)).mean(0)
        m = L.max()
        out.append(L - (m + np.log(np.exp(L - m).sum())))
    return np.array(out)


def rows_proba(net, device, meta, rows, sgf_cache, store_path: str | None = None) -> dict:
    """meta 列號 → 10 級機率。驗證集存檔有的直接讀，沒有的從 SGF 現算。"""
    import os
    out = {}
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    if store_path and os.path.exists(store_path):
        z = np.load(store_path)
        srows, sproba = z["rows"], z["proba"]          # 只讀一次：npz 每次取鍵都會重新解壓整個陣列
        pos = {int(r): i for i, r in enumerate(srows)}
        hit = [r for r in rows if int(r) in pos]
        for r in hit:
            out[int(r)] = sproba[pos[int(r)]]
        rows = np.array([r for r in rows if int(r) not in pos], dtype=np.int64)
    if len(rows):
        fi, ri, co = (meta.columns.get_loc(c) for c in ("file_idx", "row", "color"))
        games = [(sgf_cache.get(int(meta.iat[r, fi]), int(meta.iat[r, ri])), meta.iat[r, co]) for r in rows]
        P = game_proba(net, games, device)
        for r, p in zip(rows, P):
            out[int(r)] = p
    return out
