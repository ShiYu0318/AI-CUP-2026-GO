"""以棋力為條件的走法模型（Maia 式），在我們自己的 100 萬盤上訓練。

為什麼：KataGo HumanSL 是現成的同類模型，加進集成有效（RESULTS.md §2.32），
但它的棋力尺度與我們的標籤不一致——真實 0~9 級的平均峰值只落在 1.4~5.4——
而這個對應只能靠小小的校準集去學，增益因此隨校準量縮水。
在自己的資料上訓練，尺度天生對齊；而且推論時 **10 種棋力一次前向**
（KataGo 每種棋力要另跑一次），所以能用上每一手，不必只抽 24 手。

結構：共用的殘差骨幹 → 1x1 卷積出 N_LEVELS 個通道，每個通道是一種棋力的 361 格走法 logit。
訓練時只用該盤真實棋力那個通道算交叉熵；已有棋子的點一律遮掉。
一盤棋的分數 = Σ 我方各手在 10 種棋力下的 log 機率 → 與 HumanSL 成分同一個介面。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .goboard import BLACK, normalise_color, parse_moves
from .nets import ResBlock, ResBlockGPB
from .planes import game_planes, n_planes, planes_to_float

N_LEVELS = 10
EMPTY_PLANE = 2          # planes.py 的第 2 層是空點


class SkillPolicyNet(nn.Module):
    def __init__(self, in_channels: int = 34, width: int = 96, blocks: int = 8,
                 n_levels: int = N_LEVELS, gpb_every: int = 2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlockGPB(width) if (i + 1) % gpb_every == 0 else ResBlock(width)
                                      for i in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(width, width // 2, 1, bias=False),
                                  nn.BatchNorm2d(width // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(width // 2, n_levels, 1))
        self.config = dict(in_channels=in_channels, width=width, blocks=blocks,
                           n_levels=n_levels, gpb_every=gpb_every)

    def forward(self, x):
        """x: (B, C, 19, 19) float → (B, n_levels, 361) 的 log 機率（已遮掉有子的點）。"""
        logits = self.head(self.blocks(self.stem(x))).flatten(2)
        occupied = (x[:, EMPTY_PLANE] < 0.5).flatten(1)[:, None, :]
        return F.log_softmax(logits.masked_fill(occupied, -1e4), dim=-1)


def own_turns(sgf: str, color) -> tuple[list, list]:
    """(我方落子的手數, 對應的目標點 r*19+c)。第一手（空盤）略過——沒有「前一手之後」的局面。"""
    mv = parse_moves(sgf)
    my = normalise_color(color)
    t = [i for i, (c, _, _) in enumerate(mv) if c == my and i > 0]
    return t, [mv[i][1] * 19 + mv[i][2] for i in t]


def position_samples(sgf: str, color, turns, targets):
    """指定的我方手數 → (局面 (k,C,19,19) uint8, 目標 (k,))。局面取「前一手下完之後」。"""
    planes = game_planes(sgf, color, indices=[t - 1 for t in turns])
    if planes is None or len(planes) != len(turns):
        return None, None
    return planes, np.asarray(targets, dtype=np.int64)


@torch.no_grad()
def game_loglik(net, games, device, max_moves: int = 0, batch: int = 512) -> np.ndarray:
    """多盤棋 [(sgf, color), ...] → (n_games, 2 + N_LEVELS)：[評分手數, 總手數, Σlog p 於各棋力]。

    max_moves > 0 時均勻抽那麼多手（與 HumanSL 的 24 手對照用）；0 = 我方每一手。
    """
    net.eval()
    out = []
    for sgf, color in games:
        t, tg = own_turns(sgf, color)
        n_all = len(t)
        if max_moves and len(t) > max_moves:
            j = np.unique(np.linspace(0, len(t) - 1, max_moves).astype(int))
            t, tg = [t[i] for i in j], [tg[i] for i in j]
        acc = np.zeros(N_LEVELS)
        planes, tgt = position_samples(sgf, color, t, tg) if t else (None, None)
        if planes is not None:
            for a in range(0, len(planes), batch):
                x = torch.from_numpy(planes_to_float(planes[a:a + batch])).to(device)
                lp = net(x)                                           # (b, L, 361)
                idx = torch.from_numpy(tgt[a:a + batch]).to(device)
                acc += lp.gather(2, idx[:, None, None].expand(-1, lp.shape[1], 1)).squeeze(2).sum(0).cpu().numpy()
        out.append(np.concatenate([[len(t) if planes is not None else 0, n_all], acc]))
    return np.array(out)


def load_skill_policy(path: str, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = SkillPolicyNet(**ck["config"])
    net.load_state_dict(ck["model"])
    return net.eval().to(device)
