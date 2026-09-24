"""競賽二的模型定義與題目層級推論。

## 為什麼在 `src/` 而不是訓練腳本裡

這些類別原本住在 `scripts/train_rank_mlp.py` 與 `scripts/train_rank_seq.py`，
但**推論也要用**——`scripts/predict.py`（比賽當天產生提交檔的路徑）
得 `from train_rank_seq import build_seq_net`。

那讓提交路徑依賴兩個**訓練腳本**的 import：只要有人在訓練腳本頂層
加一個重的 import 或副作用，提交路徑就跟著受影響。
而且同樣是模型，板面 CNN 的 `MultiTaskGoNet` 在 `aicup.nets`，
單局與序列模型卻在 `scripts/`——同一件事住在兩個地方。

`src/` 放「什麼是正確的」，`scripts/` 放「做一件事的流程」。
模型的定義與推論屬於前者。訓練腳本仍 re-export 這些名字，舊的呼叫不會壞。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import corn_to_proba
from .moveseq import N_MOVE_FEATURES
from .question_model import build_question_features


def pick_device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class RankMLP(nn.Module):
    """兩個 head：10 類 softmax（搭配指數 soft label）與 CORN 的 9 個條件輸出。"""

    def __init__(self, d_in: int, hidden: int = 768, n_levels: int = 10,
                 dropout: float = 0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU())
        self.cls = nn.Linear(hidden // 2, n_levels)
        self.corn = nn.Linear(hidden // 2, n_levels - 1)

    def forward(self, x):
        z = self.body(x)
        return self.cls(z), self.corn(z)

@torch.no_grad()
def mlp_question_posteriors(nets, scaler, X, ok, game_rows, device, batch: int = 4096):
    """逐題：多局後驗做幾何平均（實測 logmean 勝過算術平均 +0.030）。

    ⚠️ 這個函式會暫時把模型切到 eval，結束時**還原成原本的模式**。
    早期版本結尾是無條件 `n.train()`，害 predict.py 在校準之後
    帶著 dropout 跑測試集推論：校準在 eval、預測在 train，
    期望等級的標準差從 2.16 被壓到 0.25，切點完全遷移不過去
    （端到端 0.33，內部驗證卻有 0.48）。
    """
    was_training = [n.training for n in nets]
    for n in nets:
        n.eval()
    out = []
    for rows in game_rows:
        sel = rows[ok[rows]]
        if len(sel) == 0:
            out.append(np.full(10, 0.1)); continue
        A = torch.from_numpy(scaler.transform(np.asarray(X[sel]))).to(device)
        ps = []
        for n in nets:
            lc, lo = n(A)
            p = F.softmax(lc, dim=-1).cpu().numpy()
            pc = corn_to_proba(lo).cpu().numpy()
            pc = pc / np.maximum(pc.sum(axis=1, keepdims=True), 1e-12)
            ps.append(0.5 * p + 0.5 * pc)          # softmax 與 CORN 各半
        p = np.mean(ps, axis=0)
        g = np.exp(np.log(np.clip(p, 1e-12, None)).mean(axis=0))
        out.append(g / g.sum())
    for n, t in zip(nets, was_training):
        n.train(t)
    return np.stack(out)

class SeqStore:
    """多份序列庫沿**特徵軸**串接（每一手的向量變長），取列時才合併。

    與 `aicup.featstore.MultiStore` 同樣的理由：兩份各 3.84 GB，
    實體合併既浪費磁碟、又讓「只用我方」的實驗得重建。
    """

    def __init__(self, parts):
        n = len(parts[0])
        for i, p in enumerate(parts):
            if len(p) != n:
                raise ValueError(f"第 {i} 份序列庫有 {len(p):,} 列，與第 0 份的 {n:,} 列不符")
        self.parts = parts
        self._shape = (n, parts[0].shape[1], sum(p.shape[2] for p in parts))

    def __len__(self):
        return self._shape[0]

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, rows):
        if len(self.parts) == 1:
            return self.parts[0][rows]
        return np.concatenate([np.asarray(p[rows]) for p in self.parts], axis=-1)

class SeqRankNet(nn.Module):
    """dilated 1D 卷積讀逐手序列。

    dilation 1/2/4/8 讓感受野涵蓋整個序列（160 手），
    同時保留局部的手順關係。avg 與 max pooling 並用：
    avg 抓「整體傾向」，max 抓「有沒有出現過某種糟糕的手」——
    後者對棋力判斷特別重要（一次致命失誤就暴露等級）。
    """

    def __init__(self, n_feat: int = N_MOVE_FEATURES, width: int = 128,
                 n_levels: int = 10, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Conv1d(n_feat, width, 5, padding=2)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, 5, padding=2 * r, dilation=r),
                nn.GroupNorm(8, width), nn.GELU(),
                nn.Conv1d(width, width, 1),
                nn.GroupNorm(8, width), nn.GELU())
            for r in (1, 2, 4, 8)])
        self.head = nn.Sequential(nn.Linear(2 * width, 256), nn.GELU(),
                                  nn.Dropout(dropout))
        self.cls = nn.Linear(256, n_levels)
        self.orn = nn.Linear(256, n_levels - 1)

    def forward(self, x, mask):
        h = self.stem(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        m = mask.unsqueeze(1).float()
        h = h * m
        avg = h.sum(-1) / m.sum(-1).clamp(min=1)
        mx = (h + (m - 1) * 1e4).max(-1).values
        z = self.head(torch.cat([avg, mx], dim=1))
        return self.cls(z), self.orn(z)

class SeqRankGRU(nn.Module):
    """雙向 GRU 讀逐手序列——**刻意換一個歸納偏置**，不是為了更強。

    集成的收益來自成分之間的分歧。目前三個集成成員（base / full / w192）
    都是同一個 dilated CNN，只差寬度與資料量，分歧其實很小，
    卻已經值 +0.0079（8/8，t=9.15）。換成遞迴狀態（無固定感受野、
    對「累積到目前為止的棋風」天然敏感）應該產生更不一樣的錯誤。

    pooling 刻意與 `SeqRankNet` **完全相同**（masked avg + max），
    所以兩者的差別只有 encoder 一項——這樣比較才有解釋力。
    """

    def __init__(self, n_feat: int = N_MOVE_FEATURES, width: int = 128,
                 n_levels: int = 10, dropout: float = 0.1, layers: int = 2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, width), nn.GELU())
        self.rnn = nn.GRU(width, width // 2, num_layers=layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * width, 256), nn.GELU(),
                                  nn.Dropout(dropout))
        self.cls = nn.Linear(256, n_levels)
        self.orn = nn.Linear(256, n_levels - 1)

    def forward(self, x, mask):
        h, _ = self.rnn(self.inp(x))          # (B, T, width)
        h = h.transpose(1, 2)                 # 與 CNN 的 (B, C, T) 對齊
        m = mask.unsqueeze(1).float()
        h = h * m
        avg = h.sum(-1) / m.sum(-1).clamp(min=1)
        mx = (h + (m - 1) * 1e4).max(-1).values
        z = self.head(torch.cat([avg, mx], dim=1))
        return self.cls(z), self.orn(z)

ARCHS = {"cnn": SeqRankNet, "gru": SeqRankGRU}

def build_seq_net(n_feat: int, ck_args: dict):
    """由 checkpoint 存下來的 args 重建網路。

    `arch` 是後加的，舊 checkpoint 沒有這個鍵——預設 "cnn" 才不會
    讓既有的三個集成成員載不起來。
    """
    arch = ck_args.get("arch", "cnn")
    if arch not in ARCHS:
        raise ValueError(f"未知的序列模型 arch={arch!r}，可用：{sorted(ARCHS)}")
    return ARCHS[arch](n_feat=n_feat, width=ck_args["width"])

def seq_question_posteriors(nets, S, ok, lens, game_rows, device, batch: int = 256):
    """逐題：多局後驗做幾何平均（與 train_rank_mlp 的聚合方式一致）。

    **先算完所有棋局，再逐題聚合**——不要逐題呼叫模型。
    逐題的話每次只有 5~20 局，1660 題 × 3 個網路 = 4980 次前向，
    全部是小 batch。卷積還撐得住，但 GRU 在 MPS 上每次前向要發
    160×2×2 次序列 kernel，啟動開銷完全主導：
    實測逐題版算一個 seed 的 GRU 後驗要**超過兩小時**，
    而整批版只要幾十秒。數值上兩者完全相同（同樣的幾何平均）。

    ⚠️ 結束時還原原本的 train/eval 模式。無條件切回 train 會讓後續推論
    帶著 dropout 跑（見 tests/test_eval_mode.py）。
    """
    was = [n.training for n in nets]
    for n in nets:
        n.eval()

    sels = [rows[ok[rows]] for rows in game_rows]
    flat = np.concatenate([s_ for s_ in sels if len(s_)]) if any(len(s_) for s_ in sels) \
        else np.zeros(0, dtype=np.int64)
    uniq, inv = np.unique(flat, return_inverse=True)

    P = np.empty((len(uniq), 10), dtype=np.float64)
    T = S.shape[1]
    with torch.no_grad():
        for i in range(0, len(uniq), batch):
            sel = uniq[i:i + batch]
            x = torch.from_numpy(np.asarray(S[sel], dtype=np.float16)).to(device).float()
            mk = torch.from_numpy(
                np.arange(T)[None, :] < lens[sel][:, None]).to(device)
            ps = []
            for n in nets:
                lc, lo = n(x, mk)
                pc = corn_to_proba(lo).cpu().numpy()
                ps.append(0.5 * torch.softmax(lc, -1).cpu().numpy()
                          + 0.5 * pc / np.maximum(pc.sum(1, keepdims=True), 1e-12))
            P[i:i + len(sel)] = np.mean(ps, axis=0)
    for n, t in zip(nets, was):
        n.train(t)

    logP = np.log(np.clip(P, 1e-12, None))
    out, at = [], 0
    for s_ in sels:
        if len(s_) == 0:
            out.append(np.full(10, 0.1)); continue
        g = np.exp(logP[inv[at:at + len(s_)]].mean(axis=0))
        at += len(s_)
        out.append(g / g.sum())
    return np.stack(out)


def seq_stores_of(ck_args: dict, cache_dir: str = "cache") -> list:
    """序列 checkpoint 訓練時用的序列庫路徑（順序照舊），由它自己的 args 推導。

    ⚠️ 不要在呼叫端另外維護「第 i 個變體用哪些庫」的清單。eval_blend 曾經那樣做，
    清單長度與變體數一對不上就全部套第一組——shape／qual 變體被配到 2 視角的庫，
    只是因為後驗剛好都命中快取才沒炸（RESULTS.md §2.29）。
    """
    import os
    paths = [ck_args.get("sequences", "cache/sequences.npy")] + \
        list(ck_args.get("extra_sequences", []) or [])
    return [os.path.join(cache_dir, os.path.basename(p)) for p in paths]
