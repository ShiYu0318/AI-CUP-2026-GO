"""在手工特徵上做度量學習 —— 目前競賽一增益最大的單一改動。

背景：原本的 `PlayerBaseline` 是「標準化 + L2 正規化 + 平均」，
那是**完全沒有學習**的。但訓練集有 6,405 個身分的標籤，
用它們學一個判別式投影，實測把 400 人候選庫的檢索分數從
**0.6547 推到 0.8084（+0.1538）**，而且在 MPS 上只要 1.7 分鐘。

為什麼這麼有效：手工特徵的 168 個維度尺度與資訊量差很多，
餘弦距離等於給每個維度同等權重；而 ArcFace 學到的投影會
把「能區分人」的方向放大、把「只反映棋力或雜訊」的方向壓縮。

輸入只有 168 維，成本遠低於盤面 CNN，兩者可以並存並融合。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmbedder(nn.Module):
    """168 維手工特徵 -> 判別式 embedding。"""

    def __init__(self, in_dim: int, embed_dim: int = 512, hidden: int = 1024,
                 dropout: float = 0.2, layers: int = 2):
        super().__init__()
        blocks, d = [], in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        blocks.append(nn.Linear(d, embed_dim))
        self.body = nn.Sequential(*blocks)
        self.bn_neck = nn.BatchNorm1d(embed_dim)   # Re-ID 標準：度量與分類分作用在 BN 前後
        self.embed_dim = embed_dim

    def forward(self, x):
        e = self.body(x)
        return e, self.bn_neck(e)


class FeatureScaler:
    """特徵標準化。統計量**只能**由訓練資料估計，避免洩漏。"""

    def __init__(self, eps: float = 1e-8):
        self.mu = None
        self.sd = None
        self.eps = eps

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float64) - self.mu) / self.sd).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mu": self.mu, "sd": self.sd, "eps": self.eps}

    def load_state_dict(self, d: dict) -> "FeatureScaler":
        self.mu, self.sd, self.eps = d["mu"], d["sd"], d["eps"]
        return self


@torch.no_grad()
def embed_features(model: FeatureEmbedder, scaler: FeatureScaler, X: np.ndarray,
                   device, batch: int = 8192) -> np.ndarray:
    """把原始手工特徵轉成 L2 正規化的 embedding。"""
    model.eval()
    Z = scaler.transform(X)
    out = []
    for i in range(0, len(Z), batch):
        t = torch.from_numpy(Z[i:i + batch]).to(device)
        e = model(t)[0]
        out.append(F.normalize(e, dim=1).cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros((0, model.embed_dim), np.float32)


def player_template(emb: np.ndarray) -> np.ndarray:
    """多局 embedding -> 玩家向量。L2 -> 平均 -> 再 L2。

    順序很重要：實測「先平均再標準化」會掉 0.26 分（docs/02 §10.8）。
    這裡輸入已是 L2 正規化的 embedding，所以只需平均後再正規化一次。
    """
    if len(emb) == 0:
        return np.zeros(1, dtype=np.float32)
    v = emb.mean(axis=0)
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v).astype(np.float32)


class PKFeatureSampler:
    """P 位玩家 × K 局的 batch 取樣器（batch-hard mining 的前提）。"""

    def __init__(self, labels: np.ndarray, p: int = 64, k: int = 4,
                 min_games: int = 2, seed: int = 0):
        self.p, self.k = p, k
        self.rng = np.random.default_rng(seed)
        by: dict[int, list[int]] = {}
        for i, l in enumerate(labels):
            by.setdefault(int(l), []).append(i)
        self.by = {l: np.asarray(v) for l, v in by.items() if len(v) >= min_games}
        self.keys = list(self.by)
        self.remap = {l: i for i, l in enumerate(self.keys)}

    @property
    def n_classes(self) -> int:
        return len(self.keys)

    def class_counts(self) -> torch.Tensor:
        return torch.tensor([len(self.by[l]) for l in self.keys], dtype=torch.float)

    def sample(self):
        """回傳 (樣本索引, 重新編號後的標籤)。"""
        sel = self.rng.choice(self.keys, min(self.p, len(self.keys)), replace=False)
        idx, lab = [], []
        for l in sel:
            pool = self.by[l]
            take = self.rng.choice(pool, self.k, replace=len(pool) < self.k)
            idx.extend(int(x) for x in take)
            lab.extend([self.remap[l]] * self.k)
        return np.asarray(idx), np.asarray(lab, dtype=np.int64)
