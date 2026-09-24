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

    # ⚠️ fit / transform 都**分塊**做。舊版把整份轉成 float64 再運算：
    # 78 萬局 x 1164 維時 fit 峰值約 14 GB、transform 約 18 GB，
    # 在 24 GB 的機器上一跑就換頁（2026-09-24 度量模型訓練 20 分鐘沒前進）。
    # 逐元素運算分塊與整份結果逐位元相同；fit 的加總順序不同，差在捨入誤差。
    CHUNK = 50_000

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        n = len(X)
        s = np.zeros(X.shape[1], dtype=np.float64)
        for a in range(0, n, self.CHUNK):
            s += np.asarray(X[a:a + self.CHUNK], dtype=np.float64).sum(axis=0)
        self.mu = s / n
        ss = np.zeros_like(s)
        for a in range(0, n, self.CHUNK):
            d = np.asarray(X[a:a + self.CHUNK], dtype=np.float64) - self.mu
            ss += (d * d).sum(axis=0)
        self.sd = np.sqrt(ss / n) + self.eps
        return self

    def transform(self, X: np.ndarray, inplace: bool = False) -> np.ndarray:
        """`inplace=True` 時直接覆寫 X（必須是 float32 陣列），省下一整份的記憶體。"""
        X = np.asarray(X)
        if inplace and X.dtype != np.float32:
            raise ValueError("inplace 只支援 float32 陣列")
        out = X if inplace else np.empty(X.shape, dtype=np.float32)
        for a in range(0, len(X), self.CHUNK):
            b = a + self.CHUNK
            out[a:b] = ((np.asarray(X[a:b], dtype=np.float64) - self.mu)
                        / self.sd).astype(np.float32)
        return out

    def state_dict(self) -> dict:
        return {"mu": self.mu, "sd": self.sd, "eps": self.eps}

    def load_state_dict(self, d: dict) -> "FeatureScaler":
        self.mu, self.sd, self.eps = d["mu"], d["sd"], d["eps"]
        return self


@torch.no_grad()
def embed_features(model: FeatureEmbedder, scaler: FeatureScaler, X: np.ndarray,
                   device, batch: int = 8192) -> np.ndarray:
    """把原始手工特徵轉成 L2 正規化的 embedding。

    ⚠️ 結束時還原**原本**的 train/eval 模式，不要無條件切回 train。
    競賽二曾因為同樣的寫法讓推論帶著 dropout 跑，端到端分數從 0.48 掉到 0.33
    （見 tests/test_eval_mode.py）。
    """
    was_training = model.training
    model.eval()
    Z = scaler.transform(X)
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), batch):
            t = torch.from_numpy(Z[i:i + batch]).to(device)
            e = model(t)[0]
            out.append(F.normalize(e, dim=1).cpu().numpy())
    model.train(was_training)
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


# --------------------------------------------------------------------------
# 特徵子集
# --------------------------------------------------------------------------

def identity_feature_columns(drop_global_position: bool = True) -> list[int]:
    """回傳適合「玩家辨識」的特徵欄位索引。

    實測（3 訓練 seed × 3 評估池 × 5 種等級組成，全為正）：
    **把 55 維的「全局落子分布」(`all_cell*`) 拿掉，平均提升 +0.0161。**

    | 組成 | 168 維 | 113 維（去掉全局分布） |
    |---|---:|---:|
    | uniform | 0.8317 | **0.8513** |
    | natural | 0.8310 | 0.8413 |
    | skewed | 0.7953 | 0.8122 |
    | dan_only | 0.7837 | 0.8080 |
    | kyu_only | 0.8912 | 0.9008 |

    為什麼：**整局的落子分布主要由棋局走勢決定，不是由你決定；
    而開局下哪裡是你的選擇。** 全局分布相對開局分布是冗餘且帶雜訊的。
    """
    from .features import FEATURE_NAMES
    if not drop_global_position:
        return list(range(len(FEATURE_NAMES)))
    return [i for i, n in enumerate(FEATURE_NAMES) if not n.startswith("all_cell")]
