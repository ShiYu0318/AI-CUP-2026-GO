"""對齊官方評分函數的損失，以及開放集合檢索用的 margin 損失。

競賽二：純 CrossEntropy 不知道等級是有序的——它把「6D 判成 5D」和
「6D 判成 D」視為同樣的錯誤，但官方評分前者給 exp(-1)、後者給 0。
這裡提供兩個修正：指數 soft label 與 CORN ordinal head。

競賽一：官方 baseline 用隨機取樣的 triplet，絕大多數配對早已滿足 margin、
梯度接近 0。這裡提供 batch-hard triplet 與 sub-center ArcFace。
⚠️ ArcFace 的分類頭只在訓練期使用，推論時必須丟棄、只留 embedding
（候選庫是未見身分，分類頭沒有對應的輸出單元）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 競賽二
# --------------------------------------------------------------------------

def exponential_soft_labels(targets: torch.Tensor, n_classes: int,
                            decay: float = 1.0) -> torch.Tensor:
    """q_k ∝ exp(-decay * |k - y|)。

    decay=1.0 時，soft label 的形狀與官方評分函數完全一致
    （相鄰級的權重比例恰為 exp(-1)），等於讓訓練目標就是評分目標。
    """
    device = targets.device
    k = torch.arange(n_classes, device=device).view(1, -1).float()
    d = (k - targets.view(-1, 1).float()).abs()
    q = torch.exp(-decay * d)
    return q / q.sum(dim=1, keepdim=True)


class SoftLabelCrossEntropy(nn.Module):
    """以指數 soft label 取代 one-hot 的交叉熵。"""

    def __init__(self, n_classes: int, decay: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.decay = decay

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        q = exponential_soft_labels(targets, self.n_classes, self.decay)
        return -(q * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


class CornLoss(nn.Module):
    """CORN ordinal loss（arXiv:2111.08851）。

    模型輸出 n_classes-1 個條件 logit，第 k 個代表 P(y > k | y > k-1)。
    每個子任務只用符合條件的樣本訓練，因此能保證 rank 單調一致。
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.shape[1] == self.n_classes - 1
        total = logits.new_zeros(())
        n_used = 0
        for k in range(self.n_classes - 1):
            mask = targets >= k if k > 0 else torch.ones_like(targets, dtype=torch.bool)
            if not mask.any():
                continue
            y = (targets[mask] > k).float()
            total = total + F.binary_cross_entropy_with_logits(logits[mask, k], y)
            n_used += 1
        return total / max(n_used, 1)


def corn_to_proba(logits: torch.Tensor) -> torch.Tensor:
    """把 CORN 的條件 logit 轉成各等級的機率分布。"""
    p_gt = torch.sigmoid(logits)                       # P(y > k | y > k-1)
    cum = torch.cumprod(p_gt, dim=1)                   # P(y > k)
    ones = cum.new_ones(cum.size(0), 1)
    cum = torch.cat([ones, cum], dim=1)                # P(y > -1) = 1
    probs = cum[:, :-1] - cum[:, 1:]
    last = cum[:, -1:]
    return torch.cat([probs, last], dim=1).clamp_min(0)


# --------------------------------------------------------------------------
# 競賽一
# --------------------------------------------------------------------------

class SubCenterArcFace(nn.Module):
    """Sub-center ArcFace（ECCV 2020）+ 可選的 dynamic margin。

    每個身分有 K 個 sub-center，樣本只需靠近任一個即可——
    一位棋手在不同對手、不同時期本來就有多種風格（類內多模態），
    sub-center 正對症。

    dynamic margin：訓練集每人局數從 1 到 1,800 極長尾，
    margin 隨類別樣本數自適應（Landmark Retrieval 2020 1st, arXiv:2009.05132）。
    """

    def __init__(self, embed_dim: int, n_classes: int, k: int = 3,
                 scale: float = 30.0, margin: float = 0.3,
                 class_counts: torch.Tensor | None = None,
                 margin_range: tuple[float, float] = (0.05, 0.5)):
        super().__init__()
        self.n_classes, self.k, self.scale = n_classes, k, scale
        self.weight = nn.Parameter(torch.empty(n_classes * k, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        if class_counts is None:
            m = torch.full((n_classes,), float(margin))
        else:
            c = class_counts.float().clamp_min(1.0)
            r = c.pow(-0.25)
            lo, hi = margin_range
            r = (r - r.min()) / (r.max() - r.min() + 1e-12)
            m = lo + r * (hi - lo)
        self.register_buffer("margin", m)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None):
        x = F.normalize(embeddings, dim=1)
        w = F.normalize(self.weight, dim=1)
        cos = (x @ w.t()).view(-1, self.n_classes, self.k).amax(dim=2)
        if labels is None:
            return cos * self.scale
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        m = self.margin[labels].view(-1, 1)
        theta = torch.acos(cos)
        onehot = F.one_hot(labels, self.n_classes).bool()
        target = torch.cos(theta + m)
        return torch.where(onehot, target, cos) * self.scale


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                            margin: float = 0.3) -> torch.Tensor:
    """Batch-hard triplet（arXiv:1703.07737）。

    每個 anchor 在 batch 內取最難的正例與最難的負例。
    這正是官方 baseline 缺的東西——它均勻隨機挑兩位玩家，
    絕大多數 triplet 早已滿足 margin，梯度接近 0。
    需搭配 P 位玩家 x K 局的 batch 結構使用。
    """
    x = F.normalize(embeddings, dim=1)
    dist = torch.cdist(x, x, p=2)
    same = labels.view(-1, 1) == labels.view(1, -1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)

    pos = dist.masked_fill(~same | eye, float("-inf")).amax(dim=1)
    neg = dist.masked_fill(same, float("inf")).amin(dim=1)
    valid = torch.isfinite(pos) & torch.isfinite(neg)
    if not valid.any():
        return dist.new_zeros(())
    return F.relu(pos[valid] - neg[valid] + margin).mean()

class PairwiseRankLoss(nn.Module):
    """RankNet 式的成對排序損失：**直接優化順序**。

    為什麼需要它：官方分數 = 完全正確 + e^-1 × 相鄰。實測把期望等級軸上的
    切點換成 oracle（直接在評分集擬合）只能再拿 +0.0172，
    而排序若完美可以拿到 0.9962——也就是說
    **分數幾乎全部由「期望等級對真實等級的排序」決定**（Spearman 現為 0.9037，
    見 RESULTS.md §2.10）。

    但現行損失（soft-label CE ＋ CORN）都是**分類**損失：
    它們在乎「這題是不是 4 級」，不在乎「這題該排在那題前面」。
    這裡補上直接的排序項——對批次內每一組真實等級不同的配對 (i, j)，
    要求期望等級的差與真實等級的差同號。

    margin 用真實等級差來縮放：差 3 級的配對，期望等級也該差得更開。
    只有差 1 級的配對才是最難的，給它們的 margin 最小。
    """

    def __init__(self, n_classes: int, margin_scale: float = 0.5):
        super().__init__()
        self.register_buffer("levels", torch.arange(n_classes, dtype=torch.float32))
        self.margin_scale = float(margin_scale)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ev = (torch.softmax(logits, dim=-1) * self.levels).sum(-1)
        dy = targets[:, None].float() - targets[None, :].float()
        mask = dy > 0                                   # 只取 y_i > y_j 的配對
        if not mask.any():
            return logits.sum() * 0.0
        de = ev[:, None] - ev[None, :]
        margin = self.margin_scale * dy
        # softplus(margin - de)：期望等級的差沒拉到 margin 就有損失
        return nn.functional.softplus(margin[mask] - de[mask]).mean()
