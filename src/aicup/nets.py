"""共用 backbone 的多任務網路。

一個 backbone 同時服務兩個子題（docs/02 §5.1）：
  - 玩家辨識 head：8,007 類 sub-center ArcFace（僅訓練期使用）
  - 棋力預測 head：softmax + CORN 雙路

為什麼共用：實測 player_id 與 rank 一對一（8,007 位玩家各只出現在單一等級檔）。
棋力是玩家辨識的主要混淆因子，顯式建模能讓玩家 embedding 更乾淨；
反過來，8,007 類的監督訊號遠比 10 類密集，對標籤稀疏的棋力任務是強正則化。

相對官方 baseline 的架構修正：
  - 全域池化改用 GeM（可學的 p，介於 max 與 average 之間）
  - 加入 BN-Neck（Re-ID 標準做法：分類損失與度量損失分別作用在 BN 前後）
  - 集合層級用 attention pooling，而非官方的簡單平均
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class GeM(nn.Module):
    """Generalized-Mean pooling（arXiv:1711.02512）。p 可學，介於 avg(p=1) 與 max(p→∞)。"""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        x = x.clamp_min(self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p).flatten(1)


class GoTrunk(nn.Module):
    """盤面 -> 向量。輸入 (B, C, 19, 19)。"""

    def __init__(self, in_channels: int = 34, width: int = 128, blocks: int = 10,
                 embed_dim: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.pool = GeM()
        self.proj = nn.Linear(width, embed_dim)
        self.bn_neck = nn.BatchNorm1d(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        h = self.blocks(self.stem(x))
        h = self.proj(self.pool(h))
        return h, self.bn_neck(h)          # (度量用, 分類用)


class AttentionPool(nn.Module):
    """把可變數量的局向量聚合成單一向量（gated attention MIL, arXiv:1802.04712）。

    不同棋局的資訊量差很多（短局、投降局 vs 長局），
    attention 能自動降權低資訊量的棋局，比固定平均好。
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x, mask=None):
        """x: (B, N, D)，mask: (B, N) 為 True 表示有效。"""
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return (w * x).sum(dim=1), w.squeeze(-1)


class MultiTaskGoNet(nn.Module):
    """共用 backbone，兩個子題各一組 head。

    rank_logits  10 類 softmax（搭配指數 soft label）
    corn_logits  9 個條件輸出（保證 rank 單調一致）
    embedding    L2 正規化前的度量向量，交給 SubCenterArcFace 訓練
    """

    def __init__(self, in_channels: int = 34, width: int = 128, blocks: int = 10,
                 embed_dim: int = 512, n_levels: int = 10, n_quality: int = 0):
        super().__init__()
        self.trunk = GoTrunk(in_channels, width, blocks, embed_dim)
        self.rank_head = nn.Linear(embed_dim, n_levels)
        self.corn_head = nn.Linear(embed_dim, n_levels - 1)
        self.set_pool = AttentionPool(embed_dim)
        # 棋形品質輔助 head：顯式回歸「愚形率、棄子率、官子填地率…」等
        # 已知與棋力高度相關的量（愚形的相關係數達 -0.964）。
        # 棋力標籤一局只有 1 個、訊號稀疏；這些連續量提供密集的監督訊號，
        # 思路同 KataGo 的 ownership/score 輔助目標（arXiv:1902.10565）。
        self.quality_head = nn.Linear(embed_dim, n_quality) if n_quality else None

    def forward(self, x):
        metric, cls = self.trunk(x)
        out = {
            "embedding": metric,
            "rank_logits": self.rank_head(cls),
            "corn_logits": self.corn_head(cls),
        }
        if self.quality_head is not None:
            out["quality"] = self.quality_head(cls)
        return out

    def forward_set(self, x, mask=None):
        """x: (B, N, C, 19, 19) —— 一題的 N 局棋，聚合後再出題目層級預測。"""
        B, N = x.shape[:2]
        out = self.forward(x.flatten(0, 1))
        emb = out["embedding"].view(B, N, -1)
        pooled, attn = self.set_pool(emb, mask)
        return {
            "set_embedding": pooled,
            "attention": attn,
            "rank_logits": self.rank_head(self.trunk.bn_neck(pooled)),
            "corn_logits": self.corn_head(self.trunk.bn_neck(pooled)),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
