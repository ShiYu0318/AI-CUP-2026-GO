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


class GlobalPoolBias(nn.Module):
    """KataGo 的 global pooling bias（arXiv:1902.10565）。

    把整張盤面的全域統計（平均與最大）壓成一個向量，再 broadcast 回去當偏置。
    純卷積的感受野是局部的，看不到「整盤形勢」；圍棋的很多判斷（這塊該不該棄、
    現在是該搶大場還是收官）需要全域資訊。KataGo 的 ablation 顯示這個結構
    對學習速度有顯著幫助。
    """

    def __init__(self, channels: int, pooled: int = 32):
        super().__init__()
        self.pool_conv = nn.Conv2d(channels, pooled, 1, bias=False)
        self.bn = nn.BatchNorm2d(pooled)
        self.fc = nn.Linear(pooled * 2, channels)

    def forward(self, x):
        g = F.relu(self.bn(self.pool_conv(x)))
        stats = torch.cat([g.mean(dim=(2, 3)), g.amax(dim=(2, 3))], dim=1)
        return x + self.fc(stats).unsqueeze(-1).unsqueeze(-1)


class ResBlockGPB(nn.Module):
    """帶 global pooling bias 的殘差區塊。"""

    def __init__(self, ch: int, pooled: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.gpb = GlobalPoolBias(ch, pooled)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.gpb(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class BoardTransformerBlock(nn.Module):
    """把 19x19 當成 361 個 token 的 self-attention 區塊（ResTNet 的作法）。

    純卷積的感受野隨深度線性成長，抓不到征子（ladder）、環形（cycle）這類
    需要沿著長路徑推理的 pattern —— 而那正是區分棋力的關鍵。
    [ResTNet, IJCAI 2025](https://arxiv.org/abs/2410.05347) 的做法是把
    residual block 與 transformer block **交錯堆疊**：
    19x19 圍棋勝率 53.6% -> 60.9%，環形 pattern 的 MSE 2.58 -> 1.07。

    361 個 token 的 self-attention 在 19x19 上是可負擔的
    （相較影像的數千 token 小得多）。
    """

    def __init__(self, channels: int, heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio), nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels))
        # 可學的位置編碼：圍棋盤面的角、邊、中腹性質差很多，位置資訊必須保留
        self.pos = nn.Parameter(torch.zeros(1, 19 * 19, channels))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2) + self.pos          # (B, 361, C)
        h = self.norm1(t)
        qkv = self.qkv(h).view(B, H * W, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, H * W, C)
        t = t + self.proj(a)
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1, 2).view(B, C, H, W)


class MAB(nn.Module):
    """Multihead Attention Block（Set Transformer, arXiv:1810.00825）。"""

    def __init__(self, dim_q: int, dim_kv: int, dim: int, heads: int = 4):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.q = nn.Linear(dim_q, dim)
        self.k = nn.Linear(dim_kv, dim)
        self.v = nn.Linear(dim_kv, dim)
        self.o = nn.Linear(dim, dim)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def forward(self, q, kv, mask=None):
        B, Nq, _ = q.shape
        Nk = kv.shape[1]
        h = self.heads
        Q = self.q(q).view(B, Nq, h, -1).transpose(1, 2)
        K = self.k(kv).view(B, Nk, h, -1).transpose(1, 2)
        V = self.v(kv).view(B, Nk, h, -1).transpose(1, 2)
        attn_mask = None
        if mask is not None:
            attn_mask = mask.view(B, 1, 1, Nk).expand(B, h, Nq, Nk)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, Nq, self.dim)
        out = self.ln0(self.q(q) + self.o(out))
        return self.ln1(out + self.ff(out))


class SetTransformerPool(nn.Module):
    """ISAB + PMA 的集合聚合（arXiv:1810.00825）。

    相對於簡單平均或 gated attention，它讓集合內的元素**互相參照**
    （這局的判斷可以參考其他局），並用可學的 seed 向量做池化。

    圍棋棋力估計的現行 SOTA（ICLR 2025, arXiv:2502.17109）明確採用
    inducing-point attention，讓模型自動聚焦在數百手中真正洩漏棋力的少數關鍵手。
    """

    def __init__(self, dim: int, heads: int = 4, n_inducing: int = 16, n_seeds: int = 1):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, dim) * 0.02)
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, dim) * 0.02)
        self.mab0 = MAB(dim, dim, dim, heads)      # inducing <- set
        self.mab1 = MAB(dim, dim, dim, heads)      # set <- inducing
        self.pma = MAB(dim, dim, dim, heads)       # seeds <- set
        self.n_seeds = n_seeds

    def forward(self, x, mask=None):
        """x: (B, N, D)；mask: (B, N)，True 表示有效。回傳 (B, D)。"""
        B = x.shape[0]
        ind = self.inducing.expand(B, -1, -1)
        h = self.mab0(ind, x, mask)                # 先把集合壓到 m 個 inducing point
        h = self.mab1(x, h)                        # 再把資訊散回每個元素
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        pooled = self.pma(self.seeds.expand(B, -1, -1), h, mask)
        return pooled.mean(dim=1)


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
                 embed_dim: int = 512, global_pool_bias: bool = True,
                 gpb_every: int = 2, transformer_every: int = 0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        # 每 gpb_every 個區塊插一個帶 global pooling bias 的版本（KataGo 的做法），
        # 每 transformer_every 個區塊插一個盤面 self-attention（ResTNet 的做法）
        layers = []
        for i in range(blocks):
            if global_pool_bias and (i + 1) % gpb_every == 0:
                layers.append(ResBlockGPB(width))
            else:
                layers.append(ResBlock(width))
            if transformer_every and (i + 1) % transformer_every == 0:
                layers.append(BoardTransformerBlock(width))
        self.blocks = nn.Sequential(*layers)
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
                 embed_dim: int = 512, n_levels: int = 10, n_quality: int = 0,
                 pool: str = "set_transformer", global_pool_bias: bool = True,
                 transformer_every: int = 0):
        super().__init__()
        self.trunk = GoTrunk(in_channels, width, blocks, embed_dim,
                             global_pool_bias=global_pool_bias,
                             transformer_every=transformer_every)
        self.rank_head = nn.Linear(embed_dim, n_levels)
        self.corn_head = nn.Linear(embed_dim, n_levels - 1)
        # 集合聚合：set_transformer 讓集合內元素互相參照（ICLR 2025 圍棋棋力 SOTA 的做法）；
        # attention 是較輕量的 gated MIL pooling
        self.pool_kind = pool
        self.set_pool = (SetTransformerPool(embed_dim) if pool == "set_transformer"
                         else AttentionPool(embed_dim))
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
        if self.pool_kind == "set_transformer":
            pooled = self.set_pool(emb, mask)
            attn = torch.zeros(B, N, device=emb.device)
        else:
            pooled, attn = self.set_pool(emb, mask)
        return {
            "set_embedding": pooled,
            "attention": attn,
            "rank_logits": self.rank_head(self.trunk.bn_neck(pooled)),
            "corn_logits": self.corn_head(self.trunk.bn_neck(pooled)),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
