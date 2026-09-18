"""檢索後處理：k-reciprocal re-ranking 與 query expansion。

這兩招在人物 Re-ID 上是公認的免費增益，但 ⚠️ **本任務尚未證實有效**。

原因值得說清楚：k-reciprocal re-ranking 的原理是利用
「同一身分在 gallery 中有多個條目、彼此互為近鄰」這個結構。
人物 Re-ID 的 gallery 每個人有數十張影像，結構豐富；
**本競賽是 400 位候選各聚合成單一 template 的 1-對-1 匹配，
沒有這個結構可利用**。

實測（合成資料，100 身分 × 32 維）：

    每身分 gallery 條目數    1 個        3 個      5 個      10 個
    k-recip 相對原始的變化   -0.004    +0.013    -0.011    +0.013

差異全部落在雜訊範圍內，看不出穩定增益。

**因此**：
  - 這個模組提供正確的實作，但**啟用前必須在真實 embedding 與模擬候選庫上驗證**
  - 若要讓 k-reciprocal 有發揮空間，可考慮**每位候選保留多個子 template**
    （例如 100 局切成 5 組各 20 局），讓 gallery 具備 5-per-identity 的結構
  - 超參（k1 / k2 / lambda / alpha）必須在**多個由 held-out 玩家組成的模擬候選庫**
    上調，不可在測試池上調——public 與 private 兩池互斥且難度不同

αQE 的風險較低（只是加權擴充 query 向量），但同樣要驗證。
"""

from __future__ import annotations

import numpy as np


def _to_distance(sim: np.ndarray) -> np.ndarray:
    """餘弦相似度 -> 距離。輸入已 L2 正規化時，這是單調轉換。"""
    return 1.0 - np.asarray(sim, dtype=np.float64)


def alpha_query_expansion(query: np.ndarray, gallery: np.ndarray,
                          top_k: int = 3, alpha: float = 3.0) -> np.ndarray:
    """αQE（arXiv:1711.02512）：用最相似的前 k 個 gallery 向量加權擴充 query。

    權重為相似度的 alpha 次方，讓高相似度的鄰居主導擴充，
    避免把不相關的候選混進來。
    """
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    sim = q @ g.T
    idx = np.argsort(-sim, axis=1)[:, :top_k]

    out = np.empty_like(q)
    for i in range(len(q)):
        nb = g[idx[i]]
        w = np.clip(sim[i, idx[i]], 0.0, None) ** alpha
        expanded = q[i] + (w[:, None] * nb).sum(axis=0)
        n = np.linalg.norm(expanded)
        out[i] = expanded / (n if n > 0 else 1.0)
    return out


def _k_reciprocal_set(rank_all: np.ndarray, i: int, k: int) -> np.ndarray:
    """i 的 k-reciprocal 近鄰：i 的前 k 名中，也把 i 收進自己前 k 名的那些。"""
    forward = rank_all[i, : k + 1]
    return np.asarray([j for j in forward if i in rank_all[j, : k + 1]], dtype=np.int64)


def k_reciprocal_rerank(query_emb: np.ndarray, gallery_emb: np.ndarray,
                        k1: int = 20, k2: int = 6, lambda_value: float = 0.3) -> np.ndarray:
    """k-reciprocal re-ranking（Zhong et al., arXiv:1701.08398）。

    把原始距離與 Jaccard 距離加權融合：
        d_final = (1 - lambda) * d_jaccard + lambda * d_original

    ⚠️ query 與 gallery 必須放進同一個 probe 集合計算完整的距離矩陣——
    只給 query-gallery 的相似度是不夠的，因為 k-reciprocal 需要
    query 彼此之間、以及 gallery 彼此之間的鄰居關係。

    回傳與 (n_query, n_gallery) 同形狀的「新相似度」（越大越相似）。
    """
    q = np.asarray(query_emb, dtype=np.float64)
    g = np.asarray(gallery_emb, dtype=np.float64)
    nq, ng = len(q), len(g)
    feats = np.concatenate([q, g], axis=0)

    d = _to_distance(feats @ feats.T)
    np.fill_diagonal(d, 0.0)
    d = d / np.maximum(d.max(axis=0, keepdims=True), 1e-12)   # 逐列正規化，同原論文實作

    n = nq + ng
    rank_all = np.argsort(d, axis=1)

    V = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        kset = _k_reciprocal_set(rank_all, i, k1)
        expanded = list(kset)
        for j in kset:                       # 擴充：重疊夠多的鄰居的鄰居也納入
            sub = _k_reciprocal_set(rank_all, int(j), int(round(k1 / 2)))
            if len(np.intersect1d(sub, kset)) > 2.0 / 3 * len(sub):
                expanded.extend(sub.tolist())
        expanded = np.unique(np.asarray(expanded, dtype=np.int64))
        w = np.exp(-d[i, expanded])
        V[i, expanded] = (w / w.sum()).astype(np.float32)

    if k2 > 1:                               # local query expansion：前 k2 個鄰居的 V 取平均
        V = np.stack([V[rank_all[i, :k2]].mean(axis=0) for i in range(n)])

    jaccard = np.empty((nq, ng), dtype=np.float64)
    Vg = V[nq:]
    for i in range(nq):
        vi = V[i][None, :]
        mn = np.minimum(vi, Vg).sum(axis=1)
        mx = np.maximum(vi, Vg).sum(axis=1)
        jaccard[i] = 1.0 - mn / np.maximum(mx, 1e-12)

    final = (1.0 - lambda_value) * jaccard + lambda_value * d[:nq, nq:]
    return -final


def rerank_similarity(query_emb: np.ndarray, gallery_emb: np.ndarray, *,
                      use_qe: bool = True, qe_top_k: int = 3, qe_alpha: float = 3.0,
                      use_kr: bool = True, k1: int = 20, k2: int = 6,
                      lambda_value: float = 0.3) -> np.ndarray:
    """完整的後處理：αQE -> k-reciprocal re-ranking -> 回傳新的相似度矩陣。"""
    q = np.asarray(query_emb, dtype=np.float64)
    g = np.asarray(gallery_emb, dtype=np.float64)
    if use_qe:
        q = alpha_query_expansion(q, g, top_k=qe_top_k, alpha=qe_alpha)
    if not use_kr:
        return q @ g.T
    return k_reciprocal_rerank(q, g, k1=k1, k2=k2, lambda_value=lambda_value)
