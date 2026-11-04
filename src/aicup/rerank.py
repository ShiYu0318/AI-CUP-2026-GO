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


# ---------------------------------------------------------------------------
# 全域指派：Sinkhorn 平衡
# ---------------------------------------------------------------------------

def sinkhorn_assignment(sim: np.ndarray, eps: float = 0.0075,
                        tol: float = 1e-3, max_iters: int = 2000) -> np.ndarray:
    """把相似度矩陣做 Sinkhorn 正規化（近似雙隨機），回傳新的排序分數。

    **為什麼有用**：官方規格是每池 400 題、400 位候選。
    每題獨立取 argmax 完全沒用到這個全域約束——
    一位「長得像很多人」的候選（hub）可以同時被多題選為第一名，
    而真正的答案卻被擠掉。Sinkhorn 強制每位候選的總指派量大致相等，
    等於自動做 hubness 校正。

    實測（15 次評估，5 種組成 × 3 seed）：
        獨立 argmax     0.9233 ±0.0227
        Sinkhorn        0.9552 ±0.0196   +0.0319（15/15 勝）

    ⚠️ **eps 必須用最壞情境挑，不能用最好情境。**
    官方沒有明說「每位候選正好是一題的答案」。把假設違反的情境也測過：

        eps      一一對應   三成干擾項   兩成五答案重複   最差
        0.003    +0.0534    -0.0039      +0.0033        -0.0039
        0.0075   +0.0442    +0.0033      +0.0100        +0.0033  ← 唯一全正
        0.02     +0.0006    -0.0147      -0.0133        -0.0147
        匈牙利    +0.0438    +0.0191      -0.0964        -0.0964  ← 會崩

    所以預設 **eps=0.0075**：在一一對應成立時與匈牙利算法持平（+0.0442 vs +0.0438），
    但假設錯了也不會變差。匈牙利**強制**一一對應，答案有重複時會把正解硬推給別題。

    ⚠️ **另一個邊界：增益取決於基礎準確率。** 合成實驗（400 身分、5 seed）：

        基礎 Top-1   1.00    0.79      0.40      0.21      0.13
        Sinkhorn     +0.000  +0.0830   +0.0171   -0.0096   -0.0148

    基礎 Top-1 ≳ 0.4 才有益，約 0.8 時增益最大，低於 0.3 反而有害——
    模型太弱時，強制平衡只是把錯誤重新分配。
    本專案實際 Top-1 為 0.906，落在安全區間。
    **若哪天第一階段檢索退化到 Top-1 < 0.4，要把這個後處理關掉**
    （`bijection_consistency` 可當作現場的代理指標：
    準確率低時相異率也會明顯偏低）。

    eps 小的時候收斂很慢，所以用**收斂條件**停止而不是固定迭代次數。
    不過排序遠比矩陣本身早穩定——實測 400×400、eps=0.0075：

        迭代     50       100      200      500      1000
        列和誤差 9.7e-2   2.9e-2   1.5e-2   3.6e-3   1.8e-3
        top-1 變動  —      2 題     1 題     0 題     0 題

    200 次之後名次就完全不動了，分數波動 ±0.002 在雜訊內。

    sim:  (n_query, n_gallery) 相似度，越大越像
    回傳: 同形狀的分數矩陣，逐列排序即得名次
    """
    sim = np.asarray(sim, dtype=np.float64)
    if sim.ndim != 2:
        raise ValueError(f"sim 必須是二維矩陣，收到 {sim.shape}")
    if eps <= 0:
        raise ValueError("eps 必須為正；eps → 0 會退化成硬指派（見上方風險表）")
    K = np.exp((sim - sim.max()) / eps)
    u = np.ones(K.shape[0])
    v = np.ones(K.shape[1])
    target = K.shape[1] / K.shape[0]          # 每列該分到的總量
    for _ in range(max_iters):
        u = 1.0 / np.maximum(K @ v, 1e-300)
        v = 1.0 / np.maximum(K.T @ u, 1e-300)
        rs = u * (K @ v)
        if np.abs(rs - target).max() <= tol * target:
            break
    return u[:, None] * K * v[None, :]


def bijection_consistency(sim: np.ndarray) -> float:
    """診斷：獨立 argmax 的 top-1 有多接近一個排列（1.0 = 完全相異）。

    一一對應若成立，top-1 預測應該接近一個排列，重複只來自模型誤差。
    比賽當天可以拿真實測試集的這個值與模擬測試集比較：
    明顯偏低就代表「每位候選正好是一題的答案」不成立。
    """
    top1 = np.asarray(sim).argmax(axis=1)
    return len(np.unique(top1)) / len(top1)


def pool_classifier_scores(gallery_emb: np.ndarray, gallery_label: np.ndarray, query_embs: list,
                           n_classes: int, device="cpu", iters: int = 300, wd: float = 1e-4) -> np.ndarray:
    """用候選池本身訓練 n_classes 類的多項式邏輯迴歸，回傳 (題數, 類別數) 的分數。

    每位候選人的 100 盤就是現成的帶標籤訓練資料（官方 Q&A Q10：候選池可用於校正、正規化等用途）。
    模板餘弦只用到每位候選人的「平均」，分類器還用到了他棋局分布的形狀與彼此的邊界。
    每題的分數 = 該題各盤 log 機率的平均。接 `sinkhorn_assignment(eps≈0.05)`。

    實測（4 seed x 5 種組成、20 池逐池配對）：模板餘弦＋Sinkhorn 0.9945 →
    分類器＋Sinkhorn 0.05 **0.9969**（9 勝 9 平 2 負），最壞組成 0.9825 → 0.9904（RESULTS.md §2.35）。
    """
    import torch
    X = torch.from_numpy(np.asarray(gallery_emb, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(gallery_label, dtype=np.int64)).to(device)
    W = torch.zeros(X.shape[1], n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        out = []
        for E in query_embs:
            if len(E) == 0:
                out.append(np.zeros(n_classes)); continue
            q = torch.from_numpy(np.asarray(E, dtype=np.float32)).to(device)
            out.append(torch.log_softmax(q @ W + b, -1).mean(0).cpu().numpy())
    return np.stack(out)
