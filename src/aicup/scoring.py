"""官方評分函數，以及對齊評分函數的解碼策略。

兩個子題的官方公式（來源：AIdea 競賽頁與官方 tutorial README）：

競賽二（棋力預測）
    score_i = 1            完全正確
            = exp(-1)      相鄰一級（且非完全正確）
            = 0            其餘
    相鄰的定義是 LEVELS 順序上的索引相鄰。

競賽一（玩家辨識）
    score_i = exp(-(r-1))  正解位於 Top-5 的第 r 名（r 為 1-based）
            = 0            不在 Top-5

最終分數皆為所有題目的算術平均。
"""

from __future__ import annotations

import numpy as np

from . import N_LEVELS

E_INV = float(np.exp(-1.0))
TOP_K = 5


# --------------------------------------------------------------------------
# 競賽二：棋力預測
# --------------------------------------------------------------------------

def rank_score_matrix(n_levels: int = N_LEVELS) -> np.ndarray:
    """S[c, k] = 預測等級 c、真實等級 k 時的官方得分。"""
    idx = np.arange(n_levels)
    d = np.abs(idx[:, None] - idx[None, :])
    return np.where(d == 0, 1.0, np.where(d == 1, E_INV, 0.0))


def rank_score(pred_idx, true_idx) -> np.ndarray:
    """逐題計算競賽二的官方得分。輸入為等級索引（0..9）。"""
    pred_idx = np.asarray(pred_idx)
    true_idx = np.asarray(true_idx)
    d = np.abs(pred_idx - true_idx)
    return np.where(d == 0, 1.0, np.where(d == 1, E_INV, 0.0))


def rank_final_score(pred_idx, true_idx) -> float:
    """競賽二的最終分數（所有題目平均）。"""
    return float(np.mean(rank_score(pred_idx, true_idx)))


def decode_expected_score(probs: np.ndarray, score_matrix: np.ndarray | None = None) -> np.ndarray:
    """期望分數解碼：argmax_c sum_k p_k * S(c, k)。

    這是給定後驗機率下，能最大化官方分數期望值的解碼方式，
    與 argmax(probs) 不同——後者忽略了「相鄰一級可得 exp(-1) 部分分」。

    probs: (..., n_levels) 的後驗機率
    回傳:  (...) 的預測索引
    """
    probs = np.asarray(probs, dtype=np.float64)
    S = rank_score_matrix(probs.shape[-1]) if score_matrix is None else score_matrix
    # expected[..., c] = sum_k probs[..., k] * S[c, k]
    expected = probs @ S.T
    return expected.argmax(axis=-1)


def decode_argmax(probs: np.ndarray) -> np.ndarray:
    """基準解碼：直接取機率最大的等級（官方 baseline 的做法）。"""
    return np.asarray(probs).argmax(axis=-1)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """溫度縮放後轉成機率。temperature > 1 會讓分布更平緩。"""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(logits: np.ndarray, true_idx, grid=None) -> float:
    """在驗證集上挑一個讓「期望分數解碼」得分最高的溫度。

    注意：必須在 player-disjoint 的驗證題目上做，不可用測試集或 Public LB。
    """
    if grid is None:
        grid = np.concatenate([np.linspace(0.5, 3.0, 26), np.linspace(3.5, 10.0, 14)])
    true_idx = np.asarray(true_idx)
    best_t, best_s = 1.0, -1.0
    for t in grid:
        p = apply_temperature(logits, float(t))
        s = rank_final_score(decode_expected_score(p), true_idx)
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t


# --------------------------------------------------------------------------
# 競賽一：玩家辨識
# --------------------------------------------------------------------------

def player_score_from_rank(zero_based_rank) -> np.ndarray:
    """由「正解的 0-based 名次」計算競賽一的官方得分。

    0-based 名次 0 => 官方 r=1 => 得 1.0
    名次 >= TOP_K 視為落榜，得 0。
    """
    r = np.asarray(zero_based_rank)
    return np.where(r < TOP_K, np.exp(-r.astype(np.float64)), 0.0)


def player_final_score(zero_based_rank) -> float:
    """競賽一的最終分數（所有題目平均）。"""
    return float(np.mean(player_score_from_rank(zero_based_rank)))


def ranks_from_similarity(sim: np.ndarray, true_col: np.ndarray | None = None) -> np.ndarray:
    """由相似度矩陣求出正解的 0-based 名次。

    sim:       (n_query, n_gallery)，值越大越相似
    true_col:  每個 query 的正解在 gallery 中的欄位索引；
               預設為 arange(n_query)（第 i 個 query 對應第 i 個 gallery）
    """
    sim = np.asarray(sim)
    n = sim.shape[0]
    if true_col is None:
        true_col = np.arange(n)
    order = np.argsort(-sim, axis=1)
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        ranks[i] = int(np.where(order[i] == true_col[i])[0][0])
    return ranks


def top_k_from_similarity(sim: np.ndarray, gallery_names, k: int = TOP_K):
    """由相似度矩陣取出每個 query 的 Top-k 候選名稱（保證相異）。"""
    sim = np.asarray(sim)
    order = np.argsort(-sim, axis=1)[:, :k]
    names = np.asarray(gallery_names, dtype=object)
    return names[order]


def retrieval_report(sim: np.ndarray, true_col: np.ndarray | None = None) -> dict:
    """一次回報競賽一的常用指標。"""
    ranks = ranks_from_similarity(sim, true_col)
    return {
        "official_score": player_final_score(ranks),
        "top1": float((ranks == 0).mean()),
        "top5": float((ranks < TOP_K).mean()),
        "median_rank": float(np.median(ranks) + 1),
        "n_query": int(len(ranks)),
    }


# --------------------------------------------------------------------------
# 序數回歸：把棋力當連續量預測，再最佳化切點
# --------------------------------------------------------------------------

def optimise_thresholds(scores, true_idx, n_levels: int = N_LEVELS,
                        n_grid: int = 400, **_legacy) -> np.ndarray:
    """在校準集上求出讓**官方分數**最大的 n_levels-1 個切點。**用動態規劃，精確且確定性。**

    ⚠️ 這裡刻意不用 Nelder-Mead 之類的數值最佳化。
    目標函數對切點是**分段常數**的（切點在兩個相鄰樣本之間移動時分數不變），
    梯度處處為零，Nelder-Mead 在上面本質是隨機遊走：
    輸入最後幾個 bit 的差異就會走到完全不同的解。
    實測用 Nelder-Mead 時，同一個模型在同一份資料上重跑，
    分數會在 0.20 ~ 0.48 之間跳動，而且切點會退化（兩個切點重合、
    導致某些等級幾乎分不到任何題目）。

    正確的做法是把它看成「把排序後的分數軸切成 n_levels 段」的分割問題，
    用 DP 求全域最佳解：
        best[k][i] = 前 i 個樣本切成 k 段的最大總分
    候選切點限制在 n_grid 個分位數上，複雜度 O(n_grid^2 * n_levels)。

    scores: 連續預測值（越大代表棋力越強）
    回傳：長度 n_levels-1 的遞增切點
    """
    scores = np.asarray(scores, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    n = len(scores)
    if n == 0:
        return np.arange(1, n_levels, dtype=np.float64)

    order = np.argsort(scores, kind="stable")
    y = true_idx[order]
    s = scores[order]

    # 候選切點：分位數格點（含 0 與 n，允許空的區段）
    cuts = np.unique(np.linspace(0, n, min(n_grid, n) + 1).astype(np.int64))
    m = len(cuts)

    # gain[c][a][b] = 把排序後第 cuts[a]..cuts[b] 個樣本都判成等級 c 的總分
    S = rank_score_matrix(n_levels)                      # (pred, true)
    per_class = S[:, y]                                  # (n_levels, n)
    csum = np.concatenate([np.zeros((n_levels, 1)), np.cumsum(per_class, axis=1)], axis=1)

    NEG = -1e18
    best = np.full((n_levels + 1, m), NEG)
    back = np.zeros((n_levels + 1, m), dtype=np.int64)
    best[0, 0] = 0.0
    for k in range(1, n_levels + 1):
        # gains[a, b] = csum[k-1, cuts[b]] - csum[k-1, cuts[a]]
        g = csum[k - 1, cuts][None, :] - csum[k - 1, cuts][:, None]
        cand = best[k - 1][:, None] + g                  # (a, b)
        cand = np.where(np.arange(m)[:, None] <= np.arange(m)[None, :], cand, NEG)
        best[k] = cand.max(axis=0)
        back[k] = cand.argmax(axis=0)

    # 回溯出切點位置
    idx = m - 1
    pos = []
    for k in range(n_levels, 0, -1):
        a = back[k, idx]
        pos.append(cuts[a])
        idx = a
    pos = sorted(pos)[1:]                                # 去掉起點 0

    # 把「樣本位置」轉成實際的切點數值（取相鄰樣本的中點）
    th = []
    for p in pos:
        if p <= 0:
            th.append(s[0] - 1e-6)
        elif p >= n:
            th.append(s[-1] + 1e-6)
        else:
            th.append(0.5 * (s[p - 1] + s[p]))
    th = np.asarray(th, dtype=np.float64)
    if len(th) < n_levels - 1:                           # 保險：補滿
        th = np.concatenate([th, np.full(n_levels - 1 - len(th), s[-1] + 1e-6)])
    return np.sort(th[: n_levels - 1])


def apply_thresholds(scores, thresholds) -> np.ndarray:
    """用切點把連續分數轉成等級索引。"""
    return np.searchsorted(np.sort(np.asarray(thresholds)),
                           np.asarray(scores, dtype=np.float64))


def expected_level(probs: np.ndarray) -> np.ndarray:
    """由後驗機率算期望等級（連續值）。可餵給 optimise_thresholds。"""
    probs = np.asarray(probs, dtype=np.float64)
    return probs @ np.arange(probs.shape[-1], dtype=np.float64)

def estimate_prior_em(probs, source_prior=None, iters: int = 100,
                      tol: float = 1e-7, floor: float = 1e-3,
                      weights=None, shrink: float = 0.0):
    """由**未標註**的測試後驗估計測試集的類別先驗（Saerens–Latinne–Decaestecker EM）。

    為什麼需要：官方測試集是 400 位玩家各一題，它的等級分布是**玩家層級**的，
    而我們的驗證題目用的是驗證玩家的自然分布（rank 3 佔 19.8%、rank 4 佔 5.4%）。
    實測若真實測試集是等級平衡的，同一套系統會從 0.5703 掉到 0.5331——
    **落差全部來自先驗不匹配，與模型好壞無關**（見 RESULTS.md 的先驗穩健性一節）。

    固定押注（一律用自然、或一律用均勻）在最壞情況下差 0.010，
    而且賭錯的代價比賭對的收益大。EM 不用賭：它直接從測試後驗把先驗估出來。

    ⚠️ 這是 transductive 的，只用到測試輸入的**後驗**，沒有用到任何標籤。

    floor 是估計先驗的下限。EM 可以把某個等級推到 0，之後除法會爆掉，
    而且真實測試集裡某個等級剛好一題都沒有的機率極低。

    shrink 把估計往 source_prior 拉回 shrink 的比例。官方測試集只有 400 題，
    EM 在這個樣本量下對稀有等級會過衝（合成檢查裡真值 0.02 被估成 0.001），
    收縮是對「估計本身有雜訊」最直接的保險。

    weights 是每一題的權重，給重要性加權的評估用——
    評估時測試先驗是用權重模擬的，EM 也必須看同一個加權分布才一致。
    """
    P = np.asarray(probs, dtype=np.float64)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)
    n_levels = P.shape[1]
    src = (np.full(n_levels, 1.0 / n_levels) if source_prior is None
           else np.asarray(source_prior, dtype=np.float64))
    src = np.maximum(src, floor); src /= src.sum()

    sw = (np.ones(len(P)) if weights is None
          else np.asarray(weights, dtype=np.float64))
    sw = sw / max(sw.sum(), 1e-300)

    pri = src.copy()
    for _ in range(iters):
        w = P * (pri / src)[None, :]
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
        new = np.maximum(sw @ w, floor)
        new /= new.sum()
        done = np.abs(new - pri).max() < tol
        pri = new
        if done:
            break
    if shrink > 0:
        pri = (1 - shrink) * pri + shrink * src
        pri /= pri.sum()
    return pri
