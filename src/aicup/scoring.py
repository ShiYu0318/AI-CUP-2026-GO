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
