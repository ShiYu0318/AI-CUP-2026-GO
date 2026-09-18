"""Stage-2 重排：第一階段檢索 Top-N，第二階段用 pair 特徵學一個重排器。

為什麼值得做：官方評分是 Top-5 加權 exp(-(r-1))，第 1 名值 1.0、第 2 名只值 0.368。
**名次順序本身就值錢**，而第一階段的餘弦相似度只是單一分數，
沒有利用「這個 query 的相似度分布長什麼樣」「這個候選是不是跟誰都像」等資訊。

參考 [Shopee 2nd place](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238022)：
第一階段檢索候選、第二階段用 GBDT 對 (query, candidate) pair 重排。

⚠️ **訓練資料必須來自 held-out 玩家組成的多個模擬池**，絕不可用測試池。
   public 與 private 兩池互斥且難度不同，在一池調出來的重排器要驗證跨池穩定。
"""

from __future__ import annotations

import numpy as np

TOP_N = 20


def pair_features(sim: np.ndarray, top_n: int = TOP_N,
                  query_sizes: np.ndarray | None = None,
                  gallery_sizes: np.ndarray | None = None,
                  extra_sim: dict[str, np.ndarray] | None = None):
    """由相似度矩陣建立 (query, candidate) 的 pair 特徵。

    sim: (n_query, n_gallery)
    回傳 (X, pair_index)，pair_index[i] = (query_idx, gallery_idx)

    特徵設計的重點是**相對量**而非絕對相似度：
      rank            這個候選在該 query 的第幾名
      sim             原始相似度
      gap_to_top      與該 query 第 1 名的差距 —— 分辨「穩贏」與「難分難捨」
      gap_to_next     與下一名的差距
      z_in_query      在該 query 的相似度分布中的 z 分數
      z_in_gallery    這個候選在**所有 query** 上的相似度 z 分數
                      —— 抓 hubness：跟誰都像的候選應該降權
      margin_ratio    (sim - mean) / (top1 - mean)
    """
    nq, ng = sim.shape
    top_n = min(top_n, ng)
    order = np.argsort(-sim, axis=1)[:, :top_n]

    q_mean = sim.mean(axis=1, keepdims=True)
    q_std = sim.std(axis=1, keepdims=True) + 1e-8
    g_mean = sim.mean(axis=0, keepdims=True)
    g_std = sim.std(axis=0, keepdims=True) + 1e-8
    top1 = sim.max(axis=1, keepdims=True)

    rows, pairs = [], []
    for i in range(nq):
        cand = order[i]
        s = sim[i, cand]
        for r, (j, sv) in enumerate(zip(cand, s)):
            nxt = s[r + 1] if r + 1 < len(s) else s[-1]
            f = [
                float(r),                                   # rank
                float(sv),                                  # sim
                float(top1[i, 0] - sv),                     # gap_to_top
                float(sv - nxt),                            # gap_to_next
                float((sv - q_mean[i, 0]) / q_std[i, 0]),   # z_in_query
                float((sv - g_mean[0, j]) / g_std[0, j]),   # z_in_gallery（hubness）
                float((sv - q_mean[i, 0]) /
                      max(top1[i, 0] - q_mean[i, 0], 1e-8)),  # margin_ratio
                float(q_std[i, 0]),                         # 該 query 的相似度離散度
                float(g_std[0, j]),                         # 該候選的相似度離散度
            ]
            if query_sizes is not None:
                f.append(float(query_sizes[i]))             # 該題的局數（5~20）
            if gallery_sizes is not None:
                f.append(float(gallery_sizes[j]))
            if extra_sim:
                for _, m in sorted(extra_sim.items()):
                    f.append(float(m[i, j]))                # 其他模型的相似度
            rows.append(f)
            pairs.append((i, j))
    return np.asarray(rows, dtype=np.float32), np.asarray(pairs, dtype=np.int64)


def feature_names(query_sizes=None, gallery_sizes=None, extra_sim=None) -> list[str]:
    n = ["rank", "sim", "gap_to_top", "gap_to_next", "z_in_query",
         "z_in_gallery", "margin_ratio", "query_sim_std", "gallery_sim_std"]
    if query_sizes is not None:
        n.append("n_query_games")
    if gallery_sizes is not None:
        n.append("n_gallery_games")
    if extra_sim:
        n += [f"sim_{k}" for k in sorted(extra_sim)]
    return n


def pair_labels(pairs: np.ndarray, true_col: np.ndarray | None = None) -> np.ndarray:
    """pair 是否為正解。預設第 i 個 query 的正解是第 i 個 gallery。"""
    if true_col is None:
        return (pairs[:, 0] == pairs[:, 1]).astype(np.int8)
    return (pairs[:, 1] == true_col[pairs[:, 0]]).astype(np.int8)


def rerank_scores(model, sim: np.ndarray, top_n: int = TOP_N, **kw) -> np.ndarray:
    """用重排器的機率取代 Top-N 內的相似度；Top-N 之外維持原順序且必定排在後面。

    回傳與 sim 同形狀的新分數矩陣（越大越前面）。
    """
    X, pairs = pair_features(sim, top_n=top_n, **kw)
    if len(X) == 0:
        return sim
    p = model.predict_proba(X)[:, 1]
    out = np.full_like(sim, -np.inf, dtype=np.float64)
    # Top-N 之外：用原始相似度壓到一個必定低於重排分數的區間
    lo = sim.min() - 1.0
    out[:] = lo + (sim - sim.min()) * 1e-6
    out[pairs[:, 0], pairs[:, 1]] = p + 1.0      # 重排過的必定排在未重排的前面
    return out
