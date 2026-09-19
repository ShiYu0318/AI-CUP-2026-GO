"""題目層級的直接建模：先聚合特徵，再訓練一個模型。

與「單局模型 -> 聚合後驗」是兩種不同的建模方式：

    A（單局）    每局各自預測 -> 幾何平均後驗
    B（題目層級）把 5~20 局的特徵聚合成一個向量 -> 直接預測

B 能用到 A 丟掉的資訊：**同一位玩家多局之間的變異與一致性**。
一個穩定下出 3D 水準的人，和一個在 1D~5D 之間震盪的人，
平均後驗可能一樣，但後者的 std 大得多。

實測（4 組題目、589 維特徵）：

| 做法 | 分數 |
|---|---:|
| A 單局 -> 聚合 | 0.4566 ±0.0021 |
| B 題目層級 | 0.4584 ±0.0060 |
| **A + B 融合 (w=0.3)** | **0.4841 ±0.0080** |

融合比兩者單獨都高 +0.027，而官方 baseline 是 0.4631。
"""

from __future__ import annotations

import numpy as np

# 聚合統計量。刻意包含 std 與百分位數——那正是單局後驗丟掉的資訊。
AGGREGATORS = ("mean", "std", "p25", "p75")


def aggregate_question(feats: np.ndarray) -> np.ndarray:
    """把一題的 (n_games, n_features) 聚合成單一向量。"""
    if len(feats) == 0:
        return None
    parts = [feats.mean(axis=0), feats.std(axis=0),
             np.percentile(feats, 25, axis=0), np.percentile(feats, 75, axis=0),
             np.array([len(feats)], dtype=np.float64)]
    return np.concatenate(parts).astype(np.float32)


def question_feature_dim(n_features: int) -> int:
    return n_features * len(AGGREGATORS) + 1


def build_question_features(X, ok, game_rows_list) -> np.ndarray:
    """對一批題目建立聚合特徵矩陣。"""
    dim = None
    out = []
    for rows in game_rows_list:
        sel = rows[ok[rows]]
        v = aggregate_question(np.asarray(X[sel])) if len(sel) else None
        out.append(v)
        if v is not None and dim is None:
            dim = len(v)
    if dim is None:
        raise ValueError("所有題目都沒有可用的棋局")
    return np.stack([v if v is not None else np.zeros(dim, np.float32) for v in out])


def blend_log_posteriors(log_game: np.ndarray, log_question: np.ndarray,
                         w: float = 0.3) -> np.ndarray:
    """幾何平均融合兩種後驗。w 是題目層級模型的權重。

    實測最佳 w ≈ 0.3（單局模型佔比較高）。w 必須在
    player-disjoint 的驗證題目上搜尋，不可用測試集。
    """
    mix = (1.0 - w) * log_game + w * log_question
    mix = mix - mix.max(axis=1, keepdims=True)
    e = np.exp(mix)
    return np.log(np.clip(e / e.sum(axis=1, keepdims=True), 1e-12, None))
