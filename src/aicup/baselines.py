"""以手工特徵為基礎的兩個子題 baseline。

這兩個模型的用途不是奪冠，而是：
  1. 提供一個誠實的「地板」——任何深度模型若贏不過它，就是實作有問題
  2. 提供一個與深度模型誤差結構不同的 ensemble 成分
  3. 讓整條驗證管線在深度模型還沒寫好之前就能跑通

兩者都只需 CPU。
"""

from __future__ import annotations

import numpy as np

from . import N_LEVELS
from .scoring import (
    decode_argmax, decode_expected_score, rank_final_score,
    ranks_from_similarity, retrieval_report,
)


# --------------------------------------------------------------------------
# 競賽二：棋力預測
# --------------------------------------------------------------------------

class RankBaseline:
    """單局分類器 + 題目層級聚合 + 對齊評分函數的解碼。"""

    def __init__(self, **kwargs):
        from sklearn.ensemble import HistGradientBoostingClassifier
        params = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                      l2_regularization=1.0, early_stopping=False, random_state=0)
        params.update(kwargs)
        self.model = HistGradientBoostingClassifier(**params)

    def fit(self, X, y) -> "RankBaseline":
        self.model.fit(X, y)
        return self

    def game_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    @staticmethod
    def aggregate(probs: np.ndarray, how: str = "mean") -> np.ndarray:
        """把同一題多局的後驗聚合成一個後驗。

        mean     算術平均（官方 baseline 的做法）
        logmean  幾何平均（等價於平均 log-prob，對單局的過度自信較不敏感）
        """
        if how == "mean":
            p = probs.mean(axis=0)
        elif how == "logmean":
            p = np.exp(np.log(np.clip(probs, 1e-12, None)).mean(axis=0))
        else:
            raise ValueError(how)
        return p / p.sum()

    def question_proba(self, feats_per_question, how: str = "mean") -> np.ndarray:
        """所有題目的棋局一次批次預測再切分。

        逐題呼叫 predict_proba 會讓 4,000 棵樹的固定開銷乘上題數；
        批次化之後同樣的評估從數分鐘降到數秒。
        """
        sizes = [len(X) for X in feats_per_question]
        out = np.full((len(sizes), N_LEVELS), 1.0 / N_LEVELS)
        nonempty = [i for i, n in enumerate(sizes) if n > 0]
        if not nonempty:
            return out

        flat = np.concatenate([feats_per_question[i] for i in nonempty], axis=0)
        probs = self.game_proba(flat)
        cuts = np.cumsum([sizes[i] for i in nonempty])[:-1]
        for i, block in zip(nonempty, np.split(probs, cuts)):
            out[i] = self.aggregate(block, how)
        return out

    def question_proba_from_probs(self, probs_per_question, how: str = "mean") -> np.ndarray:
        """已有逐局後驗時直接聚合，讓 mean 與 logmean 共用同一次預測。"""
        out = np.full((len(probs_per_question), N_LEVELS), 1.0 / N_LEVELS)
        for i, p in enumerate(probs_per_question):
            if len(p):
                out[i] = self.aggregate(p, how)
        return out

    def evaluate(self, feats_per_question, true_rank_id, how: str = "mean") -> dict:
        P = self.question_proba(feats_per_question, how)
        s_arg = rank_final_score(decode_argmax(P), true_rank_id)
        s_exp = rank_final_score(decode_expected_score(P), true_rank_id)
        return {
            "aggregation": how,
            "score_argmax": s_arg,
            "score_expected": s_exp,
            "decoding_gain": s_exp - s_arg,
            "question_top1": float((decode_argmax(P) == np.asarray(true_rank_id)).mean()),
            "n_questions": len(true_rank_id),
        }


# --------------------------------------------------------------------------
# 競賽一：玩家辨識
# --------------------------------------------------------------------------

class PlayerBaseline:
    """標準化 + L2 正規化 + 平均成 player template + 餘弦檢索。

    這是 face template 的標準作法：
    逐局 embedding -> L2 normalize -> 平均 -> 再 L2 normalize。
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mu = None
        self.sd = None

    def fit(self, X) -> "PlayerBaseline":
        """標準化統計量只由 gallery 估計，避免用到 query 的資訊。"""
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + self.eps
        return self

    def embed(self, X) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        n = np.linalg.norm(Z, axis=-1, keepdims=True)
        return Z / np.maximum(n, self.eps)

    def template(self, X) -> np.ndarray:
        """多局 -> 單一玩家向量（L2 normalize -> mean -> 再 L2 normalize）。"""
        e = self.embed(X).mean(axis=0)
        return e / max(float(np.linalg.norm(e)), self.eps)

    def build(self, gallery_feats: dict, query_feats: dict, players: list):
        """回傳 (相似度矩陣, 玩家順序)。列為 query、行為 gallery，順序一致。"""
        allX = np.concatenate([gallery_feats[p] for p in players], axis=0)
        self.fit(allX)
        G = np.stack([self.template(gallery_feats[p]) for p in players])
        Q = np.stack([self.template(query_feats[p]) for p in players])
        return Q @ G.T, players

    def evaluate(self, gallery_feats: dict, query_feats: dict, players: list) -> dict:
        sim, _ = self.build(gallery_feats, query_feats, players)
        rep = retrieval_report(sim)
        rep["n_candidates"] = len(players)
        return rep


def score_by_query_size(sim: np.ndarray, query_sizes) -> dict:
    """依每題局數分組回報分數。

    實測局數是最強的分數驅動因素（5 局 0.157 vs 20 局 0.458），
    分組回報能看出方法是在「少局數」還是「多局數」的題目上贏。
    """
    from .scoring import player_score_from_rank
    ranks = ranks_from_similarity(sim)
    sizes = np.asarray(query_sizes)
    sc = player_score_from_rank(ranks)
    out = {}
    for lo, hi, label in [(5, 8, "5-8"), (9, 12, "9-12"), (13, 16, "13-16"), (17, 20, "17-20")]:
        m = (sizes >= lo) & (sizes <= hi)
        if m.any():
            out[label] = {"n": int(m.sum()), "score": float(sc[m].mean()),
                          "top1": float((ranks[m] == 0).mean())}
    return out
