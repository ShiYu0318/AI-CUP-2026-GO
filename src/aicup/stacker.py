"""學習式融合（stacker）：取代固定權重的幾何平均。

**動機**：Oracle（逐題挑最好的成分）= 0.7342，固定權重融合只有 0.5639。
中間 0.17 的缺口代表三個成分的**分歧是有用的訊號**，
但固定權重無法表達「這一題該多信誰」。

實測（8 組 player-disjoint，校準/評分玩家不重疊）：

    固定權重融合              0.5639 ±0.0104
    LogisticRegression     0.5727 ±0.0124   +0.0088（8/8 勝，t=5.00）

C 從 0.3 到 3.0 是平台（+0.008~+0.009，全部 8/8），預設取中點 1.0。

兩個對照也有資訊：
  - 加「成分間分歧」特徵：−0.002。logistic 已經看得到完整後驗，不需要手餵
  - GBDT stacker：−0.0068（1/8）。與本專案其他地方一致——
    在同質的連續特徵上，軸對齊切分輸給線性/分散式表徵
  - 只用 3 個期望等級 + 局數的極簡版（4 維）：−0.0066。
    **增益來自完整的後驗形狀，不是只靠「該多信誰」**

⚠️ **防洩漏**：stacker 只能在**校準玩家**上擬合，在**評分玩家**上套用。
三個成分本身是在訓練玩家上訓練的，所以它們在驗證玩家上的預測本來就是 out-of-fold。
"""
from __future__ import annotations

import numpy as np

from .scoring import expected_level

DEFAULT_C = 1.0


def _probs(log_post: np.ndarray) -> np.ndarray:
    p = np.exp(np.asarray(log_post, dtype=np.float64))
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)


def part_names(parts: dict) -> tuple:
    """stacker 用到的成分名稱（已排序）。訓練與推論必須完全一致。"""
    return tuple(sorted(parts))


def stack_features(parts: dict, sizes) -> np.ndarray:
    """把各成分的後驗組成 stacker 的輸入。

    每個成分貢獻：10 維 log 後驗 + 期望等級 + 最大機率（信心）+ 熵。
    再加上 log(題目局數)——它決定了後驗該有多可信。

    parts: {名稱: (n_questions, n_levels) 的 log 後驗}，**順序會影響欄位順序**，
           所以呼叫端必須保證訓練與推論時用同一個順序（這裡用 sorted 固定）。
    """
    if not parts:
        raise ValueError("parts 不可為空")
    n = len(next(iter(parts.values())))
    cols = []
    for name in sorted(parts):
        lp = np.asarray(parts[name], dtype=np.float64)
        if len(lp) != n:
            raise ValueError(f"成分 {name} 有 {len(lp)} 題，與其他成分的 {n} 題不符")
        p = _probs(lp)
        cols.append(lp)
        cols.append(expected_level(p)[:, None])
        cols.append(p.max(axis=1)[:, None])
        cols.append(-(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)[:, None])
    sizes = np.asarray(sizes, dtype=np.float64).reshape(-1)
    cols.append(np.log(np.maximum(sizes, 1.0))[:, None])
    return np.concatenate(cols, axis=1)


class Stacker:
    """在校準題目上擬合，輸出「期望等級」給切點解碼使用。"""

    def __init__(self, C: float = DEFAULT_C, n_levels: int = 10):
        self.C = C
        self.n_levels = n_levels
        self.model = None
        self.mu = None
        self.sd = None
        self.parts = None          # 擬合時用到的成分名稱，推論時要對得上

    def fit(self, X: np.ndarray, y, parts=None) -> "Stacker":
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError(f"X 有 {len(X)} 列，y 有 {len(y)} 個，不符")
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        self.model = LogisticRegression(max_iter=2000, C=self.C)
        self.model.fit((X - self.mu) / self.sd, y)
        self.parts = part_names(parts) if parts is not None else None
        return self

    def check_parts(self, parts: dict) -> None:
        """推論時的成分集合必須與擬合時一致，否則欄位會整個錯位。

        少一個成分不會讓形狀對不上（只是欄數變少會直接拋錯），
        但**換一個同名不同來源的成分**就會安靜地算錯——所以明確擋掉。
        """
        if self.parts is None:
            return
        got = part_names(parts)
        if got != self.parts:
            raise ValueError(
                f"stacker 擬合時的成分是 {self.parts}，推論時卻是 {got}。"
                "兩邊必須完全一致，否則特徵欄位會對錯。")

    def expected_level(self, X: np.ndarray) -> np.ndarray:
        """回傳每題的期望等級（連續值），交給 `optimise_thresholds` 解碼。"""
        if self.model is None:
            raise RuntimeError("Stacker 尚未 fit")
        P = self.model.predict_proba((np.asarray(X, dtype=np.float64) - self.mu) / self.sd)
        # LogisticRegression 只會輸出訓練時見過的類別，缺的等級要補零再重新正規化，
        # 否則 expected_level 會對錯欄位加權。
        full = np.zeros((len(P), self.n_levels))
        full[:, self.model.classes_] = P
        full /= np.maximum(full.sum(axis=1, keepdims=True), 1e-300)
        return expected_level(full)
