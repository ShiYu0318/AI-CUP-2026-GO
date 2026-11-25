"""GBDT 集成：LightGBM / XGBoost / CatBoost / sklearn HistGB 的統一介面 + stacking。

為什麼要這個：先前只用了 sklearn 的 HistGradientBoosting 單模型。
競賽的標準做法是多個 GBDT 實作各自訓練、用 **GroupKFold（以玩家分組）** 產生
out-of-fold 預測，再疊一層 meta-learner。不同實作的偏誤結構不同，融合幾乎必定有增益。

⚠️ **分組必須是玩家，不是棋局。** 用隨機 KFold 會讓同一位玩家的棋局橫跨 fold，
OOF 預測被身分洩漏污染，stacking 權重會學歪（實測洩漏率 100%）。

本機可能只裝了 sklearn；LightGBM / XGBoost / CatBoost 在 Kaggle 與 Colab 都預裝。
未安裝的 backend 會自動略過，不會讓整個流程失敗。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import N_LEVELS


def available_backends(verbose: bool = False) -> list[str]:
    """回傳這台機器上實際可用的 GBDT 實作。

    ⚠️ 這裡攔的是 `Exception`，不是只有 `ImportError`。
    套件「裝好了但載不起來」是很常見的情況：例如 macOS 上
    `pip install lightgbm` 會成功，但 import 時才會因為缺 `libomp.dylib`
    丟 `OSError`。只攔 ImportError 的話，整條預測管線會在比賽當天直接崩掉，
    而不是安靜地少用一個 backend。
    """
    out = ["hist"]                       # sklearn 一定有
    for name, mod in (("lgbm", "lightgbm"), ("xgb", "xgboost"), ("cat", "catboost")):
        try:
            __import__(mod)
            out.append(name)
        except Exception as e:           # noqa: BLE001 —— 見上方說明
            if verbose:
                print(f"  backend {name} 不可用：{type(e).__name__}: {e}")
    return out


def make_model(backend: str, n_classes: int = N_LEVELS, seed: int = 0, **kw):
    """依 backend 名稱建立一個多類別分類器。參數是各家在表格資料上的常用起點。"""
    if backend == "hist":
        from sklearn.ensemble import HistGradientBoostingClassifier
        p = dict(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                 l2_regularization=1.0, early_stopping=False, random_state=seed)
        p.update(kw)
        return HistGradientBoostingClassifier(**p)
    if backend == "lgbm":
        import lightgbm as lgb
        p = dict(n_estimators=1200, learning_rate=0.05, num_leaves=127,
                 min_child_samples=50, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.7, reg_lambda=1.0, objective="multiclass",
                 num_class=n_classes, n_jobs=-1, random_state=seed, verbose=-1)
        p.update(kw)
        return lgb.LGBMClassifier(**p)
    if backend == "xgb":
        import xgboost as xgb
        p = dict(n_estimators=1200, learning_rate=0.05, max_depth=8,
                 min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
                 reg_lambda=1.0, objective="multi:softprob", num_class=n_classes,
                 tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0)
        p.update(kw)
        return xgb.XGBClassifier(**p)
    if backend == "cat":
        from catboost import CatBoostClassifier
        p = dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                 loss_function="MultiClass", random_seed=seed, verbose=0,
                 allow_writing_files=False)
        p.update(kw)
        return CatBoostClassifier(**p)
    raise ValueError(f"未知的 backend：{backend}")


def player_group_folds(player_ids: np.ndarray, rank_ids: np.ndarray,
                       n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """以玩家為單位、依等級分層的 fold 指派。回傳與樣本等長的 fold 編號。

    同一位玩家的所有棋局必定落在同一個 fold —— 這是避免身分洩漏的唯一正確做法。
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(player_ids, kind="stable")
    uniq, first = np.unique(player_ids[order], return_index=True)
    urank = rank_ids[order][first]

    fold_of_player = {}
    for r in np.unique(urank):
        ids = uniq[urank == r]
        rng.shuffle(ids)
        for i, pid in enumerate(ids):
            fold_of_player[pid] = i % n_folds
    return np.array([fold_of_player[p] for p in player_ids], dtype=np.int8)


@dataclass
class OofResult:
    """一個 backend 的 out-of-fold 結果。"""
    backend: str
    oof: np.ndarray                      # (n_samples, n_classes)
    models: list = field(default_factory=list)
    seconds: float = 0.0


def fit_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray, backend: str,
            seed: int = 0, verbose: bool = True, **kw) -> OofResult:
    """對一個 backend 做 K-fold，回傳 OOF 機率與各 fold 的模型。"""
    n_classes = int(y.max()) + 1
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    models = []
    t0 = time.time()
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        m = make_model(backend, n_classes, seed=seed + f, **kw)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])
        models.append(m)
        if verbose:
            acc = (oof[va].argmax(1) == y[va]).mean()
            print(f"    [{backend}] fold {f} 單局 acc {acc:.4f}"
                  f"  {(time.time() - t0) / 60:.1f} 分", flush=True)
    return OofResult(backend=backend, oof=oof, models=models, seconds=time.time() - t0)


def predict_mean(models: list, X: np.ndarray) -> np.ndarray:
    """K 個 fold 模型的平均預測。"""
    acc = None
    for m in models:
        p = m.predict_proba(X)
        acc = p if acc is None else acc + p
    return acc / len(models)


class StackingBlender:
    """把多個 backend 的 OOF 機率疊成一層 meta-learner。

    輸入是各 backend 的機率串接；meta-learner 用多項式邏輯斯迴歸
    （簡單、不易過擬合，且在只有數個 base model 時通常勝過再疊一層 GBDT）。
    """

    def __init__(self, C: float = 1.0):
        self.C = C
        self.meta = None
        self.backends: list[str] = []

    def fit(self, oofs: dict[str, np.ndarray], y: np.ndarray) -> "StackingBlender":
        from sklearn.linear_model import LogisticRegression
        self.backends = sorted(oofs)
        Z = np.hstack([oofs[b] for b in self.backends])
        self.meta = LogisticRegression(C=self.C, max_iter=2000, n_jobs=-1)
        self.meta.fit(Z, y)
        return self

    def predict_proba(self, probas: dict[str, np.ndarray]) -> np.ndarray:
        Z = np.hstack([probas[b] for b in self.backends])
        return self.meta.predict_proba(Z)


def weighted_blend(probas: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """加權平均融合。權重應在 OOF 上以官方分數最佳化，不可在測試集上調。"""
    total = sum(weights.get(b, 0.0) for b in probas)
    if total <= 0:
        return np.mean(list(probas.values()), axis=0)
    out = None
    for b, p in probas.items():
        w = weights.get(b, 0.0) / total
        out = w * p if out is None else out + w * p
    return out


def optimise_weights(oofs: dict[str, np.ndarray], score_fn, n_iter: int = 300,
                     seed: int = 0) -> dict[str, float]:
    """以隨機 Dirichlet 搜尋找出最大化 score_fn 的融合權重。

    score_fn 接受 (n_samples, n_classes) 的機率、回傳純量分數（越大越好）。
    請傳入**官方評分函數**，而不是 accuracy 或 log-loss。
    """
    rng = np.random.default_rng(seed)
    names = sorted(oofs)
    best_w = {b: 1.0 / len(names) for b in names}
    best = score_fn(weighted_blend(oofs, best_w))
    for _ in range(n_iter):
        w = rng.dirichlet(np.ones(len(names)))
        cand = {b: float(x) for b, x in zip(names, w)}
        s = score_fn(weighted_blend(oofs, cand))
        if s > best:
            best, best_w = s, cand
    return best_w
