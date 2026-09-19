"""全域指派（Sinkhorn）的性質測試。

官方規格是每池 400 題、400 位候選。每題獨立取 argmax 沒用到這個全域約束——
hub 候選會被多題同時選為第一名，把正解擠掉。
Sinkhorn 強制每位候選的總指派量大致相等，等於自動做 hubness 校正。
"""
import numpy as np
import pytest

from aicup.rerank import sinkhorn_assignment, bijection_consistency
from aicup.scoring import player_final_score


def _pool(n=200, dim=64, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, dim))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    Q = G + noise * rng.normal(size=G.shape)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return Q @ G.T


def _score(mat):
    order = np.argsort(-mat, axis=1)
    return player_final_score(
        np.array([np.where(order[i] == i)[0][0] for i in range(len(mat))]))


def test_output_is_approximately_doubly_stochastic():
    """收斂條件是 tol=1e-3（eps 小的時候 Sinkhorn 收斂很慢）。

    排序遠比矩陣本身早穩定——實測 200 次迭代後 top-1 就完全不動了，
    所以這裡驗的是「有收斂到宣告的容忍度」，不是機器精度。
    """
    P = sinkhorn_assignment(_pool(), tol=1e-3)
    for axis in (0, 1):
        tot = P.sum(axis=axis)
        assert np.abs(tot - tot.mean()).max() <= 2e-3 * tot.mean()


def test_tighter_tol_converges_further():
    S = _pool()
    loose = sinkhorn_assignment(S, tol=1e-2).sum(axis=1)
    tight = sinkhorn_assignment(S, tol=1e-5).sum(axis=1)
    assert (np.abs(tight - tight.mean()).max()
            < np.abs(loose - loose.mean()).max())


def test_ranking_is_stable_once_converged():
    """名次比矩陣早穩定——這是可以用寬鬆 tol 的理由。"""
    S = _pool()
    a = sinkhorn_assignment(S, tol=1e-3).argmax(axis=1)
    b = sinkhorn_assignment(S, tol=1e-6).argmax(axis=1)
    assert (a == b).mean() > 0.99


def test_shape_and_finiteness_preserved():
    S = _pool(n=50)
    P = sinkhorn_assignment(S)
    assert P.shape == S.shape
    assert np.isfinite(P).all() and (P >= 0).all()


def test_beats_independent_argmax_when_base_accuracy_is_high():
    """基礎 Top-1 高時必須有增益——這是採用它的唯一理由。"""
    gains = []
    for seed in range(5):
        S = _pool(noise=0.25, seed=seed)
        gains.append(_score(sinkhorn_assignment(S)) - _score(S))
    assert np.mean(gains) > 0.02, f"平均增益只有 {np.mean(gains):.4f}"


def test_reduces_hub_candidates():
    """核心機制：把「被多題同時選為第一名」的 hub 打散。"""
    S = _pool(noise=0.35)
    assert bijection_consistency(sinkhorn_assignment(S)) > bijection_consistency(S)


def test_eps_must_be_positive():
    with pytest.raises(ValueError):
        sinkhorn_assignment(_pool(n=20), eps=0.0)


def test_rejects_non_matrix():
    with pytest.raises(ValueError):
        sinkhorn_assignment(np.ones(10))


def test_deterministic():
    S = _pool(n=80)
    assert np.array_equal(sinkhorn_assignment(S), sinkhorn_assignment(S))


def test_bijection_consistency_range():
    n = 40
    assert bijection_consistency(np.eye(n)) == 1.0          # 完美排列
    allsame = np.zeros((n, n)); allsame[:, 3] = 1.0         # 全部指向同一個
    assert bijection_consistency(allsame) == 1.0 / n
