"""EM 先驗估計的性質檢查。

這支的存在有一半是**紀錄一個失敗**：EM 在本任務上實測比固定先驗差
（見 RESULTS.md §2.7）。但函式本身要是正確的——若哪天後驗變得夠可分，
它才可能派上用場，而到時候不該還要重新懷疑實作。
"""
import numpy as np
import pytest

from aicup.scoring import estimate_prior_em


def _posteriors(y, sharp=0.9, n_levels=10):
    """給定真標籤造一組「很可分」的後驗（相鄰等級留一點質量）。"""
    P = np.full((len(y), n_levels), (1 - sharp) / n_levels)
    P[np.arange(len(y)), y] += sharp
    return P / P.sum(axis=1, keepdims=True)


def test_recovers_prior_when_posteriors_are_separable():
    rng = np.random.default_rng(0)
    true = np.array([0.30, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.15, 0.15])
    y = rng.choice(10, size=20000, p=true)
    est = estimate_prior_em(_posteriors(y, sharp=0.95),
                            source_prior=np.full(10, 0.1))
    assert np.abs(est - true).max() < 0.02


def test_always_a_valid_distribution():
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(10), size=500)
    est = estimate_prior_em(P, source_prior=np.full(10, 0.1))
    assert est.shape == (10,)
    assert np.isclose(est.sum(), 1.0)
    assert (est > 0).all(), "floor 應該擋住 0，否則後續除法會爆掉"


def test_shrink_pulls_towards_source():
    rng = np.random.default_rng(2)
    y = rng.choice(10, size=4000, p=[0.5] + [0.5 / 9] * 9)
    P = _posteriors(y)
    src = np.full(10, 0.1)
    far = estimate_prior_em(P, source_prior=src, shrink=0.0)
    near = estimate_prior_em(P, source_prior=src, shrink=0.8)
    assert np.abs(near - src).max() < np.abs(far - src).max()


def test_weights_change_the_estimate():
    """加權必須真的有效——評估時測試先驗是用權重模擬的，
    EM 若忽略權重就會看到與評分端不同的分布。"""
    rng = np.random.default_rng(3)
    y = rng.choice(10, size=4000, p=np.full(10, 0.1))
    P = _posteriors(y)
    w = np.where(y == 0, 50.0, 1.0)           # 把 rank 0 的權重拉高
    plain = estimate_prior_em(P, source_prior=np.full(10, 0.1))
    weighted = estimate_prior_em(P, source_prior=np.full(10, 0.1), weights=w)
    assert weighted[0] > plain[0] + 0.1


@pytest.mark.parametrize("n", [1, 5, 50])
def test_tiny_samples_do_not_crash(n):
    rng = np.random.default_rng(4)
    P = rng.dirichlet(np.ones(10), size=n)
    est = estimate_prior_em(P, source_prior=np.full(10, 0.1))
    assert np.isfinite(est).all() and np.isclose(est.sum(), 1.0)
