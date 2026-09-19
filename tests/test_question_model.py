"""題目層級建模的測試：聚合統計量與後驗融合。"""
import numpy as np
import pytest

from aicup.question_model import (
    AGGREGATORS, aggregate_question, question_feature_dim,
    build_question_features, blend_log_posteriors,
)


def test_aggregate_dim_matches_formula():
    feats = np.random.default_rng(0).normal(size=(7, 20))
    v = aggregate_question(feats)
    assert len(v) == question_feature_dim(20) == 20 * len(AGGREGATORS) + 1


def test_aggregate_last_element_is_game_count():
    feats = np.zeros((9, 5))
    assert aggregate_question(feats)[-1] == 9.0


def test_aggregate_mean_block_matches():
    feats = np.random.default_rng(1).normal(size=(6, 8))
    v = aggregate_question(feats)
    assert np.allclose(v[:8], feats.mean(0), atol=1e-6)


def test_aggregate_std_block_distinguishes_consistency():
    """穩定與震盪的玩家，mean 相同但 std 不同——這正是單局後驗丟掉的資訊。"""
    steady = np.full((10, 3), 5.0)
    swinging = np.tile(np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]), (5, 1))
    a, b = aggregate_question(steady), aggregate_question(swinging)
    assert np.allclose(a[:3], b[:3])                 # mean 相同
    assert not np.allclose(a[3:6], b[3:6])           # std 不同


def test_aggregate_empty_returns_none():
    assert aggregate_question(np.zeros((0, 4))) is None


def test_build_question_features_handles_empty_question():
    X = np.random.default_rng(2).normal(size=(50, 6))
    ok = np.ones(50, dtype=bool); ok[10:20] = False
    rows = [np.arange(0, 5), np.arange(10, 20), np.arange(30, 40)]
    Q = build_question_features(X, ok, rows)
    assert Q.shape == (3, question_feature_dim(6))
    assert np.allclose(Q[1], 0.0)                    # 全被過濾掉的題目回傳零向量
    assert np.isfinite(Q).all()


def test_blend_returns_valid_log_posteriors():
    rng = np.random.default_rng(3)
    a = np.log(rng.dirichlet(np.ones(10), 20))
    b = np.log(rng.dirichlet(np.ones(10), 20))
    out = blend_log_posteriors(a, b, 0.3)
    p = np.exp(out)
    assert np.allclose(p.sum(1), 1.0)
    assert np.isfinite(out).all()


def test_blend_endpoints_recover_inputs():
    rng = np.random.default_rng(4)
    a = np.log(rng.dirichlet(np.ones(6), 15))
    b = np.log(rng.dirichlet(np.ones(6), 15))
    assert np.allclose(np.exp(blend_log_posteriors(a, b, 0.0)), np.exp(a), atol=1e-8)
    assert np.allclose(np.exp(blend_log_posteriors(a, b, 1.0)), np.exp(b), atol=1e-8)


def test_blend_is_geometric_not_arithmetic():
    """幾何平均要求兩個模型都同意；算術平均會被單一過度自信主導。"""
    a = np.log(np.array([[0.98, 0.01, 0.01]]))
    b = np.log(np.array([[0.02, 0.49, 0.49]]))
    geo = np.exp(blend_log_posteriors(a, b, 0.5))
    arith = 0.5 * np.exp(a) + 0.5 * np.exp(b)
    assert geo[0, 0] < arith[0, 0]
