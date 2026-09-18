"""手工特徵度量學習的測試。"""
import numpy as np
import pytest
import torch

from aicup.feature_metric import (
    FeatureEmbedder, FeatureScaler, PKFeatureSampler, embed_features, player_template,
)


def test_scaler_uses_only_fitted_data():
    rng = np.random.default_rng(0)
    A = rng.normal(loc=5, scale=2, size=(200, 10))
    sc = FeatureScaler().fit(A)
    assert np.allclose(sc.mu, A.mean(0))
    Z = sc.transform(A)
    assert abs(Z.mean()) < 1e-5 and abs(Z.std() - 1) < 1e-2


def test_scaler_roundtrip():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(50, 8))
    sc = FeatureScaler().fit(A)
    back = FeatureScaler().load_state_dict(sc.state_dict())
    assert np.allclose(sc.transform(A), back.transform(A))


def test_scaler_handles_constant_column():
    A = np.ones((30, 4)); A[:, 1] = np.arange(30)
    Z = FeatureScaler().fit(A).transform(A)
    assert np.isfinite(Z).all()


def test_embedder_output_shape_and_norm():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(64, 20)).astype(np.float32)
    sc = FeatureScaler().fit(X)
    m = FeatureEmbedder(20, embed_dim=32, hidden=64)
    E = embed_features(m, sc, X, torch.device("cpu"))
    assert E.shape == (64, 32)
    assert np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-5)


def test_embedder_is_differentiable():
    m = FeatureEmbedder(12, embed_dim=16, hidden=32)
    x = torch.randn(8, 12, requires_grad=True)
    m(x)[0].sum().backward()
    assert x.grad is not None


def test_player_template_is_unit_norm():
    rng = np.random.default_rng(3)
    E = rng.normal(size=(12, 16))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    t = player_template(E)
    assert float(np.linalg.norm(t)) == pytest.approx(1.0, rel=1e-6)


def test_player_template_averages_then_normalises():
    E = np.array([[1.0, 0.0], [0.0, 1.0]])
    t = player_template(E)
    assert np.allclose(t, [2 ** -0.5, 2 ** -0.5])


def test_pk_sampler_batch_structure():
    rng = np.random.default_rng(4)
    lab = np.repeat(np.arange(50), 6)
    rng.shuffle(lab)
    s = PKFeatureSampler(lab, p=8, k=4, seed=0)
    idx, y = s.sample()
    assert len(idx) == 32
    assert len(set(y.tolist())) == 8
    assert np.bincount(y).max() == 4


def test_pk_sampler_drops_players_with_too_few_games():
    lab = np.array([0, 0, 0, 1, 2, 2])          # 身分 1 只有一局
    s = PKFeatureSampler(lab, p=2, k=2, min_games=2)
    assert 1 not in s.by
    assert s.n_classes == 2


def test_pk_sampler_class_counts_match():
    lab = np.array([0] * 5 + [1] * 3)
    s = PKFeatureSampler(lab, p=2, k=2)
    counts = s.class_counts()
    assert sorted(counts.tolist()) == [3.0, 5.0]
