"""Stage-2 重排的測試：pair 特徵的數學性質與重排的排序保證。"""
import numpy as np
import pytest

from aicup.rerank_stage2 import (
    pair_features, pair_labels, feature_names, rerank_scores, TOP_N,
)


@pytest.fixture
def sim():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(10, 40))
    s[np.arange(10), np.arange(10)] += 2.0      # 讓對角線是正解
    return s


def test_pair_features_shape(sim):
    X, pairs = pair_features(sim, top_n=15)
    assert X.shape[0] == 10 * 15
    assert pairs.shape == (10 * 15, 2)
    assert np.isfinite(X).all()


def test_pair_features_optional_columns(sim):
    base, _ = pair_features(sim, top_n=5)
    withq, _ = pair_features(sim, top_n=5, query_sizes=np.full(10, 12))
    assert withq.shape[1] == base.shape[1] + 1
    both, _ = pair_features(sim, top_n=5, query_sizes=np.full(10, 12),
                            gallery_sizes=np.full(40, 100))
    assert both.shape[1] == base.shape[1] + 2


def test_feature_names_match_width(sim):
    X, _ = pair_features(sim, top_n=5, query_sizes=np.full(10, 12),
                         gallery_sizes=np.full(40, 100))
    assert len(feature_names(query_sizes=np.zeros(1), gallery_sizes=np.zeros(1))) == X.shape[1]


def test_rank_column_is_zero_based_and_ordered(sim):
    X, _ = pair_features(sim, top_n=8)
    for i in range(10):
        block = X[i * 8:(i + 1) * 8, 0]
        assert list(block) == list(range(8))


def test_gap_to_top_is_zero_for_first(sim):
    X, _ = pair_features(sim, top_n=6)
    assert np.allclose(X[::6, 2], 0.0)          # 每個 query 的第一名 gap_to_top = 0


def test_gap_to_top_is_non_negative(sim):
    X, _ = pair_features(sim, top_n=10)
    assert (X[:, 2] >= -1e-6).all()


def test_pair_labels_default_diagonal(sim):
    _, pairs = pair_features(sim, top_n=20)
    y = pair_labels(pairs)
    assert y.sum() > 0
    for (qi, gi), lab in zip(pairs, y):
        assert lab == int(qi == gi)


def test_pair_labels_with_explicit_true_column(sim):
    _, pairs = pair_features(sim, top_n=20)
    true_col = np.arange(10)[::-1]
    y = pair_labels(pairs, true_col)
    for (qi, gi), lab in zip(pairs, y):
        assert lab == int(gi == true_col[qi])


def test_rerank_keeps_shape_and_is_finite(sim):
    class _Stub:
        def predict_proba(self, X):
            p = 1.0 / (1.0 + np.exp(X[:, 0]))    # rank 越小機率越高
            return np.stack([1 - p, p], axis=1)
    out = rerank_scores(_Stub(), sim, top_n=10)
    assert out.shape == sim.shape
    assert np.isfinite(out).all()


def test_reranked_candidates_outrank_untouched_ones(sim):
    """Top-N 之外的候選必定排在重排過的候選之後——否則會把好候選擠掉。"""
    class _Stub:
        def predict_proba(self, X):
            p = np.full(len(X), 0.01)            # 即使機率很低
            return np.stack([1 - p, p], axis=1)
    top_n = 5
    out = rerank_scores(_Stub(), sim, top_n=top_n)
    order_before = np.argsort(-sim, axis=1)[:, :top_n]
    for i in range(sim.shape[0]):
        top_after = set(np.argsort(-out[i])[:top_n].tolist())
        assert top_after == set(order_before[i].tolist())


def test_perfect_reranker_puts_truth_first(sim):
    """若重排器完美辨識正解，Top-1 必為正解。"""
    class _Oracle:
        def __init__(self, pairs): self.pairs = pairs
        def predict_proba(self, X):
            p = (self.pairs[:, 0] == self.pairs[:, 1]).astype(float)
            return np.stack([1 - p, p], axis=1)
    _, pairs = pair_features(sim, top_n=20)
    out = rerank_scores(_Oracle(pairs), sim, top_n=20)
    assert (out.argmax(axis=1) == np.arange(10)).all()
