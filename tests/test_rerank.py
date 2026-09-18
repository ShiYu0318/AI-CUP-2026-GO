"""檢索後處理的測試。

注意：這裡只測「數學性質正確」，不測「分數會提升」——
k-reciprocal 在本任務（1-對-1 匹配）是否有效尚未證實，見模組 docstring。
"""
import numpy as np
import pytest

from aicup.rerank import (
    _to_distance, alpha_query_expansion, k_reciprocal_rerank, rerank_similarity,
)


@pytest.fixture
def emb():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(30, 16)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    q = g + rng.normal(scale=0.3, size=(30, 16))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q, g


def test_to_distance_is_monotone_decreasing():
    s = np.array([[1.0, 0.5, -1.0]])
    d = _to_distance(s)
    assert d[0, 0] < d[0, 1] < d[0, 2]


def test_qe_output_is_l2_normalised(emb):
    q, g = emb
    out = alpha_query_expansion(q, g)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_qe_preserves_shape(emb):
    q, g = emb
    assert alpha_query_expansion(q, g, top_k=5).shape == q.shape


def test_qe_with_alpha_zero_is_plain_average(emb):
    """alpha=0 時所有鄰居等權重——結果應與手算一致。"""
    q, g = emb
    out = alpha_query_expansion(q, g, top_k=2, alpha=0.0)
    sim = q @ g.T
    idx = np.argsort(-sim, axis=1)[:, :2]
    manual = q + g[idx].sum(axis=1)
    manual /= np.linalg.norm(manual, axis=1, keepdims=True)
    assert np.allclose(out, manual)


def test_kr_output_shape_and_finite(emb):
    q, g = emb
    out = k_reciprocal_rerank(q, g, k1=10, k2=3)
    assert out.shape == (len(q), len(g))
    assert np.isfinite(out).all()


def test_kr_is_deterministic(emb):
    q, g = emb
    a = k_reciprocal_rerank(q, g, k1=10, k2=3)
    b = k_reciprocal_rerank(q, g, k1=10, k2=3)
    assert np.array_equal(a, b)


def test_kr_lambda_one_uses_column_normalised_distance(emb):
    """lambda=1 時只採用原始距離，但那是**逐列正規化後**的距離。

    原論文的實作會把距離矩陣除以每一欄的最大值，這是一種 hubness 修正
    （壓抑那些「跟誰都近」的 gallery 條目）。副作用是：即使 lambda=1，
    排序也不會回到原始餘弦——這是刻意的設計，不是 bug。
    """
    q, g = emb
    out = k_reciprocal_rerank(q, g, k1=10, k2=1, lambda_value=1.0)

    feats = np.concatenate([q, g], axis=0)
    d = _to_distance(feats @ feats.T)
    np.fill_diagonal(d, 0.0)
    d = d / np.maximum(d.max(axis=0, keepdims=True), 1e-12)
    expected = -d[: len(q), len(q):]
    assert np.allclose(out, expected)


def test_kr_lambda_changes_the_result(emb):
    """lambda 必須真的有作用——否則融合權重是無效參數。"""
    q, g = emb
    a = k_reciprocal_rerank(q, g, k1=10, k2=3, lambda_value=0.0)
    b = k_reciprocal_rerank(q, g, k1=10, k2=3, lambda_value=1.0)
    assert not np.allclose(a, b)


def test_rerank_similarity_without_postprocessing_equals_cosine(emb):
    q, g = emb
    out = rerank_similarity(q, g, use_qe=False, use_kr=False)
    assert np.allclose(out, q @ g.T)


def test_rerank_similarity_shape(emb):
    q, g = emb
    assert rerank_similarity(q, g, k1=10, k2=3).shape == (len(q), len(g))


def test_kr_handles_small_gallery():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(3, 8)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    g = rng.normal(size=(5, 8)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    out = k_reciprocal_rerank(q, g, k1=3, k2=2)
    assert out.shape == (3, 5) and np.isfinite(out).all()
