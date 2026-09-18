"""官方評分函數的測試。

這些測試是整個專案的地基：若評分實作錯了，所有實驗結論都會跟著錯。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aicup import LEVELS, RANK_TO_ID
from aicup.scoring import (
    E_INV, rank_score, rank_score_matrix, rank_final_score,
    decode_expected_score, decode_argmax, apply_temperature, fit_temperature,
    player_score_from_rank, player_final_score,
    ranks_from_similarity, top_k_from_similarity, retrieval_report,
)


# ---------------- 競賽二 ----------------

def test_levels_order_is_weak_to_strong():
    assert LEVELS == ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
    assert RANK_TO_ID["D"] == 0 and RANK_TO_ID["6D"] == 9


def test_rank_score_exact_values():
    assert rank_score(3, 3) == 1.0
    assert rank_score(3, 4) == pytest.approx(E_INV)
    assert rank_score(3, 2) == pytest.approx(E_INV)
    assert rank_score(3, 5) == 0.0
    assert rank_score(0, 9) == 0.0


def test_rank_score_boundary_levels_have_one_neighbour():
    # D (索引 0) 只有 C 一個鄰居；6D (索引 9) 只有 5D 一個鄰居
    S = rank_score_matrix()
    assert (S[0] > 0).sum() == 2      # 自己 + 一個鄰居
    assert (S[9] > 0).sum() == 2
    assert (S[5] > 0).sum() == 3      # 中間等級有兩個鄰居


def test_rank_score_matrix_is_symmetric():
    S = rank_score_matrix()
    assert np.allclose(S, S.T)


def test_rank_final_score_all_correct_is_one():
    y = np.arange(10)
    assert rank_final_score(y, y) == 1.0


def test_rank_final_score_all_adjacent_is_e_inv():
    true = np.arange(9)
    pred = true + 1
    assert rank_final_score(pred, true) == pytest.approx(E_INV)


def test_expected_decoding_beats_argmax_when_uncertain():
    """經典案例：機率分散在 0 與 2，argmax 選 0，但選 1 的期望分數更高。"""
    p = np.zeros(10)
    p[0], p[1], p[2] = 0.40, 0.20, 0.40
    assert decode_argmax(p) == 0
    assert decode_expected_score(p) == 1
    S = rank_score_matrix()
    assert (S[1] @ p) > (S[0] @ p)


def test_expected_decoding_matches_argmax_when_confident():
    p = np.zeros(10); p[7] = 0.95; p[6] = 0.05
    assert decode_expected_score(p) == decode_argmax(p) == 7


def test_expected_decoding_never_worse_in_expectation():
    """隨機後驗上，期望分數解碼的期望得分不可能低於 argmax。"""
    rng = np.random.default_rng(0)
    S = rank_score_matrix()
    P = rng.dirichlet(np.ones(10) * 0.5, size=2000)
    exp_gain = (P @ S.T)[np.arange(2000), decode_expected_score(P)]
    arg_gain = (P @ S.T)[np.arange(2000), decode_argmax(P)]
    assert (exp_gain + 1e-12 >= arg_gain).all()
    assert exp_gain.mean() > arg_gain.mean()


def test_decode_handles_batch():
    P = np.eye(10)
    assert (decode_expected_score(P) == np.arange(10)).all()


def test_temperature_scaling_is_a_valid_distribution():
    logits = np.random.default_rng(1).normal(size=(50, 10))
    p = apply_temperature(logits, 2.0)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert (p >= 0).all()


def test_fit_temperature_returns_positive_and_improves_or_ties():
    rng = np.random.default_rng(2)
    true = rng.integers(0, 10, size=400)
    logits = np.eye(10)[true] * 3.0 + rng.normal(scale=2.0, size=(400, 10))
    t = fit_temperature(logits, true)
    assert t > 0
    base = rank_final_score(decode_expected_score(apply_temperature(logits, 1.0)), true)
    tuned = rank_final_score(decode_expected_score(apply_temperature(logits, t)), true)
    assert tuned >= base - 1e-12


# ---------------- 競賽一 ----------------

def test_player_score_exact_values():
    # 官方：第 r 名得 exp(-(r-1))，這裡輸入是 0-based
    got = player_score_from_rank(np.arange(6))
    expect = [1.0, 0.36787944, 0.13533528, 0.04978707, 0.01831564, 0.0]
    assert np.allclose(got, expect, atol=1e-7)


def test_player_score_outside_top5_is_zero():
    assert player_score_from_rank(5) == 0.0
    assert player_score_from_rank(399) == 0.0


def test_player_final_score_all_top1_is_one():
    assert player_final_score(np.zeros(400, dtype=int)) == 1.0


def test_ranks_from_similarity_identity():
    sim = np.eye(5) * 10 + np.random.default_rng(3).normal(scale=0.01, size=(5, 5))
    assert (ranks_from_similarity(sim) == 0).all()


def test_ranks_from_similarity_reversed():
    """正解永遠排最後一名時，名次應為 n-1。"""
    n = 6
    sim = np.zeros((n, n))
    for i in range(n):
        sim[i] = 1.0
        sim[i, i] = -1.0          # 正解相似度最低
    assert (ranks_from_similarity(sim) == n - 1).all()


def test_top_k_returns_distinct_names():
    rng = np.random.default_rng(4)
    sim = rng.normal(size=(20, 50))
    names = [f"pub_p_{i:04d}" for i in range(50)]
    top = top_k_from_similarity(sim, names, k=5)
    assert top.shape == (20, 5)
    for row in top:
        assert len(set(row)) == 5          # 官方要求 Top-5 必須相異


def test_retrieval_report_fields():
    sim = np.eye(10)
    rep = retrieval_report(sim)
    assert rep["official_score"] == 1.0
    assert rep["top1"] == 1.0 and rep["top5"] == 1.0
    assert rep["n_query"] == 10
