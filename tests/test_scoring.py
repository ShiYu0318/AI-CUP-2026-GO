"""官方評分函數的測試。

這些測試是整個專案的地基：若評分實作錯了，所有實驗結論都會跟著錯。
"""
import numpy as np
import pytest

from aicup import LEVELS, RANK_TO_ID
from aicup.scoring import (
    E_INV, rank_score, rank_score_matrix, rank_final_score,
    decode_expected_score, decode_argmax, apply_temperature, fit_temperature,
    player_score_from_rank, player_final_score,
    ranks_from_similarity, top_k_from_similarity, retrieval_report,
    optimise_thresholds, apply_thresholds,
)
from aicup import N_LEVELS


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


# ---------------- 聚合順序（實測增益最大的單一改動，鎖住避免被「簡化」掉） ----------------

def test_player_template_normalises_each_game_before_averaging():
    """template 必須是「單局標準化 -> 單局 L2 -> 平均 -> 再 L2」。

    真實資料實測：改成「先平均再標準化」會從 0.6490 掉到 0.3886（-0.26），
    比任何特徵工程的影響都大。這個測試鎖住正確的順序。
    """
    from aicup.baselines import PlayerBaseline
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8)) * np.array([100.0, 1, 1, 1, 1, 1, 1, 1])  # 第 0 維尺度極大
    pb = PlayerBaseline().fit(X)
    t = pb.template(X[:10])

    # 正確順序：每一局先標準化並 L2，再平均
    Z = (X[:10] - pb.mu) / pb.sd
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    expect = Z.mean(0)
    expect = expect / np.linalg.norm(expect)
    assert np.allclose(t, expect)

    # 錯誤順序（先平均再標準化）必須產生不同的結果
    wrong = (X[:10].mean(0) - pb.mu) / pb.sd
    wrong = wrong / np.linalg.norm(wrong)
    assert not np.allclose(t, wrong)


def test_player_template_is_unit_norm():
    from aicup.baselines import PlayerBaseline
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 12))
    pb = PlayerBaseline().fit(X)
    assert float(np.linalg.norm(pb.template(X[:7]))) == pytest.approx(1.0)


def test_player_fit_uses_gallery_statistics_only():
    """標準化統計量只能來自 gallery——用到 query 就是洩漏。"""
    from aicup.baselines import PlayerBaseline
    rng = np.random.default_rng(2)
    gallery = rng.normal(size=(30, 6))
    pb = PlayerBaseline().fit(gallery)
    assert np.allclose(pb.mu, gallery.mean(0))
    assert np.allclose(pb.sd, gallery.std(0) + pb.eps)


def test_batched_question_proba_matches_per_question_loop():
    """批次化不能改變數值——這是效率優化，不是行為改變。"""
    from aicup.baselines import RankBaseline

    class _Stub(RankBaseline):
        def __init__(self):
            pass
        def game_proba(self, X):
            X = np.asarray(X, dtype=float)
            z = np.exp(X[:, :10] - X[:, :10].max(axis=1, keepdims=True))
            return z / z.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(0)
    feats = [rng.normal(size=(int(rng.integers(5, 21)), 12)) for _ in range(20)]
    feats.append(np.zeros((0, 12)))                 # 含一個空題目
    m = _Stub()
    batched = m.question_proba(feats, "mean")
    manual = np.stack([
        m.aggregate(m.game_proba(X), "mean") if len(X) else np.full(10, 0.1)
        for X in feats])
    assert np.allclose(batched, manual)


# ---------------- GBDT 集成 ----------------

def test_player_group_folds_keep_player_together():
    """同一位玩家的所有棋局必須落在同一個 fold——否則 OOF 被身分洩漏污染。"""
    from aicup.gbdt import player_group_folds
    rng = np.random.default_rng(0)
    pid = np.repeat([f"p{i}" for i in range(80)], 15)
    rk = np.repeat(rng.integers(0, 10, 80), 15)
    folds = player_group_folds(pid, rk, n_folds=5, seed=1)
    for p in np.unique(pid):
        assert len(set(folds[pid == p])) == 1


def test_player_group_folds_are_stratified_by_rank():
    """每個 fold 都應涵蓋所有等級，否則某些 fold 學不到某些類別。"""
    from aicup.gbdt import player_group_folds
    rng = np.random.default_rng(1)
    pid = np.repeat([f"p{i}" for i in range(200)], 5)
    rk = np.repeat(rng.integers(0, 10, 200), 5)
    folds = player_group_folds(pid, rk, n_folds=5, seed=0)
    for f in range(5):
        assert len(set(rk[folds == f])) >= 8


def test_available_backends_always_has_hist():
    from aicup.gbdt import available_backends
    assert "hist" in available_backends()


def test_weighted_blend_normalises():
    from aicup.gbdt import weighted_blend
    a = np.array([[0.8, 0.2]]); b = np.array([[0.2, 0.8]])
    out = weighted_blend({"a": a, "b": b}, {"a": 3.0, "b": 1.0})
    assert np.allclose(out.sum(1), 1.0)
    assert out[0, 0] == pytest.approx(0.65)


def test_weighted_blend_falls_back_to_mean_on_zero_weights():
    from aicup.gbdt import weighted_blend
    a = np.array([[1.0, 0.0]]); b = np.array([[0.0, 1.0]])
    out = weighted_blend({"a": a, "b": b}, {"a": 0.0, "b": 0.0})
    assert np.allclose(out, [[0.5, 0.5]])


def test_optimise_weights_finds_better_than_uniform():
    """b 完全正確、a 完全錯誤時，搜尋出的權重必須偏向 b。"""
    from aicup.gbdt import optimise_weights, weighted_blend
    y = np.array([0, 1, 0, 1])
    good = np.eye(2)[y]
    bad = np.eye(2)[1 - y]
    w = optimise_weights({"a": bad, "b": good},
                         lambda p: float((p.argmax(1) == y).mean()), n_iter=200)
    assert w["b"] > w["a"]


def test_stacking_blender_roundtrip():
    from aicup.gbdt import StackingBlender
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    oofs = {"a": np.eye(3)[y] * 0.7 + 0.1, "b": rng.dirichlet(np.ones(3), 200)}
    st = StackingBlender().fit(oofs, y)
    p = st.predict_proba(oofs)
    assert p.shape == (200, 3)
    assert np.allclose(p.sum(1), 1.0)
    assert (p.argmax(1) == y).mean() > 0.8


# ---------------- 序數回歸 + 最佳化切點 ----------------

def test_apply_thresholds_maps_to_levels():
    from aicup.scoring import apply_thresholds
    th = [0.5, 1.5, 2.5]
    assert list(apply_thresholds([0.0, 1.0, 2.0, 9.0], th)) == [0, 1, 2, 3]


def test_apply_thresholds_sorts_input():
    from aicup.scoring import apply_thresholds
    a = apply_thresholds([1.2], [2.5, 0.5, 1.5])
    b = apply_thresholds([1.2], [0.5, 1.5, 2.5])
    assert a == b


def test_expected_level_matches_manual():
    from aicup.scoring import expected_level
    p = np.zeros((2, 10)); p[0, 3] = 1.0; p[1, 2] = p[1, 4] = 0.5
    assert np.allclose(expected_level(p), [3.0, 3.0])


def test_optimise_thresholds_returns_sorted_and_right_length():
    from aicup.scoring import optimise_thresholds
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 500)
    s = y + rng.normal(scale=1.0, size=500)
    th = optimise_thresholds(s, y, n_restarts=3)
    assert len(th) == 9
    assert (np.diff(th) >= 0).all()


def test_optimise_thresholds_beats_naive_on_shifted_scores():
    """分數被壓縮/平移時，最佳化切點必須勝過直接四捨五入。"""
    from aicup.scoring import optimise_thresholds, apply_thresholds
    rng = np.random.default_rng(1)
    y = rng.integers(0, 10, 2000)
    s = 2.0 + 0.4 * y + rng.normal(scale=0.3, size=2000)     # 刻意壓縮到 2~6
    naive = np.clip(np.round(s), 0, 9).astype(int)
    th = optimise_thresholds(s, y, n_restarts=4)
    assert rank_final_score(apply_thresholds(s, th), y) > rank_final_score(naive, y)


# ---------------------------------------------------------------------------
# 切點最佳化：這一組測試是為了擋住一個真實踩過的坑
#
# 舊版 optimise_thresholds 用 Nelder-Mead。但官方分數對切點是**分段常數**的，
# 梯度處處為零，Nelder-Mead 在上面本質是隨機遊走：
#   - 輸入最後幾個 bit 的差異（例如 MPS vs CPU 推論）就會走到完全不同的解
#   - 解會退化：實測出現過 [..., 4.88, 4.88, ...] 兩個切點重合，
#     導致該等級分不到任何題目，端到端分數從 0.48 掉到 0.33
# 換成動態規劃後是全域最佳且確定性的。以下測試鎖住這兩個性質。
# ---------------------------------------------------------------------------

def _synthetic_scores(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_LEVELS, n)
    return 2.0 + 0.4 * y + rng.normal(scale=0.5, size=n), y


def test_optimise_thresholds_is_deterministic():
    """同樣輸入重複呼叫必須逐位元相同。"""
    s, y = _synthetic_scores()
    assert np.array_equal(optimise_thresholds(s, y), optimise_thresholds(s, y))


def test_optimise_thresholds_is_stable_under_tiny_perturbation():
    """加上 1e-12 的擾動不該讓切點大幅跳動（Nelder-Mead 版本會）。"""
    s, y = _synthetic_scores()
    a = optimise_thresholds(s, y)
    b = optimise_thresholds(s + 1e-12, y)
    assert np.abs(a - b).max() < 1e-6


def test_optimise_thresholds_never_degenerate():
    """切點必須嚴格遞增：重複的切點代表有等級永遠不會被預測到。"""
    for seed in range(5):
        s, y = _synthetic_scores(seed=seed)
        th = optimise_thresholds(s, y)
        assert len(th) == N_LEVELS - 1
        assert np.all(np.diff(th) > 0), f"seed={seed} 切點退化：{th}"


def test_optimise_thresholds_beats_naive_rounding():
    """最佳化切點要明顯勝過直接四捨五入。"""
    s, y = _synthetic_scores()
    opt = rank_final_score(apply_thresholds(s, optimise_thresholds(s, y)), y)
    naive = rank_final_score(np.clip(np.round(s), 0, N_LEVELS - 1).astype(int), y)
    assert opt > naive + 0.05


def test_optimise_thresholds_is_globally_optimal_on_tiny_case():
    """在小案例上與暴力枚舉的最佳解一致（DP 宣稱全域最佳，這裡驗證）。"""
    import itertools
    rng = np.random.default_rng(3)
    n_levels = 3
    s = np.sort(rng.normal(size=12))
    y = rng.integers(0, n_levels, 12)

    S = rank_score_matrix(n_levels)

    def score(th):
        return S[apply_thresholds(s, np.asarray(th)), y].mean()

    best = score(optimise_thresholds(s, y, n_levels=n_levels))
    # 暴力：切點只可能落在相鄰樣本之間，枚舉所有位置組合
    mids = np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]])
    brute = max(score(c) for c in itertools.combinations(mids, n_levels - 1))
    assert best >= brute - 1e-9, f"DP {best} 低於暴力最佳 {brute}"


def test_available_backends_survives_broken_install():
    """套件『裝好了但載不起來』時，只能少一個 backend，不能讓整支管線崩掉。

    真實情況：macOS 上 `pip install lightgbm` 會成功，但 import 時因為缺
    `libomp.dylib` 丟 OSError（不是 ImportError）。早期版本只攔 ImportError，
    等於在比賽當天才會炸。
    """
    import builtins

    from aicup.gbdt import available_backends

    real_import = builtins.__import__

    def boom(name, *a, **kw):
        if name in ("lightgbm", "xgboost", "catboost"):
            raise OSError("dlopen: libomp.dylib not found")
        return real_import(name, *a, **kw)

    builtins.__import__ = boom
    try:
        assert available_backends() == ["hist"]
    finally:
        builtins.__import__ = real_import
