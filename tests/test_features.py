import os
"""手工特徵的測試：維度一致、無 NaN、對稱不變、顏色語意正確。"""
from aicup.data import TRAINING_DIR
import numpy as np
import pandas as pd
import pytest

from aicup.features import (
    FEATURE_NAMES, N_FEATURES, N_FOLD, game_features, batch_features, _FOLD_LUT,
)
from aicup.goboard import SIZE


@pytest.fixture(scope="module")
def sample():
    d = pd.read_csv(os.path.join(TRAINING_DIR, "train_4D.csv"), nrows=60)
    return d


def test_feature_names_match_length():
    assert len(FEATURE_NAMES) == N_FEATURES
    assert len(set(FEATURE_NAMES)) == N_FEATURES        # 不可有重複名稱


def test_fold_table_is_8_fold_symmetric():
    """折疊表必須對 8 種對稱等價的點給出同一個編號。"""
    for r in range(SIZE):
        for c in range(SIZE):
            v = _FOLD_LUT[r, c]
            for rr, cc in [(r, SIZE-1-c), (SIZE-1-r, c), (SIZE-1-r, SIZE-1-c),
                           (c, r), (c, SIZE-1-r), (SIZE-1-c, r), (SIZE-1-c, SIZE-1-r)]:
                assert _FOLD_LUT[rr, cc] == v


def test_fold_has_55_cells():
    assert N_FOLD == 55
    assert len(np.unique(_FOLD_LUT)) == 55


def test_features_shape_and_finite(sample):
    X, ok = batch_features(sample.sgf_content.tolist(), sample.color.tolist())
    assert X.shape == (len(sample), N_FEATURES)
    assert ok.all()
    assert np.isfinite(X).all()


def test_histograms_sum_to_one(sample):
    X, ok = batch_features(sample.sgf_content.tolist(), sample.color.tolist())
    i0 = FEATURE_NAMES.index("open_cell0")
    i1 = FEATURE_NAMES.index("all_cell0")
    assert np.allclose(X[ok, i0:i0 + N_FOLD].sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(X[ok, i1:i1 + N_FOLD].sum(axis=1), 1.0, atol=1e-5)


def test_is_black_flag_matches_color(sample):
    X, ok = batch_features(sample.sgf_content.tolist(), sample.color.tolist())
    i = FEATURE_NAMES.index("is_black")
    expect = (sample.color.to_numpy() == "B").astype(float)
    assert np.allclose(X[ok, i], expect[ok])


def test_swapping_color_changes_features(sample):
    """同一局棋，站在黑方與白方的角度看，特徵必須不同。"""
    sgf = sample.sgf_content.iloc[0]
    a = game_features(sgf, "B")
    b = game_features(sgf, "W")
    assert not np.allclose(a, b)


def test_capture_features_are_consistent(sample):
    """己方提子 + 對方提子，不可能超過總手數。"""
    X, ok = batch_features(sample.sgf_content.tolist(), sample.color.tolist())
    made = X[ok, FEATURE_NAMES.index("captures_made")]
    taken = X[ok, FEATURE_NAMES.index("captures_taken")]
    n = X[ok, FEATURE_NAMES.index("n_moves")]
    assert (made >= 0).all() and (taken >= 0).all()
    assert (made + taken <= n).all()


def test_stone_ratio_in_unit_interval(sample):
    X, ok = batch_features(sample.sgf_content.tolist(), sample.color.tolist())
    r = X[ok, FEATURE_NAMES.index("stone_ratio")]
    assert ((r >= 0) & (r <= 1)).all()


def test_liberty_means_are_plausible(sample):
    """圍棋棋串的平均氣數實務上落在 1~10 之間。"""
    X, ok = batch_features(sample.sgf_content.tolist(), sample.color.tolist())
    for name in ("my_libs_mean", "opp_libs_mean"):
        v = X[ok, FEATURE_NAMES.index(name)]
        assert (v > 0.5).all() and (v < 12).all()


def test_rates_are_within_unit_interval(sample):
    X, ok = batch_features(sample.sgf_content.tolist(), sample.color.tolist())
    for name in ("resp_local", "resp_far", "my_atari_rate", "opp_atari_rate",
                 "center_frac", "endgame_line01"):
        v = X[ok, FEATURE_NAMES.index(name)]
        assert ((v >= 0) & (v <= 1)).all(), name


def test_too_short_game_returns_none():
    assert game_features("(;SZ[19];B[dd];W[pp])", "B") is None


def test_malformed_sgf_is_handled():
    X, ok = batch_features(["garbage", "(;SZ[19])"], ["B", "W"])
    assert not ok.any()
    assert np.isfinite(X).all()


def test_deterministic(sample):
    sgf, col = sample.sgf_content.iloc[3], sample.color.iloc[3]
    assert np.array_equal(game_features(sgf, col), game_features(sgf, col))


def test_parallel_matches_serial(sample):
    """平行版必須與單進程版產生完全相同的結果。"""
    from aicup.features import parallel_batch_features
    s = sample.sgf_content.tolist() * 40          # 撐過 chunk 門檻才會真的分工
    c = sample.color.tolist() * 40
    Xs, oks = batch_features(s, c)
    Xp, okp = parallel_batch_features(s, c, nproc=4, chunk=200)
    assert np.array_equal(oks, okp)
    assert np.allclose(Xs, Xp)


def test_parallel_falls_back_for_small_input(sample):
    from aicup.features import parallel_batch_features
    s = sample.sgf_content.tolist()[:5]
    c = sample.color.tolist()[:5]
    X, ok = parallel_batch_features(s, c, nproc=4, chunk=2000)
    assert X.shape == (5, N_FEATURES)


# ---------------- 局面判斷品質族 ----------------

QUALITY_FEATURES = ["wasted_fill", "cap_per_move", "abandon_atari",
                    "self_atari_rate", "atari_given_rate"]


def test_quality_features_exist():
    for n in QUALITY_FEATURES:
        assert n in FEATURE_NAMES


def test_quality_rates_are_non_negative(sample):
    X, ok = batch_features(sample.sgf_content.tolist(), sample.color.tolist())
    for n in QUALITY_FEATURES:
        v = X[ok, FEATURE_NAMES.index(n)]
        assert (v >= 0).all(), n
        assert np.isfinite(v).all(), n


def test_quality_rates_are_bounded_by_one(sample):
    """這些都是「每手一次」的比率，不可能超過 1。"""
    X, ok = batch_features(sample.sgf_content.tolist(), sample.color.tolist())
    for n in ("wasted_fill", "abandon_atari", "self_atari_rate"):
        v = X[ok, FEATURE_NAMES.index(n)]
        assert (v <= 1.0).all(), n


def test_cap_per_move_matches_captures_made(sample):
    """cap_per_move 必須等於 captures_made / 我方手數。"""
    X, ok = batch_features(sample.sgf_content.tolist(), sample.color.tolist())
    made = X[ok, FEATURE_NAMES.index("captures_made")]
    n_mine = X[ok, FEATURE_NAMES.index("n_mine")]
    per = X[ok, FEATURE_NAMES.index("cap_per_move")]
    assert np.allclose(per, made / np.maximum(n_mine, 1), atol=1e-5)


def test_quality_features_differ_by_color(sample):
    """站在黑方與白方看同一局，品質特徵必須不同。"""
    sgf = sample.sgf_content.iloc[0]
    a = game_features(sgf, "B")
    b = game_features(sgf, "W")
    ii = [FEATURE_NAMES.index(n) for n in QUALITY_FEATURES]
    assert not np.allclose(a[ii], b[ii])


def test_wasted_fill_is_zero_when_no_endgame_selffill():
    """一局只有開局手時，不可能出現「官子填自己的地」。"""
    import pandas as pd
    d = pd.read_csv(os.path.join(TRAINING_DIR, "train_1D.csv"), nrows=3)
    # 只取前 40 手，構造一個沒有官子的殘局
    sgf = d.sgf_content.iloc[0]
    moves = sgf.split(";")[:42]
    short = ";".join(moves) + ")"
    f = game_features(short, d.color.iloc[0])
    if f is not None:
        assert f[FEATURE_NAMES.index("wasted_fill")] == 0.0


def test_empty_triangle_detection():
    """愚形偵測：三子佔 2x2 的三角、第四點空 -> True；第四點有子 -> False。"""
    from aicup.features import _makes_empty_triangle
    from aicup.goboard import Board, BLACK, SIZE

    b = Board()
    b.play(BLACK, 5, 5)
    b.play(BLACK, 5, 6)
    b.play(BLACK, 6, 5)
    # (6,6) 為空 -> 剛下的 (6,5) 構成愚形
    assert _makes_empty_triangle(b.stones, 6, 5, BLACK)

    b.play(BLACK, 6, 6)          # 補上第四點就不是愚形（成為實地的方塊）
    assert not _makes_empty_triangle(b.stones, 6, 6, BLACK)


def test_empty_triangle_needs_same_colour():
    from aicup.features import _makes_empty_triangle
    from aicup.goboard import Board, BLACK, WHITE

    b = Board()
    b.play(BLACK, 5, 5)
    b.play(WHITE, 5, 6)          # 對方的子不構成我方愚形
    b.play(BLACK, 6, 5)
    assert not _makes_empty_triangle(b.stones, 6, 5, BLACK)


def test_empty_triangle_handles_board_edge():
    from aicup.features import _makes_empty_triangle
    from aicup.goboard import Board, BLACK
    b = Board()
    b.play(BLACK, 0, 0)
    assert _makes_empty_triangle(b.stones, 0, 0, BLACK) in (True, False)   # 不可越界

def test_new_shape_features_bounded(sample):
    X, ok = batch_features(sample.sgf_content.tolist(), sample.color.tolist())
    for n in ("empty_triangle", "contact_early"):
        v = X[ok, FEATURE_NAMES.index(n)]
        assert ((v >= 0) & (v <= 1)).all(), n
