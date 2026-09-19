"""逐手序列特徵的測試。

589 維手工特徵是整局的統計摘要——愚形率是 0.13，但愚形出現在第 30 手
還是第 150 手完全看不出來。這個模組保留時序，實測為手工特徵帶來 +0.05。
"""
import numpy as np
import pytest

from aicup.moveseq import (MAX_MOVES, MIN_MY_MOVES, N_MOVE_FEATURES,
                           batch_move_sequences, move_sequence, sequence_lengths)


def _sgf(n=120):
    cs = "abcdefghijklmnopqrs"
    mv = "".join(f";{'B' if i % 2 == 0 else 'W'}"
                 f"[{cs[(i * 7 % 361) % 19]}{cs[(i * 7 % 361) // 19]}]"
                 for i in range(n))
    return "(;SZ[19]" + mv + ")"


def test_shape_and_dtype():
    v = move_sequence(_sgf(), "B")
    assert v.shape == (MAX_MOVES, N_MOVE_FEATURES) and v.dtype == np.float32


def test_all_features_normalised():
    """全部欄位必須落在 [0, 1]——float16 的精度才夠，且尺度一致。"""
    v = move_sequence(_sgf(), "B")
    assert v.min() >= 0.0 and v.max() <= 1.0


def test_only_my_moves_are_filled():
    """一局 120 手、雙方各半，我方應該剛好 60 手。"""
    assert int(sequence_lengths(move_sequence(_sgf(120), "B")[None])[0]) == 60


def test_perspective_matters():
    """黑白視角必須不同，否則整個特徵沒有意義。"""
    assert not np.array_equal(move_sequence(_sgf(), "B"), move_sequence(_sgf(), "W"))


@pytest.mark.parametrize("c", ["b", " B ", "black"])
def test_colour_spelling_insensitive(c):
    assert np.array_equal(move_sequence(_sgf(), c), move_sequence(_sgf(), "B"))


def test_unknown_colour_raises():
    with pytest.raises(ValueError):
        move_sequence(_sgf(), "X")


def test_too_short_returns_none():
    assert move_sequence("(;SZ[19];B[aa];W[bb])", "B") is None


def test_too_few_own_moves_returns_none():
    """我方手數不足時回傳 None，與 game_features 的判斷一致，遮罩才對得起來。"""
    cs = "abcdefghijklmnopqrs"
    mv = "".join(f";{'B' if i == 0 else 'W'}[{cs[i % 19]}{cs[i // 19]}]" for i in range(40))
    assert move_sequence("(;SZ[19]" + mv + ")", "B") is None


def test_truncates_at_max_moves():
    """超長對局要截斷，不能越界。"""
    v = move_sequence(_sgf(500), "B")
    assert v.shape[0] == MAX_MOVES
    assert int(sequence_lengths(v[None])[0]) <= MAX_MOVES


def test_batch_marks_failures_without_raising():
    """單局壞掉不該讓整批失敗。"""
    S, ok = batch_move_sequences([_sgf(), "(;SZ[19];B[aa])", "not an sgf"],
                                 ["B", "B", "B"])
    assert S.shape == (3, MAX_MOVES, N_MOVE_FEATURES) and S.dtype == np.float16
    assert ok.tolist() == [True, False, False]
    assert not S[1].any() and not S[2].any()


def test_move_number_is_increasing():
    """第 0 維是手數，必須單調遞增——序列模型靠它定位。"""
    v = move_sequence(_sgf(), "B")
    n = int(sequence_lengths(v[None])[0])
    assert np.all(np.diff(v[:n, 0]) > 0)


def test_min_my_moves_constant_is_respected():
    v = move_sequence(_sgf(), "B")
    assert int(sequence_lengths(v[None])[0]) >= MIN_MY_MOVES
