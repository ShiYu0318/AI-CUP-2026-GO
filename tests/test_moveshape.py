"""棋形序列的不變式。

最重要的一條：**有效遮罩必須與 `moveseq` 完全一致**。
`--extra-sequences` 是把多份庫沿特徵軸接起來、遮罩取交集，
兩邊的有效判斷一旦分岔，訓練集會悄悄少掉棋局而且沒有任何錯誤訊息。
"""
import numpy as np
import pytest

from aicup.moveseq import MAX_MOVES, batch_move_sequences, move_sequence
from aicup.moveshape import (N_SHAPE_FEATURES, batch_shape_sequences,
                             shape_sequence)

SGF = ("(;GM[1]SZ[19]" + "".join(
    f";{'B' if i % 2 == 0 else 'W'}[{chr(97 + (i * 3) % 19)}{chr(97 + (i * 5) % 19)}]"
    for i in range(120)) + ")")


def test_shape_and_move_masks_agree():
    sgfs = [SGF, SGF, "(;GM[1]SZ[19];B[aa];W[bb])", ""]
    colors = ["B", "W", "B", "B"]
    _, ok_shape = batch_shape_sequences(sgfs, colors)
    _, ok_move = batch_move_sequences(sgfs, colors)
    assert list(ok_shape) == list(ok_move)


def test_shape_is_one_hot_over_relations():
    v = shape_sequence(SGF, "B")
    assert v is not None and v.shape == (MAX_MOVES, N_SHAPE_FEATURES)
    live = v.sum(axis=-1) > 0
    # 欄 0..6 是「與最近的我方棋子」的手法類別，互斥且必有一個
    assert np.all(v[live, :7].sum(axis=-1) == 1.0)
    # 欄 7..9 是「與最近的對方棋子」，至多一個
    assert np.all(v[live, 7:10].sum(axis=-1) <= 1.0)
    assert set(np.unique(v[:, 10])) <= {0.0, 1.0}      # 愚形是二元的


def test_padding_after_my_moves_is_zero():
    v = shape_sequence(SGF, "B")
    n = int((np.abs(v).sum(axis=-1) > 0).sum())
    assert np.all(v[n:] == 0.0)


def test_empty_triangle_detected():
    # 愚形空三角：B 在 (0,0) (0,1) (1,0)，(1,1) 空
    sgf = ("(;GM[1]SZ[19];B[aa];W[pp];B[ba];W[pq];B[ab];W[qq]"
           + "".join(f";{'B' if i % 2 == 0 else 'W'}[{chr(97 + 5 + i % 10)}"
                     f"{chr(97 + 5 + (i * 7) % 10)}]" for i in range(60)) + ")")
    v = shape_sequence(sgf, "B")
    assert v is not None
    assert v[:, 10].max() == 1.0, "應該偵測到空三角"


@pytest.mark.parametrize("color", ["B", "W"])
def test_shape_handles_both_colors(color):
    v = shape_sequence(SGF, color)
    assert v is not None and np.isfinite(v).all()
