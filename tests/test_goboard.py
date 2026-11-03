import os
"""SGF 解析與棋盤邏輯的測試（提子、氣、座標系）。"""
from aicup.data import TRAINING_DIR
import numpy as np

from aicup.goboard import (
    SIZE, EMPTY, BLACK, WHITE, Board, parse_moves, replay, iter_boards,
)


def test_parse_coordinates_row_col_order():
    """SGF 的 [xy]：x 是欄、y 是列。B[pd] => col=15, row=3。"""
    mv = parse_moves("(;SZ[19];B[pd];W[dp])")
    assert mv == [(BLACK, 3, 15), (WHITE, 15, 3)]


def test_parse_ignores_sz_property():
    assert len(parse_moves("(;SZ[19];B[aa])")) == 1


def test_parse_corner_extremes():
    mv = parse_moves("(;B[aa];W[ss])")
    assert mv[0][1:] == (0, 0)
    assert mv[1][1:] == (SIZE - 1, SIZE - 1)


def test_single_stone_capture_in_corner():
    """白佔角 (0,0)，黑下 (0,1) 與 (1,0) 即可提。"""
    b = Board()
    b.play(WHITE, 0, 0)
    b.play(BLACK, 0, 1)
    assert b.stones[0] == WHITE          # 還有一口氣
    n = b.play(BLACK, 1, 0)
    assert n == 1
    assert b.stones[0] == EMPTY
    assert b.n_captured[WHITE] == 1


def test_group_capture():
    """白兩子連成一串，黑圍滿六口氣後整串被提。"""
    b = Board()
    b.play(WHITE, 5, 5); b.play(WHITE, 5, 6)
    for r, c in [(4, 5), (4, 6), (6, 5), (6, 6), (5, 4)]:
        b.play(BLACK, r, c)
        assert b.stones[5 * SIZE + 5] == WHITE
    n = b.play(BLACK, 5, 7)
    assert n == 2
    assert b.stones[5 * SIZE + 5] == EMPTY and b.stones[5 * SIZE + 6] == EMPTY


def test_capture_does_not_remove_own_stones():
    b = Board()
    b.play(BLACK, 3, 3)
    b.play(WHITE, 3, 4)
    b.play(BLACK, 3, 5)
    assert b.stones[3 * SIZE + 3] == BLACK


def test_liberties_of_isolated_stone():
    b = Board()
    b.play(BLACK, 9, 9)
    _, libs = b._group_and_liberties(9 * SIZE + 9)
    assert len(libs) == 4
    b2 = Board(); b2.play(BLACK, 0, 0)
    _, libs2 = b2._group_and_liberties(0)
    assert len(libs2) == 2               # 角上只有兩口氣


def test_liberty_planes_one_hot_per_stone():
    """每顆盤上的子必須恰好出現在一個氣分層平面。"""
    b = Board()
    for r, c, col in [(3, 3, BLACK), (3, 4, BLACK), (9, 9, WHITE), (0, 0, WHITE)]:
        b.play(col, r, c)
    p = b.liberty_planes(buckets=4)
    assert p.shape == (8, SIZE, SIZE)
    n_stones = int((b.stones != EMPTY).sum())
    assert int(p.sum()) == n_stones
    assert p.max() == 1


def test_liberty_planes_bucket_assignment():
    b = Board()
    b.play(WHITE, 0, 0)                  # 角上孤子 = 2 氣 -> bucket index 1
    p = b.liberty_planes(buckets=4)
    white_base = 4
    assert p[white_base + 1, 0, 0] == 1
    assert p[white_base + 0, 0, 0] == 0


def test_planes_bw_matches_board():
    b = Board()
    b.play(BLACK, 1, 2); b.play(WHITE, 5, 5)
    p = b.planes_bw()
    assert p.shape == (2, SIZE, SIZE)
    assert p[0, 1, 2] == 1 and p[1, 5, 5] == 1
    assert p.sum() == 2


def test_stone_count_invariant_on_real_game():
    """盤上子數 = 總手數 - 被提子數。這個不變量抓得到絕大多數提子邏輯錯誤。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(TRAINING_DIR, "train_6D.csv"), nrows=20)
    for sgf in df.sgf_content:
        mv = parse_moves(sgf)
        b = replay(mv)
        on_board = int((b.stones != EMPTY).sum())
        captured = b.n_captured[BLACK] + b.n_captured[WHITE]
        assert on_board + captured == len(mv)


def test_iter_boards_length():
    mv = parse_moves("(;B[aa];W[bb];B[cc])")
    assert len(list(iter_boards(mv))) == 3


def test_replay_upto():
    mv = parse_moves("(;B[aa];W[bb];B[cc])")
    b = replay(mv, upto=2)
    assert int((b.stones != EMPTY).sum()) == 2
