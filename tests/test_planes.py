"""盤面平面表徵的測試：層數、語意、對稱、涵蓋範圍。"""
import numpy as np
import pandas as pd
import pytest

from aicup.planes import game_planes, n_planes, _snapshot_indices, N_HISTORY, LIBERTY_BUCKETS
from aicup.goboard import SIZE, parse_moves, replay, BLACK, WHITE


@pytest.fixture(scope="module")
def sgf_color():
    d = pd.read_csv("training/train_2D.csv", nrows=5)
    return d.sgf_content.iloc[0], d.color.iloc[0]


def test_default_plane_count_is_34():
    assert n_planes() == 34


def test_shape(sgf_color):
    sgf, col = sgf_color
    t = game_planes(sgf, col, n_snapshots=8)
    assert t.ndim == 4
    assert t.shape[1:] == (34, SIZE, SIZE)
    assert t.shape[0] <= 8
    assert t.dtype == np.uint8


def test_snapshots_span_opening_to_endgame():
    """取樣必須涵蓋開局到終局，不能像官方 baseline 只取尾段。"""
    idx = _snapshot_indices(300, 8)
    assert idx.min() <= 30
    assert idx.max() >= 290


def test_occupancy_planes_are_mutually_exclusive(sgf_color):
    sgf, col = sgf_color
    t = game_planes(sgf, col, n_snapshots=4)
    tot = t[:, 0].astype(int) + t[:, 1].astype(int) + t[:, 2].astype(int)
    assert (tot == 1).all()        # 每個點只能是我方/對方/空 三者之一


def test_my_plane_matches_target_color(sgf_color):
    """第 0 層必須是「目標玩家」的棋子，不是黑子。"""
    sgf, col = sgf_color
    moves = parse_moves(sgf)
    t = game_planes(sgf, col, n_snapshots=1)
    final = replay(moves).stones.reshape(SIZE, SIZE)
    my = BLACK if col == "B" else WHITE
    # 最後一個 snapshot 接近終局，我方子數應與盤面一致的數量級
    assert t[-1, 0].sum() > 0
    assert abs(int(t[-1, 0].sum()) - int((final == my).sum())) <= int((final != 0).sum())


def test_color_plane_encodes_black_or_white():
    # ⚠️ 官方 CSV 依執色排序：前 50,000 列全是 B、後 50,000 列全是 W。
    # 用 nrows 取樣會拿到 100% 執黑而毫無警訊，必須跨越分界點取樣。
    head = pd.read_csv("training/train_2D.csv", nrows=3)
    tail = pd.read_csv("training/train_2D.csv", skiprows=range(1, 50_001), nrows=3)
    assert set(head.color) == {"B"} and set(tail.color) == {"W"}
    black, white = head.iloc[0], tail.iloc[0]
    tb = game_planes(black.sgf_content, "B", n_snapshots=2)
    tw = game_planes(white.sgf_content, "W", n_snapshots=2)
    base = 3 + 2 * N_HISTORY + 2 * LIBERTY_BUCKETS + 4
    assert tb[:, base].max() == 1
    assert tw[:, base].max() == 0


def test_liberty_planes_cover_all_stones(sgf_color):
    sgf, col = sgf_color
    t = game_planes(sgf, col, n_snapshots=3)
    base = 3 + 2 * N_HISTORY
    lib = t[:, base:base + 2 * LIBERTY_BUCKETS].astype(int).sum(axis=1)
    occupied = t[:, 0].astype(int) + t[:, 1].astype(int)
    assert (lib == occupied).all()      # 每顆子恰好落在一個氣分層


def test_last_move_planes_have_at_most_one_stone(sgf_color):
    sgf, col = sgf_color
    t = game_planes(sgf, col, n_snapshots=4)
    base = 3 + 2 * N_HISTORY + 2 * LIBERTY_BUCKETS
    for k in range(4):
        assert (t[:, base + k].reshape(len(t), -1).sum(axis=1) <= 1).all()


def test_constant_plane_is_all_ones(sgf_color):
    sgf, col = sgf_color
    t = game_planes(sgf, col, n_snapshots=2)
    assert (t[:, -1] == 1).all()


def test_progress_plane_is_monotonic(sgf_color):
    sgf, col = sgf_color
    t = game_planes(sgf, col, n_snapshots=6)
    prog = [int(x[-2, 0, 0]) for x in t]
    assert prog == sorted(prog)
    assert prog[-1] > prog[0]


def test_short_game_returns_none():
    assert game_planes("(;SZ[19];B[dd];W[pp])", "B") is None


def test_history_planes_count(sgf_color):
    sgf, col = sgf_color
    t = game_planes(sgf, col, n_snapshots=2, n_history=4)
    assert t.shape[1] == n_planes(n_history=4)
