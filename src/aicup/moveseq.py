"""逐手序列特徵：保留「什麼時候發生」。

`features.py` 的 589 維是**整局的統計摘要**。愚形率是 0.13，
但愚形出現在第 30 手還是第 150 手完全看不出來——時序資訊被整個丟掉。

這裡每一手抽一個小向量，讓下游的 1D 卷積讀序列。
實測（6 組獨立題目抽樣、玩家不重疊）：

    589 維手工特徵 MLP      0.3631 ±0.0149
    逐手序列 1D-CNN 單獨    0.3831 ±0.0094   +0.0200（6/6 勝，t=3.64）
    兩者融合 w=0.5          0.4146 ±0.0146   +0.0515（6/6 勝，t=5.74）

組內殘差相關 +0.530，互補性明確。
**每一手只用 12 個純量、只訓練 2.8 萬局，就為 589 維手工特徵帶來 +0.05。**

比板面 CNN 便宜兩個數量級：不需要 34 層 19×19 的張量，
一局只要 160×12 個 float16。
"""
from __future__ import annotations

import numpy as np

from .features import normalise_color
from .goboard import BLACK, EMPTY, SIZE, WHITE, Board, parse_moves

MAX_MOVES = 160          # 中位數約 200 手；前 160 手已涵蓋開局到中盤收束
N_MOVE_FEATURES = 12
MIN_MY_MOVES = 15

# 每一點的「第幾線」（0 = 第一線，上限 4）
_edge = np.minimum(np.arange(SIZE), SIZE - 1 - np.arange(SIZE))
_LINE_LUT = np.minimum(np.minimum.outer(_edge, _edge), 4).astype(np.float32)

_STAR = {(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)}


def move_sequence(sgf: str, color: str | int) -> np.ndarray | None:
    """回傳 (MAX_MOVES, N_MOVE_FEATURES) float32，只填**我方**的手，其餘補零。

    無法解析、太短、或我方的手少於 MIN_MY_MOVES 時回傳 None
    （與 `game_features` 的判斷一致，兩邊的有效遮罩才對得起來）。
    """
    my = normalise_color(color)
    moves = parse_moves(sgf)
    if len(moves) < 30:
        return None

    board = Board()
    out = np.zeros((MAX_MOVES, N_MOVE_FEATURES), dtype=np.float32)
    k = 0
    prev_mine: tuple[int, int] | None = None

    for t, (col, r, c) in enumerate(moves):
        if k >= MAX_MOVES:
            break
        captured = board.play(col, r, c)
        if col != my:
            continue
        idx = r * SIZE + c
        group, libs = board._group_and_liberties(idx)

        neigh = []
        if r > 0:            neigh.append(board.stones[(r - 1) * SIZE + c])
        if r < SIZE - 1:     neigh.append(board.stones[(r + 1) * SIZE + c])
        if c > 0:            neigh.append(board.stones[r * SIZE + c - 1])
        if c < SIZE - 1:     neigh.append(board.stones[r * SIZE + c + 1])

        d_prev = (abs(r - prev_mine[0]) + abs(c - prev_mine[1])) if prev_mine else 0
        d_opp = (abs(r - moves[t - 1][1]) + abs(c - moves[t - 1][2])
                 if t > 0 and moves[t - 1][0] != my else 0)

        out[k] = (
            t / 200.0,                                   # 手數（相對）
            _LINE_LUT[r, c] / 4.0,                       # 第幾線
            min(len(libs), 6) / 6.0,                     # 這手所屬棋串的氣
            min(len(group), 8) / 8.0,                    # 棋串大小
            min(captured, 4) / 4.0,                      # 這手提了幾子
            1.0 if len(libs) == 1 else 0.0,              # 自己走進打吃
            min(d_prev, 18) / 18.0,                      # 與自己前一手的距離（脫先程度）
            min(d_opp, 18) / 18.0,                       # 與對手前一手的距離（應手程度）
            sum(1 for v in neigh if v == my) / 4.0,
            sum(1 for v in neigh if v not in (my, EMPTY)) / 4.0,
            sum(1 for v in neigh if v == EMPTY) / 4.0,
            1.0 if (r, c) in _STAR else 0.0,
        )
        prev_mine = (r, c)
        k += 1

    return out if k >= MIN_MY_MOVES else None


def batch_move_sequences(sgfs, colors):
    """一批棋局 → (N, MAX_MOVES, N_MOVE_FEATURES) float16 與布林有效遮罩。

    float16 就夠：所有欄位都已正規化到 [0, 1]，
    而 (1M, 160, 12) 用 float32 要 7.7 GB、float16 只要 3.8 GB。
    """
    n = len(sgfs)
    out = np.zeros((n, MAX_MOVES, N_MOVE_FEATURES), dtype=np.float16)
    ok = np.zeros(n, dtype=bool)
    for i, (s, c) in enumerate(zip(sgfs, colors)):
        try:
            v = move_sequence(s, c)
        except Exception:                    # 單局壞掉不該讓整批失敗
            v = None
        if v is not None:
            out[i] = v.astype(np.float16)
            ok[i] = True
    return out, ok


def sequence_lengths(seq: np.ndarray) -> np.ndarray:
    """由補零推回每局實際的我方手數（給 mask 用）。"""
    return (np.abs(seq).sum(axis=-1) > 0).sum(axis=-1)
