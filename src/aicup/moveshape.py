"""逐手**棋形**特徵：補上 `moveseq.py` 丟掉的「排列」。

`moveseq.py` 的 12 維裡有鄰居的**數量**（我方幾顆、對方幾顆、空幾個），
但沒有**排列**。而圍棋裡好壞形的差別正是排列——
愚形空三角、虎口、竹節，鄰居數量可能一樣，棋形天差地遠。

更直接的問題：那 12 維用**曼哈頓距離**表示與自己前一手的關係，
而 **尖（對角，曼哈頓 2）與 跳（一間，曼哈頓 2）距離相同**，
飛（1,2，曼哈頓 3）與 大跳（0,3，曼哈頓 3）也相同。
這四種是圍棋裡完全不同的手法，卻被壓成同一個數字。

這裡改成把「這手與最近的自己人的相對位置」離散成**手法類別**
（接／尖／跳／飛／象步／大飛／遠），這是圍棋實際使用的詞彙，
也是業餘與段位差異最明顯的地方。

12 維，與 `moveseq` 同樣的長度與遮罩規則，用 `--extra-sequences` 沿特徵軸接上去。
"""
from __future__ import annotations

import numpy as np

from .features import normalise_color
from .goboard import EMPTY, SIZE, Board, parse_moves
from .moveseq import MAX_MOVES, MIN_MY_MOVES

N_SHAPE_FEATURES = 12
_R = 4                                   # 只在半徑 4 的視窗內找最近的棋子

# 9×9 視窗內每一格到中心的 Chebyshev 距離與 (|dr|,|dc|)，預先算好
_off = np.arange(-_R, _R + 1)
_DR = np.abs(_off)[:, None] * np.ones((1, 2 * _R + 1), dtype=int)
_DC = np.ones((2 * _R + 1, 1), dtype=int) * np.abs(_off)[None, :]
# 排序鍵 = Chebyshev × 10 + Manhattan。
# 只用 Chebyshev 的話 接(0,1) 與 尖(1,1) 同分，平手會由掃描順序決定——那沒有意義。
# 加上 Manhattan 破平手後，接(10+1) 勝過 尖(10+2)，飛(20+3) 勝過 象步(20+4)，
# 與圍棋裡「先看有沒有貼著的子」的直覺一致。
_CHEB = np.maximum(_DR, _DC) * 10 + (_DR + _DC)
_CHEB[_R, _R] = 9999                      # 中心（這手自己）不算

# 手法類別：把 sorted(|dr|,|dc|) 對到一個槽
#   0 接(0,1)  1 尖(1,1)  2 跳(0,2)  3 飛(1,2)  4 象步(2,2)  5 大跳/大飛(0,3)(1,3)  6 遠
_RELATION = {(0, 1): 0, (1, 1): 1, (0, 2): 2, (1, 2): 3,
             (2, 2): 4, (0, 3): 5, (1, 3): 5}
N_RELATION = 7


def _nearest(win: np.ndarray, target: int) -> tuple[int, int] | None:
    """視窗內距離中心最近的 `target` 色棋子，回傳 sorted(|dr|,|dc|)。"""
    hit = win == target
    if not hit.any():
        return None
    d = np.where(hit, _CHEB, 9999)
    k = int(d.argmin())
    if d.flat[k] >= 9999:
        return None
    a, b = _DR.flat[k], _DC.flat[k]
    return (min(a, b), max(a, b))


def shape_sequence(sgf: str, color: str | int,
                   max_moves: int = MAX_MOVES) -> np.ndarray | None:
    """(max_moves, 12) float32。有效條件與 `move_sequence` **完全一致**，
    兩份遮罩才對得起來（否則 `--extra-sequences` 的交集會悄悄少掉棋局）。"""
    my = normalise_color(color)
    opp = 3 - my
    moves = parse_moves(sgf)
    if len(moves) < 30:
        return None

    board = Board()
    out = np.zeros((max_moves, N_SHAPE_FEATURES), dtype=np.float32)
    k = 0
    for _t, (col, r, c) in enumerate(moves):
        if k >= max_moves:
            break
        # 先看盤面（這手還沒下），再落子——「與已有棋子的關係」才是這手的手法
        r0, r1 = max(0, r - _R), min(SIZE, r + _R + 1)
        c0, c1 = max(0, c - _R), min(SIZE, c + _R + 1)
        if col == my:
            win = np.full((2 * _R + 1, 2 * _R + 1), -1, dtype=np.int8)
            win[r0 - (r - _R):r1 - (r - _R), c0 - (c - _R):c1 - (c - _R)] = \
                board.stones.reshape(SIZE, SIZE)[r0:r1, c0:c1]
        board.play(col, r, c)
        if col != my:
            continue

        v = out[k]
        rel = _nearest(win, my)
        v[_RELATION.get(rel, 6) if rel is not None else 6] = 1.0

        ro = _nearest(win, opp)
        if ro == (0, 1):
            v[7] = 1.0                                   # 貼／碰：緊貼對方
        elif ro == (1, 1):
            v[8] = 1.0                                   # 尖沖：斜接觸
        elif ro is not None and max(ro) <= 3:
            v[9] = 1.0                                   # 在對方附近但不接觸

        # 愚形空三角：某個 2×2 方格裡有三顆我方的子、剩下一格是空的。
        # ⚠️ 不能只檢查「剛下的子是三角的頂點」——補上第三顆的那一手
        # 往往才是愚形發生的時刻，而那一手並不在頂點上。
        st = board.stones.reshape(SIZE, SIZE)
        for dr in (-1, 1):
            for dc in (-1, 1):
                ra, ca = r + dr, c + dc
                if not (0 <= ra < SIZE and 0 <= ca < SIZE):
                    continue
                others = (st[ra, c], st[r, ca], st[ra, ca])
                if sum(o == my for o in others) == 2 and \
                        sum(o == EMPTY for o in others) == 1:
                    v[10] = 1.0
                    break
            if v[10]:
                break

        v[11] = min(int((win == my).sum()), 8) / 8.0      # 半徑 4 內的我方厚度
        k += 1

    return out if k >= MIN_MY_MOVES else None


def batch_shape_sequences(sgfs, colors, max_moves: int = MAX_MOVES):
    n = len(sgfs)
    out = np.zeros((n, max_moves, N_SHAPE_FEATURES), dtype=np.float16)
    ok = np.zeros(n, dtype=bool)
    for i, (s, c) in enumerate(zip(sgfs, colors)):
        try:
            v = shape_sequence(s, c, max_moves=max_moves)
        except Exception:
            v = None
        if v is not None:
            out[i] = v.astype(np.float16)
            ok[i] = True
    return out, ok
