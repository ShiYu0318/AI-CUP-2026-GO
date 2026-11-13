"""逐手的「落子位置＋落子前的局部盤面」序列（競賽一的身分特徵）。

moveseq／moveshape 的逐手特徵沒有**落子的實際座標**，也看不到周圍的棋形；
但「在哪裡下、在什麼形狀下選這手」正是個人簽名（帳本中最強的身分特徵是「開局順序族」）。

每一手我方落子 52 個值（uint8，0~255，模型端乘 1/255）：
  - 0、1：列、行座標（(r+1)·13、(c+1)·13，13~247；從 1 起算，空棋盤角落的一手也不會整列為 0）
  - 2~26：落子**前**、以落子點為中心的 5x5 我方子（255／0；盤外為 0）
  - 27~51：同上，對方子
有效判斷與 moveseq 一致（總手數 < 30 或我方手數 < MIN_MY_MOVES 視為失敗），各序列庫的遮罩才對得起來。
"""
from __future__ import annotations

import numpy as np

from .features import normalise_color
from .goboard import EMPTY, SIZE, Board, parse_moves
from .moveseq import MAX_MOVES, MIN_MY_MOVES

R = 2                                   # 5x5
N_PATCH_FEATURES = 2 + 2 * (2 * R + 1) ** 2
PATCH_SCALE = 1.0 / 255.0


def patch_sequence(sgf: str, color, max_moves: int = MAX_MOVES) -> np.ndarray | None:
    my = normalise_color(color)
    moves = parse_moves(sgf)
    if len(moves) < 30:
        return None
    board = Board()
    out = np.zeros((max_moves, N_PATCH_FEATURES), dtype=np.uint8)
    w = 2 * R + 1
    k = 0
    for col, r, c in moves:
        if k >= max_moves:
            break
        if col == my:
            st = board.stones.reshape(SIZE, SIZE)
            own = np.zeros((w, w), np.uint8); opp = np.zeros((w, w), np.uint8)
            r0, r1, c0, c1 = max(r - R, 0), min(r + R + 1, SIZE), max(c - R, 0), min(c + R + 1, SIZE)
            blk = st[r0:r1, c0:c1]
            sl = (slice(r0 - (r - R), r1 - (r - R)), slice(c0 - (c - R), c1 - (c - R)))
            own[sl] = (blk == my) * 255
            opp[sl] = ((blk != my) & (blk != EMPTY)) * 255
            out[k, 0], out[k, 1] = (r + 1) * 13, (c + 1) * 13     # 從 1 起算：保證每手至少一個非零值
            out[k, 2:2 + w * w] = own.ravel(); out[k, 2 + w * w:] = opp.ravel()
            k += 1
        board.play(col, r, c)
    return out if k >= MIN_MY_MOVES else None


def batch_patch_sequences(sgfs, colors, max_moves: int = MAX_MOVES):
    """一批棋局 → (N, max_moves, N_PATCH_FEATURES) uint8 與有效遮罩。"""
    n = len(sgfs)
    out = np.zeros((n, max_moves, N_PATCH_FEATURES), dtype=np.uint8)
    ok = np.zeros(n, dtype=bool)
    for i, (s, c) in enumerate(zip(sgfs, colors)):
        try:
            v = patch_sequence(s, c, max_moves=max_moves)
        except Exception:
            v = None
        if v is not None:
            out[i] = v; ok[i] = True
    return out, ok
