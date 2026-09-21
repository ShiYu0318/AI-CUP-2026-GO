"""逐手**局面判斷品質**：把最強的整局訊號補上時間軸。

## 為什麼是這幾個

`features.py` 裡與棋力相關係數最高的幾個特徵：

    territory_rate  -0.959  下在「自己已有子、對方沒子」的區域的比例
    invade_rate     +0.953  下在對方子較多處的比例
    empty_triangle  -0.936  愚形
    wasted_fill     -0.747  在官子階段把子下進自己已包圍的空點
    cap_per_move    +0.654  每手提到多少子

`moveseq` 的 12 維裡有提子數，`moveshape` 補上了愚形，
但**最強的兩個（territory / invade）序列模型完全看不到**。

誠實地說：territory / invade / wasted 這三個**可以**從 `moveseq` 既有的
鄰居計數推導出來（我方鄰居 > 0 且對方鄰居 = 0 就是 territory），
所以它們是「把模型可以自己算的東西直接給它」。
真正**無法**從既有 24 維推導的是：`atari_given`、`connect`、
`opp_lib_drop`（都需要對方棋串的氣數與棋串身分）以及半徑 3 的厚薄——
8 維裡有 5 維是全新的資訊。

而這正是當初做序列模型的理由：整局統計說「territory_rate 是 0.31」，
但「在自己地裡補棋」發生在**布局**還是**收官**，意義天差地遠——
前者是厚實，後者是純粹的浪費。

## 8 維／手

    0 own_density    半徑 3 內我方子數（正規化）
    1 opp_density    半徑 3 內對方子數
    2 territory      在自己地盤下棋（四鄰有我方、無對方）——與 features.py 同判準
    3 invade         侵入（四鄰對方多於我方）——與 features.py 同判準
    4 wasted_fill    四鄰全是我方且沒提到子——實打實的浪費一手
    5 atari_given    這手讓對方某條棋串只剩 1 氣
    6 connect        這手把兩條以上原本分開的我方棋串接起來
    7 opp_lib_drop   這手讓相鄰對方棋串總氣數減少多少（正規化）

有效條件與 `move_sequence` **完全一致**，兩份遮罩才對得起來。
"""
from __future__ import annotations

import numpy as np

from .features import normalise_color
from .goboard import EMPTY, SIZE, Board, parse_moves
from .moveseq import MAX_MOVES, MIN_MY_MOVES

N_QUALITY_FEATURES = 8
_R = 3
_off = np.arange(-_R, _R + 1)
_IN = (np.abs(_off)[:, None] + np.abs(_off)[None, :]) <= 2 * _R


def _neighbours(r: int, c: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < SIZE and 0 <= cc < SIZE:
            yield rr * SIZE + cc


def quality_sequence(sgf: str, color: str | int,
                     max_moves: int = MAX_MOVES) -> np.ndarray | None:
    my = normalise_color(color)
    opp = 3 - my
    moves = parse_moves(sgf)
    if len(moves) < 30:
        return None

    board = Board()
    out = np.zeros((max_moves, N_QUALITY_FEATURES), dtype=np.float32)
    k = 0
    for col, r, c in moves:
        if k >= max_moves:
            break
        if col != my:
            board.play(col, r, c)
            continue

        st = board.stones.reshape(SIZE, SIZE)
        r0, r1 = max(0, r - _R), min(SIZE, r + _R + 1)
        c0, c1 = max(0, c - _R), min(SIZE, c + _R + 1)
        win = st[r0:r1, c0:c1]
        n_own = int((win == my).sum())
        n_opp = int((win == opp).sum())

        # 落子**前**相鄰的對方棋串總氣數（用來算這手減了多少）
        before, seen = 0, set()
        for idx in _neighbours(r, c):
            if board.stones[idx] == opp and idx not in seen:
                grp, libs = board._group_and_liberties(idx)
                seen |= set(grp)
                before += len(libs)
        own_adj = [i for i in _neighbours(r, c) if board.stones[i] == my]
        groups_before = set()
        for idx in own_adj:
            grp, _ = board._group_and_liberties(idx)
            groups_before.add(min(grp))
        # ⚠️ territory / invade / wasted 的判準要與 `features.py` **完全一致**
        # （四個正交鄰居，不是半徑 3 的窗）。第一版用半徑 3，
        # 結果「自地」只觸發 0.12%、「廢手」完全不觸發——範圍放太大就永遠不成立。
        nb = list(_neighbours(r, c))
        near_my = sum(1 for i in nb if board.stones[i] == my)
        near_opp = sum(1 for i in nb if board.stones[i] == opp)
        all_mine = bool(nb) and near_my == len(nb)

        taken = board.play(col, r, c)

        after, atari, seen = 0, 0.0, set()
        for idx in _neighbours(r, c):
            if board.stones[idx] == opp and idx not in seen:
                grp, libs = board._group_and_liberties(idx)
                seen |= set(grp)
                after += len(libs)
                if len(libs) == 1:
                    atari = 1.0

        v = out[k]
        v[0] = min(n_own, 12) / 12.0
        v[1] = min(n_opp, 12) / 12.0
        v[2] = 1.0 if (near_my > 0 and near_opp == 0) else 0.0
        v[3] = 1.0 if near_opp > near_my else 0.0
        v[4] = 1.0 if (all_mine and taken == 0) else 0.0
        v[5] = atari
        v[6] = 1.0 if len(groups_before) >= 2 else 0.0
        v[7] = min(max(before - after, 0), 6) / 6.0
        k += 1

    return out if k >= MIN_MY_MOVES else None


def batch_quality_sequences(sgfs, colors, max_moves: int = MAX_MOVES):
    n = len(sgfs)
    out = np.zeros((n, max_moves, N_QUALITY_FEATURES), dtype=np.float16)
    ok = np.zeros(n, dtype=bool)
    for i, (s, c) in enumerate(zip(sgfs, colors)):
        try:
            v = quality_sequence(s, c, max_moves=max_moves)
        except Exception:
            v = None
        if v is not None:
            out[i] = v.astype(np.float16)
            ok[i] = True
    return out, ok
