"""SGF 解析與圍棋盤面（含提子與氣的計算）。

本競賽的 SGF 極單純（實測全 100 萬局）：
  - 只有 SZ / B / W 三種屬性，沒有 KM / RE / HA / PB / PW / DT
  - 沒有虛手（;B[] 或 ;W[] 出現率 0%）
  - 沒有讓子，全部從空盤開始
  - 沒有分支變化
所以不需要 sgfmill，自己解析更快、也順便能算氣。

座標系統：SGF 的 [xy] 中 x 是「欄」、y 是「列」，皆以 'a' 為 0。
本模組一律用 (row, col) = (y, x) 存取，與 numpy 的索引順序一致。
"""

from __future__ import annotations

import re
from typing import Iterator, List, Tuple

import numpy as np

SIZE = 19
EMPTY, BLACK, WHITE = 0, 1, 2
_MOVE_RE = re.compile(r";([BW])\[([a-s])([a-s])\]")

# 預先算好每個點的鄰居（flat index），避免內迴圈重複算邊界
_NEIGHBORS: List[Tuple[int, ...]] = []
for _r in range(SIZE):
    for _c in range(SIZE):
        _n = []
        if _r > 0:      _n.append((_r - 1) * SIZE + _c)
        if _r < SIZE-1: _n.append((_r + 1) * SIZE + _c)
        if _c > 0:      _n.append(_r * SIZE + _c - 1)
        if _c < SIZE-1: _n.append(_r * SIZE + _c + 1)
        _NEIGHBORS.append(tuple(_n))


def parse_moves(sgf: str) -> List[Tuple[int, int, int]]:
    """把 SGF 字串解析成 [(color, row, col), ...]。

    color 為 BLACK(1) 或 WHITE(2)。回傳順序即實際手順。
    """
    out = []
    for c, xs, ys in _MOVE_RE.findall(sgf):
        col = ord(xs) - 97
        row = ord(ys) - 97
        out.append((BLACK if c == "B" else WHITE, row, col))
    return out


class Board:
    """19x19 圍棋盤，支援提子與氣的計算。

    內部用長度 361 的一維 uint8 陣列，鄰居查表加速。
    不檢查自殺與打劫——本競賽的棋譜都是合法對局，不需要驗證。
    """

    __slots__ = ("stones", "n_captured")

    def __init__(self) -> None:
        self.stones = np.zeros(SIZE * SIZE, dtype=np.uint8)
        self.n_captured = [0, 0, 0]          # index by color

    def copy(self) -> "Board":
        b = Board()
        b.stones = self.stones.copy()
        b.n_captured = list(self.n_captured)
        return b

    # -------- 群組與氣 --------

    def _group_and_liberties(self, idx: int):
        """回傳 (該點所屬棋串的所有點, 氣的點集合)。"""
        color = self.stones[idx]
        stones = self.stones
        seen = {idx}
        stack = [idx]
        libs = set()
        while stack:
            p = stack.pop()
            for q in _NEIGHBORS[p]:
                v = stones[q]
                if v == EMPTY:
                    libs.add(q)
                elif v == color and q not in seen:
                    seen.add(q)
                    stack.append(q)
        return seen, libs

    def play(self, color: int, row: int, col: int) -> int:
        """落子並處理提子，回傳這一手提掉的子數。"""
        idx = row * SIZE + col
        self.stones[idx] = color
        opp = WHITE if color == BLACK else BLACK
        captured = 0
        checked = set()
        for q in _NEIGHBORS[idx]:
            if self.stones[q] == opp and q not in checked:
                group, libs = self._group_and_liberties(q)
                checked |= group
                if not libs:
                    for p in group:
                        self.stones[p] = EMPTY
                    captured += len(group)
        self.n_captured[opp] += captured
        return captured

    # -------- 輸出表徵 --------

    def planes_bw(self) -> np.ndarray:
        """(2, 19, 19)：黑子平面、白子平面。"""
        s = self.stones.reshape(SIZE, SIZE)
        return np.stack([(s == BLACK), (s == WHITE)]).astype(np.uint8)

    def liberty_planes(self, buckets: int = 4) -> np.ndarray:
        """(2*buckets, 19, 19)：黑/白各自「氣數為 1, 2, ..., >=buckets」的分層平面。

        氣的分層是 AI CUP 2023 跨多支隊伍獨立實作、公認增益最大的特徵
        （見 docs/02 附錄 A.3）。它編碼了棋塊死活的緊迫程度，
        而棋力的核心差異正是對死活的判斷力。
        """
        out = np.zeros((2 * buckets, SIZE * SIZE), dtype=np.uint8)
        visited = np.zeros(SIZE * SIZE, dtype=bool)
        stones = self.stones
        for idx in range(SIZE * SIZE):
            if stones[idx] == EMPTY or visited[idx]:
                continue
            group, libs = self._group_and_liberties(idx)
            b = min(len(libs), buckets) - 1          # 0-based bucket
            base = 0 if stones[idx] == BLACK else buckets
            for p in group:
                visited[p] = True
                out[base + b, p] = 1
        return out.reshape(2 * buckets, SIZE, SIZE)

    def as_array(self) -> np.ndarray:
        return self.stones.reshape(SIZE, SIZE).copy()


def replay(moves, upto: int | None = None) -> Board:
    """重播前 upto 手，回傳當下盤面。"""
    b = Board()
    for i, (color, r, c) in enumerate(moves):
        if upto is not None and i >= upto:
            break
        b.play(color, r, c)
    return b


def iter_boards(moves) -> Iterator[Tuple[int, Board]]:
    """逐手 yield (手序, 落子後的盤面)。盤面物件會被重複使用，需要保留請自行 copy。"""
    b = Board()
    for i, (color, r, c) in enumerate(moves):
        b.play(color, r, c)
        yield i, b
