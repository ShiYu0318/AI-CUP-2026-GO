"""深度模型的盤面平面表徵。

相對官方 baseline 的三個修正（見 docs/02 §3.2、§3.3）：

1. **canonical 我方/對方，而非黑/白**
   官方棋力 baseline 的特徵完全沒有執色資訊，模型被要求評某位玩家的棋力，
   卻不知道要評哪一方——這是欠定的任務。改用「我方/對方」後問題消失，
   同時等效於免費的顏色對稱增強。

2. **均勻取樣涵蓋全局，而非只取最後 250 個盤面**
   實測 27.5% 的棋局超過 250 手，官方做法會把這些棋局的開局整段丟掉，
   而布局是棋力的強訊號。

3. **加入氣的分層平面**
   AI CUP 2023 跨多支公開隊伍獨立實作、公認增益最大的特徵。
   它編碼棋塊死活的緊迫程度，而棋力的核心差異正是對死活的判斷力。

平面配置（預設 34 層）：
    0        我方棋子
    1        對方棋子
    2        空點
    3..10    我方最近 n_history 手的盤面
    11..18   對方最近 n_history 手的盤面
    19..22   我方棋串的氣 = 1 / 2 / 3 / >=4
    23..26   對方棋串的氣 = 1 / 2 / 3 / >=4
    27..30   最後一手、倒數第 2/3/4 手的位置
    31       我方執色（全 1 = 黑）
    32       手數進度（當前手數 / 總手數，broadcast）
    33       全 1 的邊界常數平面（讓卷積能感知盤面邊界）
"""

from __future__ import annotations

import numpy as np

from .goboard import SIZE, EMPTY, BLACK, WHITE, Board, parse_moves

N_HISTORY = 8
LIBERTY_BUCKETS = 4
N_LAST_MOVES = 4


def n_planes(n_history: int = N_HISTORY, buckets: int = LIBERTY_BUCKETS,
             n_last: int = N_LAST_MOVES) -> int:
    return 3 + 2 * n_history + 2 * buckets + n_last + 3


def progress_plane(n_history: int = N_HISTORY, buckets: int = LIBERTY_BUCKETS,
                   n_last: int = N_LAST_MOVES) -> int:
    """「手數進度」那一層的索引。它是**唯一**被量化到 0~255 的通道。"""
    return 3 + 2 * n_history + 2 * buckets + n_last + 1


def planes_to_float(t, n_history: int = N_HISTORY, buckets: int = LIBERTY_BUCKETS,
                    n_last: int = N_LAST_MOVES) -> np.ndarray:
    """uint8 板面 → float32，並把手數進度那一層除回 [0, 1]。

    ⚠️ **每一個消費端都必須經過這裡。**
    `game_planes` 為了省記憶體回傳 uint8，其中手數進度量化成 0~255；
    其餘 33 層都是 0/1。若直接 `.float()` 餵進網路，
    等於有一層的尺度是其他層的 **255 倍**——實測後果是
    rank 損失卡在 ln(10)=2.303、題目分數停在隨機水準（0.19），
    而且**沒有任何錯誤訊息**，看起來只像「模型學不起來」。

    這個函式存在的理由就是「別處不要自己寫 .float()」。
    """
    a = np.asarray(t, dtype=np.float32)
    idx = progress_plane(n_history, buckets, n_last)
    if a.shape[-3] > idx:
        a[..., idx, :, :] /= 255.0
    return a


def _snapshot_indices(n_moves: int, n_snapshots: int, min_move: int = 20) -> np.ndarray:
    """在整局中均勻取樣 n_snapshots 個手序（含開局與終局）。"""
    if n_moves <= min_move:
        return np.zeros(n_snapshots, dtype=np.int64)
    return np.unique(np.linspace(min_move, n_moves - 1, n_snapshots).astype(np.int64))


def game_planes(sgf: str, color: str | int, *, n_snapshots: int = 8,
                n_history: int = N_HISTORY, buckets: int = LIBERTY_BUCKETS,
                n_last: int = N_LAST_MOVES) -> np.ndarray | None:
    """把一局棋轉成 (n_snapshots, C, 19, 19) 的 uint8 張量。

    回傳 None 表示棋局太短或無法解析。
    """
    my = BLACK if (color == "B" or color == BLACK) else WHITE
    opp = WHITE if my == BLACK else BLACK

    moves = parse_moves(sgf)
    n = len(moves)
    if n < 30:
        return None

    want = set(_snapshot_indices(n, n_snapshots).tolist())
    C = n_planes(n_history, buckets, n_last)
    out = []

    board = Board()
    history: list[np.ndarray] = []          # 每手之後的 (2,19,19)：我方、對方
    for i, (col, r, c) in enumerate(moves):
        board.play(col, r, c)
        s = board.stones.reshape(SIZE, SIZE)
        history.append(np.stack([(s == my), (s == opp)]).astype(np.uint8))
        if len(history) > n_history:
            history.pop(0)

        if i not in want:
            continue

        p = np.zeros((C, SIZE, SIZE), dtype=np.uint8)
        cur = history[-1]
        p[0], p[1] = cur[0], cur[1]
        p[2] = (s == EMPTY).astype(np.uint8)

        # 歷史：不足時前面補空盤（對齊到最近一手在最後）
        pad = n_history - len(history)
        for h in range(n_history):
            if h < pad:
                continue
            frame = history[h - pad]
            p[3 + h] = frame[0]
            p[3 + n_history + h] = frame[1]

        base = 3 + 2 * n_history
        lib = board.liberty_planes(buckets)          # (2*buckets, 19, 19)，黑在前
        if my == BLACK:
            p[base:base + 2 * buckets] = lib
        else:
            p[base:base + buckets] = lib[buckets:]
            p[base + buckets:base + 2 * buckets] = lib[:buckets]

        base += 2 * buckets
        for k in range(n_last):
            j = i - k
            if j >= 0:
                _, rr, cc = moves[j]
                p[base + k, rr, cc] = 1

        base += n_last
        p[base] = 1 if my == BLACK else 0
        # 手數進度量化到 uint8（省記憶體）。**讀取端必須用 `planes_to_float`
        # 除回 [0,1]**，否則這一層的尺度是其他層的 255 倍。
        p[base + 1] = int(round(255 * (i + 1) / n))
        p[base + 2] = 1
        out.append(p)

    if not out:
        return None
    return np.stack(out)
