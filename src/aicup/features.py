"""手工特徵：不需要 GPU、不需要圍棋引擎，卻已能超越官方 baseline。

實測（docs/02 §1.2）：
  競賽二  題目層級 0.3496（官方深度學習 baseline 0.4631）
  競賽一  400 人候選庫 0.3181（官方深度學習 baseline 0.2845）

特徵分三族（消融見 docs/02 §10.5）：
  A. 行為統計量 —— 應手距離、脫先率、線偏好、跳躍幅度。最強的單一族，
     而且是真正的「決策風格」而非背下來的定式，泛化性最好。
  B. 死活統計量 —— 提子、被提、己方棋串的氣、打吃頻率、棋形破碎度。
     需要真實棋盤才算得出來，是本模組相對於早期原型的新增部分。
  C. 位置分布 —— 以 8 重對稱折疊到 55 格的直方圖（開局 / 全局各一份）。
     實測折疊優於不折疊（0.4697 vs 0.3989），代表落子方位不帶身分訊號。
"""

from __future__ import annotations

import numpy as np

from .goboard import SIZE, EMPTY, BLACK, WHITE, Board, parse_moves

# ---- 8 重對稱折疊表 ----
_FOLD = {}
_CELLS = []
for _r in range(SIZE):
    for _c in range(SIZE):
        _a, _b = min(_r, SIZE - 1 - _r), min(_c, SIZE - 1 - _c)
        _k = (min(_a, _b), max(_a, _b))
        if _k not in _FOLD:
            _FOLD[_k] = len(_CELLS)
            _CELLS.append(_k)
N_FOLD = len(_CELLS)                      # 55

_FOLD_LUT = np.empty((SIZE, SIZE), dtype=np.int16)
for _r in range(SIZE):
    for _c in range(SIZE):
        _a, _b = min(_r, SIZE - 1 - _r), min(_c, SIZE - 1 - _c)
        _FOLD_LUT[_r, _c] = _FOLD[(min(_a, _b), max(_a, _b))]

_LINE_LUT = np.empty((SIZE, SIZE), dtype=np.int8)
for _r in range(SIZE):
    for _c in range(SIZE):
        _LINE_LUT[_r, _c] = min(_r, _c, SIZE - 1 - _r, SIZE - 1 - _c)

PHASES = ((0, 40), (40, 120), (120, 10 ** 9))
OPENING_MOVES = 15
_LIBERTY_SAMPLE_EVERY = 10                # 每 N 手取一次死活快照，控制成本


def _scalar_names() -> list[str]:
    n = ["n_moves", "n_mine", "is_black"]
    for p in range(len(PHASES)):
        n += [f"p{p}_line_mean", f"p{p}_line_std"]
        n += [f"p{p}_line{L}" for L in range(5)]
        n += [f"p{p}_jump_mean", f"p{p}_tenuki_rate"]
    n += ["resp_mean", "resp_local", "resp_far"]
    n += ["first_line", "corners_first6", "spread_x", "spread_y", "center_frac",
          "endgame_line01"]
    # 死活族
    n += ["captures_made", "captures_taken", "capture_events",
          "my_libs_mean", "opp_libs_mean", "my_groups_mean", "opp_groups_mean",
          "my_atari_rate", "opp_atari_rate", "final_my_stones", "final_opp_stones",
          "stone_ratio"]
    return n


SCALAR_NAMES = _scalar_names()
FEATURE_NAMES = (SCALAR_NAMES
                 + [f"open_cell{i}" for i in range(N_FOLD)]
                 + [f"all_cell{i}" for i in range(N_FOLD)])
N_FEATURES = len(FEATURE_NAMES)


def _group_stats(board: Board, color: int):
    """回傳 (棋串數, 平均氣, 處於打吃的棋串比例)。"""
    stones = board.stones
    visited = np.zeros(SIZE * SIZE, dtype=bool)
    n_groups, libs_sum, n_atari = 0, 0, 0
    for idx in range(SIZE * SIZE):
        if stones[idx] != color or visited[idx]:
            continue
        group, libs = board._group_and_liberties(idx)
        for p in group:
            visited[p] = True
        n_groups += 1
        libs_sum += len(libs)
        if len(libs) == 1:
            n_atari += 1
    if n_groups == 0:
        return 0.0, 0.0, 0.0
    return float(n_groups), libs_sum / n_groups, n_atari / n_groups


def game_features(sgf: str, color: str | int) -> np.ndarray | None:
    """從一局 SGF 抽出固定長度的特徵向量。無法解析或太短時回傳 None。"""
    my = BLACK if (color == "B" or color == BLACK) else WHITE
    opp = WHITE if my == BLACK else BLACK

    moves = parse_moves(sgf)
    n = len(moves)
    if n < 30:
        return None
    mine = [(i, r, c) for i, (col, r, c) in enumerate(moves) if col == my]
    if len(mine) < 15:
        return None

    rows = np.array([r for _, r, _ in mine])
    cols = np.array([c for _, _, c in mine])
    order = np.array([i for i, _, _ in mine])
    lines = _LINE_LUT[rows, cols].astype(np.float64)

    sc: list[float] = [float(n), float(len(mine)), 1.0 if my == BLACK else 0.0]

    for lo, hi in PHASES:
        sel = (order >= lo) & (order < hi)
        if sel.any():
            ls = lines[sel]
            sc += [float(ls.mean()), float(ls.std())]
            sc += [float((ls == L).mean()) for L in range(5)]
            pr, pc = rows[sel], cols[sel]
            if len(pr) > 1:
                d = np.abs(np.diff(pr)) + np.abs(np.diff(pc))
                sc += [float(d.mean()), float((d > 6).mean())]
            else:
                sc += [0.0, 0.0]
        else:
            sc += [0.0] * 9

    # 應手：己手與「前一手對方落子」的曼哈頓距離
    resp = [abs(r - moves[i - 1][1]) + abs(c - moves[i - 1][2])
            for i, r, c in mine if i > 0 and moves[i - 1][0] == opp]
    if resp:
        a = np.array(resp, dtype=np.float64)
        sc += [float(a.mean()), float((a <= 3).mean()), float((a >= 10).mean())]
    else:
        sc += [0.0, 0.0, 0.0]

    sc.append(float(_LINE_LUT[rows[0], cols[0]]))
    sc.append(float(len({(r < 9, c < 9) for r, c in zip(rows[:6], cols[:6])})))
    sc += [float(rows.std()), float(cols.std())]
    sc.append(float(((np.abs(rows - 9) <= 4) & (np.abs(cols - 9) <= 4)).mean()))
    tail = order >= int(0.75 * n)
    sc.append(float((lines[tail] <= 1).mean()) if tail.any() else 0.0)

    # ---- 死活族：重播全局，定期取快照 ----
    board = Board()
    cap_made = cap_taken = cap_events = 0
    my_libs, opp_libs, my_grp, opp_grp, my_at, opp_at = [], [], [], [], [], []
    for i, (col, r, c) in enumerate(moves):
        taken = board.play(col, r, c)
        if taken:
            cap_events += 1
            if col == my:
                cap_made += taken
            else:
                cap_taken += taken
        if i % _LIBERTY_SAMPLE_EVERY == 0 and i >= 20:
            g, l, a = _group_stats(board, my)
            my_grp.append(g); my_libs.append(l); my_at.append(a)
            g, l, a = _group_stats(board, opp)
            opp_grp.append(g); opp_libs.append(l); opp_at.append(a)

    def _m(x):
        return float(np.mean(x)) if x else 0.0

    final_my = int((board.stones == my).sum())
    final_opp = int((board.stones == opp).sum())
    sc += [float(cap_made), float(cap_taken), float(cap_events),
           _m(my_libs), _m(opp_libs), _m(my_grp), _m(opp_grp),
           _m(my_at), _m(opp_at), float(final_my), float(final_opp),
           final_my / max(final_my + final_opp, 1)]

    # ---- 位置分布（8 重對稱折疊）----
    fold = _FOLD_LUT[rows, cols]
    all_h = np.bincount(fold, minlength=N_FOLD).astype(np.float64)
    open_h = np.bincount(fold[:OPENING_MOVES], minlength=N_FOLD).astype(np.float64)
    all_h /= max(all_h.sum(), 1.0)
    open_h /= max(open_h.sum(), 1.0)

    return np.concatenate([np.array(sc, dtype=np.float32),
                           open_h.astype(np.float32),
                           all_h.astype(np.float32)])


def batch_features(sgfs, colors) -> tuple[np.ndarray, np.ndarray]:
    """對一批棋局抽特徵。回傳 (特徵矩陣, 成功的布林遮罩)。"""
    out = np.zeros((len(sgfs), N_FEATURES), dtype=np.float32)
    ok = np.zeros(len(sgfs), dtype=bool)
    for i, (s, c) in enumerate(zip(sgfs, colors)):
        try:
            v = game_features(s, c)
        except Exception:
            v = None
        if v is not None:
            out[i] = v
            ok[i] = True
    return out, ok
