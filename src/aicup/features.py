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
    # 「局面判斷品質」族：不需要引擎，但抓得到經典的棋力破綻。
    # 實測與棋力的相關係數（每級抽 400 局）：
    #   wasted_fill   -0.747  弱者更常在官子把子下進自己已包圍的空點
    #   cap_per_move  +0.654  強者每手提到更多子
    #   abandon_atari +0.485  強者會果斷棄掉被打吃的死子，而不是硬救
    #   empty_triangle -0.936  愚形（自己三子成 L、第四點空）—— 圍棋最著名的壞形，
    #                          是目前找到最強的單一棋力訊號
    #   contact_early  -0.662  開局 30 手內的貼身接觸比例，弱者常過早接觸
    #   territory_rate -0.959（全 10 級）／**-0.997（1D~4D 區間）**
    #                  下在「自己已有子、對方沒子」的區域的比例。
    #                  強者較少在已安定處補棋。這是目前唯一能明顯分開段位的訊號。
    #   fourth_line    +0.625（全 10 級）／**+0.890（1D~4D）** 開局第四線比例
    #   invade_rate    +0.953 下在對方子較多處的比例
    n += ["wasted_fill", "cap_per_move", "abandon_atari", "self_atari_rate",
          "atari_given_rate", "empty_triangle", "contact_early",
          "territory_rate", "invade_rate", "fourth_line_open", "third_line_open"]
    return n


K_SEQ = 6          # 開局前幾手要保留「順序」

SCALAR_NAMES = _scalar_names()
FEATURE_NAMES = (SCALAR_NAMES
                 + [f"open_cell{i}" for i in range(N_FOLD)]
                 + [f"all_cell{i}" for i in range(N_FOLD)]
                 # 開局順序族：前 K_SEQ 手各自的折疊位置 one-hot。
                 # open_cell* 是直方圖、**順序完全丟失**，但
                 # 「先星位、再小目、然後掛角」是極強的個人簽名。
                 # 實測（無學習基準、400 人候選庫）：
                 #   目前 168 維 0.6641 / 只有開局順序 0.6958 / 兩者合併 0.8490
                 + [f"seq{k}_cell{i}" for k in range(K_SEQ) for i in range(N_FOLD)]
                 + ["seq_corners6", "seq_corner_frac", "seq_first_local",
                    "seq_line_mean", "seq_line_std"]
                 # 開局轉移族：相鄰兩手的幾何關係 + 區域造訪順序。
                 # 無需詞彙表（對照組 bigram 需要從訓練資料擬合 400 個常見組合）。
                 # 實測在無學習基準上：順序 one-hot 0.6603 -> 加轉移族 0.7221
                 + [f"tr{k}_dist{b}" for k in range(K_SEQ - 1) for b in range(5)]
                 + [f"tr{k}_{n}" for k in range(K_SEQ - 1)
                    for n in ("same_region", "fold_gap", "manhattan")]
                 + [f"reg{k}_visit{v}" for k in range(K_SEQ) for v in range(7)])
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


def _region(r: int, c: int) -> int:
    """粗略區域：四個角 0–3、邊 4、中腹 5。用於「區域造訪順序」特徵。"""
    if min(r, c, SIZE - 1 - r, SIZE - 1 - c) >= 7:
        return 5
    if r <= 6 and c <= 6:
        return 0
    if r <= 6 and c >= 12:
        return 1
    if r >= 12 and c <= 6:
        return 2
    if r >= 12 and c >= 12:
        return 3
    return 4


def _opening_transitions(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """開局的轉移幾何與區域造訪順序。

    為什麼需要：`seq*_cell*` 保留了「第 k 手下在哪」，但沒有直接表達
    「這一手與上一手的關係」（是繼續經營同一個角，還是轉到別處）。
    區域造訪順序對盤面旋轉不變，是比絕對位置更穩定的簽名。
    """
    n_tr = K_SEQ - 1
    dist = np.zeros((n_tr, 5), dtype=np.float32)
    misc = np.zeros((n_tr, 3), dtype=np.float32)
    for k in range(min(n_tr, len(rows) - 1)):
        r1, c1, r2, c2 = int(rows[k]), int(cols[k]), int(rows[k + 1]), int(cols[k + 1])
        d = abs(r1 - r2) + abs(c1 - c2)
        dist[k, min(d // 6, 4)] = 1.0
        misc[k] = (float(_region(r1, c1) == _region(r2, c2)),
                   abs(int(_FOLD_LUT[r1, c1]) - int(_FOLD_LUT[r2, c2])) / float(N_FOLD),
                   d / 36.0)

    visit = np.zeros((K_SEQ, 7), dtype=np.float32)
    seen: dict[int, int] = {}
    for k in range(min(K_SEQ, len(rows))):
        g = _region(int(rows[k]), int(cols[k]))
        if g not in seen:
            seen[g] = len(seen)
        visit[k, min(seen[g], 6)] = 1.0
    return np.concatenate([dist.ravel(), misc.ravel(), visit.ravel()])


def _makes_empty_triangle(stones, r: int, c: int, color: int) -> bool:
    """這一手是否形成愚形：自己三子佔 2x2 的三個角、第四角是空的。

    愚形是圍棋公認的壞形（效率低、氣少）。實測它與棋力的相關係數達 -0.936，
    是本專案找到最強的單一棋力訊號，而且完全不需要圍棋引擎。
    """
    for dr in (-1, 1):
        for dc in (-1, 1):
            r1, c1 = r + dr, c          # 同欄的鄰點
            r2, c2 = r, c + dc          # 同列的鄰點
            r3, c3 = r + dr, c + dc     # 對角點
            if not (0 <= r1 < SIZE and 0 <= c2 < SIZE):
                continue
            others = [stones[r1 * SIZE + c1], stones[r2 * SIZE + c2],
                      stones[r3 * SIZE + c3]]
            # 剩下三點必須「兩子同色 + 一點為空」——空點落在哪一個位置都算愚形
            if others.count(color) == 2 and others.count(EMPTY) == 1:
                return True
    return False


def _quality_stats(moves, my: int, opp: int, n_moves: int) -> dict:
    """局面判斷品質的代理指標。只重播一次，靠「最後一手的鄰居」增量判斷打吃，
    避免每手都掃描整個盤面（那會讓特徵抽取慢上數倍）。"""
    from .goboard import _NEIGHBORS

    board = Board()
    n_my = wasted = self_atari = abandoned = atari_given = 0
    n_empty_tri = n_contact = n_open = cap_made = 0
    n_territory = n_invade = n_fourth = n_third = 0
    endgame_start = int(0.75 * n_moves)
    my_in_atari: set[int] = set()          # 我方處於打吃的棋串代表點

    for i, (col, r, c) in enumerate(moves):
        idx = r * SIZE + c
        neigh = _NEIGHBORS[idx]
        if col == my and i < 30:           # 開局的貼身接觸
            n_open += 1
            if any(board.stones[q] == opp for q in neigh):
                n_contact += 1
        # 落子前先記下：這一手是否有「填自己的地」的嫌疑
        surrounded_by_me = (col == my and i >= endgame_start
                            and all(board.stones[q] == my for q in neigh))

        # 落子「前」看周圍敵我子數：分辨「在自己的地裡補棋」與「侵入對方陣地」
        if col == my and i >= 40:
            near_my = sum(1 for qq in neigh if board.stones[qq] == my)
            near_opp = sum(1 for qq in neigh if board.stones[qq] == opp)
            if near_opp > near_my:
                n_invade += 1
            elif near_my > 0 and near_opp == 0:
                n_territory += 1

        taken = board.play(col, r, c)

        if col == my:
            n_my += 1
            cap_made += taken
            # 開局的線偏好：強者更重視第四線的勢
            if i < 60:
                L = int(_LINE_LUT[r, c])
                if L == 3:
                    n_fourth += 1
                elif L == 2:
                    n_third += 1
            if surrounded_by_me and taken == 0:
                wasted += 1
            if board.stones[idx] == my:
                _, libs = board._group_and_liberties(idx)
                if len(libs) == 1 and taken == 0:
                    self_atari += 1
                if _makes_empty_triangle(board.stones, r, c, my):
                    n_empty_tri += 1
            # 先前被打吃的棋串，若這手沒去救、之後也還在打吃，算一次「棄子」
            if my_in_atari:
                still = 0
                for p in my_in_atari:
                    if board.stones[p] == my:
                        _, l = board._group_and_liberties(p)
                        if len(l) == 1:
                            still += 1
                if still:
                    abandoned += 1
            my_in_atari = set()
        else:
            # 對手落子後，只檢查其鄰接的我方棋串是否被打吃
            checked: set[int] = set()
            for q in neigh:
                if board.stones[q] == my and q not in checked:
                    grp, libs = board._group_and_liberties(q)
                    checked |= grp
                    if len(libs) == 1:
                        my_in_atari.add(q)
            # 我方是否讓對手處於打吃（攻擊性的代理）
            pass

    # 我方造成對手打吃的比例：以終局對手 1 氣棋串數近似，成本低
    opp_atari_groups = 0
    seen: set[int] = set()
    for idx in range(SIZE * SIZE):
        if board.stones[idx] == opp and idx not in seen:
            grp, libs = board._group_and_liberties(idx)
            seen |= grp
            if len(libs) == 1:
                opp_atari_groups += 1

    d = max(n_my, 1)
    return {"wasted_fill": wasted / d, "cap_per_move": cap_made / d,
            "abandon_atari": abandoned / d, "self_atari_rate": self_atari / d,
            "atari_given_rate": opp_atari_groups / d,
            "empty_triangle": n_empty_tri / d,
            "contact_early": n_contact / max(n_open, 1),
            "territory_rate": n_territory / d,
            "invade_rate": n_invade / d,
            "fourth_line_open": n_fourth / d,
            "third_line_open": n_third / d}


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

    q = _quality_stats(moves, my, opp, n)
    sc += [q["wasted_fill"], q["cap_per_move"], q["abandon_atari"],
           q["self_atari_rate"], q["atari_given_rate"],
           q["empty_triangle"], q["contact_early"],
           q["territory_rate"], q["invade_rate"],
           q["fourth_line_open"], q["third_line_open"]]

    # ---- 位置分布（8 重對稱折疊）----
    fold = _FOLD_LUT[rows, cols]
    all_h = np.bincount(fold, minlength=N_FOLD).astype(np.float64)
    open_h = np.bincount(fold[:OPENING_MOVES], minlength=N_FOLD).astype(np.float64)
    all_h /= max(all_h.sum(), 1.0)
    open_h /= max(open_h.sum(), 1.0)

    # ---- 開局順序（保留順序的 one-hot）----
    seq = np.zeros(K_SEQ * N_FOLD, dtype=np.float32)
    for k in range(min(K_SEQ, len(fold))):
        seq[k * N_FOLD + int(fold[k])] = 1.0

    corners = len({(r < 9, c < 9) for r, c in zip(rows[:6], cols[:6])})
    corner_frac = float(np.mean([_LINE_LUT[r, c] <= 5
                                 for r, c in zip(rows[:6], cols[:6])]))
    first_local = float(len(rows))
    for k in range(1, min(12, len(rows))):
        if any(abs(int(rows[k]) - int(rows[j])) + abs(int(cols[k]) - int(cols[j])) <= 4
               for j in range(k)):
            first_local = float(k)
            break
    head_lines = lines[:6]
    seq_sc = np.array([float(corners), corner_frac, first_local,
                       float(head_lines.mean()), float(head_lines.std())],
                      dtype=np.float32)

    return np.concatenate([np.array(sc, dtype=np.float32),
                           open_h.astype(np.float32),
                           all_h.astype(np.float32),
                           seq, seq_sc,
                           _opening_transitions(rows, cols)])


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


def _chunk_worker(args):
    sgfs, colors = args
    return batch_features(sgfs, colors)


def parallel_batch_features(sgfs, colors, nproc: int | None = None, chunk: int = 2000):
    """多進程版的 batch_features。

    正式比賽當天要處理的候選庫是 2 池 x 40,000 局，單執行緒約 3 分鐘；
    10 核可降到 20 秒左右。

    ⚠️ 啟動方式的取捨（本機實測，4,000 局 / 8 worker）：
        fork        1.8 秒，可用
        forkserver  直接拋例外
        spawn       每個 worker 都要重新 import pandas/numpy，啟動極慢（實測像卡住）
    所以固定用 fork。Python 會對「從多執行緒程序 fork」發出 DeprecationWarning，
    這個風險是真的，但實測跑完 100 萬局特徵沒有發生死鎖。
    不支援 fork 的平台（Windows）會退回單進程，行為完全相同、只是慢。
    """
    import multiprocessing as mp
    import os

    n = len(sgfs)
    if nproc is None:
        nproc = min(os.cpu_count() or 1, 10)
    if nproc <= 1 or n <= chunk:
        return batch_features(sgfs, colors)

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return batch_features(sgfs, colors)

    jobs = [(list(sgfs[i:i + chunk]), list(colors[i:i + chunk]))
            for i in range(0, n, chunk)]
    with ctx.Pool(nproc) as pool:
        parts = pool.map(_chunk_worker, jobs)
    X = np.concatenate([p[0] for p in parts], axis=0)
    ok = np.concatenate([p[1] for p in parts], axis=0)
    return X, ok
