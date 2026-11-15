"""Strength Estimator（Chen、Shih、Wu，ICLR 2025）的逐局面強度分數。

預訓練權重：https://rlg.iis.sinica.edu.tw/papers/strength-estimator/ 的 go.tar.gz（se_go.pt，
MiniZero 的 TorchScript，20 塊 x 256 通道，野狐 3-5k ~ 9d 訓練）。程式：github.com/rlglab/strength-estimator。

輸入編碼與 MiniZero `GoEnv::getFeatures` 相同（18 個 19x19 平面）：
  0~15：最近 8 個局面（含當前），偶數＝輪到下的一方的子、奇數＝另一方的子；不足 8 個補 0
  16：輪黑下為 1；17：輪白下為 1
第 pos 個局面＝第 pos 手**落子前**的盤面（官方 evaluator 對全部局面的 score 取平均）。
輸出：score（強度分數 β，越大越強）、weight（訓練時的局面權重）、policy（野狐人類 SL policy）。

每盤存一個與執色無關的向量（SE_FEATURES），之後依題目的執色挑我方／對方的欄位。
"""
from __future__ import annotations

import hashlib

import numpy as np

from .goboard import BLACK, EMPTY, SIZE, WHITE, Board, parse_moves

PHASES = ((0, 50), (50, 150), (150, 10_000))
# 依「輪到誰下」分開：score_to_move_B 是黑方要下的局面（白剛下完）的平均分數
SE_FEATURES = (
    ["score_all", "score_wmean", "score_sd"]
    + [f"score_ph{i}" for i in range(len(PHASES))]
    + ["score_tomove_B", "score_tomove_W"]
    + ["logp_B", "logp_W", "top1_B", "top1_W", "n_pos"]
)
N_SE = len(SE_FEATURES)


def game_hash(sgf: str) -> str:
    return hashlib.sha1(sgf.encode()).hexdigest()[:20]


def encode_game(sgf: str, max_moves: int = 400, stride: int = 1):
    """一盤棋 → (planes (n,18,19,19) uint8, moves (n,) 落子點 r*19+c, mover (n,) 1=黑 2=白, idx (n,) 手序)。

    stride > 1 時只取第 0、stride、2·stride… 個局面（奇數 stride 讓黑白輪流被取到）。"""
    moves = parse_moves(sgf)[:max_moves]
    n = len(moves)
    planes = np.zeros((n, 18, SIZE, SIZE), np.uint8)
    hist = []                                   # 每個局面的 stones（落子前）
    board = Board()
    acts = np.zeros(n, np.int64); mover = np.zeros(n, np.int8)
    for i, (col, r, c) in enumerate(moves):
        hist.append(board.stones.reshape(SIZE, SIZE).copy())
        acts[i] = r * SIZE + c; mover[i] = col
        board.play(col, r, c)
    keep = np.arange(0, n, stride)
    planes = planes[keep]
    for o, i in enumerate(keep):
        tm = mover[i]; op = WHITE if tm == BLACK else BLACK
        for k in range(8):
            j = i - k
            if j < 0:
                break
            st = hist[j]
            planes[o, 2 * k] = st == tm
            planes[o, 2 * k + 1] = st == op
        planes[o, 16 if tm == BLACK else 17] = 1
    return planes, acts[keep], mover[keep], keep


def load_se(path: str, device, half: bool = False):
    import torch
    m = torch.jit.load(path, map_location="cpu").eval().to(device)
    return m.half() if half else m


def score_games(model, sgfs, device, batch: int = 512, max_moves: int = 400, stride: int = 1,
                half: bool = False):
    """多盤棋 → (len(sgfs), N_SE) float32；解析失敗或空盤為 NaN。"""
    import torch
    out = np.full((len(sgfs), N_SE), np.nan, np.float32)
    enc = []
    for s in sgfs:
        try:
            enc.append(encode_game(s, max_moves, stride))
        except Exception:
            enc.append(None)
    # 把所有局面串起來成批推論
    lens = [0 if e is None else len(e[1]) for e in enc]
    tot = sum(lens)
    if tot == 0:
        return out
    P = np.concatenate([e[0] for e in enc if e is not None and len(e[1])])
    A = np.concatenate([e[1] for e in enc if e is not None and len(e[1])])
    score = np.empty(tot, np.float32); weight = np.empty(tot, np.float32)
    logp = np.empty(tot, np.float32); top1 = np.empty(tot, np.float32)
    with torch.no_grad():
        for a in range(0, tot, batch):
            x = torch.from_numpy(P[a:a + batch]).to(device)
            x = x.half() if half else x.float()
            o = model(x)
            lg = torch.log_softmax(o["policy_logit"].float(), 1)
            act = torch.from_numpy(A[a:a + batch]).to(device)
            score[a:a + batch] = o["score"].float().view(-1).cpu().numpy()
            weight[a:a + batch] = o["weight"].float().view(-1).cpu().numpy()
            logp[a:a + batch] = lg.gather(1, act[:, None]).view(-1).cpu().numpy()
            top1[a:a + batch] = (lg.argmax(1) == act).float().cpu().numpy()
    i0 = 0
    for g, (e, n) in enumerate(zip(enc, lens)):
        if n == 0:
            continue
        sl = slice(i0, i0 + n); i0 += n
        s, w, lp, t1, mv, idx = score[sl], weight[sl], logp[sl], top1[sl], e[2], e[3]
        f = [s.mean(), (s * w).sum() / max(w.sum(), 1e-9), s.std()]
        for lo, hi in PHASES:
            m = (idx >= lo) & (idx < hi)
            f.append(s[m].mean() if m.any() else s.mean())
        b = mv == BLACK
        f += [s[b].mean() if b.any() else np.nan, s[~b].mean() if (~b).any() else np.nan]
        f += [lp[b].mean() if b.any() else np.nan, lp[~b].mean() if (~b).any() else np.nan,
              t1[b].mean() if b.any() else np.nan, t1[~b].mean() if (~b).any() else np.nan, len(mv) and int(idx[-1]) + 1]
        out[g] = f
    return out


def side_features(f: np.ndarray, color) -> np.ndarray:
    """與執色無關的盤向量 → 以 color 為「我方」的向量：
    [score_all, score_wmean, score_sd, ph0, ph1, ph2, score_我方要下, score_對方要下,
     logp_我方, logp_對方, top1_我方, top1_對方]"""
    from .features import normalise_color
    me_black = normalise_color(color) == BLACK
    i = {k: j for j, k in enumerate(SE_FEATURES)}
    a, b = ("B", "W") if me_black else ("W", "B")
    return np.array([f[i["score_all"]], f[i["score_wmean"]], f[i["score_sd"]],
                     f[i["score_ph0"]], f[i["score_ph1"]], f[i["score_ph2"]],
                     f[i[f"score_tomove_{a}"]], f[i[f"score_tomove_{b}"]],
                     f[i[f"logp_{a}"]], f[i[f"logp_{b}"]],
                     f[i[f"top1_{a}"]], f[i[f"top1_{b}"]]], np.float32)


N_SIDE = 12
_ = EMPTY
