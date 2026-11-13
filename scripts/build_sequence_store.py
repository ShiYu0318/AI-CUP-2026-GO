"""一次算好全部 100 萬局的逐手序列特徵，存成 float16 memmap。

與 `build_feature_store.py` 平行的存在：那份存的是**整局的統計摘要**（589 維），
這份存的是**逐手的序列**（160 × 12），保留「什麼時候發生」。
實測序列模型單獨就勝過 589 維手工特徵，融合後 +0.05（見 RESULTS.md §2.1c）。

用法：
    python3 scripts/build_sequence_store.py [--nproc 10] [--limit N]

產出：
    cache/sequences.npy      (1_000_000, 160, 12) float16，約 3.8 GB
    cache/sequences_ok.npy   布林遮罩

⚠️ 順序與 `load_metadata()` 一致，且有效判斷與 `game_features` 相同
（手數 < 30 或我方手數 < 15 就算失敗），兩份遮罩才對得起來。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from aicup.data import TRAINING_DIR  # noqa: E402
from aicup.data import training_files                          # noqa: E402
from aicup.features import normalise_color                     # noqa: E402
from aicup.goboard import BLACK                                # noqa: E402
from aicup.moveseq import (MAX_MOVES, N_MOVE_FEATURES,         # noqa: E402
                           batch_move_sequences)
from aicup.moveshape import (N_SHAPE_FEATURES,                 # noqa: E402
                             batch_shape_sequences)
from aicup.movequality import (N_QUALITY_FEATURES,             # noqa: E402
                               batch_quality_sequences)
from aicup.movepatch import N_PATCH_FEATURES, batch_patch_sequences  # noqa: E402

CACHE_DIR = "cache"


def _flip(colors):
    return ["W" if normalise_color(c) == BLACK else "B" for c in colors]


def _worker(args):
    file_idx, path, limit, opponent, max_moves, kind = args
    d = pd.read_csv(path, usecols=["color", "sgf_content"], nrows=limit)
    colors = d["color"].tolist()
    if opponent:
        colors = _flip(colors)
    t0 = time.time()
    fn = {"shape": batch_shape_sequences, "quality": batch_quality_sequences,
          "patch": batch_patch_sequences}.get(kind, batch_move_sequences)
    S, ok = fn(d["sgf_content"].tolist(), colors, max_moves=max_moves)
    np.savez(os.path.join(CACHE_DIR, f"_seq{file_idx:02d}.npz"), S=S, ok=ok)
    return file_idx, len(d), int(ok.sum()), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=None, help="每個等級只處理前 N 局（除錯用）")
    ap.add_argument("--training-dir", default=TRAINING_DIR)
    ap.add_argument("--opponent", action="store_true",
                    help="抽對手視角的序列（執色反轉）。對手視角在整局統計上值 +0.05，"
                         "序列上很可能也有")
    ap.add_argument("--max-moves", type=int, default=MAX_MOVES,
                    help="每局取前幾手。預設 160；棋局手數中位數約 200，"
                         "所以預設會截掉尾盤")
    ap.add_argument("--quality", action="store_true",
                    help="改抽**局面判斷品質**序列（movequality，8 維）。"
                         "序列模型目前看不到 territory / invade（相關 −0.959 / +0.953，"
                         "是整個特徵集裡最強的兩個），也看不到打吃、連接、減氣")
    ap.add_argument("--shape", action="store_true",
                    help="改抽**棋形**序列（moveshape，12 維）而不是 moveseq 的 12 維。"
                         "手法類別（接／尖／跳／飛）取代曼哈頓距離——"
                         "尖與跳的曼哈頓距離都是 2，在 moveseq 裡被混為一談")
    ap.add_argument("--patch", action="store_true",
                    help="改抽**落子座標＋落子前 5x5 局部盤面**（movepatch，52 維 uint8）。"
                         "競賽一的身分特徵：moveseq／moveshape 沒有座標也看不到周圍棋形")
    ap.add_argument("--out", default="cache/sequences.npy")
    args = ap.parse_args()
    kind = ("patch" if args.patch else "quality" if args.quality
            else ("shape" if args.shape else "move"))
    if args.patch and args.out == "cache/sequences.npy":
        args.out = "cache/seqpatch_opp.npy" if args.opponent else "cache/seqpatch.npy"
    if args.quality and args.out == "cache/sequences.npy":
        args.out = "cache/seqqual_opp.npy" if args.opponent else "cache/seqqual.npy"
    elif args.shape and args.out == "cache/sequences.npy":
        args.out = "cache/seqshape_opp.npy" if args.opponent else "cache/seqshape.npy"
    elif args.opponent and args.out == "cache/sequences.npy":
        args.out = "cache/sequences_opp.npy"

    # macOS 的 Python 3.13 預設 spawn 會讓 worker 卡住；本工作純數值運算，fork 安全。
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    os.makedirs(CACHE_DIR, exist_ok=True)
    files = training_files(args.training_dir)
    jobs = [(i, p, args.limit, args.opponent, args.max_moves, kind)
            for i, p in enumerate(files)]

    t0 = time.time()
    with ctx.Pool(min(args.nproc, len(jobs))) as pool:
        for fi, n, n_ok, el in pool.imap_unordered(_worker, jobs):
            print(f"  [{os.path.basename(files[fi]):>16}] {n:,} 局，成功 {n_ok:,}"
                  f"（{n_ok / max(n, 1):.4f}），{el / 60:.1f} 分", flush=True)

    # 逐檔寫進磁碟上的 memmap：不在記憶體裡一次串接（1M x 160 x 52 的峰值會是兩倍）。
    n_out = {"shape": N_SHAPE_FEATURES, "quality": N_QUALITY_FEATURES,
             "patch": N_PATCH_FEATURES}.get(kind, N_MOVE_FEATURES)
    sizes, dtype = [], None
    for fi in range(len(files)):
        with np.load(os.path.join(CACHE_DIR, f"_seq{fi:02d}.npz")) as z:
            sizes.append(len(z["ok"])); dtype = z["S"].dtype
    S = np.lib.format.open_memmap(args.out, mode="w+", dtype=dtype,
                                  shape=(sum(sizes), args.max_moves, n_out))
    oks, a = [], 0
    for fi in range(len(files)):
        with np.load(os.path.join(CACHE_DIR, f"_seq{fi:02d}.npz")) as z:
            assert z["S"].shape[1:] == (args.max_moves, n_out)
            S[a:a + sizes[fi]] = z["S"]; oks.append(z["ok"]); a += sizes[fi]
    S.flush()
    ok = np.concatenate(oks)
    np.save(args.out.replace(".npy", "_ok.npy"), ok)
    for fi in range(len(files)):
        os.remove(os.path.join(CACHE_DIR, f"_seq{fi:02d}.npz"))

    print(f"\n完成：{S.shape} {S.dtype}，成功率 {ok.mean():.4f}，"
          f"{S.nbytes / 1e9:.2f} GB，總耗時 {(time.time() - t0) / 60:.1f} 分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
