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
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup.data import training_files                          # noqa: E402
from aicup.moveseq import (MAX_MOVES, N_MOVE_FEATURES,         # noqa: E402
                           batch_move_sequences)

CACHE_DIR = "cache"


def _worker(args):
    file_idx, path, limit = args
    d = pd.read_csv(path, usecols=["color", "sgf_content"], nrows=limit)
    t0 = time.time()
    S, ok = batch_move_sequences(d["sgf_content"].tolist(), d["color"].tolist())
    np.savez(os.path.join(CACHE_DIR, f"_seq{file_idx:02d}.npz"), S=S, ok=ok)
    return file_idx, len(d), int(ok.sum()), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=None, help="每個等級只處理前 N 局（除錯用）")
    ap.add_argument("--training-dir", default="training")
    ap.add_argument("--out", default="cache/sequences.npy")
    args = ap.parse_args()

    # macOS 的 Python 3.13 預設 spawn 會讓 worker 卡住；本工作純數值運算，fork 安全。
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    os.makedirs(CACHE_DIR, exist_ok=True)
    files = training_files(args.training_dir)
    jobs = [(i, p, args.limit) for i, p in enumerate(files)]

    t0 = time.time()
    with ctx.Pool(min(args.nproc, len(jobs))) as pool:
        for fi, n, n_ok, el in pool.imap_unordered(_worker, jobs):
            print(f"  [{os.path.basename(files[fi]):>16}] {n:,} 局，成功 {n_ok:,}"
                  f"（{n_ok / max(n, 1):.4f}），{el / 60:.1f} 分", flush=True)

    parts, oks = [], []
    for fi in range(len(files)):
        z = np.load(os.path.join(CACHE_DIR, f"_seq{fi:02d}.npz"))
        parts.append(z["S"]); oks.append(z["ok"])
    S = np.concatenate(parts); ok = np.concatenate(oks)
    assert S.shape[1:] == (MAX_MOVES, N_MOVE_FEATURES)

    np.save(args.out, S)
    np.save(args.out.replace(".npy", "_ok.npy"), ok)
    for fi in range(len(files)):
        os.remove(os.path.join(CACHE_DIR, f"_seq{fi:02d}.npz"))

    print(f"\n完成：{S.shape} {S.dtype}，成功率 {ok.mean():.4f}，"
          f"{S.nbytes / 1e9:.2f} GB，總耗時 {(time.time() - t0) / 60:.1f} 分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
