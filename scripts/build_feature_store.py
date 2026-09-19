"""一次算好全部 100 萬局的手工特徵，存成 float32 memmap。

之後所有實驗（棋力預測、玩家辨識、各種等級組成的候選庫）都直接吃這份快取，
不必重複解析 SGF。10 個等級檔案對應 10 個 worker，天然平行。

用法：
    python3 scripts/build_feature_store.py [--nproc 10] [--limit N]

產出：
    cache/features.npy   (1_000_000, 589) float32，順序與 load_metadata() 一致
    cache/features_ok.npy 布林遮罩，標記哪些棋局成功抽出特徵

加上 --opponent 會改抽**對手視角**的特徵（同一盤棋、執色反轉），
存成 cache/features_opp.npy。

為什麼要抽對手：SGF 同時包含雙方的手，但 `game_features` 只用「我方」的手
（`mine = [... if col == my]`）。圍棋伺服器會把你配對給實力相近的人，
所以**對手的表現反映的是配對分級，而分級反映你的穩定實力**；
你自己在這一局的表現則是繞著你的平均值震盪。
對手因此是一個雜訊更低的估計量——實測「只用對手」的分數比「只用我方」還高。
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

from aicup.data import training_files          # noqa: E402
from aicup.features import (N_FEATURES, batch_features,          # noqa: E402
                            normalise_color)
from aicup.goboard import BLACK                     # noqa: E402

CACHE_DIR = "cache"


def _flip(colors):
    """執色反轉，用來抽對手視角。normalise_color 會擋掉無法辨識的值。"""
    return ["W" if normalise_color(c) == BLACK else "B" for c in colors]


def _worker(args):
    file_idx, path, limit, opponent = args
    d = pd.read_csv(path, usecols=["color", "sgf_content"], nrows=limit)
    colors = d["color"].tolist()
    if opponent:
        colors = _flip(colors)
    t0 = time.time()
    X, ok = batch_features(d["sgf_content"].tolist(), colors)
    out = os.path.join(CACHE_DIR, f"_part{file_idx:02d}.npz")
    np.savez(out, X=X, ok=ok)
    return file_idx, len(d), int(ok.sum()), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=None, help="每個等級只處理前 N 局（除錯用）")
    ap.add_argument("--training-dir", default="training")
    ap.add_argument("--opponent", action="store_true",
                    help="抽對手視角的特徵（執色反轉），存成 features_opp.npy")
    args = ap.parse_args()
    stem = "features_opp" if args.opponent else "features"

    # macOS 的 Python 3.13 預設用 spawn，每個 worker 都要重新 import 並可能卡住。
    # 本工作是純數值運算、沒有持有執行緒，用 fork 既安全又省下重複 import 的成本。
    try:
        ctx = mp.get_context("fork")
    except ValueError:                       # 不支援 fork 的平台（Windows）
        ctx = mp.get_context("spawn")

    os.makedirs(CACHE_DIR, exist_ok=True)
    files = training_files(args.training_dir)
    jobs = [(i, p, args.limit, args.opponent) for i, p in enumerate(files)]

    t0 = time.time()
    with ctx.Pool(min(args.nproc, len(jobs))) as pool:
        for fi, n, n_ok, el in pool.imap_unordered(_worker, jobs):
            print(f"  [{os.path.basename(files[fi]):>16}] {n:,} 局，成功 {n_ok:,}"
                  f"（{n_ok / max(n,1):.4f}），{el / 60:.1f} 分", flush=True)

    Xs, oks = [], []
    for fi in range(len(files)):
        z = np.load(os.path.join(CACHE_DIR, f"_part{fi:02d}.npz"))
        Xs.append(z["X"]); oks.append(z["ok"])
    X = np.concatenate(Xs); ok = np.concatenate(oks)
    assert X.shape[1] == N_FEATURES

    np.save(os.path.join(CACHE_DIR, f"{stem}.npy"), X)
    np.save(os.path.join(CACHE_DIR, f"{stem}_ok.npy"), ok)
    for fi in range(len(files)):
        os.remove(os.path.join(CACHE_DIR, f"_part{fi:02d}.npz"))

    print(f"\n完成：{X.shape} float32，成功率 {ok.mean():.4f}，"
          f"{X.nbytes / 1e6:.0f} MB，總耗時 {(time.time() - t0) / 60:.1f} 分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
