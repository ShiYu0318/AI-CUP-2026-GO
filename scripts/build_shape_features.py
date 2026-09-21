"""把逐手棋形序列聚合成**整局**的統計，接到 589 維手工特徵後面。

## 為什麼

棋形序列（`cache/seqshape.npy`）目前只餵給序列模型。但 stacker 裡最強的成分
是**單局 MLP**，而它讀的 589 維手工特徵裡**沒有手法類別**——
只有鄰居的數量，沒有排列（見 RESULTS.md §2.17）。

線性探針顯示棋形帶來的是**新資訊**（合併後比既有特徵高 +11.6%），
所以把它接到單局模型的輸入上是合理的一步，而且成本只有幾分鐘：
序列庫已經建好，聚合純粹是算術，不用重新解析 SGF。

## 每個通道抽四個統計

    mean, std, 前三分之一的 mean, 後三分之一的 mean

前後段分開是刻意的——**愚形出現在布局還是收官，意義完全不同**，
而整局 mean 把這件事抹平了。這正是 `moveseq.py` 當初存在的理由，
只是這次用在整局特徵上。

12 通道 × 4 統計 = 48 維／視角。

用法：
    python3 scripts/build_shape_features.py                 # 我方視角
    python3 scripts/build_shape_features.py --opponent      # 對手視角
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup.moveshape import (N_SHAPE_FEATURES,      # noqa: E402
                             N_SHAPE_STATS as N_STATS, aggregate_shape)

# ⚠️ 聚合邏輯放在 `aicup.moveshape`，因為 `predict.py` 推論時也要用同一份。
# 訓練用的特徵庫是這個函式算的，推論若用另一份實作，
# 欄位語意只要差一點點就會**靜默**失準。
aggregate = aggregate_shape


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", action="store_true")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--source", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = args.source or ("cache/seqshape_opp.npy" if args.opponent
                          else "cache/seqshape.npy")
    out = args.out or ("cache/features_shape_opp.npy" if args.opponent
                       else "cache/features_shape.npy")

    S = np.load(src, mmap_mode="r")
    ok = np.load(src.replace(".npy", "_ok.npy"))
    n, T, C = S.shape
    assert C == N_SHAPE_FEATURES, f"通道數 {C} 與 moveshape 的 {N_SHAPE_FEATURES} 不符"

    F = np.zeros((n, C * N_STATS), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, args.chunk):
        j = min(i + args.chunk, n)
        F[i:j] = aggregate(np.asarray(S[i:j]))
        if (i // args.chunk) % 10 == 0:
            print(f"  {j:,}/{n:,}（{time.time() - t0:.0f} 秒）", flush=True)

    np.save(out, F)
    np.save(out.replace(".npy", "_ok.npy"), ok)
    print(f"完成：{F.shape} {F.dtype}，{F.nbytes / 1e6:.0f} MB，"
          f"有效 {ok.mean():.4f}，{(time.time() - t0) / 60:.1f} 分 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
