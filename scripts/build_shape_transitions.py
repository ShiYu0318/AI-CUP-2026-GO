"""手法**轉移**統計：接→跳、尖→飛……7×7 = 49 維／視角。

## 為什麼

競賽一最強的身分特徵是「開局**轉移**族」——相鄰兩手的幾何關係
（實測順序 one-hot 0.6603 → 加轉移族 0.7221，§1.4）。
而棋形統計剛剛讓競賽一從 0.9599 跳到 0.9745（§2.23），
最大的增益在段位（+0.0264）——因為**風格**不像**品質**那樣會隨棋力收斂。

把同一個邏輯套上去：手法的**轉移**（下完一個接之後傾向下什麼）
比手法的**頻率**更能刻畫一個人的節奏。

## 成本幾乎為零

手法類別已經在 `cache/seqshape.npy` 的第 0~6 欄（one-hot）。
這支只是對那 7 欄做 argmax 再數相鄰配對——**純 numpy，不用重新解析 SGF**。

用法：
    python3 scripts/build_shape_transitions.py [--opponent]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup.moveshape import N_RELATION           # noqa: E402


def transitions(block: np.ndarray) -> np.ndarray:
    """(n, T, C) → (n, N_RELATION**2)，每局的手法轉移頻率。"""
    X = np.asarray(block[:, :, :N_RELATION], dtype=np.float32)
    live = X.sum(-1) > 0                                  # 手法 one-hot 必有一個為 1
    rel = X.argmax(-1)
    a, b = rel[:, :-1], rel[:, 1:]
    ok = live[:, :-1] & live[:, 1:]                       # 兩手都要是真的手
    idx = a * N_RELATION + b
    n, T1 = idx.shape
    out = np.zeros((n, N_RELATION * N_RELATION), dtype=np.float32)
    # 逐列 bincount：T 只有 160，迴圈成本可忽略，而且比 scatter 好讀
    for i in range(n):
        m = ok[i]
        if not m.any():
            continue
        c = np.bincount(idx[i][m], minlength=N_RELATION * N_RELATION)
        out[i] = c / c.sum()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", action="store_true")
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()

    src = "cache/seqshape_opp.npy" if args.opponent else "cache/seqshape.npy"
    out = ("cache/features_shapetrans_opp.npy" if args.opponent
           else "cache/features_shapetrans.npy")
    S = np.load(src, mmap_mode="r")
    ok = np.load(src.replace(".npy", "_ok.npy"))
    n = len(S)
    F = np.zeros((n, N_RELATION * N_RELATION), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, args.chunk):
        j = min(i + args.chunk, n)
        F[i:j] = transitions(np.asarray(S[i:j]))
        if (i // args.chunk) % 10 == 0:
            print(f"  {j:,}/{n:,}（{time.time() - t0:.0f} 秒）", flush=True)
    np.save(out, F)
    np.save(out.replace(".npy", "_ok.npy"), ok)
    print(f"完成：{F.shape}，{F.nbytes / 1e6:.0f} MB，"
          f"{(time.time() - t0) / 60:.1f} 分 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
