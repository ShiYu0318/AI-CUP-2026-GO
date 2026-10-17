"""用走法模型（`aicup.skillpolicy`）為驗證集每一盤算 10 種棋力下的對數似然，存成一個檔。

eval_blend 的 `skill` 成分讀這個檔，所有 seed 共用——一位玩家的棋在不同 seed 的題目裡
會一再出現，逐題重算是浪費。存檔記錄 checkpoint 的內容指紋，換模型時會拒絕使用舊檔。

全量約 9 小時，所以**每個批次都存進度**（`*.partial.npz`，原子寫入），中斷後重跑會接續。
⚠️ 第一版只在最後寫檔，2026-10-13 跑到 120,000 / 216,418 盤時程序被終止，5 小時的結果全部遺失。

    python3 scripts/build_skill_store.py --ckpt cache/skill_policy/best.pt --max-moves 48
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from aicup.data import load_metadata, make_player_split
from aicup.dataset import SgfCache
from aicup.oofcache import file_fingerprint
from aicup.rank_nets import pick_device
from aicup.skillpolicy import game_loglik, load_skill_policy


def store_path(ckpt: str, max_moves: int) -> str:
    return os.path.join(os.path.dirname(ckpt), f"val_loglik_m{max_moves}.npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="cache/skill_policy/best.pt")
    ap.add_argument("--max-moves", type=int, default=48, help="每盤評分的我方手數（0 = 全部）")
    ap.add_argument("--chunk", type=int, default=2000, help="每幾盤回報一次進度")
    args = ap.parse_args()

    dev = pick_device("auto")
    net = load_skill_policy(args.ckpt, dev)
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    rows = np.flatnonzero(split.mask(meta))
    cache = SgfCache()
    fp = file_fingerprint(args.ckpt)
    path = store_path(args.ckpt, args.max_moves)
    partial = path.replace(".npz", ".partial.npz")
    out = np.zeros((len(rows), 12), dtype=np.float32)
    done = np.zeros(len(rows), dtype=bool)
    if os.path.exists(partial):
        z = np.load(partial)
        if str(z["fingerprint"]) == fp and np.array_equal(z["rows"], rows):
            out, done = z["loglik"], z["done"]
            print(f"從進度檔接續：已完成 {done.sum():,} 盤", flush=True)
        else:
            print("進度檔屬於另一顆 checkpoint 或另一組棋，忽略", flush=True)

    def save(target, **extra):
        tmp = target + ".tmp"
        with open(tmp, "wb") as fh:                    # 傳檔案物件，np.savez 才不會自己補副檔名
            np.savez(fh, rows=rows, loglik=out, done=done, fingerprint=fp,
                     max_moves=args.max_moves, **extra)
        os.replace(tmp, target)

    t0 = time.time(); n_new = 0
    for a in range(0, len(rows), args.chunk):
        if done[a:a + args.chunk].all():
            continue
        rr = rows[a:a + args.chunk]
        games = [(cache.get(int(meta.iat[r, meta.columns.get_loc("file_idx")]),
                            int(meta.iat[r, meta.columns.get_loc("row")])),
                  meta.iat[r, meta.columns.get_loc("color")]) for r in rr]
        out[a:a + len(rr)] = game_loglik(net, games, dev, max_moves=args.max_moves)
        done[a:a + len(rr)] = True
        save(partial)
        n_new += len(rr); el = time.time() - t0
        print(f"  {done.sum():,}/{len(rows):,}  剩約 {(~done).sum() * el / n_new / 60:.0f} 分", flush=True)
    save(path)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"已寫出 {path}（{len(rows):,} 盤，這次 {(time.time() - t0) / 60:.0f} 分）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
