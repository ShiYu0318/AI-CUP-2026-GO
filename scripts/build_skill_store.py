"""用走法模型（`aicup.skillpolicy`）為驗證集每一盤算 10 種棋力下的對數似然，存成一個檔。

eval_blend 的 `skill` 成分讀這個檔，所有 seed 共用——一位玩家的棋在不同 seed 的題目裡
會一再出現，逐題重算是浪費。存檔記錄 checkpoint 的內容指紋，換模型時會拒絕使用舊檔。

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
    out = np.zeros((len(rows), 12), dtype=np.float32)
    t0 = time.time()
    for a in range(0, len(rows), args.chunk):
        rr = rows[a:a + args.chunk]
        games = [(cache.get(int(meta.iat[r, meta.columns.get_loc("file_idx")]),
                            int(meta.iat[r, meta.columns.get_loc("row")])),
                  meta.iat[r, meta.columns.get_loc("color")]) for r in rr]
        out[a:a + len(rr)] = game_loglik(net, games, dev, max_moves=args.max_moves)
        el = time.time() - t0; done = a + len(rr)
        print(f"  {done:,}/{len(rows):,}  剩約 {(len(rows) - done) * el / done / 60:.0f} 分", flush=True)
    path = store_path(args.ckpt, args.max_moves)
    np.savez(path, rows=rows, loglik=out, fingerprint=file_fingerprint(args.ckpt),
             max_moves=args.max_moves)
    print(f"已寫出 {path}（{len(rows):,} 盤，{(time.time() - t0) / 60:.0f} 分）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
