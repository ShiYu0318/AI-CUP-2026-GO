"""用落子時間盤面模型（`aicup.timeplanes`）為驗證集每一盤算 10 級機率，存成一個檔。

eval_blend 的 `timeplanes` 成分讀這個檔（所有 seed 共用）。存檔記錄 checkpoint 指紋，換模型會拒用舊檔；
每個批次都存進度（`*.partial.npz`），中斷後重跑會接續。

    python3 scripts/build_timeplanes_store.py --ckpt cache/timeplanes/w96b8.pt
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from aicup.data import load_metadata, make_player_split
from aicup.dataset import SgfCache
from aicup.oofcache import file_fingerprint
from aicup.rank_nets import pick_device
from aicup.timeplanes import game_proba, load_timeplanes


def store_path(ckpt: str) -> str:
    return os.path.splitext(ckpt)[0] + "_val.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="cache/timeplanes/w96b8.pt")
    ap.add_argument("--chunk", type=int, default=5000)
    args = ap.parse_args()
    dev = pick_device("auto")
    net = load_timeplanes(args.ckpt, dev)
    meta = load_metadata()
    rows = np.flatnonzero(make_player_split(meta, val_frac=0.20, seed=42).mask(meta))
    cache = SgfCache()
    fp = file_fingerprint(args.ckpt)
    path = store_path(args.ckpt); partial = path.replace(".npz", ".partial.npz")
    out = np.zeros((len(rows), 10), np.float32); done = np.zeros(len(rows), bool)
    if os.path.exists(partial):
        z = np.load(partial)
        if str(z["fingerprint"]) == fp:
            out, done = z["proba"], z["done"]
            print(f"從進度檔接續：已完成 {done.sum():,} 盤", flush=True)

    def save(target):
        tmp = target + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, rows=rows, proba=out, done=done, fingerprint=fp)
        os.replace(tmp, target)

    fi, ri, co = (meta.columns.get_loc(c) for c in ("file_idx", "row", "color"))
    t0 = time.time(); n_new = 0
    for a in range(0, len(rows), args.chunk):
        if done[a:a + args.chunk].all():
            continue
        rr = rows[a:a + args.chunk]
        games = [(cache.get(int(meta.iat[r, fi]), int(meta.iat[r, ri])), meta.iat[r, co]) for r in rr]
        out[a:a + len(rr)] = game_proba(net, games, dev)
        done[a:a + len(rr)] = True; save(partial)
        n_new += len(rr)
        print(f"  {done.sum():,}/{len(rows):,}  剩約 {(~done).sum() * (time.time() - t0) / n_new / 60:.0f} 分", flush=True)
    save(path)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"已寫出 {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
