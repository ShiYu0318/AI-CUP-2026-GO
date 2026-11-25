"""算 Strength Estimator（ICLR 2025）的逐盤強度特徵，存成只追加的 JSONL（可中斷續跑）。

    python3 scripts/build_se_store.py --model ~/se_model_dl/x/se_go.pt --pool --test-dir data

每盤一行 {"k": sha1(sgf)[:20], "f": SE_FEATURES}；與執色無關，題目端用 side_features() 挑我方／對方。
棋池與 HumanSL 相同（cache/humansl/pool.npz），校準題才能共用。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from aicup.se_strength import N_SE, game_hash, load_se, score_games  # noqa: E402

STORE = "cache/se/games.jsonl"


class SEStore:
    def __init__(self, path: str = STORE):
        self.path = path
        self.data: dict[str, np.ndarray] = {}
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    f = np.asarray(r["f"], dtype=np.float32)
                    if len(f) == N_SE:
                        self.data[r["k"]] = f

    def __contains__(self, k):
        return k in self.data

    def get(self, k):
        return self.data.get(k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import torch
    from build_humansl_store import test_items
    from aicup.humansl import POOL_FILE

    os.makedirs(os.path.dirname(args.store), exist_ok=True)
    store = SEStore(args.store)
    sgfs = []
    if args.test_dir:                                   # 測試集先算，早點能檢查
        sgfs += [s for s, _ in test_items(args.test_dir)]
    if args.pool:
        from aicup.data import load_metadata
        from aicup.dataset import SgfCache
        meta, cache = load_metadata(), SgfCache()
        for i in np.load(POOL_FILE)["rows"]:
            sgfs.append(cache.get(int(meta.at[i, "file_idx"]), int(meta.at[i, "row"])))
    todo, seen = [], set()
    for s in sgfs:
        k = game_hash(s)
        if k not in store and k not in seen:
            todo.append(s); seen.add(k)
    print(f"共 {len(sgfs)} 盤，待算 {len(todo)} 盤（stride {args.stride}）", flush=True)
    dev = torch.device(args.device)
    model = load_se(args.model, dev, half=args.device != "cpu")
    t0 = time.time()
    with open(args.store, "a") as fh:
        for a in range(0, len(todo), args.chunk):
            part = todo[a:a + args.chunk]
            F = score_games(model, part, dev, batch=256, stride=args.stride, half=args.device != "cpu")
            for s, f in zip(part, F):
                fh.write(json.dumps({"k": game_hash(s), "f": [None if np.isnan(x) else round(float(x), 5) for x in f]}) + "\n")
            fh.flush()
            done = a + len(part); el = time.time() - t0
            if (a // args.chunk) % 10 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}  {el / 60:.1f} 分，剩約 {el / done * (len(todo) - done) / 60:.0f} 分", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
