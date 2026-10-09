"""用 KataGo HumanSL 為一批棋算每盤特徵，寫進可續跑的儲存檔（`aicup.humansl`）。

兩種來源：

    # 驗證集玩家的固定棋池（每等級 P 位玩家、每位 G 盤），供融合評估用
    python3 scripts/build_humansl_store.py --pool --players-per-rank 80 --games 20

    # 官方測試檔（rank_prediction_test_{public,private}.csv 的全部棋）
    python3 scripts/build_humansl_store.py --test-dir mock_tests

很慢（每盤約 2.5 秒，GPU 與其他程式共用時），所以：
- 結果逐盤追加寫入，中斷後重跑會跳過已完成的盤；
- 棋池的處理順序在各等級之間輪流，停在任何時候已完成的部分都是各等級平均的。
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from aicup.data import load_metadata, make_player_split
from aicup.dataset import SgfCache
from aicup.humansl import HumanSLEngine, HumanSLStore, game_key
from aicup.submission import resolve_test_file

POOL_FILE = "cache/humansl/pool.npz"


def build_pool(players_per_rank: int, games: int, seed: int = 1) -> np.ndarray:
    """驗證集（與所有模型的訓練集不重疊）中，每等級 players_per_rank 位、每位 games 盤的 meta 列索引。

    回傳的順序是**一次完成一位玩家的全部盤數**，玩家則在各等級之間輪流
    （等級 0 的第 1 位、等級 1 的第 1 位……等級 9 的第 1 位、等級 0 的第 2 位……）。
    這樣跑到任何時候停下來，已完成的都是「完整的玩家、各等級數量相近」，可以直接拿來評估。
    """
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    vm = meta[split.mask(meta)]
    rng = np.random.default_rng(seed)
    by_rank = []
    for r in range(10):
        sub = vm[vm["rank_id"] == r]
        cnt = sub["player_id"].value_counts()
        cands = cnt[cnt >= games].index.to_numpy()
        by_rank.append([rng.choice(sub.index[sub["player_id"] == p].to_numpy(), games, replace=False)
                        for p in rng.choice(cands, min(players_per_rank, len(cands)), replace=False)])
    order = [g for i in range(max(map(len, by_rank))) for r in range(10) if i < len(by_rank[r])
             for g in by_rank[r][i]]
    return np.asarray(order, dtype=np.int64)


def test_items(test_dir: str):
    for tag in ("public", "private"):
        df = pd.read_csv(resolve_test_file(test_dir, "rank", tag))
        for i in range(1, 21):
            if f"sgf_{i}" not in df:
                break
            for sgf, col in zip(df[f"sgf_{i}"], df[f"color_{i}"]):
                if isinstance(sgf, str) and sgf.strip():
                    yield sgf, col


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true", help="驗證集玩家的固定棋池")
    ap.add_argument("--players-per-rank", type=int, default=80)
    ap.add_argument("--games", type=int, default=20, help="棋池中每位玩家的盤數")
    ap.add_argument("--test-dir", default=None, help="官方測試檔所在目錄")
    ap.add_argument("--limit", type=int, default=0, help="最多處理幾盤（0 = 全部；冒煙測試用）")
    ap.add_argument("--moves", type=int, default=24, help="每盤評分的我方手數")
    ap.add_argument("--store", default="cache/humansl/games.jsonl")
    args = ap.parse_args()
    if not (args.pool or args.test_dir):
        ap.error("請指定 --pool 或 --test-dir")

    for d in (os.path.dirname(args.store), os.path.dirname(POOL_FILE)):
        os.makedirs(d or ".", exist_ok=True)
    store = HumanSLStore(args.store)
    items = []
    if args.pool:
        rows = build_pool(args.players_per_rank, args.games)
        np.savez(POOL_FILE, rows=rows, players_per_rank=args.players_per_rank, games=args.games)
        meta, cache = load_metadata(), SgfCache()
        for i in rows:
            rec = meta.loc[i]
            sgf = cache.get(int(rec["file_idx"]), int(rec["row"]))
            items.append((game_key(sgf, rec["color"]), sgf, rec["color"]))
    if args.test_dir:
        items += [(game_key(s, c), s, c) for s, c in test_items(args.test_dir)]
    seen, todo = set(), []
    for it in items:
        if it[0] not in store and it[0] not in seen:
            seen.add(it[0]); todo.append(it)
    n_done = len({it[0] for it in items}) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"共 {len(items):,} 盤，已完成 {n_done:,}，這次要算 {len(todo):,}", flush=True)
    if not todo:
        return 0

    t0, n = time.time(), 0
    with open(args.store, "a") as fh:
        for key, f in HumanSLEngine(n_moves=args.moves).run(todo):
            store.append(key, f, fh)
            n += 1
            if n % 200 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n:,}/{len(todo):,}  {el / n:.2f} 秒/盤  剩約 {(len(todo) - n) * el / n / 3600:.1f} 小時",
                      flush=True)
    print(f"完成 {n:,} 盤，{(time.time() - t0) / 60:.0f} 分；儲存檔共 {len(store):,} 盤", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
