"""上限量測：**每題的盤數**對棋力分數的影響，一路量到 100 盤。

## 為什麼要知道

官方競賽一的候選庫是「400 人**各 100 譜**」。若競賽二的 400 位查詢玩家
就是那 400 位候選人（`scripts/check_crosstask_link.py` 會在當天回答），
我們就能先認出每道棋力題是誰，再用那位候選人的 100 盤取代題目裡的 5~20 盤。

這支量的是那個假設成立時的**上限**：假設身分認得完全正確能拿到多少分。
上限不夠大的話，後面那套串接就不必做。

## 刻意只用單局成分

完整管線（單局＋題目層級＋3 個序列變體）跑 100 盤 × 550 位玩家會很重，
而這裡要看的是**盤數的邊際效益曲線**，不是絕對分數。
單局成分足以呈現曲線形狀，成本約八分之一。
（絕對值會比完整管線低約 0.05，比較的是相對關係。）

⚠️ 不要在有訓練在跑的時候開這支。實測三個重工作同時跑會把 swap 吃滿，
CPU 掉到 1.7%，所有工作一起變慢——`auto_loop.py` 的記憶體預算排程器
就是為了防這個，手動啟動會繞過它。

用法：
    python3 scripts/eval_more_games.py --sizes 5 10 20 50 100 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aicup.benchmark import _player_rows                        # noqa: E402
from aicup.data import load_metadata, make_player_split         # noqa: E402
from aicup.scoring import (apply_thresholds, expected_level,    # noqa: E402
                           optimise_thresholds, rank_final_score)
import eval_blend as EB                                         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[5, 10, 20, 50, 100])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--max-players", type=int, default=400,
                    help="每個盤數用幾位玩家出題（官方是 400 題）")
    ap.add_argument("--json-out", default="artifacts/more_games.json")
    args = ap.parse_args()

    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    comp = EB.Components(EB.pick_device())
    rank_of = meta.groupby("player_id")["rank_id"].first()
    rows = _player_rows(meta, split.val_players)

    out = {}
    print(f"{'每題盤數':<10}{'題數':>7}{'分數':>9}", flush=True)
    for size in args.sizes:
        # 盤數夠的玩家才能出題。⚠️ 大盤數會篩掉譜數少的玩家，
        # 而那群人的等級分布與全體不同——所以**所有盤數都用同一批玩家**，
        # 否則量到的是「玩家換了」而不是「盤數變了」。
        elig = sorted(p for p, idx in rows.items() if len(idx) >= max(args.sizes))
        rng0 = np.random.default_rng(12345)
        if len(elig) > args.max_players:
            elig = [elig[i] for i in
                    sorted(rng0.choice(len(elig), args.max_players, replace=False))]
        per_seed = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            gr = [rng.choice(rows[p], size=size, replace=False) for p in elig]
            y = np.array([int(rank_of[p]) for p in elig])
            pid = np.array(elig, dtype=object)
            q = SimpleNamespace(game_rows=gr, true_rank_id=y, player_id=pid)
            lo = comp.game(q)
            ev = expected_level(np.exp(lo) /
                                np.exp(lo).sum(axis=1, keepdims=True))
            up = np.array(sorted(set(pid)))
            sc = []
            for sp in range(args.splits):
                rr = np.random.default_rng(90000 + 31 * seed + sp)
                u = up.copy(); rr.shuffle(u)
                cs = set(u[:len(u) // 2].tolist())
                inc = np.array([p in cs for p in pid])
                cal, sco = np.flatnonzero(inc), np.flatnonzero(~inc)
                th = optimise_thresholds(ev[cal], y[cal])
                sc.append(rank_final_score(apply_thresholds(ev[sco], th), y[sco]))
            per_seed.append(float(np.mean(sc)))
        m = float(np.mean(per_seed))
        out[size] = {"mean": m, "std": float(np.std(per_seed, ddof=1))
                     if len(per_seed) > 1 else 0.0, "n_questions": len(elig)}
        print(f"{size:<10}{len(elig):>7}{m:>9.4f}", flush=True)

    if len(args.sizes) >= 2:
        a, b = min(args.sizes), max(args.sizes)
        print(f"\n{a} 盤 → {b} 盤：{out[b]['mean'] - out[a]['mean']:+.4f}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"已寫出 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
