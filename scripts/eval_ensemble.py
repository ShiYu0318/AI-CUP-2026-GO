"""挑序列變體的集成組合——**貪婪前向選擇，用最壞情況收斂**。

## 為什麼需要這支

集成的收益來自**分歧**，不是每個成分都更強。實測 `seq_full`（+0.0053）與
`seq_w192`（+0.0061）各自都沒過 0.005 門檻，合起來卻是 +0.0079（8/8，t=9.15）。
所以「哪幾個該進集成」是一個獨立的問題，不能從個別分數推出來。

## 兩個刻意的選擇

**1. 貪婪前向，不是窮舉所有子集。**
k 個候選有 2^k 個子集，用 8 個 seed 去挑最高的那個，
挑到的多半是雜訊——這是標準的選擇偏誤。貪婪一次只加一個，
比較次數是 O(k²) 而不是 O(2^k)，而且每一步都要通過門檻才繼續。

**2. 收斂條件是「平均增益 ≥ 門檻**且**勝場 ≥ min_wins」。**
只看平均會選到「平均高一點但一半 seed 會輸」的組合。
本專案一貫用最壞情況挑（Sinkhorn 的 ε、三變體 vs 五變體都是這樣決定的）。

每個成分的後驗都吃 `artifacts/oof/` 的快取，所以跑過一次之後
比較不同組合是秒級的。

用法：
    python3 scripts/eval_ensemble.py --seeds 8 \\
        --base cache/rank_seq_opp/models.pt cache/rank_seq_opp_full/models.pt \\
               cache/rank_seq_w192/models.pt \\
        --candidates cache/rank_seq_gru/models.pt cache/rank_seq_shape/models.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aicup.benchmark import build_rank_questions                 # noqa: E402
from aicup.data import load_metadata, make_player_split          # noqa: E402
from aicup.oofcache import OofCache                              # noqa: E402
from eval_blend import (GAME, QUESTION, SEQ, SEQS, Components,   # noqa: E402
                        pick_device, score_blend)

DEFAULT_STORES = ["cache/sequences.npy", "cache/sequences_opp.npy"]


def tag_of(ckpt: str) -> str:
    return f"seq_{os.path.basename(os.path.dirname(ckpt))}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--base", nargs="*", default=list(SEQ),
                    help="目前上線中的序列變體（預設讀 eval_blend 的預設集成）")
    ap.add_argument("--candidates", nargs="*", default=[],
                    help="要考慮加進集成的候選 checkpoint")
    ap.add_argument("--stores", nargs="*", default=None,
                    help="序列庫；每個 checkpoint 一組時用 ';' 分隔，"
                         "只給一組就套用到全部")
    ap.add_argument("--floor", type=float, default=0.005)
    ap.add_argument("--min-wins", type=int, default=7,
                    help="至少要贏幾個 seed 才算真的有增益（預設 7/8）")
    ap.add_argument("--stacker", type=float, default=1.0)
    ap.add_argument("--cache-dir", default="artifacts/oof")
    ap.add_argument("--json-out", default="artifacts/ensemble_select.json")
    args = ap.parse_args()

    allck = list(dict.fromkeys(list(args.base) + list(args.candidates)))
    if args.stores:
        groups = [g.split(",") for g in args.stores]
        stores = groups if len(groups) == len(allck) else [groups[0]] * len(allck)
    else:
        by_default = {c: s for c, s in zip(SEQ, SEQS)}
        stores = [by_default.get(c, DEFAULT_STORES) for c in allck]
    store_of = dict(zip(allck, stores))

    comp = Components(pick_device())
    cache = OofCache(args.cache_dir)
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)

    print(f"基礎集成 {len(args.base)} 個，候選 {len(args.candidates)} 個", flush=True)
    per_seed = []          # [(parts_all, y, cal, sco, sizes)]
    t0 = time.time()
    for seed in range(args.seeds):
        q = build_rank_questions(meta, split, n_per_player=3, seed=seed)
        parts, y, pid = {}, None, None
        comps = [("game", comp.game, GAME), ("question", comp.question, QUESTION)]
        for ck in allck:
            comps.append((tag_of(ck),
                          (lambda qq, c=ck: comp.seq(qq, c, store_of[c])), ck))
        for name, fn, path in comps:
            tag = f"{name}__{os.path.basename(os.path.dirname(path))}"
            lo, y, pid = cache.get(tag, q, lambda f=fn, qq=q: f(qq),
                                   model_paths=[path])
            parts[name] = lo
        up = np.array(sorted(set(pid)))
        rr = np.random.default_rng(1000 + seed); rr.shuffle(up)
        cs = set(up[:len(up) // 2].tolist())
        inc = np.array([p in cs for p in pid])
        per_seed.append((parts, y, np.flatnonzero(inc), np.flatnonzero(~inc),
                         np.array([len(r) for r in q.game_rows], dtype=float)))
        print(f"  seed {seed} 後驗就緒", flush=True)
    print(f"{cache.summary()}，{time.time() - t0:.0f} 秒\n", flush=True)

    def evaluate(chosen):
        keep = {"game", "question"} | {tag_of(c) for c in chosen}
        out = []
        for parts, y, cal, sco, sizes in per_seed:
            sub = {k: v for k, v in parts.items() if k in keep}
            out.append(score_blend(sub, {}, y, cal, sco,
                                   stacker_c=args.stacker, sizes=sizes))
        return np.array(out)

    def report(name, a, ref=None):
        line = f"  {name:<44} {a.mean():.4f} ±{a.std(ddof=1):.4f}"
        if ref is not None:
            d = a - ref
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
            line += (f"  {d.mean():+.4f}"
                     f"（{int((d > 0).sum())}/{len(d)}，t={t:.2f}）")
        print(line, flush=True)

    cur = list(args.base)
    ref = evaluate(cur)
    report("基礎集成", ref)
    log = {"base": [tag_of(c) for c in cur], "base_score": float(ref.mean()),
           "steps": []}

    print("\n=== 每個候選單獨加進基礎集成 ===")
    for c in args.candidates:
        report(f"＋{tag_of(c)}", evaluate(cur + [c]), ref)

    print(f"\n=== 貪婪前向選擇（門檻 {args.floor:+.3f}，至少 "
          f"{args.min_wins}/{args.seeds} 勝）===")
    rest = list(args.candidates)
    while rest:
        scored = []
        for c in rest:
            a = evaluate(cur + [c])
            d = a - ref
            scored.append((d.mean(), int((d > 0).sum()), c, a))
        scored.sort(reverse=True, key=lambda x: x[0])
        gain, wins, c, a = scored[0]
        ok = gain >= args.floor and wins >= args.min_wins
        print(f"  最佳候選 {tag_of(c):<28} {gain:+.4f}（{wins}/{args.seeds}）"
              f" → {'採用' if ok else '停止'}")
        log["steps"].append({"candidate": tag_of(c), "gain": float(gain),
                             "wins": wins, "accepted": bool(ok),
                             "score": float(a.mean())})
        if not ok:
            break
        cur.append(c); rest.remove(c); ref = a

    print(f"\n最終集成（{len(cur)} 個）：{[tag_of(c) for c in cur]}")
    print(f"分數 {ref.mean():.4f} ±{ref.std(ddof=1):.4f}"
          f"（基礎集成 {log['base_score']:.4f}）")
    if len(cur) > len(args.base):
        print("\n要上線的話，把 predict.py 的 --rank-seq 改成：")
        print("  " + " ".join(cur))
    log["final"] = [tag_of(c) for c in cur]
    log["final_score"] = float(ref.mean())
    log["final_paths"] = cur
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        print(f"已寫出 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
