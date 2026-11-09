"""KataGo HumanSL 加進生產集成的配對評估。

HumanSL 每盤約 2.5 秒，不可能為 eval_blend 的全部題目（約 20 萬盤）都算一遍。
所以這裡改在**固定棋池**（`build_humansl_store.py --pool`）上出題：
只用已經算完全部 20 盤的玩家，局數照官方分布（5/10/15/20 各 1/4），
其餘成分（單局、題目層級、序列、板面）照生產設定現算（進 OOF 快取）。

比較的是同一批題目上「生產 stacker」與「生產 stacker ＋ HumanSL」，
校準與評分的玩家不重疊（每個 seed 5 種切分），逐 seed 配對。

    python3 scripts/eval_humansl.py --seeds 8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

from aicup.data import load_metadata
from aicup.dataset import SgfCache
from aicup.humansl import (FEATURE_NAMES, PROFILES, HumanSLStore, pool_players, pool_questions,
                           question_log_posterior)
from aicup.oofcache import OofCache
from aicup.scoring import apply_thresholds, optimise_thresholds, rank_final_score
from aicup.stacker import Stacker, stack_features

SIZES = (5, 10, 15, 20)


def _eval_blend():
    spec = importlib.util.spec_from_file_location(
        "eval_blend", os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_blend.py"))
    eb = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["eval_blend.py"]
    try:
        spec.loader.exec_module(eb)
    finally:
        sys.argv = argv
    return eb


def hs_extra(feats_per_q) -> np.ndarray:
    """論文的另一半特徵：各盤平均的首選一致率（10 profile）與目數損失的 4 個統計量。"""
    lo, hi = FEATURE_NAMES.index("top1_" + PROFILES[0]), FEATURE_NAMES.index("n_used")
    out = []
    for fs in feats_per_q:
        v = [f[lo:hi] for f in fs if f is not None and not np.isnan(f[lo:hi]).any()]
        out.append(np.mean(v, axis=0) if v else np.zeros(hi - lo))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--per-player", type=int, default=3, help="每位玩家每個 seed 出幾題")
    ap.add_argument("--mode", choices=["sum", "mean"], default="sum")
    ap.add_argument("--players", type=int, default=0,
                    help="只用棋池順序上前 N 位已完成的玩家（0 = 全部）。比較不同設定時要固定，"
                         "否則棋池還在長，兩次評估的題目不同、基準也不同")
    ap.add_argument("--extra", action="store_true",
                    help="除了對數似然，再把首選一致率與目數損失當成 stacker 的額外特徵")
    ap.add_argument("--store", default="cache/humansl/games.jsonl")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--opponent", action="store_true",
                    help="另外加入「對手落子」的 HumanSL 成分（humansl_opp），報告 生產／＋我方／＋我方＋對方 三者")
    args = ap.parse_args()

    eb = _eval_blend()
    meta = load_metadata()
    store = HumanSLStore(args.store)
    done, keys = pool_players(meta, store, SgfCache())
    okeys = None
    if args.opponent:
        done_o, okeys = pool_players(meta, store, SgfCache(), opponent=True)
        done = {p: rs for p, rs in done.items() if p in done_o}
    n_pool = len(np.load("cache/humansl/pool.npz")["rows"]) // 20
    if args.players:
        done = dict(list(done.items())[:args.players])          # 棋池順序（各等級輪流）
    done = {p: done[p] for p in sorted(done)}
    ranks = np.bincount([int(meta.at[rs[0], "rank_id"]) for rs in done.values()], minlength=10)
    print(f"棋池 {n_pool} 位玩家，HumanSL 已完成 {len(done)} 位；各等級 {ranks.tolist()}", flush=True)
    if len(done) < 50:
        print("完成的玩家太少，先不評估"); return 1

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    comp = eb.Components(dev)
    oof = OofCache()
    res = {"base": [], "hs": []} | ({"hs2": []} if args.opponent else {})
    acc = {k: {s: [] for s in SIZES} for k in res}
    for seed in range(args.seeds):
        q = pool_questions(meta, done, args.per_player, 7000 + seed, SIZES)
        gr, pid = q.game_rows, list(q.player_id)
        parts = {}
        comps = [("game", comp.game, [eb.GAME]), ("question", comp.question, [eb.QUESTION])]
        if eb.BOARD:
            comps.append(("boardset", comp.board, [eb.BOARD]))
        for ck, st in zip(eb.SEQ, eb.SEQS):
            comps.append((f"seq_{os.path.basename(os.path.dirname(ck))}",
                          (lambda qq, c=ck, s=st: comp.seq(qq, c, s)), [ck]))
        for name, fn, paths in comps:
            tag = f"hspool_{name}__{os.path.basename(os.path.dirname(paths[0]))}"
            parts[name], _, _ = oof.get(tag, q, lambda f=fn: f(q), model_paths=paths)
        fq = [[store.get(keys[int(r)]) for r in g] for g in gr]
        hs = question_log_posterior(fq, args.mode)
        ex = hs_extra(fq) if args.extra else None
        hso = (question_log_posterior([[store.get(okeys[int(r)]) for r in g] for g in gr], args.mode)
               if args.opponent else None)
        sizes = np.array([len(g) for g in gr], dtype=float)
        y = q.true_rank_id; up = np.array(sorted(set(pid)))
        row = {k: [] for k in res}
        for sp in range(args.splits):
            rr = np.random.default_rng(90000 + 31 * seed + sp); u = up.copy(); rr.shuffle(u)
            cs = set(u[:len(u) // 2].tolist()); inc = np.array([p in cs for p in q.player_id])
            cal, sco = np.flatnonzero(inc), np.flatnonzero(~inc)
            cfgs = [("base", parts), ("hs", {**parts, "humansl": hs})]
            if hso is not None:
                cfgs.append(("hs2", {**parts, "humansl": hs, "humansl_opp": hso}))
            for k, ps in cfgs:
                F = stack_features(ps, sizes)
                if k == "hs" and ex is not None:
                    F = np.hstack([F, ex])
                ev = Stacker(C=1.0).fit(F[cal], y[cal], parts=ps).expected_level(F)
                th = optimise_thresholds(ev[cal], y[cal]); pr = apply_thresholds(ev[sco], th)
                row[k].append(rank_final_score(pr, y[sco]))
                for s in SIZES:
                    m = sizes[sco] == s
                    acc[k][s].append(float((pr[m] == y[sco][m]).mean()))
        for k in res:
            res[k].append(float(np.mean(row[k])))
        print(f"  seed {seed}  生產 {res['base'][-1]:.4f}  ＋HumanSL {res['hs'][-1]:.4f}"
              + (f"  ＋對手 {res['hs2'][-1]:.4f}" if args.opponent else "")
              + f"  （{len(y)} 題）", flush=True)

    a, b = np.array(res["base"]), np.array(res["hs"]); d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    print(f"\n=== {len(done)} 位玩家、{args.seeds} seed x {args.splits} 切分"
          f"（mode={args.mode}{'＋額外特徵' if args.extra else ''}）===")
    print(f"  生產 stacker      {a.mean():.4f}")
    print(f"  ＋HumanSL         {b.mean():.4f}")
    print(f"  配對差            {d.mean():+.4f} ±{d.std(ddof=1):.4f}，{(d > 0).sum()}/{len(d)} 勝，t={d.mean() / se:.2f}")
    print("  完全猜中率：" + "　".join(f"{s} 盤 {np.mean(acc['base'][s]):.3f}→{np.mean(acc['hs'][s]):.3f}"
                                  for s in SIZES))
    if args.opponent:
        c = np.array(res["hs2"]); d2 = c - b; se2 = d2.std(ddof=1) / np.sqrt(len(d2))
        print(f"  ＋我方＋對方      {c.mean():.4f}")
        print(f"  對方的額外增益    {d2.mean():+.4f} ±{d2.std(ddof=1):.4f}，{(d2 > 0).sum()}/{len(d2)} 勝，t={d2.mean() / se2:.2f}")
        print("  完全猜中率（＋對方）：" + "　".join(f"{s} 盤 {np.mean(acc['hs2'][s]):.3f}" for s in SIZES))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"players": len(done), "mode": args.mode, "base": res["base"], "hs": res["hs"],
                       "acc": {k: {str(s): float(np.mean(v)) for s, v in acc[k].items()} for k in acc}},
                      fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
