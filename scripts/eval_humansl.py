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
from types import SimpleNamespace

import numpy as np
import torch

from aicup.data import load_metadata
from aicup.dataset import SgfCache
from aicup.humansl import HumanSLStore, N_PROFILES, FEATURE_NAMES, game_key, question_loglik
from aicup.oofcache import OofCache
from aicup.scoring import apply_thresholds, optimise_thresholds, rank_final_score
from aicup.stacker import Stacker, stack_features

POOL_FILE = "cache/humansl/pool.npz"
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


def hs_part(feats_per_q, mode: str) -> np.ndarray:
    """一題的 HumanSL 成分（log 後驗形式）。

    sum：所有手數的對數似然加總——盤數越多、證據越強（生成式分類的原意）。
    mean：除以總手數的每手平均——尺度穩定，盤數交給 stacker 的 log(盤數) 特徵。
    """
    n_used = FEATURE_NAMES.index("n_used")
    out = []
    for fs in feats_per_q:
        L = question_loglik(fs)
        if mode == "mean":
            L = L / max(sum(f[n_used] for f in fs if f is not None), 1.0)
        out.append(L - (L.max() + np.log(np.exp(L - L.max()).sum())))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--per-player", type=int, default=3, help="每位玩家每個 seed 出幾題")
    ap.add_argument("--mode", choices=["sum", "mean"], default="sum")
    ap.add_argument("--store", default="cache/humansl/games.jsonl")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    eb = _eval_blend()
    meta = load_metadata()
    store = HumanSLStore(args.store)
    rows = np.load(POOL_FILE)["rows"]
    cache_sgf = SgfCache()
    keys = {int(r): game_key(cache_sgf.get(int(meta.at[r, "file_idx"]), int(meta.at[r, "row"])),
                             meta.at[r, "color"]) for r in rows}
    by_player: dict = {}
    for r in rows:
        by_player.setdefault(meta.at[r, "player_id"], []).append(int(r))
    done = {p: rs for p, rs in by_player.items() if all(keys[r] in store for r in rs)}
    ranks = np.bincount([int(meta.at[rs[0], "rank_id"]) for rs in done.values()], minlength=10)
    print(f"棋池 {len(by_player)} 位玩家，HumanSL 已完成 {len(done)} 位；各等級 {ranks.tolist()}", flush=True)
    if len(done) < 50:
        print("完成的玩家太少，先不評估"); return 1

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    comp = eb.Components(dev)
    oof = OofCache()
    res = {"base": [], "hs": []}
    acc = {k: {s: [] for s in SIZES} for k in res}
    for seed in range(args.seeds):
        rng = np.random.default_rng(7000 + seed)
        gr, y, pid = [], [], []
        for p in sorted(done):
            rs = done[p]
            for _ in range(args.per_player):
                n = int(rng.choice(SIZES))
                gr.append(np.array(sorted(rng.choice(rs, n, replace=False)), dtype=np.int64))
                y.append(int(meta.at[rs[0], "rank_id"])); pid.append(p)
        q = SimpleNamespace(game_rows=gr, true_rank_id=np.array(y), player_id=np.array(pid, dtype=object))
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
        hs = hs_part([[store.get(keys[int(r)]) for r in g] for g in gr], args.mode)
        sizes = np.array([len(g) for g in gr], dtype=float)
        y = q.true_rank_id; up = np.array(sorted(set(pid)))
        row = {"base": [], "hs": []}
        for sp in range(args.splits):
            rr = np.random.default_rng(90000 + 31 * seed + sp); u = up.copy(); rr.shuffle(u)
            cs = set(u[:len(u) // 2].tolist()); inc = np.array([p in cs for p in q.player_id])
            cal, sco = np.flatnonzero(inc), np.flatnonzero(~inc)
            for k, ps in (("base", parts), ("hs", {**parts, "humansl": hs})):
                F = stack_features(ps, sizes)
                ev = Stacker(C=1.0).fit(F[cal], y[cal], parts=ps).expected_level(F)
                th = optimise_thresholds(ev[cal], y[cal]); pr = apply_thresholds(ev[sco], th)
                row[k].append(rank_final_score(pr, y[sco]))
                for s in SIZES:
                    m = sizes[sco] == s
                    acc[k][s].append(float((pr[m] == y[sco][m]).mean()))
        for k in res:
            res[k].append(float(np.mean(row[k])))
        print(f"  seed {seed}  生產 {res['base'][-1]:.4f}  ＋HumanSL {res['hs'][-1]:.4f}"
              f"  （{len(y)} 題）", flush=True)

    a, b = np.array(res["base"]), np.array(res["hs"]); d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    print(f"\n=== {len(done)} 位玩家、{args.seeds} seed x {args.splits} 切分（mode={args.mode}）===")
    print(f"  生產 stacker      {a.mean():.4f}")
    print(f"  ＋HumanSL         {b.mean():.4f}")
    print(f"  配對差            {d.mean():+.4f} ±{d.std(ddof=1):.4f}，{(d > 0).sum()}/{len(d)} 勝，t={d.mean() / se:.2f}")
    print("  完全猜中率：" + "　".join(f"{s} 盤 {np.mean(acc['base'][s]):.3f}→{np.mean(acc['hs'][s]):.3f}"
                                  for s in SIZES))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"players": len(done), "mode": args.mode, "base": res["base"], "hs": res["hs"],
                       "acc": {k: {str(s): float(np.mean(v)) for s, v in acc[k].items()} for k in acc}},
                      fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
