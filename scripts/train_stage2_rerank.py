"""競賽一：訓練並評估 Stage-2 重排器。

協定（刻意設計成不會自欺）：
  - 用 N 個**訓練池**（不同 seed 的模擬候選庫）產生 pair 特徵訓練重排器
  - 在另外 M 個**未見的評估池**上比較「重排前 vs 重排後」的官方分數
  - 訓練池與評估池的 seed 完全不重疊

如果重排後沒有穩定超過重排前（超出 seed 之間的標準差），就不要用。

用法：
    python3 scripts/train_stage2_rerank.py --train-pools 4 --eval-pools 3
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup.data import load_metadata, make_player_split        # noqa: E402
from aicup.benchmark import build_player_pool                  # noqa: E402
from aicup.baselines import PlayerBaseline                     # noqa: E402
from aicup.rerank_stage2 import (pair_features, pair_labels,   # noqa: E402
                                 rerank_scores, feature_names, TOP_N)
from aicup.scoring import retrieval_report                     # noqa: E402


def build_sim(meta, split, X, ok, composition: str, seed: int):
    """建一個模擬候選庫，回傳 (相似度矩陣, 每題局數, 每位候選的 gallery 局數)。"""
    pool = build_player_pool(meta, split, composition=composition, seed=seed)
    ps = [p for p in pool.players
          if ok[pool.gallery_rows[p]].sum() >= 20 and ok[pool.query_rows[p]].sum() >= 1]
    gf = {p: np.asarray(X[pool.gallery_rows[p][ok[pool.gallery_rows[p]]]]) for p in ps}
    qf = {p: np.asarray(X[pool.query_rows[p][ok[pool.query_rows[p]]]]) for p in ps}
    sim, _ = PlayerBaseline().build(gf, qf, ps)
    return sim, np.array([len(qf[p]) for p in ps]), np.array([len(gf[p]) for p in ps])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pools", type=int, default=4)
    ap.add_argument("--eval-pools", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--composition", default="uniform")
    ap.add_argument("--train-compositions", nargs="*", default=None,
                    help="訓練池的等級組成。預設只用 --composition；"
                         "傳多個會混合訓練，對未見組成更穩健（真實候選庫組成未知）")
    ap.add_argument("--cross-composition", action="store_true",
                    help="評估涵蓋 5 種組成，測跨分布穩定性")
    ap.add_argument("--out", default="cache/stage2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    X = np.load("cache/features.npy", mmap_mode="r")
    ok = np.load("cache/features_ok.npy")
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)

    # ---- 訓練池 ----
    train_comps = args.train_compositions or [args.composition]
    print(f"建立訓練池：組成 {train_comps} × 各 {args.train_pools} 個 seed", flush=True)
    Xs, ys = [], []
    for ci, comp in enumerate(train_comps):
        for s in range(args.train_pools):
            sim, qn, gn = build_sim(meta, split, X, ok, comp, seed=100 + ci * 50 + s)
            f, pairs = pair_features(sim, top_n=args.top_n,
                                     query_sizes=qn, gallery_sizes=gn)
            Xs.append(f); ys.append(pair_labels(pairs))
            print(f"  {comp:<10} seed {s}: {sim.shape[0]} 題 -> {len(f):,} 個 pair",
                  flush=True)
    Xtr = np.concatenate(Xs); ytr = np.concatenate(ys)
    print(f"訓練 pair {len(Xtr):,}，正例率 {ytr.mean():.4f}", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         early_stopping=False, random_state=0)
    clf.fit(Xtr, ytr)

    # ---- 評估池（seed 與訓練完全不重疊）----
    comps = ([args.composition] if not args.cross_composition
             else ["uniform", "natural", "skewed", "dan_only", "kyu_only"])
    report = {}
    print("\n=== 重排前 vs 重排後（評估池的 seed 與訓練不重疊）===", flush=True)
    print(f"{'組成':<10} {'重排前':>8} {'重排後':>8} {'差異':>9}", flush=True)
    for comp in comps:
        before, after = [], []
        for s in range(args.eval_pools):
            sim, qn, gn = build_sim(meta, split, X, ok, comp, seed=900 + s)
            before.append(retrieval_report(sim)["official_score"])
            new = rerank_scores(clf, sim, top_n=args.top_n,
                                query_sizes=qn, gallery_sizes=gn)
            after.append(retrieval_report(new)["official_score"])
        b, a = float(np.mean(before)), float(np.mean(after))
        sd = float(np.std(before))
        report[comp] = {"before": b, "after": a, "delta": a - b, "before_std": sd}
        flag = "✅" if a - b > 2 * sd else ("≈" if a - b > -2 * sd else "❌")
        print(f"{comp:<10} {b:>8.4f} {a:>8.4f} {a - b:>+9.4f}  {flag}"
              f" (seed 標準差 ±{sd:.4f})", flush=True)

    names = feature_names(query_sizes=np.zeros(1), gallery_sizes=np.zeros(1))
    print("\n=== pair 特徵重要性（permutation，evaluation pool）===", flush=True)
    from sklearn.inspection import permutation_importance
    sim, qn, gn = build_sim(meta, split, X, ok, args.composition, seed=999)
    f, pairs = pair_features(sim, top_n=args.top_n, query_sizes=qn, gallery_sizes=gn)
    r = permutation_importance(clf, f, pair_labels(pairs), n_repeats=3,
                               random_state=0, scoring="average_precision")
    for i in np.argsort(-r.importances_mean)[:8]:
        print(f"  {names[i]:<18} {r.importances_mean[i]:+.4f}", flush=True)

    with open(os.path.join(args.out, "model.pkl"), "wb") as fh:
        pickle.dump({"model": clf, "top_n": args.top_n}, fh)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    gains = [v["delta"] for v in report.values()]
    print(f"\n各組成增益：min {min(gains):+.4f} / 中位 {np.median(gains):+.4f}"
          f" / max {max(gains):+.4f}", flush=True)
    if np.median(gains) <= 2 * report[comps[0]]["before_std"]:
        print("⚠️ 中位增益未超過 2 倍 seed 標準差 —— **不建議啟用**", flush=True)
    else:
        print("✅ 增益穩定超過雜訊 —— 可以啟用", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
