"""跑完兩個子題的手工特徵 baseline，產出可追蹤的離線分數。

這支腳本就是 67 天沒有 leaderboard 期間的儀表板。每次改動後重跑它，
比較的是 player-disjoint、題目層級、官方公式的分數——不是單局 accuracy。

用法：
    python3 scripts/run_baselines.py [--train-per-rank 20000] [--seeds 3]

產出：
    cache/baseline_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup import LEVELS                                    # noqa: E402
from aicup.data import load_metadata, make_player_split     # noqa: E402
from aicup.benchmark import build_rank_questions, build_player_pool  # noqa: E402
from aicup.baselines import RankBaseline, PlayerBaseline, score_by_query_size  # noqa: E402

CACHE = "cache"


def load_feature_store():
    fx = os.path.join(CACHE, "features.npy")
    fo = os.path.join(CACHE, "features_ok.npy")
    if not os.path.exists(fx):
        raise SystemExit("找不到 cache/features.npy，請先執行 scripts/build_feature_store.py")
    return np.load(fx, mmap_mode="r"), np.load(fo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-rank", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3, help="候選庫重複抽樣次數")
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    X, ok = load_feature_store()
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=args.split_seed)
    val_mask = split.mask(meta)
    report: dict = {"config": vars(args)}
    print(f"特徵庫 {X.shape}，成功率 {ok.mean():.4f}")
    print(f"切分：訓練玩家 {len(split.train_players):,} / 驗證玩家 {len(split.val_players):,}\n")

    # ---------------- 競賽二 ----------------
    print("=" * 62)
    print("競賽二：棋力預測")
    print("=" * 62)
    rng = np.random.default_rng(0)
    rank_id = meta["rank_id"].to_numpy()
    tr_rows = []
    for lv in range(len(LEVELS)):
        cand = np.flatnonzero((~val_mask) & ok & (rank_id == lv))
        take = min(args.train_per_rank, len(cand))
        tr_rows.append(rng.choice(cand, size=take, replace=False))
    tr_rows = np.concatenate(tr_rows)
    print(f"訓練局數 {len(tr_rows):,}")

    t0 = time.time()
    model = RankBaseline().fit(np.asarray(X[tr_rows]), rank_id[tr_rows])
    print(f"訓練完成 {time.time() - t0:.0f} 秒")

    q = build_rank_questions(meta, split, n_per_player=3, seed=0)
    feats = [np.asarray(X[r[ok[r]]]) for r in q.game_rows]
    print(f"驗證題數 {len(q):,}，每題局數 mean {q.n_games.mean():.1f}\n")

    report["rank"] = {}
    for how in ("mean", "logmean"):
        res = model.evaluate(feats, q.true_rank_id, how=how)
        report["rank"][how] = res
        print(f"  聚合={how:<8} argmax {res['score_argmax']:.4f}"
              f" | 期望解碼 {res['score_expected']:.4f}"
              f" | 增益 {res['decoding_gain']:+.4f}")

    # 單局 accuracy 當對照，說明聚合貢獻多大
    va_rows = np.flatnonzero(val_mask & ok)[:50000]
    single = float((model.game_proba(np.asarray(X[va_rows])).argmax(1) == rank_id[va_rows]).mean())
    report["rank"]["single_game_top1"] = single
    print(f"  （對照）單局 top-1 accuracy {single:.4f}\n")

    # ---------------- 競賽一 ----------------
    print("=" * 62)
    print("競賽一：玩家辨識")
    print("=" * 62)
    report["player"] = {}
    for comp in ("uniform", "natural", "skewed", "dan_only", "kyu_only"):
        scores, tops = [], []
        detail = None
        for s in range(args.seeds):
            pool = build_player_pool(meta, split, composition=comp, seed=s)
            gf = {p: np.asarray(X[pool.gallery_rows[p][ok[pool.gallery_rows[p]]]])
                  for p in pool.players}
            qf = {p: np.asarray(X[pool.query_rows[p][ok[pool.query_rows[p]]]])
                  for p in pool.players}
            keep = [p for p in pool.players if len(gf[p]) >= 10 and len(qf[p]) >= 1]
            pb = PlayerBaseline()
            sim, _ = pb.build(gf, qf, keep)
            from aicup.scoring import retrieval_report
            rep = retrieval_report(sim)
            scores.append(rep["official_score"]); tops.append(rep["top1"])
            if s == 0:
                detail = score_by_query_size(sim, [len(qf[p]) for p in keep])
        report["player"][comp] = {
            "n_candidates": len(keep),
            "score_mean": float(np.mean(scores)), "score_std": float(np.std(scores)),
            "top1_mean": float(np.mean(tops)),
            "by_query_size": detail,
        }
        print(f"  {comp:<10} {len(keep):>3} 人  分數 {np.mean(scores):.4f} ±{np.std(scores):.4f}"
              f"  Top-1 {np.mean(tops):.4f}")

    print("\n  依每題局數分組（composition=uniform, seed=0）:")
    for k, v in (report["player"]["uniform"]["by_query_size"] or {}).items():
        print(f"    {k:>6} 局: n={v['n']:>3}  分數 {v['score']:.4f}  Top-1 {v['top1']:.4f}")

    out = os.path.join(CACHE, "baseline_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n報告已寫出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
