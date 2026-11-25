"""測試賽前提出的跨題綜效主張：用棋力後驗重新加權候選庫。

想法：若 query 被判為 6D，而某候選被判為 D，兩者相容性極低，可以降權。
候選庫每人有 100 局，棋力估計應該相當可靠；query 只有 5~20 局，較不準。

    相容性 w_c = sum_k p_query(k) * p_candidate(k)
    最終分數 = 餘弦相似度 + lambda * log(w_c)

⚠️ 這招的效益完全取決於真實候選庫的等級組成，而那在 11/04 前是未知的。
若候選庫等級同質（例如全是段位），這招會失效甚至有害。
所以這裡跨 5 種組成測試，並一定要看 lambda=0 的退路。

用法：
    python3 scripts/eval_rank_prior.py [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from aicup.data import load_metadata, make_player_split       # noqa: E402
from aicup.benchmark import build_player_pool                 # noqa: E402
from aicup.baselines import PlayerBaseline, RankBaseline      # noqa: E402
from aicup.scoring import retrieval_report                    # noqa: E402

LAMBDAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)


def _posteriors(model, blocks, how="logmean"):
    """一次批次預測所有棋局再切分——逐一呼叫 4,000 棵樹的模型開銷極大。"""
    sizes = [len(b) for b in blocks]
    flat = np.concatenate([b for b in blocks if len(b)], axis=0)
    probs = model.game_proba(flat)
    cuts = np.cumsum([n for n in sizes if n])[:-1]
    parts = np.split(probs, cuts)
    out, it = [], iter(parts)
    for n in sizes:
        out.append(RankBaseline.aggregate(next(it), how) if n else np.full(10, 0.1))
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model", default="cache/rank_model.pkl")
    args = ap.parse_args()

    X = np.load("cache/features.npy", mmap_mode="r")
    ok = np.load("cache/features_ok.npy")
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    with open(args.model, "rb") as f:
        rank_model = pickle.load(f)

    results: dict = {}
    header = "%-12s" % "組成" + "".join("  λ=%-5.2f" % l for l in LAMBDAS)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for comp in ("uniform", "natural", "skewed", "dan_only", "kyu_only"):
        per_lambda = {l: [] for l in LAMBDAS}
        for s in range(args.seeds):
            pool = build_player_pool(meta, split, composition=comp, seed=s)
            ps = [p for p in pool.players
                  if ok[pool.gallery_rows[p]].sum() >= 20 and ok[pool.query_rows[p]].sum() >= 1]
            gf = {p: np.asarray(X[pool.gallery_rows[p][ok[pool.gallery_rows[p]]]]) for p in ps}
            qf = {p: np.asarray(X[pool.query_rows[p][ok[pool.query_rows[p]]]]) for p in ps}

            sim, _ = PlayerBaseline().build(gf, qf, ps)
            Pg = _posteriors(rank_model, [gf[p] for p in ps])
            Pq = _posteriors(rank_model, [qf[p] for p in ps])
            W = np.log(np.clip(Pq @ Pg.T, 1e-12, None))
            for l in LAMBDAS:
                per_lambda[l].append(retrieval_report(sim + l * W)["official_score"])

        means = {l: float(np.mean(v)) for l, v in per_lambda.items()}
        best = max(LAMBDAS, key=lambda l: means[l])
        results[comp] = {"means": means, "best_lambda": best,
                         "gain": means[best] - means[0.0],
                         "std": {l: float(np.std(v)) for l, v in per_lambda.items()}}
        row = "%-12s" % comp + "".join("  %.4f" % means[l] for l in LAMBDAS)
        print(f"{row}   最佳 λ={best:.2f} ({means[best] - means[0.0]:+.4f})", flush=True)

    with open("cache/rank_prior_report.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    gains = [r["gain"] for r in results.values()]
    print(f"\n各組成的最佳增益：min {min(gains):+.4f} / 中位 {np.median(gains):+.4f}"
          f" / max {max(gains):+.4f}", flush=True)
    print("報告已寫出：cache/rank_prior_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
