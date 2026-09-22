"""用真實特徵評估競賽一的各種後處理選項。

回答一個具體問題：k-reciprocal re-ranking 與 αQE 在本任務（1-對-1 匹配）
到底有沒有用？合成資料上看不出穩定增益，這裡用真實的 400 人模擬候選庫來測。

同時測「每位候選保留多個子 template」的變體——
理論上這能讓 gallery 具備 multi-per-identity 的結構，
給 k-reciprocal 發揮的空間。

用法：
    python3 scripts/eval_player_postprocess.py [--seeds 3] [--composition uniform]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from aicup.data import load_metadata, make_player_split      # noqa: E402
from aicup.benchmark import build_player_pool                # noqa: E402
from aicup.baselines import PlayerBaseline                   # noqa: E402
from aicup.rerank import alpha_query_expansion, k_reciprocal_rerank  # noqa: E402
from aicup.scoring import player_final_score                 # noqa: E402


def score_from_sim(sim: np.ndarray, owner: np.ndarray, n_id: int) -> dict:
    """gallery 可能一個身分有多個條目，先取該身分的最大相似度再排名。"""
    agg = np.stack([sim[:, owner == i].max(axis=1) for i in range(n_id)], axis=1)
    order = np.argsort(-agg, axis=1)
    ranks = np.array([np.where(order[i] == i)[0][0] for i in range(n_id)])
    return {"score": player_final_score(ranks),
            "top1": float((ranks == 0).mean()),
            "top5": float((ranks < 5).mean())}


def build_embeddings(X, ok, pool, n_sub: int, rng):
    """回傳 (query 向量, gallery 向量, gallery 所屬身分, 有效玩家清單)。"""
    players = [p for p in pool.players
               if ok[pool.gallery_rows[p]].sum() >= 20 and ok[pool.query_rows[p]].sum() >= 1]
    raw = {p: np.asarray(X[pool.gallery_rows[p][ok[pool.gallery_rows[p]]]]) for p in players}
    qraw = {p: np.asarray(X[pool.query_rows[p][ok[pool.query_rows[p]]]]) for p in players}

    pb = PlayerBaseline().fit(np.concatenate([raw[p] for p in players], axis=0))
    Q = np.stack([pb.template(qraw[p]) for p in players])

    G, owner = [], []
    for i, p in enumerate(players):
        g = raw[p]
        if n_sub <= 1:
            G.append(pb.template(g)); owner.append(i)
        else:
            idx = rng.permutation(len(g))
            for chunk in np.array_split(idx, n_sub):
                if len(chunk) == 0:
                    continue
                G.append(pb.template(g[chunk])); owner.append(i)
    return Q, np.stack(G), np.asarray(owner), players


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--composition", default="uniform")
    args = ap.parse_args()

    X = np.load("cache/features.npy", mmap_mode="r")
    ok = np.load("cache/features_ok.npy")
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)

    configs = [
        ("原始餘弦",                dict(qe=False, kr=False)),
        ("+ αQE",                   dict(qe=True,  kr=False)),
        ("+ k-reciprocal",          dict(qe=False, kr=True)),
        ("+ αQE + k-reciprocal",    dict(qe=True,  kr=True)),
    ]
    results: dict = {}

    for n_sub in (1, 5):
        print(f"\n{'=' * 64}", flush=True)
        print(f"每位候選的 gallery 條目數 = {n_sub}"
              f"（{'單一 template' if n_sub == 1 else '100 局切成 5 組子 template'}）")
        print("=" * 64, flush=True)
        for label, cfg in configs:
            runs = []
            for s in range(args.seeds):
                rng = np.random.default_rng(1000 + s)
                pool = build_player_pool(meta, split, composition=args.composition, seed=s)
                Q, G, owner, players = build_embeddings(X, ok, pool, n_sub, rng)
                q = alpha_query_expansion(Q, G) if cfg["qe"] else Q
                sim = (k_reciprocal_rerank(q, G, k1=20, k2=6, lambda_value=0.3)
                       if cfg["kr"] else q @ G.T)
                runs.append(score_from_sim(sim, owner, len(players)))
            sc = np.array([r["score"] for r in runs])
            t1 = np.array([r["top1"] for r in runs])
            results[f"sub{n_sub}_{label}"] = {"score_mean": float(sc.mean()),
                                              "score_std": float(sc.std()),
                                              "top1_mean": float(t1.mean())}
            print(f"  {label:<22} 分數 {sc.mean():.4f} ±{sc.std():.4f}   Top-1 {t1.mean():.4f}", flush=True)

    with open("cache/postprocess_report.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n報告已寫出：cache/postprocess_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
