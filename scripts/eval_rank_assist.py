"""競賽一：用**競賽二的棋力預測**輔助玩家匹配。

## 想法

競賽一目前 Top-1 0.9475、Top-5 0.9862——錯的那 5% 裡，
有一部分應該是把查詢配到了**棋力明顯不同**的候選身上。

而棋力資訊在這個任務裡特別便宜：
**候選庫每人有 100 譜**，棋力預測會非常準（查詢只有 5~20 譜，
但期望等級對真實等級的 Spearman 也有 0.90）。
把「預測棋力不合」當成匹配代價的懲罰項，就能砍掉跨等級的誤配。

    cost = cosine 相似度 − λ · |E[等級|查詢] − E[等級|候選]|

⚠️ 這與 RESULTS.md §3.3 是**相反的方向**。§3.3 測的是
「把競賽一的玩家 embedding 拿給競賽二用」——失敗了。
這裡是反過來，而且候選端的資訊量（100 譜 vs 5~20 譜）大得多。

λ=0 就是現況，所以這個實驗**不可能變差**——最壞情況是掃出來 λ=0 最好。

用法：
    python3 scripts/eval_rank_assist.py --seeds 3 --compositions uniform natural
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aicup.benchmark import build_player_pool                    # noqa: E402
from aicup.data import load_metadata, make_player_split          # noqa: E402
from aicup.feature_metric import (FeatureEmbedder, FeatureScaler,  # noqa: E402
                                  embed_features, player_template)
from aicup.featstore import open_stores                          # noqa: E402
from aicup.rerank import sinkhorn_assignment                     # noqa: E402
from aicup.scoring import expected_level, ranks_from_similarity  # noqa: E402
from aicup.scoring import player_final_score                     # noqa: E402
from eval_blend import Components, pick_device                   # noqa: E402

METRIC = os.environ.get("AICUP_METRIC", "cache/feature_metric/models.pt")


def load_metric(dev):
    ck = torch.load(METRIC, map_location=dev, weights_only=False)
    sc = FeatureScaler().load_state_dict(ck["scaler"])
    models = []
    for sd in ck["models"]:
        m = FeatureEmbedder(len(sc.mu), embed_dim=ck["args"]["embed_dim"],
                            hidden=ck["args"]["hidden"])
        m.load_state_dict(sd); m.eval().to(dev)
        models.append(m)
    return models, sc, ck.get("cols")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--compositions", nargs="*",
                    default=["uniform", "natural", "dan_only"])
    ap.add_argument("--lambdas", type=float, nargs="*",
                    default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--sinkhorn", type=float, default=0.002)
    ap.add_argument("--json-out", default="artifacts/rank_assist.json")
    args = ap.parse_args()

    dev = pick_device()
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    models, sc, cols = load_metric(dev)
    Xm, okm = open_stores(["cache/features.npy"])
    comp = Components(dev)                      # 競賽二的單局模型（用來估棋力）

    def template(rows):
        A = np.asarray(Xm[rows])
        if cols is not None:
            A = A[:, cols]
        v = np.concatenate([player_template(embed_features(m, sc, A, dev))
                            for m in models])
        return v / max(float(np.linalg.norm(v)), 1e-8)

    def levels(list_of_rows):
        """每一組棋局的期望等級（用競賽二的單局模型 + 幾何平均聚合）。"""
        P = np.exp(comp.game(SimpleNamespace(game_rows=list_of_rows)))
        return expected_level(P / P.sum(axis=1, keepdims=True))

    rows: dict[str, list[float]] = {}
    t0 = time.time()
    for cmp_name in args.compositions:
        for seed in range(args.seeds):
            pool = build_player_pool(meta, split, composition=cmp_name, seed=seed)
            names = list(pool.players)
            grows = [pool.gallery_rows[p][okm[pool.gallery_rows[p]]] for p in names]
            qrows = [pool.query_rows[p][okm[pool.query_rows[p]]] for p in names]
            keep = [i for i, (g, q) in enumerate(zip(grows, qrows))
                    if len(g) > 0 and len(q) > 0]
            grows = [grows[i] for i in keep]; qrows = [qrows[i] for i in keep]

            G = np.stack([template(r) for r in grows])
            Q = np.stack([template(r) for r in qrows])
            sim = Q @ G.T
            evg, evq = levels(grows), levels(qrows)
            gap = np.abs(evq[:, None] - evg[None, :])

            for lam in args.lambdas:
                s = sim - lam * gap
                s = sinkhorn_assignment(s, eps=args.sinkhorn) if args.sinkhorn > 0 else s
                r = ranks_from_similarity(s)
                rows.setdefault(f"λ={lam:g}", []).append(player_final_score(r))
                if lam == 0.0:
                    rows.setdefault("_top1", []).append(float((r == 0).mean()))
            print(f"  {cmp_name} seed {seed}（{len(keep)} 人）完成", flush=True)

    print(f"\n=== {len(rows['λ=0'])} 次評估（{args.compositions}）"
          f"，{time.time() - t0:.0f} 秒 ===")
    base = np.array(rows["λ=0"])
    out = {}
    for k in args.lambdas:
        a = np.array(rows[f"λ={k:g}"]); d = a - base
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12) if k else 0.0
        out[f"λ={k:g}"] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                           "delta": float(d.mean()), "wins": int((d > 0).sum())}
        ex = "" if k == 0 else f"  {d.mean():+.4f}（{int((d>0).sum())}/{len(d)}，t={t:.2f}）"
        print(f"  λ={k:<6g} {a.mean():.4f} ±{a.std(ddof=1):.4f}{ex}")
    print(f"\n（λ=0 的 Top-1 = {np.mean(rows['_top1']):.4f}）")
    best = max(out, key=lambda k: out[k]["mean"])
    print(f"最佳 {best}"
          + ("　→ 棋力輔助**沒有幫助**" if best == "λ=0" else ""))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
