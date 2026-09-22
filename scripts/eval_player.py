"""競賽一的統一評估入口——輸出格式與 `eval_blend.py` 相同，可以進自駕佇列。

**為什麼要有這支**：`scripts/auto_loop.py` 的 gate 寫死用 `eval_blend.py`，
所以迴圈到目前為止**只能改進競賽二**。競賽一雖然已經是 baseline 的 3.37 倍，
但它的分數同樣會計入名次，而且愈接近天花板、每 0.01 能拉開的名次愈多。

## 協定

與競賽二一樣是「用最壞情況挑」：候選庫的等級組成**未知**，
所以每個組成都要量，取跨組成的**最小值**當作決策依據，不是平均。
實測各組成的難度差很多（dan_only 0.9052 vs kyu_only 0.9418），
只看平均會選到「在某個組成下會崩」的設定。

`best_score` 寫的是**最壞組成**的分數，讓 `auto_loop.py` 的門檻比較有意義。

用法：
    python3 scripts/eval_player.py --seeds 3
    AICUP_METRIC=cache/feature_metric_shape/models.pt python3 scripts/eval_player.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aicup.benchmark import build_player_pool                    # noqa: E402
from aicup.data import load_metadata, make_player_split          # noqa: E402
from aicup.feature_metric import (FeatureEmbedder, FeatureScaler,  # noqa: E402
                                  embed_features, player_template)
from aicup.featstore import open_stores                          # noqa: E402
from aicup.rerank import sinkhorn_assignment                     # noqa: E402
from aicup.scoring import player_final_score, ranks_from_similarity  # noqa: E402

METRIC = os.environ.get("AICUP_METRIC", "cache/feature_metric/models.pt")
FEATS = os.environ.get("AICUP_PLAYER_FEATS", "cache/features.npy").split(",")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--compositions", nargs="*",
                    default=["uniform", "natural", "skewed", "dan_only", "kyu_only"])
    ap.add_argument("--sinkhorn", type=float, default=0.002)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device = {dev}\n  度量模型 {METRIC}\n  特徵庫 {FEATS}", flush=True)
    ck = torch.load(METRIC, map_location=dev, weights_only=False)
    sc = FeatureScaler().load_state_dict(ck["scaler"])
    cols = ck.get("cols")
    models = []
    for sd in ck["models"]:
        m = FeatureEmbedder(len(sc.mu), embed_dim=ck["args"]["embed_dim"],
                            hidden=ck["args"]["hidden"])
        m.load_state_dict(sd); m.eval().to(dev)
        models.append(m)

    X, ok = open_stores(FEATS)
    if cols is not None and max(cols) >= X.shape[1]:
        raise ValueError(
            f"checkpoint 的欄位索引最大到 {max(cols)}，但特徵庫只有 {X.shape[1]} 欄。"
            f"這個模型是用別組特徵庫訓練的——請用 AICUP_PLAYER_FEATS 指定正確的組合。")

    def tpl(rows):
        A = np.asarray(X[rows])
        if cols is not None:
            A = A[:, cols]
        v = np.concatenate([player_template(embed_features(m, sc, A, dev))
                            for m in models])
        return v / max(float(np.linalg.norm(v)), 1e-8)

    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    rows, t0 = {}, time.time()
    for comp in args.compositions:
        for seed in range(args.seeds):
            pool = build_player_pool(meta, split, composition=comp, seed=seed)
            names = list(pool.players)
            g = [pool.gallery_rows[p][ok[pool.gallery_rows[p]]] for p in names]
            q = [pool.query_rows[p][ok[pool.query_rows[p]]] for p in names]
            keep = [i for i in range(len(names)) if len(g[i]) and len(q[i])]
            G = np.stack([tpl(g[i]) for i in keep])
            Q = np.stack([tpl(q[i]) for i in keep])
            sim = Q @ G.T
            if args.sinkhorn > 0:
                sim = sinkhorn_assignment(sim, eps=args.sinkhorn)
            r = ranks_from_similarity(sim)
            rows.setdefault(comp, []).append(player_final_score(r))
            rows.setdefault(f"_top1_{comp}", []).append(float((r == 0).mean()))
        print(f"  {comp:<10} {np.mean(rows[comp]):.4f}"
              f"（top-1 {np.mean(rows[f'_top1_{comp}']):.4f}）", flush=True)

    out, worst, worst_name = {}, 1.0, None
    print(f"\n=== {args.seeds} seed × {len(args.compositions)} 種組成，"
          f"{time.time() - t0:.0f} 秒  官方 baseline 0.2845 ===")
    for comp in args.compositions:
        a = np.array(rows[comp])
        out[comp] = {"mean": float(a.mean()), "std": float(a.std(ddof=1))
                     if len(a) > 1 else 0.0,
                     "top1": float(np.mean(rows[f"_top1_{comp}"]))}
        print(f"  {comp:<10} {a.mean():.4f} ±{out[comp]['std']:.4f}")
        if a.mean() < worst:
            worst, worst_name = float(a.mean()), comp
    allm = float(np.mean([out[c]["mean"] for c in args.compositions]))
    print(f"\n  平均 {allm:.4f}   **最壞組成 {worst_name} = {worst:.4f}**")
    print("  （決策看最壞組成：真實候選庫的等級組成未知，"
          "只看平均會選到在某個組成下會崩的設定）")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"results": out, "best": worst_name, "best_score": worst,
                       "mean_over_compositions": allm}, f, indent=2,
                      ensure_ascii=False)
        print(f"已寫出 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
