"""競賽一：在手工特徵上做度量學習（目前增益最大的單一改動）。

實測：400 人候選庫的檢索分數從 0.6625（標準化+L2，無學習）
推到 **0.8167**（+0.1542），而且在筆電 GPU 上只要 1~2 分鐘。

支援多 seed 集成：不同 seed 的 embedding 各自 L2 正規化後串接，
等價於相似度平均（Kaggle 檢索類競賽的標準融合法）。

用法：
    python3 scripts/train_feature_metric.py --seeds 3 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup.data import load_metadata, make_player_split        # noqa: E402
from aicup.featstore import open_stores          # noqa: E402
from aicup.benchmark import build_player_pool                  # noqa: E402
from aicup.baselines import PlayerBaseline                     # noqa: E402
from aicup.feature_metric import (FeatureEmbedder, FeatureScaler,   # noqa: E402
                                  PKFeatureSampler, embed_features, player_template,
                                  identity_feature_columns)
from aicup.losses import SubCenterArcFace, batch_hard_triplet_loss  # noqa: E402
from aicup.scoring import retrieval_report                     # noqa: E402

COMPS = ("uniform", "natural", "skewed", "dan_only", "kyu_only")


def pick_device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(models, scaler, meta, split, X, ok, device, comps=("uniform",),
             seeds: int = 2, stage2=None, cols=None):
    """多個模型的 embedding 串接後檢索。stage2 不為 None 時再套重排。"""
    out = {}
    for comp in comps:
        scores = []
        for s in range(seeds):
            pool = build_player_pool(meta, split, composition=comp, seed=300 + s)
            ps = [p for p in pool.players
                  if ok[pool.gallery_rows[p]].sum() >= 20 and ok[pool.query_rows[p]].sum() >= 1]

            def tpl(rows):
                A = np.asarray(X[rows[ok[rows]]])
                if cols is not None:
                    A = A[:, cols]
                parts = [player_template(embed_features(m, scaler, A, device))
                         for m in models]
                v = np.concatenate(parts)
                return v / max(float(np.linalg.norm(v)), 1e-8)

            G = np.stack([tpl(pool.gallery_rows[p]) for p in ps])
            Q = np.stack([tpl(pool.query_rows[p]) for p in ps])
            sim = Q @ G.T
            if stage2 is not None:
                from aicup.rerank_stage2 import rerank_scores
                qn = np.array([int(ok[pool.query_rows[p]].sum()) for p in ps])
                gn = np.array([int(ok[pool.gallery_rows[p]].sum()) for p in ps])
                sim = rerank_scores(stage2["model"], sim, top_n=stage2["top_n"],
                                    query_sizes=qn, gallery_sizes=gn)
            scores.append(retrieval_report(sim)["official_score"])
        out[comp] = float(np.mean(scores))
    return out


def train_one(Xtr, labels, sampler, in_dim, args, device, seed: int):
    torch.manual_seed(seed)
    net = FeatureEmbedder(in_dim, embed_dim=args.embed_dim, hidden=args.hidden).to(device)
    arc = SubCenterArcFace(args.embed_dim, sampler.n_classes, k=3, scale=30.0,
                           class_counts=sampler.class_counts()).to(device)
    opt = torch.optim.AdamW([{"params": net.parameters(), "lr": args.lr},
                             {"params": arc.parameters(), "lr": args.lr * 10}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.lr * 10], total_steps=args.steps, pct_start=0.1)
    net.train()
    for step in range(1, args.steps + 1):
        idx, lab = sampler.sample()
        x = torch.from_numpy(Xtr[idx]).to(device)
        y = torch.from_numpy(lab).to(device)
        e, _ = net(x)
        loss = F.cross_entropy(arc(e, y), y) + args.triplet_weight * batch_hard_triplet_loss(e, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="訓練幾個模型做集成")
    ap.add_argument("--steps", type=int, default=3000,
                    help="實測 1500 步就接近天花板，再多沒有明顯增益")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--p", type=int, default=48)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--triplet-weight", type=float, default=0.3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stage2", default="cache/stage2/model.pkl")
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--keep-global-position", action="store_true",
                    help="保留 55 維的全局落子分布。預設拿掉——實測拿掉平均 +0.0161，"
                         "5 種等級組成全為正（整局分布由棋局走勢決定，不是由人決定）")
    ap.add_argument("--out", default="cache/feature_metric")
    ap.add_argument("--extra-features", nargs="*", default=[],
                    help="額外的特徵庫。`cache/features_opp.npy` 是對手視角——"
                         "對競賽二有 +0.033，對競賽一是否有用尚待驗證")
    args = ap.parse_args()
    _t0 = time.time()      # 記錄總耗時，寫進 cache/experiments.csv

    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"device = {device}", flush=True)

    X, ok = open_stores(["cache/features.npy"] + list(args.extra_features))
    if args.extra_features:
        print(f"  特徵庫 {X.shape[1]} 維（含 {len(args.extra_features)} 份額外）"
              f"，共同有效 {ok.mean():.4f}", flush=True)
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    vm = split.mask(meta)
    pid = meta["player_id"].to_numpy()

    tr = np.flatnonzero((~vm) & ok)
    players = sorted(set(pid[tr]))
    p2l = {p: i for i, p in enumerate(players)}
    labels = np.array([p2l[p] for p in pid[tr]], dtype=np.int64)
    print(f"訓練局數 {len(tr):,}，身分 {len(players):,}", flush=True)

    ident = np.asarray(
        identity_feature_columns(drop_global_position=not args.keep_global_position))
    if args.extra_features:
        # identity_feature_columns 的索引是針對「589 維手工特徵」定義的。
        # ⚠️ 額外的特徵庫**不一定也是 589 維**——例如棋形統計只有 48 維。
        # 舊版對每一份套用同一組相對索引，遇到寬度不同的庫會直接越界。
        # 改成：589 維的套身分欄位子集，其他寬度的整份都用
        #（那些庫本來就是為了加資訊而存在，沒有「要丟掉全域位置」的問題）。
        n_base = len(ident) and int(X.parts[0].shape[1])
        pieces, off = [], 0
        for part in X.parts:
            w = int(part.shape[1])
            pieces.append(ident + off if w == n_base else np.arange(w) + off)
            off += w
        cols = np.concatenate(pieces)
    else:
        cols = ident
    print(f"特徵欄位 {len(cols)} / {X.shape[1]}", flush=True)
    Xraw = np.asarray(X[tr])[:, cols]
    scaler = FeatureScaler().fit(Xraw)
    Xtr = scaler.transform(Xraw)
    sampler = PKFeatureSampler(labels, p=args.p, k=args.k, seed=0)
    print(f"可用身分 {sampler.n_classes:,}（局數 ≥2）", flush=True)

    # 基準：無學習
    base = {}
    for comp in ("uniform",):
        sc = []
        for s in range(args.eval_seeds):
            pool = build_player_pool(meta, split, composition=comp, seed=300 + s)
            ps = [p for p in pool.players
                  if ok[pool.gallery_rows[p]].sum() >= 20 and ok[pool.query_rows[p]].sum() >= 1]
            gf = {p: np.asarray(X[pool.gallery_rows[p][ok[pool.gallery_rows[p]]]]) for p in ps}
            qf = {p: np.asarray(X[pool.query_rows[p][ok[pool.query_rows[p]]]]) for p in ps}
            sim, _ = PlayerBaseline().build(gf, qf, ps)
            sc.append(retrieval_report(sim)["official_score"])
        base[comp] = float(np.mean(sc))
    print(f"基準（標準化+L2，無學習）{base['uniform']:.4f}\n", flush=True)

    models = []
    for s in range(args.seeds):
        t0 = time.time()
        models.append(train_one(Xtr, labels, sampler, Xtr.shape[1], args, device, seed=s))
        r = evaluate(models, scaler, meta, split, X, ok, device, seeds=args.eval_seeds, cols=cols)
        print(f"  seed {s} 訓練完（{(time.time() - t0) / 60:.1f} 分）"
              f"→ 累計 {len(models)} 個模型集成：{r['uniform']:.4f}", flush=True)

    stage2 = None
    if args.stage2 and os.path.exists(args.stage2):
        with open(args.stage2, "rb") as f:
            stage2 = pickle.load(f)

    print("\n=== 各等級組成 ===", flush=True)
    print(f"{'組成':<10} {'無學習':>8} {'度量學習':>10} {'＋Stage-2':>11}", flush=True)
    r_learn = evaluate(models, scaler, meta, split, X, ok, device,
                       comps=COMPS, seeds=args.eval_seeds, cols=cols)
    r_rr = (evaluate(models, scaler, meta, split, X, ok, device, comps=COMPS,
                     seeds=args.eval_seeds, stage2=stage2, cols=cols) if stage2 else {})
    report = {}
    for comp in COMPS:
        sc = []
        for s in range(args.eval_seeds):
            pool = build_player_pool(meta, split, composition=comp, seed=300 + s)
            ps = [p for p in pool.players
                  if ok[pool.gallery_rows[p]].sum() >= 20 and ok[pool.query_rows[p]].sum() >= 1]
            gf = {p: np.asarray(X[pool.gallery_rows[p][ok[pool.gallery_rows[p]]]]) for p in ps}
            qf = {p: np.asarray(X[pool.query_rows[p][ok[pool.query_rows[p]]]]) for p in ps}
            sim, _ = PlayerBaseline().build(gf, qf, ps)
            sc.append(retrieval_report(sim)["official_score"])
        b = float(np.mean(sc))
        report[comp] = {"baseline": b, "metric": r_learn[comp],
                        "metric_rerank": r_rr.get(comp)}
        print(f"{comp:<10} {b:>8.4f} {r_learn[comp]:>10.4f}"
              f" {r_rr.get(comp, float('nan')):>11.4f}", flush=True)

    torch.save({"models": [m.state_dict() for m in models],
                "scaler": scaler.state_dict(), "cols": cols, "args": vars(args)},
               os.path.join(args.out, "models.pt"))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n已寫出 {args.out}/models.pt", flush=True)

    from aicup.experiment_log import log_experiment
    log_experiment("player", "feature_metric", vars(args),
                   {c: report[c]["metric_rerank"] or report[c]["metric"] for c in COMPS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
