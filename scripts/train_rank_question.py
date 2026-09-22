"""競賽二：訓練題目層級模型，並與單局模型融合。

這是目前把競賽二推過官方 baseline 的關鍵：
單獨 0.4584、單局模型 0.4566，但**融合後 0.4841**（baseline 0.4631）。

用法：
    python3 scripts/train_rank_mlp.py --seeds 3        # 先有單局模型
    python3 scripts/train_rank_question.py --seeds 3   # 再訓練題目層級並搜尋融合權重
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aicup.data import load_metadata, make_player_split            # noqa: E402
from aicup.featstore import open_stores          # noqa: E402
from aicup.benchmark import build_rank_questions                   # noqa: E402
from aicup.feature_metric import FeatureScaler                     # noqa: E402
from aicup.losses import SoftLabelCrossEntropy, CornLoss, corn_to_proba  # noqa: E402
from aicup.question_model import build_question_features, blend_log_posteriors  # noqa: E402
from aicup.scoring import (rank_final_score, apply_temperature,    # noqa: E402
                           fit_temperature, optimise_thresholds,
                           apply_thresholds, expected_level)
from aicup.rank_nets import (RankMLP, pick_device,                 # noqa: E402
                             mlp_question_posteriors as question_posteriors)


class _TrainPlayerSplit:
    """把 split 反轉，用「訓練玩家」出題（題目層級模型的訓練資料）。"""

    def __init__(self, split):
        self.val_players = split.train_players
        self.train_players = split.val_players

    def mask(self, meta):
        return meta["player_id"].isin(self.val_players).to_numpy()


def decode_score(log_post, y, cal, ev):
    T = fit_temperature(log_post[cal], y[cal])
    th = optimise_thresholds(expected_level(apply_temperature(log_post[cal], T)), y[cal])
    return rank_final_score(
        apply_thresholds(expected_level(apply_temperature(log_post[ev], T)), th), y[ev])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--train-per-player", type=int, default=4,
                    help="每位訓練玩家出幾題")
    ap.add_argument("--game-model", default="cache/rank_mlp/models.pt")
    ap.add_argument("--eval-question-seeds", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--extra-features", nargs="*", default=[],
                    help="額外的特徵庫，沿特徵軸串接。`cache/features_opp.npy` 是**對手視角**的特徵（同盤棋、執色反轉）：伺服器把你配對給實力相近的人，所以對手的表現反映配對分級、而分級反映你的穩定實力。實測 6 組獨立題目 +0.0491±0.0232（6/6 勝，t=5.19）")
    ap.add_argument("--out", default="cache/rank_question")
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

    t0 = time.time()
    qtr = build_rank_questions(meta, _TrainPlayerSplit(split),
                               n_per_player=args.train_per_player, seed=7)
    Xq = build_question_features(X, ok, qtr.game_rows)
    ytr = qtr.true_rank_id
    print(f"訓練題 {len(ytr):,}，聚合特徵 {Xq.shape[1]} 維 "
          f"({(time.time() - t0) / 60:.1f} 分)", flush=True)

    scaler = FeatureScaler().fit(Xq)
    Xt = torch.from_numpy(scaler.transform(Xq))
    yt = torch.from_numpy(ytr).long()
    soft = SoftLabelCrossEntropy(10, decay=1.0)
    corn = CornLoss(10)

    nets = []
    for sd in range(args.seeds):
        torch.manual_seed(sd)
        net = RankMLP(Xq.shape[1], hidden=args.hidden).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                  total_steps=args.steps, pct_start=0.1)
        t0 = time.time()
        net.train()
        for _ in range(args.steps):
            i = torch.randint(0, len(Xt), (args.batch,))
            lc, lo = net(Xt[i].to(device))
            yb = yt[i].to(device)
            loss = soft(lc, yb) + 0.5 * corn(lo, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sch.step()
        net.eval(); nets.append(net)
        print(f"  seed {sd} loss {loss.item():.4f} ({(time.time() - t0) / 60:.1f} 分)",
              flush=True)

    # 單局模型（對照與融合用）
    ck = torch.load(args.game_model, map_location=device, weights_only=False)
    gsc = FeatureScaler().load_state_dict(ck["scaler"])
    gnets = []
    for sd in ck["models"]:
        n = RankMLP(len(gsc.mu), hidden=ck["args"]["hidden"])
        n.load_state_dict(sd); n.eval().to(device)
        gnets.append(n)

    @torch.no_grad()
    def qpost(Xv):
        A = torch.from_numpy(scaler.transform(Xv)).to(device)
        ps = []
        for n in nets:
            lc, lo = n(A)
            p = F.softmax(lc, dim=-1).cpu().numpy()
            pc = corn_to_proba(lo).cpu().numpy()
            pc = pc / np.maximum(pc.sum(axis=1, keepdims=True), 1e-12)
            ps.append(0.5 * p + 0.5 * pc)
        return np.mean(ps, axis=0)

    WEIGHTS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    res = {w: [] for w in WEIGHTS}
    for qs in range(args.eval_question_seeds):
        q = build_rank_questions(meta, split, n_per_player=3, seed=qs)
        y = q.true_rank_id
        lg = np.log(np.clip(question_posteriors(gnets, gsc, X, ok, q.game_rows, device),
                            1e-12, None))
        lq = np.log(np.clip(qpost(build_question_features(X, ok, q.game_rows)), 1e-12, None))
        rng = np.random.default_rng(qs)
        perm = rng.permutation(len(y))
        cal, ev = perm[:len(y) // 2], perm[len(y) // 2:]
        for w in WEIGHTS:
            res[w].append(decode_score(blend_log_posteriors(lg, lq, w), y, cal, ev))
        print(f"  題目 seed {qs}: 單局 {res[0.0][-1]:.4f} / 題目層級 {res[1.0][-1]:.4f}"
              f" / 融合0.3 {res[0.3][-1]:.4f}", flush=True)

    print(f"\n{'題目層級權重 w':<16} {'分數':>10}", flush=True)
    for w in WEIGHTS:
        a = np.array(res[w])
        tag = "  ← 只有單局" if w == 0 else ("  ← 只有題目層級" if w == 1 else "")
        print(f"{w:<16.1f} {a.mean():>7.4f} ±{a.std():.4f}{tag}", flush=True)
    best = max(WEIGHTS, key=lambda w: np.mean(res[w]))
    print(f"\n最佳 w={best}，分數 {np.mean(res[best]):.4f} ±{np.std(res[best]):.4f}"
          f"（官方 baseline 0.4631）", flush=True)

    torch.save({"models": [n.state_dict() for n in nets],
                "scaler": scaler.state_dict(), "blend_weight": float(best),
                "args": vars(args)}, os.path.join(args.out, "models.pt"))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({str(w): {"mean": float(np.mean(res[w])), "std": float(np.std(res[w]))}
                   for w in WEIGHTS} | {"best_weight": float(best)}, f, indent=2)
    print(f"已寫出 {args.out}/models.pt", flush=True)

    from aicup.experiment_log import log_experiment
    log_experiment("rank", "question_level_blend", vars(args),
                   {"official": float(np.mean(res[best])), "best_weight": float(best)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
