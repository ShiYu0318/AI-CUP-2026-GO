"""競賽二：手工特徵上的 MLP 分類器（多 seed 集成）。

實測勝過 GBDT：0.4351 -> 0.4521（168 維）／**0.4663（589 維，已超過官方 baseline 0.4631）**。
融合 GBDT 反而更差，所以預設不融合。

損失直接對齊評分函數：
  - 指數 soft label：相鄰級的權重恰為 exp(-1)，與官方計分完全一致
  - CORN ordinal head：保證 P(y>k) 單調，避免「>3D 但不 >2D」的矛盾
推論用 logmean 聚合 + 溫度校準 + 最佳化切點解碼。

用法：
    python3 scripts/train_rank_mlp.py --seeds 3 --steps 8000
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup.data import load_metadata, make_player_split          # noqa: E402
from aicup.featstore import open_stores          # noqa: E402
from aicup.benchmark import build_rank_questions                 # noqa: E402
from aicup.feature_metric import FeatureScaler                   # noqa: E402
from aicup.losses import (SoftLabelCrossEntropy, CornLoss,       # noqa: E402
                          PairwiseRankLoss, corn_to_proba)
from aicup.scoring import (rank_final_score, apply_temperature,  # noqa: E402
                           fit_temperature, optimise_thresholds,
                           apply_thresholds, expected_level, decode_expected_score)


class RankMLP(nn.Module):
    """兩個 head：10 類 softmax（搭配指數 soft label）與 CORN 的 9 個條件輸出。"""

    def __init__(self, d_in: int, hidden: int = 768, n_levels: int = 10,
                 dropout: float = 0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU())
        self.cls = nn.Linear(hidden // 2, n_levels)
        self.corn = nn.Linear(hidden // 2, n_levels - 1)

    def forward(self, x):
        z = self.body(x)
        return self.cls(z), self.corn(z)


def pick_device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def question_posteriors(nets, scaler, X, ok, game_rows, device, batch: int = 4096):
    """逐題：多局後驗做幾何平均（實測 logmean 勝過算術平均 +0.030）。

    ⚠️ 這個函式會暫時把模型切到 eval，結束時**還原成原本的模式**。
    早期版本結尾是無條件 `n.train()`，害 predict.py 在校準之後
    帶著 dropout 跑測試集推論：校準在 eval、預測在 train，
    期望等級的標準差從 2.16 被壓到 0.25，切點完全遷移不過去
    （端到端 0.33，內部驗證卻有 0.48）。
    """
    was_training = [n.training for n in nets]
    for n in nets:
        n.eval()
    out = []
    for rows in game_rows:
        sel = rows[ok[rows]]
        if len(sel) == 0:
            out.append(np.full(10, 0.1)); continue
        A = torch.from_numpy(scaler.transform(np.asarray(X[sel]))).to(device)
        ps = []
        for n in nets:
            lc, lo = n(A)
            p = F.softmax(lc, dim=-1).cpu().numpy()
            pc = corn_to_proba(lo).cpu().numpy()
            pc = pc / np.maximum(pc.sum(axis=1, keepdims=True), 1e-12)
            ps.append(0.5 * p + 0.5 * pc)          # softmax 與 CORN 各半
        p = np.mean(ps, axis=0)
        g = np.exp(np.log(np.clip(p, 1e-12, None)).mean(axis=0))
        out.append(g / g.sum())
    for n, t in zip(nets, was_training):
        n.train(t)
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rank-loss", type=float, default=0.0,
                    help="成對排序損失的權重（見 aicup.losses.PairwiseRankLoss）")
    ap.add_argument("--train-games", type=int, default=300000,
                    help="抽多少局訓練（0 = 全部）")
    ap.add_argument("--corn-weight", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU 執行緒數。0 = 不動（torch 在這台 10 核機器上預設只用 4）。"
                         "⚠️ 與 MPS 工作並行時不要開太大——"
                         "MPS 那邊的資料載入也要 CPU，搶光反而兩邊都慢")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--extra-features", nargs="*", default=[],
                    help="額外的特徵庫，沿特徵軸串接。`cache/features_opp.npy` 是**對手視角**的特徵（同盤棋、執色反轉）：伺服器把你配對給實力相近的人，所以對手的表現反映配對分級、而分級反映你的穩定實力。實測 6 組獨立題目 +0.0491±0.0232（6/6 勝，t=5.19）")
    ap.add_argument("--out", default="cache/rank_mlp")
    args = ap.parse_args()
    _t0 = time.time()      # 記錄總耗時，寫進 cache/experiments.csv

    if args.threads > 0:
        torch.set_num_threads(args.threads)
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
    rank_id = meta["rank_id"].to_numpy()

    rng = np.random.default_rng(0)
    pool = np.flatnonzero((~vm) & ok)
    tr = pool if not args.train_games else rng.choice(
        pool, min(args.train_games, len(pool)), replace=False)
    Xraw = np.asarray(X[tr]); y = rank_id[tr]
    scaler = FeatureScaler().fit(Xraw)
    Xt = torch.from_numpy(scaler.transform(Xraw))
    yt = torch.from_numpy(y).long()
    print(f"訓練局數 {len(tr):,}，特徵 {Xraw.shape[1]} 維", flush=True)

    soft = SoftLabelCrossEntropy(10, decay=1.0)

    prank = PairwiseRankLoss(10).to(device) if args.rank_loss > 0 else None
    corn = CornLoss(10)
    nets = []
    for sd in range(args.seeds):
        torch.manual_seed(sd)
        net = RankMLP(Xraw.shape[1], hidden=args.hidden).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                  total_steps=args.steps, pct_start=0.1)
        t0 = time.time()
        net.train()
        for _ in range(args.steps):
            i = torch.randint(0, len(Xt), (args.batch,))
            lc, lo = net(Xt[i].to(device))
            yb = yt[i].to(device)
            loss = soft(lc, yb) + args.corn_weight * corn(lo, yb)
            if prank is not None:
                loss = loss + args.rank_loss * prank(lc, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sch.step()
        nets.append(net)
        print(f"  seed {sd} loss {loss.item():.4f} ({(time.time() - t0) / 60:.1f} 分)",
              flush=True)

    q = build_rank_questions(meta, split, n_per_player=3, seed=0)
    yv = q.true_rank_id
    half = len(yv) // 2
    P = question_posteriors(nets, scaler, X, ok, q.game_rows, device)
    lo_ = np.log(np.clip(P, 1e-12, None))
    T = fit_temperature(lo_[:half], yv[:half])
    th = optimise_thresholds(expected_level(apply_temperature(lo_[:half], T)), yv[:half])

    s_exp = rank_final_score(decode_expected_score(apply_temperature(lo_[half:], T)), yv[half:])
    s_th = rank_final_score(
        apply_thresholds(expected_level(apply_temperature(lo_[half:], T)), th), yv[half:])
    print(f"\n期望解碼 {s_exp:.4f} ／ 切點解碼 {s_th:.4f}"
          f"（官方 baseline 0.4631）", flush=True)

    torch.save({"models": [n.state_dict() for n in nets],
                "scaler": scaler.state_dict(), "temperature": float(T),
                "thresholds": th.tolist(), "args": vars(args)},
               os.path.join(args.out, "models.pt"))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({"expected": s_exp, "threshold": s_th,
                   "temperature": float(T), "thresholds": th.tolist()}, f, indent=2)
    print(f"已寫出 {args.out}/models.pt", flush=True)

    from aicup.experiment_log import log_experiment
    log_experiment("rank", "mlp_ensemble", vars(args),
                   {"official": s_th, "expected_decode": s_exp})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
