"""多任務訓練腳本（棋力預測 + 玩家辨識共用 backbone）。

設計重點都寫在 docs/02，這裡只列與官方 baseline 不同的關鍵處：
  - 驗證一律用 player-disjoint 的題目層級官方分數選 checkpoint，
    不是單局 accuracy、更不是訓練 loss（官方玩家辨識 baseline 用訓練 loss 選，
    在 metric learning 上這兩者經常脫鉤）
  - 損失對齊評分函數：指數 soft label + CORN
  - 玩家 head 用 sub-center ArcFace，推論時丟棄、只留 embedding
  - 8 相位 dihedral 增強（已於 docs/02 §10.2 驗證對兩個子題都安全）

用法（單機 GPU）：
    python3 scripts/train.py --device cuda --width 128 --blocks 10 --steps 20000

本機只做流程驗證：
    python3 scripts/train.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup import N_LEVELS                                         # noqa: E402
from aicup.data import load_metadata, make_player_split            # noqa: E402
from aicup.benchmark import build_rank_questions                   # noqa: E402
from aicup.dataset import GameDataset, PKPlayerSampler             # noqa: E402
from aicup.losses import SoftLabelCrossEntropy, CornLoss, SubCenterArcFace  # noqa: E402
from aicup.nets import MultiTaskGoNet, count_parameters            # noqa: E402
from aicup.scoring import decode_expected_score, decode_argmax, rank_final_score  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_rank(model, meta, questions, device, n_snapshots: int, max_q: int = 200):
    """題目層級的官方分數——這才是選 checkpoint 的依據。"""
    from aicup.dataset import QuestionDataset, collate_questions
    ds = QuestionDataset(meta, questions, n_snapshots=n_snapshots)
    n = min(max_q, len(ds))
    loader = DataLoader([ds[i] for i in range(n)], batch_size=8,
                        collate_fn=collate_questions)
    model.eval()
    probs, ys = [], []
    for x, mask, y in loader:
        out = model.forward_set(x.to(device), mask.to(device))
        probs.append(F.softmax(out["rank_logits"], dim=-1).cpu().numpy())
        ys.append(y.numpy())
    model.train()
    P = np.concatenate(probs); Y = np.concatenate(ys)
    return {
        "score_argmax": rank_final_score(decode_argmax(P), Y),
        "score_expected": rank_final_score(decode_expected_score(P), Y),
        "n_questions": int(len(Y)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--embed-dim", type=int, default=512)
    ap.add_argument("--snapshots", type=int, default=4)
    ap.add_argument("--p", type=int, default=16, help="每個 batch 的玩家數")
    ap.add_argument("--k", type=int, default=4, help="每位玩家的局數")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arcface-weight", type=float, default=1.0)
    ap.add_argument("--corn-weight", type=float, default=0.5)
    ap.add_argument("--out", default="cache/checkpoints")
    ap.add_argument("--smoke", action="store_true", help="極小設定，只驗證流程能跑通")
    args = ap.parse_args()

    if args.smoke:
        args.width, args.blocks, args.embed_dim = 16, 1, 32
        args.p, args.k, args.steps, args.eval_every = 4, 2, 4, 2
        args.snapshots, args.workers = 1, 0

    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"device = {device}")

    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    train_rows = np.flatnonzero(~split.mask(meta))
    if args.smoke:
        train_rows = train_rows[:3000]

    players = sorted(set(meta.iloc[train_rows]["player_id"]))
    p2l = {p: i for i, p in enumerate(players)}
    counts = meta.iloc[train_rows]["player_id"].value_counts()
    class_counts = torch.tensor([counts[p] for p in players], dtype=torch.float)
    print(f"訓練局數 {len(train_rows):,}，訓練身分 {len(players):,}")

    ds = GameDataset(meta, train_rows, n_snapshots=args.snapshots, augment=True,
                     player_to_label=p2l)
    sampler = PKPlayerSampler(meta, train_rows, p=args.p, k=args.k,
                              n_batches=args.steps, seed=0)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers)

    model = MultiTaskGoNet(in_channels=34, width=args.width, blocks=args.blocks,
                           embed_dim=args.embed_dim, n_levels=N_LEVELS).to(device)
    arc = SubCenterArcFace(args.embed_dim, len(players), k=3,
                           class_counts=class_counts).to(device)
    print(f"參數量 backbone {count_parameters(model):,} / arcface {count_parameters(arc):,}")

    opt = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": args.lr},
         {"params": arc.parameters(), "lr": args.lr * 10}],   # head 用較大 lr
        weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 10],
                                                total_steps=args.steps, pct_start=0.1)
    soft_ce = SoftLabelCrossEntropy(N_LEVELS, decay=1.0)
    corn = CornLoss(N_LEVELS)

    questions = build_rank_questions(meta, split, n_per_player=1, seed=0,
                                     max_questions=200 if not args.smoke else 8)
    history, best = [], -1.0
    t0 = time.time()
    model.train()

    for step, (x, y_rank, y_player) in enumerate(loader, start=1):
        # 一局多個 snapshot -> 攤平進 batch，標籤跟著複製
        B, S = x.shape[0], x.shape[1]
        x = x.flatten(0, 1).to(device)
        y_rank = y_rank.repeat_interleave(S).to(device)
        y_player = y_player.repeat_interleave(S).to(device)

        out = model(x)
        loss_rank = soft_ce(out["rank_logits"], y_rank)
        loss_corn = corn(out["corn_logits"], y_rank)
        logits = arc(out["embedding"], y_player)
        loss_arc = F.cross_entropy(logits, y_player)
        loss = loss_rank + args.corn_weight * loss_corn + args.arcface_weight * loss_arc

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        if step % max(1, args.eval_every) == 0 or step == args.steps:
            ev = evaluate_rank(model, meta, questions, device, args.snapshots,
                               max_q=8 if args.smoke else 200)
            rec = {"step": step, "loss": loss.item(), "loss_rank": loss_rank.item(),
                   "loss_corn": loss_corn.item(), "loss_arc": loss_arc.item(),
                   "minutes": (time.time() - t0) / 60, **ev}
            history.append(rec)
            print(f"  step {step:>6} loss {rec['loss']:.4f}"
                  f" (rank {rec['loss_rank']:.3f} corn {rec['loss_corn']:.3f}"
                  f" arc {rec['loss_arc']:.3f})"
                  f" | 題目分數 期望解碼 {ev['score_expected']:.4f}"
                  f" argmax {ev['score_argmax']:.4f}"
                  f" | {rec['minutes']:.1f} 分", flush=True)
            if ev["score_expected"] > best:
                best = ev["score_expected"]
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "score": best},
                           os.path.join(args.out, "best.pt"))

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n完成。最佳題目層級分數（期望解碼）{best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
