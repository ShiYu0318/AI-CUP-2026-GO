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
def evaluate_rank(model, meta, questions, device, n_snapshots: int, max_q: int = 800):
    """題目層級的官方分數——這才是選 checkpoint 的依據。

    ⚠️ 題數要夠多。官方測試集是 400 題，二項分布的標準誤約
    sqrt(0.25/400) ≈ 0.025；用 200 題評估時標準誤高達 ~0.035，
    兩個設定差 0.04 以內根本分不出來。實測就踩過這個坑：
    兩次評估得到 0.2323 與 0.1912，看起來差很多，其實在雜訊範圍內。
    """
    from aicup.dataset import QuestionDataset, collate_questions
    ds = QuestionDataset(meta, questions, n_snapshots=n_snapshots)
    n = min(max_q, len(ds))
    loader = DataLoader([ds[i] for i in range(n)], batch_size=8,
                        collate_fn=collate_questions)
    was_training = model.training      # 還原原本的模式，不要無條件切回 train
    model.eval()
    probs, ys = [], []
    with torch.no_grad():              # 少了這個會在 GPU 上白白累積計算圖，800 題足以 OOM
        for x, mask, y in loader:
            out = model.forward_set(x.to(device), mask.to(device))
            probs.append(F.softmax(out["rank_logits"], dim=-1).cpu().numpy())
            ys.append(y.numpy())
    model.train(was_training)
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
    ap.add_argument("--eval-questions", type=int, default=800,
                    help="評估題數。太少會讓 checkpoint 選擇被雜訊主導（見 evaluate_rank）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arcface-weight", type=float, default=1.0)
    ap.add_argument("--corn-weight", type=float, default=0.5)
    ap.add_argument("--balance-losses", action="store_true",
                    help="用各損失的滑動平均把量級正規化，避免大量級的損失壟斷梯度。"
                         "實測未加時 arcface 損失量級 13~17、rank 只有 2.3，"
                         "rank head 在 3,000 步內幾乎沒學到東西")
    ap.add_argument("--balance-momentum", type=float, default=0.99)
    ap.add_argument("--quality-weight", type=float, default=0.0,
                    help="棋形品質輔助 head 的權重。>0 時會從 cache/features.npy 取"
                         "愚形率等連續量當回歸目標，提供比 10 類棋力更密集的監督訊號")
    ap.add_argument("--pool", default="set_transformer",
                    choices=["attention", "set_transformer"],
                    help="集合聚合方式。set_transformer 讓多局互相參照"
                         "（ICLR 2025 圍棋棋力 SOTA 的做法）")
    ap.add_argument("--transformer-every", type=int, default=0,
                    help="每 N 個 residual block 插一個盤面 self-attention（ResTNet）。\n"
                         "純卷積抓不到征子/環形等長程 pattern，而那正是區分棋力的關鍵。\n"
                         "0 = 停用；建議從 3 開始。成本約 +1.5 倍 forward 時間")
    ap.add_argument("--out", default="cache/checkpoints")
    ap.add_argument("--resume", action="store_true",
                    help="從 --out 的 last.pt 續訓。Colab / Kaggle 的 session 會斷線，"
                         "長訓練一定要開這個")
    ap.add_argument("--save-every", type=int, default=500,
                    help="每 N 步存一次 last.pt（含 optimizer 與 scheduler 狀態）")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="時間預算（小時）。到點就存檔並正常結束，之後用 --resume 接回。"
                         "Kaggle Notebooks 的 12 小時是硬上限、到點直接砍掉，"
                         "設成 11.5 可以確保最後的 checkpoint 一定寫得出去")
    ap.add_argument("--smoke", action="store_true", help="極小設定，只驗證流程能跑通")
    args = ap.parse_args()

    if args.smoke:
        args.width, args.blocks, args.embed_dim = 16, 1, 32
        args.p, args.k, args.steps, args.eval_every = 4, 2, 4, 2
        args.eval_questions = 8
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

    quality_store = quality_cols = None
    if args.quality_weight > 0:
        from aicup.features import FEATURE_NAMES
        names = ["empty_triangle", "abandon_atari", "atari_given_rate",
                 "wasted_fill", "contact_early", "cap_per_move"]
        quality_cols = [FEATURE_NAMES.index(n) for n in names]
        quality_store = np.load("cache/features.npy", mmap_mode="r")
        print(f"棋形品質輔助目標 {len(quality_cols)} 維：{', '.join(names)}")

    ds = GameDataset(meta, train_rows, n_snapshots=args.snapshots, augment=True,
                     player_to_label=p2l, quality_store=quality_store,
                     quality_cols=quality_cols)
    sampler = PKPlayerSampler(meta, train_rows, p=args.p, k=args.k,
                              n_batches=args.steps, seed=0)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers)

    model = MultiTaskGoNet(in_channels=34, width=args.width, blocks=args.blocks,
                           embed_dim=args.embed_dim, n_levels=N_LEVELS,
                           n_quality=len(quality_cols or []),
                           pool=args.pool,
                           transformer_every=args.transformer_every).to(device)
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

    # 各損失的量級差很多（arcface 13~17 vs rank 2.3），直接相加會讓梯度被
    # 大量級的那一項主導。這裡用滑動平均把每項除以自身尺度，讓三個任務等權。
    scale = {"rank": None, "corn": None, "arc": None}

    def _balanced(name, value):
        if not args.balance_losses:
            return value
        v = float(value.detach())
        scale[name] = v if scale[name] is None else (
            args.balance_momentum * scale[name] + (1 - args.balance_momentum) * v)
        return value / max(scale[name], 1e-8)

    questions = build_rank_questions(meta, split, n_per_player=1, seed=0,
                                     max_questions=8 if args.smoke else args.eval_questions)
    history, best, start_step = [], -1.0, 0

    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        arc.load_state_dict(ck["arc"])
        opt.load_state_dict(ck["opt"])
        # ⚠️ 不能直接 load_state_dict：OneCycleLR 的 state 內含建構時的 total_steps，
        # 還原後會把上限鎖回上一次的步數，續訓第一步就丟
        # "Tried to step N times. The specified number of total steps is M"。
        # 只還原「已走到第幾步」，總步數維持這次 --steps 指定的值。
        sched.last_epoch = int(ck["sched"]["last_epoch"])
        sched._step_count = int(ck["sched"].get("_step_count", sched.last_epoch + 1))
        history, best, start_step = ck["history"], ck["best"], ck["step"]
        print(f"從 {last_path} 續訓：已完成 {start_step:,} 步，最佳分數 {best:.4f}")
        if start_step >= args.steps:
            print("已達到指定步數，無需續訓。")
            return 0
        sampler.n_batches = args.steps - start_step

    t0 = time.time()
    model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        y_quality = None
        if len(batch) == 4:
            x, y_rank, y_player, y_quality = batch
        else:
            x, y_rank, y_player = batch
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
        loss = (_balanced("rank", loss_rank)
                + args.corn_weight * _balanced("corn", loss_corn)
                + args.arcface_weight * _balanced("arc", loss_arc))
        if y_quality is not None:
            tgt = y_quality.repeat_interleave(S, dim=0).to(device)
            loss_qual = F.smooth_l1_loss(out["quality"], tgt)
            loss = loss + args.quality_weight * loss_qual

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        if step % max(1, args.eval_every) == 0 or step == args.steps:
            ev = evaluate_rank(model, meta, questions, device, args.snapshots,
                               max_q=8 if args.smoke else args.eval_questions)
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

        out_of_time = (args.max_hours > 0
                       and (time.time() - t0) / 3600 >= args.max_hours)
        if step % max(1, args.save_every) == 0 or step == args.steps or out_of_time:
            # 續訓用的完整狀態。斷線或用完時間預算後 --resume 可從這裡接回去。
            torch.save({"model": model.state_dict(), "arc": arc.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": best, "history": history,
                        "args": vars(args)}, last_path)
        if out_of_time:
            print(f"\n已達時間預算 {args.max_hours} 小時，於第 {step:,} 步存檔結束。"
                  f"\n用相同指令加 --resume 可接回（目標 {args.steps:,} 步）。", flush=True)
            break

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n完成。最佳題目層級分數（期望解碼）{best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
