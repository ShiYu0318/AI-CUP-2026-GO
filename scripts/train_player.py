"""競賽一（玩家辨識）的深度 embedding 訓練。

為什麼需要獨立的腳本：這一題的目標不是分類 accuracy，而是
**在未見身分的候選庫上的檢索分數**。官方 baseline 用「訓練 loss」選 checkpoint，
而 metric learning 的訓練 loss 與檢索能力經常脫鉤——那是它只有 0.2845 的主因之一。

本腳本每 --eval-every 步就真的建一次候選庫、算一次官方公式分數，並據此選 checkpoint。

架構（對齊 docs/02 §4.3 與 NeurIPS 2021 behavioral stylometry）：
    盤面 (34,19,19) -> GoTrunk(含 KataGo global pooling bias) -> GeM -> BN-Neck -> embedding
    訓練：Sub-center ArcFace(6,405 身分, K=3, dynamic margin) + batch-hard triplet
    取樣：P 位玩家 x K 局（batch-hard mining 才有素材）
    推論：丟棄分類頭，只用 embedding；單局標準化 -> L2 -> 平均 -> 再 L2

⚠️ 推論時**必須丟棄 ArcFace 分類頭**——候選庫是未見身分，分類頭沒有對應的輸出單元。
   （docs/02 §4.6）

用法：
    python3 scripts/train_player.py --device cuda --steps 20000 --resume
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

from aicup.data import load_metadata, make_player_split        # noqa: E402
from aicup.benchmark import build_player_pool                  # noqa: E402
from aicup.dataset import GameDataset, PKPlayerSampler         # noqa: E402
from aicup.losses import SubCenterArcFace, batch_hard_triplet_loss  # noqa: E402
from aicup.nets import MultiTaskGoNet, count_parameters        # noqa: E402
from aicup.planes import game_planes                           # noqa: E402
from aicup.scoring import retrieval_report                     # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed_games(model, meta, cache, rows, device, n_snapshots: int, batch: int = 64):
    """把一組棋局轉成 L2 正規化的 embedding。"""
    model.eval()
    out, buf = [], []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).float().to(device)
        e = model(x)["embedding"]
        out.append(F.normalize(e, dim=1).cpu())
        buf.clear()

    for r in rows:
        rec = meta.iloc[int(r)]
        t = game_planes(cache.get(int(rec["file_idx"]), int(rec["row"])),
                        rec["color"], n_snapshots=n_snapshots)
        if t is None:
            continue
        buf.append(t.mean(axis=0))          # 一局的多個 snapshot 先平均
        if len(buf) >= batch:
            flush()
    flush()
    model.train()
    return torch.cat(out) if out else torch.zeros(0, model.trunk.embed_dim)


def template(emb: torch.Tensor) -> torch.Tensor:
    """多局 -> 玩家向量。L2 -> 平均 -> 再 L2（人臉 template 的標準作法）。

    實測（docs/02 §10.8）：這個順序相對「先平均再標準化」值 +0.26 分，
    比任何特徵工程都重要。
    """
    if len(emb) == 0:
        return torch.zeros(1)
    v = emb.mean(0)
    return v / max(float(v.norm()), 1e-8)


@torch.no_grad()
def evaluate_retrieval(model, meta, cache, split, device, *, n_snapshots: int,
                       n_candidates: int, n_gallery: int, seed: int,
                       composition: str = "uniform") -> dict:
    """完整複製官方協定：候選庫 N 人、每人 n_gallery 局；query 5~20 局；交 Top-5。"""
    pool = build_player_pool(meta, split, n_candidates=n_candidates,
                             n_gallery=n_gallery, composition=composition, seed=seed)
    G, Q = [], []
    for p in pool.players:
        g = embed_games(model, meta, cache, pool.gallery_rows[p], device, n_snapshots)
        q = embed_games(model, meta, cache, pool.query_rows[p], device, n_snapshots)
        if len(g) < 5 or len(q) < 1:
            continue
        G.append(template(g)); Q.append(template(q))
    if len(G) < 10:
        return {"official_score": 0.0, "top1": 0.0, "top5": 0.0, "n_query": len(G)}
    G = torch.stack(G).numpy(); Q = torch.stack(Q).numpy()
    rep = retrieval_report(Q @ G.T)
    rep["n_candidates"] = len(G)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--embed-dim", type=int, default=512)
    ap.add_argument("--snapshots", type=int, default=4)
    ap.add_argument("--p", type=int, default=32, help="每個 batch 的玩家數")
    ap.add_argument("--k", type=int, default=4, help="每位玩家的局數")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arcface-k", type=int, default=3, help="sub-center 數")
    ap.add_argument("--arcface-scale", type=float, default=30.0)
    ap.add_argument("--triplet-weight", type=float, default=0.3)
    ap.add_argument("--rank-weight", type=float, default=0.2,
                    help="棋力 head 的權重。棋力是玩家辨識的主要混淆因子，"
                         "顯式建模能讓 embedding 更乾淨（NeurIPS 2021）")
    ap.add_argument("--min-games", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-candidates", type=int, default=200,
                    help="訓練中監控用的候選庫人數。最終請用 400 完整評估")
    ap.add_argument("--eval-gallery", type=int, default=40,
                    help="訓練中監控用的每人 gallery 局數。官方是 100")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pool", default="set_transformer", choices=["attention", "set_transformer"])
    ap.add_argument("--no-global-pool-bias", action="store_true")
    ap.add_argument("--out", default="cache/ckpt_player")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.width, args.blocks, args.embed_dim = 16, 1, 32
        args.p, args.k, args.steps = 4, 2, 4
        args.eval_every, args.save_every, args.snapshots, args.workers = 2, 2, 1, 0
        args.eval_candidates, args.eval_gallery = 12, 6

    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"device = {device}", flush=True)

    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    train_rows = np.flatnonzero(~split.mask(meta))
    if args.smoke:
        train_rows = train_rows[:2000]

    counts = meta.iloc[train_rows]["player_id"].value_counts()
    keep = set(counts[counts >= args.min_games].index)
    train_rows = train_rows[meta.iloc[train_rows]["player_id"].isin(keep).to_numpy()]
    players = sorted(keep)
    p2l = {p: i for i, p in enumerate(players)}
    class_counts = torch.tensor([counts[p] for p in players], dtype=torch.float)
    print(f"訓練局數 {len(train_rows):,}，訓練身分 {len(players):,}"
          f"（每人局數 {int(class_counts.min())}~{int(class_counts.max())}）", flush=True)

    ds = GameDataset(meta, train_rows, n_snapshots=args.snapshots, augment=True,
                     player_to_label=p2l)
    sampler = PKPlayerSampler(meta, train_rows, p=args.p, k=args.k,
                              n_batches=args.steps, seed=0)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers,
                        persistent_workers=args.workers > 0)

    model = MultiTaskGoNet(in_channels=34, width=args.width, blocks=args.blocks,
                           embed_dim=args.embed_dim, pool=args.pool,
                           global_pool_bias=not args.no_global_pool_bias).to(device)
    arc = SubCenterArcFace(args.embed_dim, len(players), k=args.arcface_k,
                           scale=args.arcface_scale, class_counts=class_counts).to(device)
    print(f"參數量 backbone {count_parameters(model):,} / arcface {count_parameters(arc):,}",
          flush=True)

    opt = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": args.lr},
         {"params": arc.parameters(), "lr": args.lr * 10}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.lr * 10], total_steps=args.steps, pct_start=0.1)

    from aicup.dataset import SgfCache
    cache = SgfCache()
    history, best, start_step = [], -1.0, 0
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); arc.load_state_dict(ck["arc"])
        opt.load_state_dict(ck["opt"])
        sched.last_epoch = int(ck["sched"]["last_epoch"])
        sched._step_count = int(ck["sched"].get("_step_count", sched.last_epoch + 1))
        history, best, start_step = ck["history"], ck["best"], ck["step"]
        print(f"從 {last_path} 續訓：已完成 {start_step:,} 步，最佳檢索分數 {best:.4f}",
              flush=True)
        if start_step >= args.steps:
            print("已達到指定步數。"); return 0
        sampler.n_batches = args.steps - start_step

    t0 = time.time()
    model.train()
    for step, (x, y_rank, y_player) in enumerate(loader, start=start_step + 1):
        B, S = x.shape[0], x.shape[1]
        x = x.flatten(0, 1).to(device)
        y_rank = y_rank.repeat_interleave(S).to(device)
        y_player = y_player.repeat_interleave(S).to(device)

        out = model(x)
        emb = out["embedding"]
        loss_arc = F.cross_entropy(arc(emb, y_player), y_player)
        loss_tri = batch_hard_triplet_loss(emb, y_player)
        loss_rank = F.cross_entropy(out["rank_logits"], y_rank)
        loss = loss_arc + args.triplet_weight * loss_tri + args.rank_weight * loss_rank

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()

        if step % max(1, args.eval_every) == 0 or step == args.steps:
            rep = evaluate_retrieval(model, meta, cache, split, device,
                                     n_snapshots=args.snapshots,
                                     n_candidates=args.eval_candidates,
                                     n_gallery=args.eval_gallery, seed=0)
            rec = {"step": step, "loss": loss.item(), "loss_arc": loss_arc.item(),
                   "loss_tri": loss_tri.item(), "loss_rank": loss_rank.item(),
                   "minutes": (time.time() - t0) / 60, **rep}
            history.append(rec)
            print(f"  step {step:>6} loss {rec['loss']:.3f}"
                  f" (arc {rec['loss_arc']:.3f} tri {rec['loss_tri']:.3f}"
                  f" rank {rec['loss_rank']:.3f})"
                  f" | 檢索 {rep['official_score']:.4f}"
                  f" Top-1 {rep['top1']:.4f} ({rep['n_candidates']} 人)"
                  f" | {rec['minutes']:.1f} 分", flush=True)
            if rep["official_score"] > best:
                best = rep["official_score"]
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "score": best}, os.path.join(args.out, "best.pt"))

        out_of_time = args.max_hours > 0 and (time.time() - t0) / 3600 >= args.max_hours
        if step % max(1, args.save_every) == 0 or step == args.steps or out_of_time:
            torch.save({"model": model.state_dict(), "arc": arc.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": best, "history": history,
                        "args": vars(args)}, last_path)
        if out_of_time:
            print(f"\n已達時間預算 {args.max_hours} 小時，於第 {step:,} 步存檔結束。"
                  f"加 --resume 可接回（目標 {args.steps:,} 步）。", flush=True)
            break

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n完成。最佳檢索分數 {best:.4f}"
          f"（監控用 {args.eval_candidates} 人候選庫；最終請用 400 人完整評估）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
