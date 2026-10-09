"""訓練以棋力為條件的走法模型（`aicup.skillpolicy`）。

    python3 scripts/train_skill_policy.py --steps 60000 --out cache/skill_policy

評估：在 KataGo HumanSL 探針的同一批 1000 盤（驗證集 50 位玩家 x 20 盤）上，
算「Σlog p 的期望等級」與真實等級的單盤 Spearman，直接和 HumanSL（24 手 0.677）、
板面 CNN（0.753）、單局 MLP（0.606）比較。checkpoint 照全手數的 Spearman 挑。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

from aicup.data import load_metadata, make_player_split
from aicup.dataset import SgfCache
from aicup.planes import planes_to_float
from aicup.rank_nets import pick_device
from aicup.scoring import expected_level
from aicup.skillpolicy import N_LEVELS, SkillPolicyNet, game_loglik, own_turns, position_samples

PROBE = "artifacts/probe_humansl2_compare.npz"


class PositionDataset(Dataset):
    """一個項目 = 一盤棋的 k 個隨機我方局面（不足 k 手時重複抽）。"""

    def __init__(self, meta, rows, k: int, n_items: int, seed: int = 0):
        self.meta, self.rows, self.k, self.n = meta, np.asarray(rows), k, n_items
        self.seed = seed
        self.cache = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.cache is None:
            self.cache = SgfCache()
        rng = np.random.default_rng((self.seed, i))
        for _ in range(10):
            r = int(self.rows[rng.integers(len(self.rows))])
            rec = self.meta.iloc[r]
            sgf = self.cache.get(int(rec["file_idx"]), int(rec["row"]))
            t, tg = own_turns(sgf, rec["color"])
            if len(t) < 4:
                continue
            j = rng.choice(len(t), self.k, replace=len(t) < self.k)
            planes, tgt = position_samples(sgf, rec["color"], [t[x] for x in sorted(j)],
                                           [tg[x] for x in sorted(j)])
            if planes is None:
                continue
            if rng.random() < 0.5:                       # 左右翻轉：盤面與目標點一起翻
                planes = planes[..., ::-1].copy()
                tgt = (tgt // 19) * 19 + (18 - tgt % 19)
            return (torch.from_numpy(planes_to_float(planes)), torch.from_numpy(tgt),
                    torch.full((len(tgt),), int(rec["rank_id"]), dtype=torch.long))
        raise RuntimeError("連續 10 盤都取不到局面")


def collate(batch):
    x, t, r = zip(*batch)
    return torch.cat(x), torch.cat(t), torch.cat(r)


def evaluate(net, meta, probe_rows, y, device, cache, max_moves: int):
    games = [(cache.get(int(meta.at[i, "file_idx"]), int(meta.at[i, "row"])), meta.at[i, "color"])
             for i in probe_rows]
    G = game_loglik(net, games, device, max_moves=max_moves)
    L = G[:, 2:] / np.maximum(G[:, :1], 1)                 # 每手平均
    P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    ev = expected_level(np.log(P))
    acc = float((L.argmax(1) == y).mean())
    return float(spearmanr(ev, y).statistic), acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--games", type=int, default=32, help="每批幾盤")
    ap.add_argument("--k", type=int, default=8, help="每盤取幾個局面")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="cache/skill_policy")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    train_rows = np.flatnonzero(~split.mask(meta))
    probe = np.load(PROBE, allow_pickle=True)
    probe_rows = probe["rows"]
    probe_y = meta.loc[probe_rows, "rank_id"].to_numpy()
    print(f"device = {device}；訓練局數 {len(train_rows):,}；評估 {len(probe_rows)} 盤", flush=True)

    net = SkillPolicyNet(width=args.width, blocks=args.blocks).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    start, best, history = 0, -1.0, []
    last = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.last_epoch = ck["step"]; start, best, history = ck["step"], ck["best"], ck["history"]
        print(f"續訓：第 {start:,} 步，最佳 {best:.4f}", flush=True)

    ds = PositionDataset(meta, train_rows, args.k, (args.steps - start) * args.games, seed=start)
    ctx = torch.multiprocessing.get_context("fork") if args.workers else None
    loader = DataLoader(ds, batch_size=args.games, num_workers=args.workers, collate_fn=collate,
                        multiprocessing_context=ctx, persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers else None)
    cache = SgfCache()
    t0 = time.time(); net.train(); run_loss = run_acc = 0.0
    for step, (x, tgt, rank) in enumerate(loader, start=start + 1):
        x, tgt, rank = x.to(device), tgt.to(device), rank.to(device)
        lp = net(x)                                                   # (B, L, 361)
        lp_r = lp[torch.arange(len(rank), device=device), rank]       # 該盤真實棋力那個 head
        loss = F.nll_loss(lp_r, tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
        run_loss += loss.item(); run_acc += (lp_r.argmax(1) == tgt).float().mean().item()
        out_of_time = args.max_hours and (time.time() - t0) / 3600 >= args.max_hours
        if step % args.eval_every == 0 or step == args.steps or out_of_time:
            sp24, a24 = evaluate(net, meta, probe_rows, probe_y, device, cache, max_moves=24)
            spall, aall = evaluate(net, meta, probe_rows, probe_y, device, cache, max_moves=0)
            net.train()
            rec = {"step": step, "loss": run_loss / args.eval_every, "move_acc": run_acc / args.eval_every,
                   "spearman24": sp24, "spearman_all": spall, "acc_all": aall,
                   "minutes": (time.time() - t0) / 60}
            history.append(rec); run_loss = run_acc = 0.0
            print(f"  step {step:>6} loss {rec['loss']:.3f} 走法命中 {rec['move_acc']:.3f} | "
                  f"單盤 Spearman 24手 {sp24:.4f} 全手數 {spall:.4f}（峰值命中 {aall:.3f}） | "
                  f"{rec['minutes']:.0f} 分", flush=True)
            if spall > best:
                best = spall
                torch.save({"model": net.state_dict(), "config": net.config, "score": best,
                            "step": step, "args": vars(args)}, os.path.join(args.out, "best.pt"))
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                        "history": history, "config": net.config, "args": vars(args)}, last)
        if out_of_time:
            print(f"已達時間預算，於第 {step:,} 步存檔結束", flush=True)
            break
    with open(os.path.join(args.out, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"完成。最佳單盤 Spearman（全手數）{best:.4f}；對照 HumanSL 24 手 0.677、板面 CNN 0.753", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
