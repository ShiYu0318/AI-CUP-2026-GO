"""競賽二：逐手序列模型（dilated 1D-CNN）。

589 維手工特徵是**整局的統計摘要**——愚形率是 0.13，但愚形出現在
第 30 手還是第 150 手完全看不出來。這支腳本訓練一個讀**序列**的模型，
補上那塊被丟掉的時序資訊。

小規模實測（6 組獨立題目抽樣、玩家不重疊）：

    589 維手工特徵 MLP      0.3631 ±0.0149
    逐手序列 1D-CNN 單獨    0.3831 ±0.0094   +0.0200（6/6 勝，t=3.64）
    兩者融合 w=0.5          0.4146 ±0.0146   +0.0515（6/6 勝，t=5.74）

組內殘差相關 +0.530 —— 互補性明確。

用法：
    python3 scripts/build_sequence_store.py          # 先建序列庫（2.3 分）
    python3 scripts/train_rank_seq.py --seeds 3
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup.data import load_metadata, make_player_split           # noqa: E402
from aicup.benchmark import build_rank_questions                  # noqa: E402
from aicup.losses import SoftLabelCrossEntropy, CornLoss, corn_to_proba  # noqa: E402
from aicup.moveseq import MAX_MOVES, N_MOVE_FEATURES              # noqa: E402
from aicup.scoring import (fit_temperature, apply_temperature,    # noqa: E402
                           expected_level, optimise_thresholds,
                           apply_thresholds, rank_final_score)


class SeqStore:
    """多份序列庫沿**特徵軸**串接（每一手的向量變長），取列時才合併。

    與 `aicup.featstore.MultiStore` 同樣的理由：兩份各 3.84 GB，
    實體合併既浪費磁碟、又讓「只用我方」的實驗得重建。
    """

    def __init__(self, parts):
        n = len(parts[0])
        for i, p in enumerate(parts):
            if len(p) != n:
                raise ValueError(f"第 {i} 份序列庫有 {len(p):,} 列，與第 0 份的 {n:,} 列不符")
        self.parts = parts
        self._shape = (n, parts[0].shape[1], sum(p.shape[2] for p in parts))

    def __len__(self):
        return self._shape[0]

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, rows):
        if len(self.parts) == 1:
            return self.parts[0][rows]
        return np.concatenate([np.asarray(p[rows]) for p in self.parts], axis=-1)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SeqRankNet(nn.Module):
    """dilated 1D 卷積讀逐手序列。

    dilation 1/2/4/8 讓感受野涵蓋整個序列（160 手），
    同時保留局部的手順關係。avg 與 max pooling 並用：
    avg 抓「整體傾向」，max 抓「有沒有出現過某種糟糕的手」——
    後者對棋力判斷特別重要（一次致命失誤就暴露等級）。
    """

    def __init__(self, n_feat: int = N_MOVE_FEATURES, width: int = 128,
                 n_levels: int = 10, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Conv1d(n_feat, width, 5, padding=2)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, 5, padding=2 * r, dilation=r),
                nn.GroupNorm(8, width), nn.GELU(),
                nn.Conv1d(width, width, 1),
                nn.GroupNorm(8, width), nn.GELU())
            for r in (1, 2, 4, 8)])
        self.head = nn.Sequential(nn.Linear(2 * width, 256), nn.GELU(),
                                  nn.Dropout(dropout))
        self.cls = nn.Linear(256, n_levels)
        self.orn = nn.Linear(256, n_levels - 1)

    def forward(self, x, mask):
        h = self.stem(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        m = mask.unsqueeze(1).float()
        h = h * m
        avg = h.sum(-1) / m.sum(-1).clamp(min=1)
        mx = (h + (m - 1) * 1e4).max(-1).values
        z = self.head(torch.cat([avg, mx], dim=1))
        return self.cls(z), self.orn(z)


def question_posteriors(nets, S, ok, lens, game_rows, device, batch: int = 256):
    """逐題：多局後驗做幾何平均（與 train_rank_mlp 的聚合方式一致）。

    ⚠️ 結束時還原原本的 train/eval 模式。無條件切回 train 會讓後續推論
    帶著 dropout 跑（見 tests/test_eval_mode.py）。
    """
    was = [n.training for n in nets]
    for n in nets:
        n.eval()
    out = []
    with torch.no_grad():
        for rows in game_rows:
            sel = rows[ok[rows]]
            if len(sel) == 0:
                out.append(np.full(10, 0.1)); continue
            x = torch.from_numpy(np.asarray(S[sel], dtype=np.float32)).to(device)
            mk = torch.from_numpy(
                np.arange(MAX_MOVES)[None, :] < lens[sel][:, None]).to(device)
            ps = []
            for n in nets:
                lc, lo = n(x, mk)
                pc = corn_to_proba(lo).cpu().numpy()
                ps.append(0.5 * torch.softmax(lc, -1).cpu().numpy()
                          + 0.5 * pc / np.maximum(pc.sum(1, keepdims=True), 1e-12))
            p = np.mean(ps, axis=0)
            g = np.exp(np.log(np.clip(p, 1e-12, None)).mean(axis=0))
            out.append(g / g.sum())
    for n, t in zip(nets, was):
        n.train(t)
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-games", type=int, default=400000,
                    help="抽多少局訓練（0 = 全部）。序列張量比特徵大，記憶體吃緊時調小")
    ap.add_argument("--sequences", default="cache/sequences.npy")
    ap.add_argument("--extra-sequences", nargs="*", default=[],
                    help="額外的序列庫，沿**特徵軸**串接（每一手的向量變長）。"
                         "`cache/sequences_opp.npy` 是對手視角的逐手序列——"
                         "對手視角在整局統計上值 +0.05")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="cache/rank_seq")
    args = ap.parse_args()

    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"device = {device}", flush=True)

    paths = [args.sequences] + list(args.extra_sequences)
    parts = [np.load(p_, mmap_mode="r") for p_ in paths]
    ok = None
    for p_ in paths:
        m = np.load(p_.replace(".npy", "_ok.npy"))
        ok = m if ok is None else (ok & m)     # 任一視角失敗就整局不用
    S = SeqStore(parts)
    if args.extra_sequences:
        print(f"  序列庫 {S.shape[-1]} 維/手（含 {len(args.extra_sequences)} 份額外）"
              f"，共同有效 {ok.mean():.4f}", flush=True)
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    vm = split.mask(meta)
    rank_id = meta["rank_id"].to_numpy()

    pool = np.flatnonzero((~vm) & ok)
    rng = np.random.default_rng(0)
    rows = (np.sort(rng.choice(pool, size=min(args.train_games, len(pool)),
                               replace=False))
            if args.train_games else pool)
    print(f"訓練局數 {len(rows):,}，序列 {MAX_MOVES}×{S.shape[-1]}", flush=True)

    Xtr = np.asarray(S[rows], dtype=np.float32)
    ltr = (np.abs(Xtr).sum(-1) > 0).sum(-1)
    ytr = torch.from_numpy(rank_id[rows]).long()
    Xt = torch.from_numpy(Xtr)
    Mt = torch.from_numpy(np.arange(MAX_MOVES)[None, :] < ltr[:, None])
    print(f"  訓練張量 {Xt.nbytes / 1e9:.2f} GB", flush=True)

    soft, corn = SoftLabelCrossEntropy(10, decay=1.0), CornLoss(10)
    nets = []
    for sd in range(args.seeds):
        torch.manual_seed(sd)
        net = SeqRankNet(n_feat=S.shape[-1], width=args.width).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                  total_steps=args.steps, pct_start=0.1)
        net.train(); t0 = time.time()
        for _ in range(args.steps):
            i = torch.randint(0, len(ytr), (args.batch,))
            lc, lo = net(Xt[i].to(device), Mt[i].to(device))
            yb = ytr[i].to(device)
            loss = soft(lc, yb) + 0.5 * corn(lo, yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sch.step()
        net.eval(); nets.append(net)
        print(f"  seed {sd} loss {loss.item():.4f} ({(time.time() - t0) / 60:.1f} 分)",
              flush=True)

    # 題目層級評估：校準與評分的玩家不重疊
    scores = []
    for qs in range(4):
        q = build_rank_questions(meta, split, n_per_player=3, seed=qs)
        used = np.unique(np.concatenate([r for r in q.game_rows]))
        lens = np.zeros(len(S), dtype=np.int64)
        lens[used] = (np.abs(np.asarray(S[used], dtype=np.float32)).sum(-1) > 0).sum(-1)
        P = question_posteriors(nets, S, ok, lens, q.game_rows, device)
        lo = np.log(np.clip(P, 1e-12, None))
        y = np.asarray(q.true_rank_id); pl = np.asarray(q.player_id)
        up = np.array(sorted(set(pl))); rr = np.random.default_rng(1000 + qs); rr.shuffle(up)
        cs = set(up[:len(up) // 2].tolist()); inc = np.array([p in cs for p in pl])
        cal, sco = np.flatnonzero(inc), np.flatnonzero(~inc)
        T = fit_temperature(lo[cal], y[cal])
        ev = expected_level(apply_temperature(lo, T))
        th = optimise_thresholds(ev[cal], y[cal])
        scores.append(rank_final_score(apply_thresholds(ev[sco], th), y[sco]))
        print(f"  題目 seed {qs}: {scores[-1]:.4f}", flush=True)

    a = np.array(scores)
    print(f"\n題目層級官方分數 {a.mean():.4f} ±{a.std(ddof=1):.4f}"
          f"（官方 baseline 0.4631）", flush=True)
    torch.save({"models": [n.state_dict() for n in nets], "args": vars(args),
                "score": float(a.mean())}, os.path.join(args.out, "models.pt"))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({"scores": scores, "mean": float(a.mean()),
                   "std": float(a.std(ddof=1))}, f, indent=2)
    print(f"已寫出 {args.out}/models.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
