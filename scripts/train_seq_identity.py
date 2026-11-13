"""競賽一：逐手序列的身分 embedding（dilated 1D-CNN ＋ Sub-center ArcFace）。

動機（RESULTS.md §2.40）：更新版測試集的新增題目每位候選人只有 10 盤、每題 1/2/3/5 盤，
現行的手工特徵度量 MLP 在單盤時只有 0.20（本地模擬），**單盤辨識力是瓶頸**。
手工特徵是整盤的統計摘要；這裡讓模型讀逐手序列，看得到「哪個時點下了什麼樣的手」。

選 checkpoint 用的是「未見身分的檢索分數」：驗證玩家依新增測試集規則組池
（600 候選 x 10 盤、400 題 x 1/2/3/5 盤、同一人可多題），模板餘弦、不做指派。

    python3 scripts/train_seq_identity.py --steps 30000 --out cache/seq_identity
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aicup.data import load_metadata, make_player_split
from aicup.losses import SubCenterArcFace
from aicup.rank_nets import SeqStore, pick_device


class SeqIdentityNet(nn.Module):
    """與 SeqRankNet 同一個主幹（dilated 1D-CNN，avg＋max pooling），輸出 L2 正規化的 embedding。"""

    def __init__(self, n_feat: int, width: int = 192, dim: int = 256, dropout: float = 0.1,
                 in_scale=None):
        super().__init__()
        # 每一維輸入的縮放：float16 的序列庫已正規化（x1），uint8 的 movepatch 要 x1/255。
        # 放在模型裡（並存進 checkpoint），訓練、評估、推論三端就不可能各縮各的。
        self.has_scale = in_scale is not None
        if self.has_scale:
            self.register_buffer("in_scale", torch.as_tensor(in_scale, dtype=torch.float32))
        self.stem = nn.Conv1d(n_feat, width, 5, padding=2)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(width, width, 5, padding=2 * r, dilation=r),
                          nn.GroupNorm(8, width), nn.GELU(),
                          nn.Conv1d(width, width, 1), nn.GroupNorm(8, width), nn.GELU())
            for r in (1, 2, 4, 8, 1, 2, 4, 8)])
        self.head = nn.Sequential(nn.Linear(2 * width, 512), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(512, dim))

    def forward(self, x, mask):
        if self.has_scale:
            x = x * self.in_scale
        h = self.stem(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        m = mask.unsqueeze(1).float()
        h = h * m
        avg = h.sum(-1) / m.sum(-1).clamp(min=1)
        mx = (h + (m - 1) * 1e4).max(-1).values
        return F.normalize(self.head(torch.cat([avg, mx], dim=1)), dim=1)


def store_scale(paths) -> list:
    """各序列庫每一維的縮放：uint8 的庫（movepatch）乘 1/255，其餘 x1。"""
    out = []
    for p in paths:
        a = np.load(p, mmap_mode="r")
        out += [1.0 / 255.0 if a.dtype == np.uint8 else 1.0] * a.shape[-1]
    return out


@torch.no_grad()
def embed(net, X: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    net.eval(); out = []
    for a in range(0, len(X), batch):
        x = torch.from_numpy(np.asarray(X[a:a + batch], dtype=np.float32)).to(device)
        m = x.abs().sum(-1) > 0
        out.append(net(x, m).cpu().numpy())
    net.train()
    return np.concatenate(out) if out else np.zeros((0, net.head[-1].out_features), np.float32)


def v2_pools(meta, split, ok, n_seeds: int = 2):
    """新增測試集規則的驗證池：每個 seed 600 位候選 x 10 盤、400 題 x 1/2/3/5 盤。"""
    pid = meta["player_id"].to_numpy()
    val = np.flatnonzero(split.mask(meta) & ok)
    by = {}
    for r in val:
        by.setdefault(pid[r], []).append(r)
    elig = [p for p, rs in by.items() if len(rs) >= 30]
    pools = []
    for s in range(n_seeds):
        rng = np.random.default_rng(500 + s)
        cands = rng.choice(elig, 600, replace=False)
        gal, rest = {}, {}
        for p in cands:
            rs = rng.permutation(by[p]); gal[p] = rs[:10]; rest[p] = rs[10:]
        qp = rng.choice(cands, 400, replace=True)
        sizes = rng.permutation(np.repeat([1, 2, 3, 5], 100))
        used = {p: 0 for p in cands}; q = []
        for p, n in zip(qp, sizes):
            q.append(rest[p][used[p]:used[p] + n]); used[p] += n
        idx = {p: i for i, p in enumerate(cands)}
        pools.append({"gal": [gal[p] for p in cands], "q": q, "truth": np.array([idx[p] for p in qp]),
                      "sizes": sizes})
    return pools


def retrieval_score(E_of, pools) -> float:
    """模板餘弦、不做指派；官方公式 e^-(r-1)（r 從 0 起算即 e^-r）。"""
    scores = []
    for pl in pools:
        G = np.stack([E_of(g).mean(0) for g in pl["gal"]]); G /= np.linalg.norm(G, axis=1, keepdims=True)
        Q = np.stack([E_of(q).mean(0) for q in pl["q"]]); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        S = Q @ G.T; t = pl["truth"]
        r = (S > S[np.arange(len(S)), t][:, None]).sum(1)
        scores.append(float(np.mean(np.where(r < 5, np.exp(-r), 0.0))))
    return float(np.mean(scores))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", default="cache/sequences.npy,cache/seqshape.npy")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=3000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-games", type=int, default=0,
                    help="只用前 N 盤訓練（0 = 全部）；加了 uint8 的大庫時用來控制記憶體")
    ap.add_argument("--stop-after-peak", type=int, default=0,
                    help="從這一步起，若評估分數連續兩次低於最佳就停止（0 = 不停）。"
                         "30k 排程的最佳通常在 12k~14k（RESULTS.md §2.40）；完整的短排程反而較差")
    ap.add_argument("--out", default="cache/seq_identity")
    args = ap.parse_args()
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    paths = args.stores.split(",")
    S = SeqStore([np.load(p, mmap_mode="r") for p in paths])
    ok = None
    for p in paths:
        m = np.load(p.replace(".npy", "_ok.npy"))
        ok = m if ok is None else (ok & m)
    meta = load_metadata(); split = make_player_split(meta, val_frac=0.20, seed=42)
    vm = split.mask(meta); pid = meta["player_id"].to_numpy()
    tr = np.flatnonzero((~vm) & ok)
    if args.train_games and args.train_games < len(tr):
        tr = np.sort(np.random.default_rng(args.seed).choice(tr, args.train_games, replace=False))
    players = sorted(set(pid[tr])); p2l = {p: i for i, p in enumerate(players)}
    y = np.array([p2l[p] for p in pid[tr]], dtype=np.int64)
    print(f"訓練 {len(tr):,} 盤、{len(players):,} 位身分、序列 {S.shape[1]}x{S.shape[-1]}", flush=True)
    # 各庫保持原本的型別分開放（float16 / uint8）；先串接會整份升成 float16，記憶體翻倍。
    Xparts = [torch.from_numpy(np.ascontiguousarray(np.asarray(p_[tr]))) for p_ in S.parts]
    Mtr = None
    for xp in Xparts:
        m_ = xp.abs().sum(-1) > 0 if xp.dtype != torch.uint8 else xp.sum(-1, dtype=torch.int32) > 0
        Mtr = m_ if Mtr is None else (Mtr | m_)
    ytr = torch.from_numpy(y)
    print(f"  訓練張量 {sum(x.nbytes for x in Xparts) / 1e9:.2f} GB（{[str(x.dtype) for x in Xparts]}）", flush=True)
    scale = store_scale(paths)

    pools = v2_pools(meta, split, ok)
    need = np.unique(np.concatenate([np.concatenate(pl["gal"] + pl["q"]) for pl in pools]))
    Xval = np.asarray(S[need], dtype=np.float16); pos = {int(r): i for i, r in enumerate(need)}

    torch.manual_seed(args.seed)
    net = SeqIdentityNet(S.shape[-1], args.width, args.dim,
                         in_scale=scale if any(v != 1.0 for v in scale) else None).to(device)
    arc = SubCenterArcFace(args.dim, len(players), k=3,
                           class_counts=torch.from_numpy(np.bincount(y, minlength=len(players)))).to(device)
    opt = torch.optim.AdamW(list(net.parameters()) + list(arc.parameters()), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, hist, t0 = -1.0, [], time.time()
    for step in range(1, args.steps + 1):
        i = torch.randint(0, len(ytr), (args.batch,))
        xb = torch.cat([xp[i].to(device).float() for xp in Xparts], dim=-1)
        emb = net(xb, Mtr[i].to(device))
        loss = F.cross_entropy(arc(emb, ytr[i].to(device)), ytr[i].to(device))
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sch.step()
        if step % args.eval_every == 0 or step == args.steps:
            Ev = embed(net, Xval, device)
            sc = retrieval_score(lambda rows: Ev[[pos[int(r)] for r in rows]], pools)
            hist.append({"step": step, "loss": float(loss.item()), "v2_score": sc,
                         "minutes": (time.time() - t0) / 60})
            print(f"  step {step:>6} loss {loss.item():.3f}  新增測試集模擬（模板）{sc:.4f}"
                  f"  ({(time.time() - t0) / 60:.0f} 分)", flush=True)
            ck = {"model": net.state_dict(), "n_feat": S.shape[-1], "width": args.width, "dim": args.dim,
                  "stores": paths, "step": step, "v2_score": sc,
                  "in_scale": scale if net.has_scale else None}
            torch.save(ck, os.path.join(args.out, "last.pt"))
            if sc > best:
                best = sc; torch.save(ck, os.path.join(args.out, "best.pt"))
            with open(os.path.join(args.out, "history.json"), "w") as fh:
                json.dump(hist, fh, indent=1)
            if (args.stop_after_peak and step >= args.stop_after_peak and len(hist) >= 2
                    and all(h["v2_score"] < best for h in hist[-2:])):
                print(f"  連續兩次低於最佳 {best:.4f}，停止", flush=True)
                break
    print(f"最佳 {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
