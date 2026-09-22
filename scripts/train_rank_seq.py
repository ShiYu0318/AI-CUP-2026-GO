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
from aicup.losses import (SoftLabelCrossEntropy, CornLoss,       # noqa: E402
                          PairwiseRankLoss, corn_to_proba)
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


class SeqRankGRU(nn.Module):
    """雙向 GRU 讀逐手序列——**刻意換一個歸納偏置**，不是為了更強。

    集成的收益來自成分之間的分歧。目前三個集成成員（base / full / w192）
    都是同一個 dilated CNN，只差寬度與資料量，分歧其實很小，
    卻已經值 +0.0079（8/8，t=9.15）。換成遞迴狀態（無固定感受野、
    對「累積到目前為止的棋風」天然敏感）應該產生更不一樣的錯誤。

    pooling 刻意與 `SeqRankNet` **完全相同**（masked avg + max），
    所以兩者的差別只有 encoder 一項——這樣比較才有解釋力。
    """

    def __init__(self, n_feat: int = N_MOVE_FEATURES, width: int = 128,
                 n_levels: int = 10, dropout: float = 0.1, layers: int = 2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, width), nn.GELU())
        self.rnn = nn.GRU(width, width // 2, num_layers=layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * width, 256), nn.GELU(),
                                  nn.Dropout(dropout))
        self.cls = nn.Linear(256, n_levels)
        self.orn = nn.Linear(256, n_levels - 1)

    def forward(self, x, mask):
        h, _ = self.rnn(self.inp(x))          # (B, T, width)
        h = h.transpose(1, 2)                 # 與 CNN 的 (B, C, T) 對齊
        m = mask.unsqueeze(1).float()
        h = h * m
        avg = h.sum(-1) / m.sum(-1).clamp(min=1)
        mx = (h + (m - 1) * 1e4).max(-1).values
        z = self.head(torch.cat([avg, mx], dim=1))
        return self.cls(z), self.orn(z)


ARCHS = {"cnn": SeqRankNet, "gru": SeqRankGRU}


def build_seq_net(n_feat: int, ck_args: dict):
    """由 checkpoint 存下來的 args 重建網路。

    `arch` 是後加的，舊 checkpoint 沒有這個鍵——預設 "cnn" 才不會
    讓既有的三個集成成員載不起來。
    """
    arch = ck_args.get("arch", "cnn")
    if arch not in ARCHS:
        raise ValueError(f"未知的序列模型 arch={arch!r}，可用：{sorted(ARCHS)}")
    return ARCHS[arch](n_feat=n_feat, width=ck_args["width"])


def question_posteriors(nets, S, ok, lens, game_rows, device, batch: int = 256):
    """逐題：多局後驗做幾何平均（與 train_rank_mlp 的聚合方式一致）。

    **先算完所有棋局，再逐題聚合**——不要逐題呼叫模型。
    逐題的話每次只有 5~20 局，1660 題 × 3 個網路 = 4980 次前向，
    全部是小 batch。卷積還撐得住，但 GRU 在 MPS 上每次前向要發
    160×2×2 次序列 kernel，啟動開銷完全主導：
    實測逐題版算一個 seed 的 GRU 後驗要**超過兩小時**，
    而整批版只要幾十秒。數值上兩者完全相同（同樣的幾何平均）。

    ⚠️ 結束時還原原本的 train/eval 模式。無條件切回 train 會讓後續推論
    帶著 dropout 跑（見 tests/test_eval_mode.py）。
    """
    was = [n.training for n in nets]
    for n in nets:
        n.eval()

    sels = [rows[ok[rows]] for rows in game_rows]
    flat = np.concatenate([s_ for s_ in sels if len(s_)]) if any(len(s_) for s_ in sels) \
        else np.zeros(0, dtype=np.int64)
    uniq, inv = np.unique(flat, return_inverse=True)

    P = np.empty((len(uniq), 10), dtype=np.float64)
    T = S.shape[1]
    with torch.no_grad():
        for i in range(0, len(uniq), batch):
            sel = uniq[i:i + batch]
            x = torch.from_numpy(np.asarray(S[sel], dtype=np.float16)).to(device).float()
            mk = torch.from_numpy(
                np.arange(T)[None, :] < lens[sel][:, None]).to(device)
            ps = []
            for n in nets:
                lc, lo = n(x, mk)
                pc = corn_to_proba(lo).cpu().numpy()
                ps.append(0.5 * torch.softmax(lc, -1).cpu().numpy()
                          + 0.5 * pc / np.maximum(pc.sum(1, keepdims=True), 1e-12))
            P[i:i + len(sel)] = np.mean(ps, axis=0)
    for n, t in zip(nets, was):
        n.train(t)

    logP = np.log(np.clip(P, 1e-12, None))
    out, at = [], 0
    for s_ in sels:
        if len(s_) == 0:
            out.append(np.full(10, 0.1)); continue
        g = np.exp(logP[inv[at:at + len(s_)]].mean(axis=0))
        at += len(s_)
        out.append(g / g.sum())
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--arch", default="cnn", choices=sorted(ARCHS),
                    help="序列 encoder。cnn = dilated 1D 卷積（預設）；\n"
                         "gru = 雙向 GRU。換 arch 是為了**製造集成分歧**，"
                         "不是預期它單獨更強")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rank-loss", type=float, default=0.0,
                    help="成對排序損失的權重。實測切點已經到頂（oracle 只剩 +0.0172），"
                         "分數幾乎全由期望等級的**排序**決定，"
                         "但 CE 與 CORN 都是分類損失、不直接管順序")
    ap.add_argument("--train-games", type=int, default=400000,
                    help="抽多少局訓練（0 = 全部）。序列張量比特徵大，記憶體吃緊時調小")
    ap.add_argument("--sequences", default="cache/sequences.npy")
    ap.add_argument("--extra-sequences", nargs="*", default=[],
                    help="額外的序列庫，沿**特徵軸**串接（每一手的向量變長）。"
                         "`cache/sequences_opp.npy` 是對手視角的逐手序列——"
                         "對手視角在整局統計上值 +0.05")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU 執行緒數。0 = 不動（torch 在這台 10 核機器上預設只用 4）。"
                         "⚠️ 與 MPS 工作並行時不要開太大——"
                         "MPS 那邊的資料載入也要 CPU，搶光反而兩邊都慢")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="cache/rank_seq")
    args = ap.parse_args()
    _t0 = time.time()      # 記錄總耗時，寫進 cache/experiments.csv

    if args.threads > 0:
        torch.set_num_threads(args.threads)
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
    print(f"訓練局數 {len(rows):,}，序列 {S.shape[1]}×{S.shape[-1]}", flush=True)

    # ⚠️ 保持 float16，**不要**在這裡升成 float32。
    # 來源序列庫本來就是 float16（所有欄位都已正規化到 [0,1]，精度夠）；
    # 升成 float32 會讓 78 萬局 × 160 手 × 24 維從 6 GB 變成 12 GB，
    # 在 26 GB 的機器上直接吃掉一半記憶體，也讓平行訓練變得不可能。
    # 只在每個 batch 搬上裝置時才轉型，成本可忽略。
    Xtr = np.asarray(S[rows], dtype=np.float16)
    ltr = (np.abs(Xtr).sum(-1) > 0).sum(-1)
    ytr = torch.from_numpy(rank_id[rows]).long()
    Xt = torch.from_numpy(Xtr)
    Mt = torch.from_numpy(np.arange(S.shape[1])[None, :] < ltr[:, None])
    print(f"  訓練張量 {Xt.nbytes / 1e9:.2f} GB（float16）", flush=True)

    soft, corn = SoftLabelCrossEntropy(10, decay=1.0), CornLoss(10)
    prank = PairwiseRankLoss(10).to(device) if args.rank_loss > 0 else None
    nets = []
    for sd in range(args.seeds):
        torch.manual_seed(sd)
        net = build_seq_net(S.shape[-1], vars(args)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                  total_steps=args.steps, pct_start=0.1)
        net.train(); t0 = time.time()
        for _ in range(args.steps):
            i = torch.randint(0, len(ytr), (args.batch,))
            lc, lo = net(Xt[i].to(device).float(), Mt[i].to(device))
            yb = ytr[i].to(device)
            loss = soft(lc, yb) + 0.5 * corn(lo, yb)
            if prank is not None:
                loss = loss + args.rank_loss * prank(lc, yb)
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

    from aicup.experiment_log import log_experiment
    log_experiment("rank", os.path.basename(args.out), vars(args),
                   {"internal_question_score": float(a.mean()),
                    "std": float(a.std(ddof=1))},
                   minutes=(time.time() - _t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
