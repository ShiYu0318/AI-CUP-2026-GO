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
import time

import numpy as np
import torch
import torch.nn as nn

from aicup.data import load_metadata, make_player_split           # noqa: E402
from aicup.benchmark import build_rank_questions                  # noqa: E402
# 模型定義與題目層級推論住在 `src/aicup/rank_nets.py`——推論路徑（predict.py）
# 也要用，不該依賴訓練腳本。這裡 re-export，既有的 `from train_rank_seq import
# build_seq_net` 仍然可用。
from aicup.rank_nets import (ARCHS, SeqRankGRU, SeqRankNet,       # noqa: E402,F401
                             SeqStore, build_seq_net, pick_device,
                             seq_question_posteriors as question_posteriors)
from aicup.losses import (SoftLabelCrossEntropy, CornLoss,       # noqa: E402
                          PairwiseRankLoss, corn_to_proba)
from aicup.moveseq import MAX_MOVES, N_MOVE_FEATURES              # noqa: E402
from aicup.scoring import (fit_temperature, apply_temperature,    # noqa: E402
                           expected_level, optimise_thresholds,
                           apply_thresholds, rank_final_score)
















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
