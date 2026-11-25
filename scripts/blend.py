"""把深度模型與手工特徵/GBDT 融合，並產出最終提交檔。

為什麼融合幾乎必定有用：兩條路徑的**誤差結構完全不同**。
手工特徵看的是統計習慣（應手距離、脫先率、開局選點、愚形率），
深度模型看的是盤面本身。實測兩者單獨的分數都不低，而錯的題目重疊度很低。

競賽一：把兩種 embedding 各自 L2 正規化後**串接**再算餘弦
        （等價於相似度加權平均，是 Kaggle 檢索類競賽的標準融合法）
競賽二：把兩種後驗做加權幾何平均，權重在驗證題目上以**官方分數**搜尋

⚠️ 所有權重都必須在 player-disjoint 的驗證題目上決定，
   絕不可用測試集或排行榜（見 docs/04-實驗總覽.md 的量測協定）。

用法：
    # 先分別訓練
    python3 scripts/train_player.py --device cuda --steps 20000
    python3 scripts/train.py        --device cuda --steps 20000
    # 再融合並評估
    python3 scripts/blend.py --player-ckpt cache/ckpt_player/best.pt \\
                             --rank-ckpt   cache/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from aicup.data import load_metadata, make_player_split          # noqa: E402
from aicup.benchmark import build_player_pool, build_rank_questions  # noqa: E402
from aicup.baselines import PlayerBaseline, RankBaseline         # noqa: E402
from aicup.dataset import SgfCache                               # noqa: E402
from aicup.nets import MultiTaskGoNet                            # noqa: E402
from aicup.planes import game_planes, planes_to_float           # noqa: E402
from aicup.scoring import (retrieval_report, decode_expected_score,   # noqa: E402
                           rank_final_score, apply_temperature, fit_temperature)


def load_net(path: str, device) -> MultiTaskGoNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    net = MultiTaskGoNet(in_channels=34, width=a["width"], blocks=a["blocks"],
                         embed_dim=a["embed_dim"],
                         n_quality=len(a.get("quality_cols") or []) or 0,
                         pool=a.get("pool", "attention"),
                         global_pool_bias=not a.get("no_global_pool_bias", False))
    net.load_state_dict(ck["model"], strict=False)
    net.eval().to(device)
    print(f"  載入 {path}（訓練時最佳分數 {ck.get('score', float('nan')):.4f}）", flush=True)
    return net


@torch.no_grad()
def deep_embed(net, meta, cache, rows, device, n_snapshots: int, batch: int = 64):
    out, buf = [], []
    def flush():
        if not buf: return
        x = torch.from_numpy(np.stack(buf)).float().to(device)
        out.append(F.normalize(net(x)["embedding"], dim=1).cpu().numpy())
        buf.clear()
    for r in rows:
        rec = meta.iloc[int(r)]
        t = game_planes(cache.get(int(rec["file_idx"]), int(rec["row"])),
                        rec["color"], n_snapshots=n_snapshots)
        if t is None: continue
        buf.append(planes_to_float(t).mean(axis=0))
        if len(buf) >= batch: flush()
    flush()
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def tpl(e: np.ndarray) -> np.ndarray:
    v = e.mean(0)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def blend_player(args, meta, split, X, ok, device):
    """競賽一：手工 embedding ‖ 深度 embedding 串接檢索。"""
    print("\n=== 競賽一：玩家辨識 ===", flush=True)
    net = load_net(args.player_ckpt, device) if args.player_ckpt else None
    cache = SgfCache()
    rows_out = {}
    for alpha in args.alphas:
        scores = []
        for s in range(args.seeds):
            pool = build_player_pool(meta, split, composition=args.composition, seed=s)
            ps = [p for p in pool.players
                  if ok[pool.gallery_rows[p]].sum() >= 20 and ok[pool.query_rows[p]].sum() >= 1]
            gh = {p: np.asarray(X[pool.gallery_rows[p][ok[pool.gallery_rows[p]]]]) for p in ps}
            qh = {p: np.asarray(X[pool.query_rows[p][ok[pool.query_rows[p]]]]) for p in ps}
            pb = PlayerBaseline().fit(np.concatenate([gh[p] for p in ps]))
            Gh = np.stack([pb.template(gh[p]) for p in ps])
            Qh = np.stack([pb.template(qh[p]) for p in ps])

            if net is None or alpha == 0.0:
                G, Q = Gh, Qh
            else:
                Gd, Qd = [], []
                for p in ps:
                    Gd.append(tpl(deep_embed(net, meta, cache, pool.gallery_rows[p],
                                             device, args.snapshots)))
                    Qd.append(tpl(deep_embed(net, meta, cache, pool.query_rows[p],
                                             device, args.snapshots)))
                Gd, Qd = np.stack(Gd), np.stack(Qd)
                # 串接前各自乘上權重 = 相似度的加權平均
                G = np.hstack([Gh * (1 - alpha) ** 0.5, Gd * alpha ** 0.5])
                Q = np.hstack([Qh * (1 - alpha) ** 0.5, Qd * alpha ** 0.5])
                G /= np.linalg.norm(G, axis=1, keepdims=True)
                Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            scores.append(retrieval_report(Q @ G.T)["official_score"])
        rows_out[alpha] = (float(np.mean(scores)), float(np.std(scores)))
        print(f"  深度權重 α={alpha:<4} 分數 {np.mean(scores):.4f} ±{np.std(scores):.4f}",
              flush=True)
        if net is None:
            break
    best = max(rows_out, key=lambda a: rows_out[a][0])
    print(f"  最佳 α={best}，分數 {rows_out[best][0]:.4f}"
          f"（純手工 {rows_out[min(rows_out)][0]:.4f} / 官方 baseline 0.2845）", flush=True)
    return {"alphas": {str(k): v for k, v in rows_out.items()}, "best_alpha": best}


def blend_rank(args, meta, split, X, ok, device):
    """競賽二：手工 GBDT 後驗 與 深度後驗 的加權幾何平均。"""
    print("\n=== 競賽二：棋力預測 ===", flush=True)
    q = build_rank_questions(meta, split, n_per_player=3, seed=0)
    y = q.true_rank_id
    half = len(y) // 2

    import pickle
    with open(args.rank_model, "rb") as f:
        hand = pickle.load(f)
    Ph = hand.question_proba([np.asarray(X[r[ok[r]]]) for r in q.game_rows], "logmean")

    if not args.rank_ckpt:
        T = fit_temperature(np.log(np.clip(Ph[:half], 1e-12, None)), y[:half])
        s = rank_final_score(decode_expected_score(
            apply_temperature(np.log(np.clip(Ph[half:], 1e-12, None)), T)), y[half:])
        print(f"  只有手工特徵：{s:.4f}", flush=True)
        return {"handcrafted_only": s}

    net = load_net(args.rank_ckpt, device)
    cache = SgfCache()
    Pd = []
    with torch.no_grad():
        for rows in q.game_rows:
            planes = []
            for r in rows:
                rec = meta.iloc[int(r)]
                t = game_planes(cache.get(int(rec["file_idx"]), int(rec["row"])),
                                rec["color"], n_snapshots=args.snapshots)
                if t is not None:
                    planes.append(planes_to_float(t).mean(axis=0))
            if not planes:
                Pd.append(np.full(10, 0.1)); continue
            x = torch.from_numpy(np.stack(planes)).float().to(device)
            p = F.softmax(net(x)["rank_logits"], dim=-1).cpu().numpy()
            g = np.exp(np.log(np.clip(p, 1e-12, None)).mean(0))
            Pd.append(g / g.sum())
    Pd = np.stack(Pd)

    out = {}
    for w in args.alphas:
        mix = np.exp((1 - w) * np.log(np.clip(Ph, 1e-12, None))
                     + w * np.log(np.clip(Pd, 1e-12, None)))
        mix /= mix.sum(1, keepdims=True)
        T = fit_temperature(np.log(np.clip(mix[:half], 1e-12, None)), y[:half])
        s = rank_final_score(decode_expected_score(
            apply_temperature(np.log(np.clip(mix[half:], 1e-12, None)), T)), y[half:])
        out[w] = s
        print(f"  深度權重 w={w:<4} 分數 {s:.4f}  (T={T:.2f})", flush=True)
    best = max(out, key=out.get)
    print(f"  最佳 w={best}，分數 {out[best]:.4f}"
          f"（純手工 {out[min(out)]:.4f} / 官方 baseline 0.4631）", flush=True)
    return {"weights": {str(k): v for k, v in out.items()}, "best_weight": best}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-ckpt", default="")
    ap.add_argument("--rank-ckpt", default="")
    ap.add_argument("--rank-model", default="cache/rank_model.pkl")
    ap.add_argument("--snapshots", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--composition", default="uniform")
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="cache/blend_report.json")
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device != "auto" else
              torch.device("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device = {device}", flush=True)

    X = np.load("cache/features.npy", mmap_mode="r")
    ok = np.load("cache/features_ok.npy")
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)

    report = {}
    report["player"] = blend_player(args, meta, split, X, ok, device)
    if os.path.exists(args.rank_model):
        report["rank"] = blend_rank(args, meta, split, X, ok, device)
    else:
        print(f"\n找不到 {args.rank_model}，略過競賽二（先跑一次 scripts/predict.py 會產生）",
              flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n報告已寫出：{args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
