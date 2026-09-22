"""端到端模擬：用**競賽一的辨識結果**去增強競賽二的棋力預測。

## 前提與上限

官方競賽一的候選庫是「400 人各 100 譜」。若競賽二的 400 位查詢玩家
就是那 400 位候選人（當天由 `scripts/check_crosstask_link.py` 判定），
我們可以先認出每道棋力題是誰，再用那位候選人的 100 譜。

盤數的邊際效益很大（`scripts/eval_more_games.py`，固定同一批玩家）：

    5 盤 0.4584 / 10 盤 0.4864 / 20 盤 0.5227 / 50 盤 0.5521 / 100 盤 0.5597

相對官方題型組成（5/10/15/20 各 25%，約 0.493）**上限是 +0.067**。

## 但辨識會錯

top-1 是 0.9475，所以約 5% 的題目會拿到**別人**的 100 盤——
那不是「少賺一點」，是把完全錯的證據當成強證據，對那些題目是災難。
所以這支量的是**淨效果**，包含辨識誤差，並比較三種用法：

    A 只用題目自己的 5~20 盤（現況）
    B 直接換成配對到的候選人的 100 盤（全押）
    C 兩者的後驗加權幾何平均，權重由**配對信心**決定（保守）

C 才是應該上線的形式：配對不確定時自動退回 A。

用法：
    python3 scripts/eval_crosstask.py --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aicup.benchmark import build_player_pool                    # noqa: E402
from aicup.data import load_metadata, make_player_split          # noqa: E402
from aicup.feature_metric import (FeatureEmbedder, FeatureScaler,  # noqa: E402
                                  embed_features, player_template)
from aicup.featstore import open_stores                          # noqa: E402
from aicup.rerank import sinkhorn_assignment                     # noqa: E402
from aicup.scoring import (apply_thresholds, expected_level,     # noqa: E402
                           optimise_thresholds, rank_final_score)
import eval_blend as EB                                          # noqa: E402

METRIC = os.environ.get("AICUP_METRIC", "cache/feature_metric/models.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--sinkhorn", type=float, default=0.002)
    ap.add_argument("--weights", type=float, nargs="*",
                    default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
                    help="混合權重：0 = 只用題目自己的盤，1 = 全用配對到的 100 盤。"
                         "**固定權重**而不是逐題信心——逐題 margin 在 Sinkhorn 之後"
                         "對『全部配錯』與『全部配對』長得一樣，不是有效的訊號")
    ap.add_argument("--no-match", action="store_true",
                    help="**假設不成立**的情境：把查詢換成候選庫裡沒有的人，"
                         "看信心門檻擋不擋得住。這是這個功能最重要的安全測試")
    ap.add_argument("--json-out", default="artifacts/crosstask.json")
    args = ap.parse_args()

    dev = EB.pick_device()
    comp = EB.Components(dev)
    Xm, okm = open_stores(["cache/features.npy"])
    ck = torch.load(METRIC, map_location=dev, weights_only=False)
    sc = FeatureScaler().load_state_dict(ck["scaler"])
    cols = ck.get("cols")
    embs = []
    for sd in ck["models"]:
        m = FeatureEmbedder(len(sc.mu), embed_dim=ck["args"]["embed_dim"],
                            hidden=ck["args"]["hidden"])
        m.load_state_dict(sd); m.eval().to(dev)
        embs.append(m)

    def tpl(rows):
        A = np.asarray(Xm[rows])
        if cols is not None:
            A = A[:, cols]
        v = np.concatenate([player_template(embed_features(m, sc, A, dev))
                            for m in embs])
        return v / max(float(np.linalg.norm(v)), 1e-8)

    def ev_of(game_rows):
        """單局成分 → 期望等級。與 eval_more_games 一致，只用單局成分。"""
        lo = comp.game(SimpleNamespace(game_rows=game_rows))
        p = np.exp(lo)
        return expected_level(p / p.sum(axis=1, keepdims=True)), lo

    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    rows, stats = {}, {"top1": [], "swapped": []}
    t0 = time.time()
    for seed in range(args.seeds):
        pool = build_player_pool(meta, split, composition="uniform", seed=seed)
        names = list(pool.players)
        g = [pool.gallery_rows[p][okm[pool.gallery_rows[p]]] for p in names]
        q = [pool.query_rows[p][okm[pool.query_rows[p]]] for p in names]
        if args.no_match:
            # 假設不成立：查詢玩家**完全不在**候選庫裡。
            # ⚠️ 必須湊滿同樣的 400 人。人數少會誇大 top-1 的碰撞、
            #    讓相異率偏低，等於高估了偵測器的安全邊際。
            #    候選人需要 ≥105 譜（只有約 700 人合格），但**查詢只需要 5~20 盤**，
            #    所以查詢玩家可以從大得多的池子裡抽。
            from aicup.benchmark import _player_rows
            allrows = _player_rows(meta, split.val_players)
            taken = set(names)
            cands = [p for p, idx in allrows.items()
                     if p not in taken and len(idx) >= 20]
            rr = np.random.default_rng(777 + seed); rr.shuffle(cands)
            pick = cands[:len(names)]
            names = pick
            q = [allrows[p][okm[allrows[p]]][:12] for p in pick]
            q = [r for r in q]
        keep = [i for i in range(len(names)) if len(g[i]) and len(q[i])]
        names = [names[i] for i in keep]; g = [g[i] for i in keep]; q = [q[i] for i in keep]

        raw = np.stack([tpl(r) for r in q]) @ np.stack([tpl(r) for r in g]).T
        # ⚠️ 全域判準一定要看 **Sinkhorn 之前**的相似度。
        # Sinkhorn 會強制產生近乎完美的一一對應，**不管真正的答案在不在候選庫裡**
        # ——套用之後的 margin 對「全部配錯」與「全部配對」長得一模一樣
        #（實測全錯時 margin 中位數 0.64、p90 0.9956）。
        from aicup.rerank import bijection_consistency
        bij_raw = bijection_consistency(raw)
        stats.setdefault("bij_raw", []).append(float(bij_raw))
        sim = sinkhorn_assignment(raw, eps=args.sinkhorn) if args.sinkhorn > 0 else raw
        stats.setdefault("top1_sim", []).append(
            float(np.mean(np.max(raw, axis=1))))
        order = np.argsort(-sim, axis=1)
        top1 = order[:, 0]
        srt = np.take_along_axis(sim, order[:, :2], axis=1)
        margin = srt[:, 0] - srt[:, 1]
        acc = float((top1 == np.arange(len(names))).mean())
        stats["top1"].append(acc)

        y = np.array([meta["rank_id"].to_numpy()[r[0]] for r in q])
        correct = (top1 == np.arange(len(names))) if not args.no_match \
            else np.zeros(len(names), dtype=bool)
        if correct.any() and (~correct).any():
            print(f"    margin：配對正確 {np.median(margin[correct]):.4f}"
                  f"（p10 {np.percentile(margin[correct], 10):.4f}）"
                  f"／配對錯誤 {np.median(margin[~correct]):.4f}"
                  f"（p90 {np.percentile(margin[~correct], 90):.4f}）", flush=True)
        elif args.no_match:
            print(f"    margin（全部都是錯的）中位數 {np.median(margin):.4f}"
                  f"，p90 {np.percentile(margin, 90):.4f}", flush=True)
        ev_q, lo_q = ev_of(q)                      # A：題目自己的 5~20 盤
        ev_m, lo_m = ev_of([g[int(j)] for j in top1])   # B：配對到的 100 盤
        cand = []
        for wt in args.weights:
            lo_c = (1 - wt) * lo_q + wt * lo_m
            pp = np.exp(lo_c)
            cand.append((f"w={wt:g}" + ("（現況）" if wt == 0 else ""),
                         expected_level(pp / pp.sum(axis=1, keepdims=True))))

        up = np.arange(len(names))
        for nm, ev in cand:
            sc_ = []
            for sp in range(args.splits):
                rr = np.random.default_rng(90000 + 31 * seed + sp)
                u = up.copy(); rr.shuffle(u)
                cal = np.sort(u[:len(u) // 2]); sco = np.sort(u[len(u) // 2:])
                th = optimise_thresholds(ev[cal], y[cal])
                sc_.append(rank_final_score(apply_thresholds(ev[sco], th), y[sco]))
            rows.setdefault(nm, []).append(float(np.mean(sc_)))
        print(f"  seed {seed}：{len(names)} 人，辨識 top-1 {acc:.4f}"
              f"，**Sinkhorn 前的 top-1 相異率 {bij_raw:.4f}**"
              f"，平均最高相似度 {stats['top1_sim'][-1]:.4f}", flush=True)

    print(f"\n=== {args.seeds} seed × {args.splits} 切分，{time.time() - t0:.0f} 秒 ===")
    base = np.array(rows["w=0（現況）"])
    out = {}
    for k, v in rows.items():
        a = np.array(v); d = a - base
        out[k] = {"mean": float(a.mean()), "delta": float(d.mean()),
                  "wins": int((d > 0).sum()), "n": len(a)}
        extra = "" if k.startswith("w=0（") else \
            f"  {d.mean():+.4f}（{int((d>0).sum())}/{len(d)}）"
        print(f"  {k:<26} {a.mean():.4f}{extra}")
    print(f"\n**全域判準**：Sinkhorn 前的 top-1 相異率 "
          f"{np.mean(stats['bij_raw']):.4f}"
          f"，平均最高相似度 {np.mean(stats['top1_sim']):.4f}")
    print(f"辨識 top-1 平均 {np.mean(stats['top1']):.4f}"
          f"——約 {1 - np.mean(stats['top1']):.1%} 的題目會拿到**別人**的 100 盤")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"results": out, "top1": float(np.mean(stats["top1"]))},
                      f, indent=2, ensure_ascii=False)
        print(f"已寫出 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
