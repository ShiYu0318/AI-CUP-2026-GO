"""競賽二：多 backend GBDT 集成（LightGBM / XGBoost / CatBoost / HistGB）+ stacking。

相對於先前的單一 HistGradientBoosting，這裡做三件事：
  1. 每個可用的 backend 各自做 **GroupKFold（以玩家分組）** 產生 OOF 機率
  2. 用 OOF 訓練一層 stacking meta-learner，並另外搜尋加權融合的最佳權重
  3. 全程以**官方評分函數**（題目層級、期望分數解碼）決定融合權重，
     不是用 accuracy 或 log-loss

⚠️ 分組必須是玩家：隨機 KFold 會讓同一位玩家的棋局橫跨 fold，
   OOF 被身分洩漏污染，stacking 權重會學歪。

用法：
    python3 scripts/train_gbdt_ensemble.py --folds 5
    python3 scripts/train_gbdt_ensemble.py --folds 5 --backends lgbm xgb cat hist
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup.data import load_metadata, make_player_split          # noqa: E402
from aicup.benchmark import build_rank_questions                 # noqa: E402
from aicup.baselines import RankBaseline                         # noqa: E402
from aicup.gbdt import (available_backends, fit_oof, predict_mean,   # noqa: E402
                        player_group_folds, StackingBlender,
                        optimise_weights, weighted_blend)
from aicup.scoring import (decode_expected_score, rank_final_score,   # noqa: E402
                           apply_temperature, fit_temperature)


def question_score(model_probs: np.ndarray, q, ok, rows_of, y, temperature: float):
    """把單局機率聚合成題目層級分數（logmean + 溫度校準 + 期望解碼）。"""
    preds = []
    for rows in q.game_rows:
        sel = rows_of[rows]
        p = model_probs[sel[sel >= 0]] if len(sel) else np.empty((0, 10))
        if len(p) == 0:
            preds.append(np.full(10, 0.1)); continue
        g = np.exp(np.log(np.clip(p, 1e-12, None)).mean(0))
        preds.append(g / g.sum())
    P = np.stack(preds)
    P = apply_temperature(np.log(np.clip(P, 1e-12, None)), temperature)
    return rank_final_score(decode_expected_score(P), y), P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--backends", nargs="*", default=None,
                    help="預設使用這台機器上所有可用的 backend")
    ap.add_argument("--max-per-rank", type=int, default=0,
                    help="每等級最多取幾局（0 = 全部）。除錯時可調小")
    ap.add_argument("--out", default="cache/gbdt")
    args = ap.parse_args()

    backends = args.backends or available_backends()
    print(f"使用 backend：{backends}", flush=True)
    os.makedirs(args.out, exist_ok=True)

    X = np.load("cache/features.npy", mmap_mode="r")
    ok = np.load("cache/features_ok.npy")
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    vm = split.mask(meta)
    rank_id = meta["rank_id"].to_numpy()
    pid = meta["player_id"].to_numpy()

    tr_rows = np.flatnonzero((~vm) & ok)
    if args.max_per_rank:
        rng = np.random.default_rng(0)
        tr_rows = np.concatenate([
            rng.choice(tr_rows[rank_id[tr_rows] == lv],
                       min(args.max_per_rank, int((rank_id[tr_rows] == lv).sum())),
                       replace=False) for lv in range(10)])
    Xtr = np.asarray(X[tr_rows]); ytr = rank_id[tr_rows]
    folds = player_group_folds(pid[tr_rows], ytr, n_folds=args.folds, seed=0)
    print(f"訓練局數 {len(tr_rows):,}，玩家 {len(np.unique(pid[tr_rows])):,}，"
          f"{args.folds} folds（以玩家分組）", flush=True)

    # --- 各 backend 的 OOF ---
    oofs, all_models = {}, {}
    for b in backends:
        print(f"\n[{b}]", flush=True)
        try:
            res = fit_oof(Xtr, ytr, folds, b, seed=0)
        except Exception as e:
            print(f"    略過（{type(e).__name__}: {e}）", flush=True)
            continue
        oofs[b] = res.oof; all_models[b] = res.models
        print(f"    OOF 單局 acc {(res.oof.argmax(1) == ytr).mean():.4f}"
              f"  耗時 {res.seconds / 60:.1f} 分", flush=True)

    if not oofs:
        raise SystemExit("沒有任何 backend 成功訓練")

    # --- 在驗證題目上評估（這才是官方指標）---
    q = build_rank_questions(meta, split, n_per_player=3, seed=0)
    va_rows = np.flatnonzero(vm & ok)
    rows_of = np.full(len(meta), -1, dtype=np.int64)
    rows_of[va_rows] = np.arange(len(va_rows))
    Xva = np.asarray(X[va_rows])
    y = q.true_rank_id
    half = len(y) // 2

    print("\n=== 各 backend 的題目層級分數 ===", flush=True)
    va_probs = {}
    for b, models in all_models.items():
        va_probs[b] = predict_mean(models, Xva)
        s, _ = question_score(va_probs[b], q, ok, rows_of, y, 0.5)
        print(f"  {b:<6} {s:.4f}", flush=True)

    # --- 融合：OOF 上找權重，驗證題目上評分 ---
    print("\n=== 融合 ===", flush=True)
    results = {}

    mean_p = np.mean(list(va_probs.values()), axis=0)
    s_mean, _ = question_score(mean_p, q, ok, rows_of, y, 0.5)
    results["simple_mean"] = s_mean
    print(f"  簡單平均            {s_mean:.4f}", flush=True)

    if len(oofs) > 1:
        def oof_score(p):
            return float((p.argmax(1) == ytr).mean())
        w = optimise_weights(oofs, oof_score, n_iter=300)
        blend = weighted_blend(va_probs, w)
        s_w, _ = question_score(blend, q, ok, rows_of, y, 0.5)
        results["weighted"] = s_w
        print(f"  加權融合（OOF 搜尋）  {s_w:.4f}   權重 "
              f"{ {k: round(v, 3) for k, v in w.items()} }", flush=True)

        st = StackingBlender().fit(oofs, ytr)
        s_st, _ = question_score(st.predict_proba(va_probs), q, ok, rows_of, y, 0.5)
        results["stacking"] = s_st
        print(f"  Stacking            {s_st:.4f}", flush=True)

    # --- 溫度校準（前半題目校準、後半評分，避免自我驗證）---
    best_name = max(results, key=results.get)
    best_probs = {"simple_mean": mean_p,
                  "weighted": weighted_blend(va_probs, w) if len(oofs) > 1 else mean_p,
                  "stacking": st.predict_proba(va_probs) if len(oofs) > 1 else mean_p}[best_name]
    _, P = question_score(best_probs, q, ok, rows_of, y, 1.0)
    T = fit_temperature(np.log(np.clip(P[:half], 1e-12, None)), y[:half])
    Pc = apply_temperature(np.log(np.clip(P[half:], 1e-12, None)), T)
    final = rank_final_score(decode_expected_score(Pc), y[half:])
    print(f"\n最佳融合：{best_name}，溫度校準 T={T:.2f}", flush=True)
    print(f"最終題目層級分數 {final:.4f}"
          f"（官方 baseline 0.4631 / 單一 HistGB 0.3973）", flush=True)

    with open(os.path.join(args.out, "models.pkl"), "wb") as f:
        pickle.dump({"models": all_models, "temperature": T,
                     "blend": best_name, "weights": w if len(oofs) > 1 else None}, f)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({"backends": list(oofs), "per_backend":
                   {b: question_score(va_probs[b], q, ok, rows_of, y, 0.5)[0]
                    for b in va_probs},
                   "blends": results, "temperature": T, "final": final},
                  f, indent=2, ensure_ascii=False)
    print(f"已寫出 {args.out}/models.pkl 與 report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
