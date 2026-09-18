"""從官方格式的測試檔產生兩個子題的提交檔，並（若有正解）離線評分。

這支腳本是 2026/11/04 的預演：測試集當天釋出後，理想情況是只需要換掉
`--test-dir`，其餘完全不動。現在就用 `scripts/make_mock_testset.py` 產生的
模擬測試集把整條路徑跑通，避免在有時間壓力時才發現格式或效能問題。

用法：
    python3 scripts/make_mock_testset.py          # 先產生模擬測試集
    python3 scripts/predict.py --test-dir mock_tests

正式使用：
    python3 scripts/predict.py --test-dir tests --out-dir .
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup import RANK_TO_ID, ID_TO_RANK                       # noqa: E402
from aicup.data import load_metadata, make_player_split        # noqa: E402
from aicup.features import batch_features, parallel_batch_features  # noqa: E402
from aicup.baselines import RankBaseline, PlayerBaseline       # noqa: E402
from aicup.scoring import (apply_temperature, decode_expected_score,   # noqa: E402
                           rank_final_score, top_k_from_similarity,
                           player_final_score, fit_temperature,
                           optimise_thresholds, apply_thresholds, expected_level)
from aicup.submission import (build_rank_submission, build_player_submission,  # noqa: E402
                              write_submission)

MAX_GAMES = 20


def _question_features(df: pd.DataFrame):
    """把一份題目檔轉成 [每題的特徵矩陣]。"""
    sgf_cols = [f"sgf_{i}" for i in range(1, MAX_GAMES + 1)]
    col_cols = [f"color_{i}" for i in range(1, MAX_GAMES + 1)]
    flat_sgf, flat_col, sizes = [], [], []
    for _, r in df.iterrows():
        n = 0
        for sc, cc in zip(sgf_cols, col_cols):
            v = r.get(sc)
            if pd.notna(v) and str(v).strip():
                flat_sgf.append(v); flat_col.append(r[cc]); n += 1
        sizes.append(n)
    X, ok = parallel_batch_features(flat_sgf, flat_col)
    out, pos = [], 0
    for n in sizes:
        block = X[pos:pos + n][ok[pos:pos + n]]
        out.append(block)
        pos += n
    return out


def _fit_rank_model(cache_path: str, train_per_rank: int):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    X = np.load("cache/features.npy", mmap_mode="r")
    ok = np.load("cache/features_ok.npy")
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    vm = split.mask(meta)
    rank_id = meta["rank_id"].to_numpy()
    rng = np.random.default_rng(0)
    if train_per_rank and train_per_rank > 0:
        rows = np.concatenate([
            rng.choice(np.flatnonzero((~vm) & ok & (rank_id == lv)),
                       size=min(train_per_rank,
                                int(((~vm) & ok & (rank_id == lv)).sum())),
                       replace=False)
            for lv in range(10)])
    else:
        rows = np.flatnonzero((~vm) & ok)          # 全量訓練玩家的棋局
    print(f"  棋力模型訓練局數 {len(rows):,}", flush=True)
    model = RankBaseline().fit(np.asarray(X[rows]), rank_id[rows])
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(model, f)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="mock_tests")
    ap.add_argument("--out-dir", default="cache/submissions")
    ap.add_argument("--train-per-rank", type=int, default=0,
                    help="每個等級取幾局訓練棋力模型。0 = 用全部訓練玩家的棋局"
                         "（實測全量 0.3973 vs 每級 2 萬局 0.3866，值得多花訓練時間）")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="題目後驗的溫度；<1 為銳化。實測最佳 0.5~0.6")
    ap.add_argument("--aggregation", default="logmean", choices=["mean", "logmean"])
    ap.add_argument("--decode", default="threshold",
                    choices=["threshold", "expected", "argmax"],
                    help="解碼方式。實測（5 次獨立切分）：最佳化切點 0.4336±0.0032 > "
                         "期望分數解碼 0.3921±0.0099 > argmax 0.3886±0.0086")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    answers = {"rank": {}, "player": {}}
    ans_path = os.path.join(args.test_dir, "_answers.json")
    if os.path.exists(ans_path):
        with open(ans_path) as f:
            answers = json.load(f)

    # ---------------- 競賽二 ----------------
    print("=" * 60, flush=True)
    print("競賽二：棋力預測", flush=True)
    print("=" * 60, flush=True)
    model = _fit_rank_model("cache/rank_model.pkl", args.train_per_rank)

    # 切點與溫度都在「held-out 玩家的模擬題目」上校準，與測試集完全無關
    calib = None
    if args.decode == "threshold":
        from aicup.data import make_player_split as _mps
        from aicup.benchmark import build_rank_questions as _brq
        _X = np.load("cache/features.npy", mmap_mode="r")
        _ok = np.load("cache/features_ok.npy")
        _meta = load_metadata()
        _sp = _mps(_meta, val_frac=0.20, seed=42)
        _q = _brq(_meta, _sp, n_per_player=3, seed=0)
        _P = model.question_proba(
            [np.asarray(_X[r[_ok[r]]]) for r in _q.game_rows], args.aggregation)
        _lo = np.log(np.clip(_P, 1e-12, None))
        _T = fit_temperature(_lo, _q.true_rank_id)
        _th = optimise_thresholds(
            expected_level(apply_temperature(_lo, _T)), _q.true_rank_id)
        calib = (_T, _th)
        print(f"  切點校準完成（{len(_q.true_rank_id)} 題）："
              f"T={_T:.2f}, 切點={np.round(_th, 2).tolist()}", flush=True)

    qids, preds, truth = [], [], []
    for tag in ("public", "private"):
        path = os.path.join(args.test_dir, f"rank_prediction_test_{tag}.csv")
        df = pd.read_csv(path)
        t0 = time.time()
        feats = _question_features(df)
        P = model.question_proba(feats, how=args.aggregation)
        lo = np.log(np.clip(P, 1e-12, None))
        if args.decode == "threshold":
            T, th = calib
            pred = apply_thresholds(expected_level(apply_temperature(lo, T)), th)
        elif args.decode == "expected":
            pred = decode_expected_score(apply_temperature(lo, args.temperature))
        else:
            pred = apply_temperature(lo, args.temperature).argmax(axis=-1)
        qids.extend(df["question_id"].tolist()); preds.extend(pred.tolist())
        truth.extend([answers["rank"].get(q) for q in df["question_id"]])
        print(f"  {tag:<8} {len(df)} 題，{time.time() - t0:.0f} 秒", flush=True)

    sub = build_rank_submission(qids, preds)
    path = write_submission(sub, os.path.join(args.out_dir, "submission_rank.csv"), "rank")
    print(f"  已寫出 {path}（{len(sub)} 列，格式驗證通過）", flush=True)
    if truth and all(t is not None for t in truth):
        y = np.array([RANK_TO_ID[t] for t in truth])
        print(f"  離線分數 {rank_final_score(np.array(preds), y):.4f}", flush=True)

    # ---------------- 競賽一 ----------------
    print("\n" + "=" * 60, flush=True)
    print("競賽一：玩家辨識", flush=True)
    print("=" * 60, flush=True)
    all_qids, all_top5, all_truth = [], [], []
    for tag in ("public", "private"):
        cand = pd.read_csv(os.path.join(
            args.test_dir, f"player_identification_candidates_{tag}.csv"))
        t0 = time.time()
        Xc, okc = parallel_batch_features(cand["sgf_content"].tolist(),
                                          cand["color"].tolist())
        pb = PlayerBaseline().fit(Xc[okc])
        names = sorted(cand["player_id"].unique())
        pid = cand["player_id"].to_numpy()
        G = np.stack([pb.template(Xc[okc & (pid == n)]) for n in names])
        print(f"  {tag:<8} 候選庫 {len(names)} 人 / {len(cand):,} 局，"
              f"{time.time() - t0:.0f} 秒")

        df = pd.read_csv(os.path.join(
            args.test_dir, f"player_identification_test_{tag}.csv"))
        feats = _question_features(df)
        Q = np.stack([pb.template(f) if len(f) else np.zeros(G.shape[1]) for f in feats])
        top5 = top_k_from_similarity(Q @ G.T, names, k=5)
        all_qids.extend(df["question_id"].tolist()); all_top5.extend(top5.tolist())
        all_truth.extend([answers["player"].get(q) for q in df["question_id"]])

    sub = build_player_submission(all_qids, all_top5)
    pools = {}
    for tag, key in (("public", "pub"), ("private", "priv")):
        c = pd.read_csv(os.path.join(
            args.test_dir, f"player_identification_candidates_{tag}.csv"))
        pools[key] = set(c["player_id"].unique())
    path = write_submission(sub, os.path.join(args.out_dir, "submission_player.csv"),
                            "player", public_candidates=pools["pub"],
                            private_candidates=pools["priv"])
    print(f"  已寫出 {path}（{len(sub)} 列，格式驗證通過）", flush=True)
    if all_truth and all(t is not None for t in all_truth):
        ranks = np.array([row.index(t) if t in row else 99
                          for row, t in zip(all_top5, all_truth)])
        print(f"  離線分數 {player_final_score(ranks):.4f}"
              f"   Top-1 {(ranks == 0).mean():.4f}   Top-5 {(ranks < 5).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
