"""逐特徵回報「與棋力的相關性」與「對玩家的區辨力」。

新增特徵後先跑這支，再決定要不要花時間重建整份特徵庫與重訓模型。
兩個指標的意義不同，一個特徵可能只對其中一個子題有用：

  rank_corr     各等級平均值與棋力索引（D=0 … 6D=9）的相關係數。
                |corr| > 0.7 算強訊號。這是競賽二在乎的。
  player_ratio  玩家「之間」的變異 / 玩家「之內」的變異。
                比值越大代表這個特徵越能穩定區分不同的人。這是競賽一在乎的。

用法：
    python3 scripts/feature_signal_report.py [--top 30]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from aicup import LEVELS                                   # noqa: E402
from aicup.data import load_metadata, make_player_split    # noqa: E402
from aicup.features import FEATURE_NAMES                   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--players", type=int, default=600, help="計算玩家區辨力用的玩家數")
    ap.add_argument("--min-games", type=int, default=30)
    args = ap.parse_args()

    X = np.load("cache/features.npy", mmap_mode="r")
    ok = np.load("cache/features_ok.npy")
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    vm = split.mask(meta)
    rank_id = meta["rank_id"].to_numpy()
    rng = np.random.default_rng(0)

    # --- 與棋力的相關性（用驗證玩家，避免與訓練資料混淆）---
    per_rank = []
    for lv in range(len(LEVELS)):
        rows = np.flatnonzero(vm & ok & (rank_id == lv))
        rows = rng.choice(rows, size=min(4000, len(rows)), replace=False)
        per_rank.append(np.asarray(X[rows]).mean(axis=0))
    M = np.stack(per_rank)
    idx = np.arange(len(LEVELS))
    rank_corr = np.array([
        0.0 if np.std(M[:, j]) < 1e-12 else np.corrcoef(idx, M[:, j])[0, 1]
        for j in range(M.shape[1])])

    # --- 對玩家的區辨力 ---
    sub = meta[vm & ok]
    counts = sub["player_id"].value_counts()
    players = counts[counts >= args.min_games].index.to_numpy()
    rng.shuffle(players)
    players = players[: args.players]
    means, within = [], []
    for pid in players:
        rows = sub.index[sub["player_id"] == pid].to_numpy()
        # ⚠️ 必須隨機抽樣。官方 CSV 依執色排序（前 5 萬列全 B、後 5 萬列全 W），
        # 直接取前 100 列會拿到幾乎單一執色的樣本，讓 is_black 之類的特徵
        # 憑空得到很高的「玩家區辨力」。這個腳本第一版就踩了這個坑。
        if len(rows) > 100:
            rows = rng.choice(rows, size=100, replace=False)
        A = np.asarray(X[rows])
        means.append(A.mean(axis=0))
        within.append(A.var(axis=0))
    between = np.stack(means).var(axis=0)
    within = np.stack(within).mean(axis=0)
    player_ratio = between / np.maximum(within, 1e-12)

    df = pd.DataFrame({"feature": FEATURE_NAMES,
                       "rank_corr": rank_corr,
                       "player_ratio": player_ratio})
    df["abs_corr"] = df["rank_corr"].abs()

    pd.set_option("display.width", 200)
    print(f"共 {len(FEATURE_NAMES)} 個特徵，"
          f"玩家 {len(players)} 位（每人取最多 100 局）\n", flush=True)
    print("=== 競賽二最有用：|與棋力的相關係數| 最高 ===", flush=True)
    print(df.nlargest(args.top, "abs_corr")[["feature", "rank_corr", "player_ratio"]]
          .to_string(index=False, float_format=lambda v: "%.3f" % v), flush=True)
    print("\n=== 競賽一最有用：玩家間/玩家內變異比 最高 ===", flush=True)
    print(df.nlargest(args.top, "player_ratio")[["feature", "player_ratio", "rank_corr"]]
          .to_string(index=False, float_format=lambda v: "%.3f" % v), flush=True)

    dead = df[(df["abs_corr"] < 0.15) & (df["player_ratio"] < 0.05)]
    print(f"\n=== 兩邊都沒用的特徵（{len(dead)} 個）===", flush=True)
    print(", ".join(dead["feature"].tolist()) or "（無）", flush=True)

    df.drop(columns=["abs_corr"]).to_csv("cache/feature_signal.csv", index=False)
    print("\n完整表格已寫出：cache/feature_signal.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
