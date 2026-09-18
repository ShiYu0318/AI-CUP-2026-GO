"""用訓練資料合成一份「格式與官方一致」的模擬測試集。

目的：測試集 2026/11/04 才釋出，但推論流程、檔案讀取、提交格式驗證
都可以現在就演練完畢。當天只要把路徑換掉即可，不必在有時間壓力時除錯。

產出（全部使用 held-out 玩家，與訓練互斥）：
    mock_tests/rank_prediction_test_public.csv      400 題
    mock_tests/rank_prediction_test_private.csv     400 題
    mock_tests/player_identification_test_public.csv
    mock_tests/player_identification_test_private.csv
    mock_tests/player_identification_candidates_public.csv    400 人 x 100 局
    mock_tests/player_identification_candidates_private.csv
    mock_tests/_answers.json                        正解（官方不會給，僅供離線評分）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup import ID_TO_RANK                                  # noqa: E402
from aicup.data import load_metadata, make_player_split, load_sgf   # noqa: E402
from aicup.benchmark import build_rank_questions, build_player_pool, MAX_GAMES  # noqa: E402

OUT = "mock_tests"


def _bulk_sgf(meta, rows) -> dict:
    """一次取回所有需要的 sgf_content。

    每題各呼叫一次 load_sgf 會讓每題都重讀一個 130 MB 的 CSV——
    400 題就是 400 次全檔讀取。這裡先把所有列合併成一次查詢。
    """
    rows = np.unique(np.concatenate([np.asarray(r) for r in rows]))
    sub = meta.iloc[rows]
    sgf = load_sgf(sub)
    return dict(zip(rows.tolist(), sgf.tolist()))


def _write_rank_split(meta, split, prefix: str, seed: int, n: int, answers: dict):
    q = build_rank_questions(meta, split, n_per_player=1, seed=seed, max_questions=n)
    lookup = _bulk_sgf(meta, q.game_rows)
    rows = []
    for i, (game_rows, y) in enumerate(zip(q.game_rows, q.true_rank_id), start=1):
        qid = f"{prefix}q_{i:04d}"
        sub = meta.iloc[game_rows]
        sgfs = [lookup[int(r)] for r in game_rows]
        rec = {"question_id": qid, "num_games": len(sgfs)}
        for j in range(MAX_GAMES):
            rec[f"sgf_{j + 1}"] = sgfs[j] if j < len(sgfs) else ""
            rec[f"color_{j + 1}"] = sub["color"].iloc[j] if j < len(sgfs) else ""
        rows.append(rec)
        answers[qid] = ID_TO_RANK[int(y)]
    return pd.DataFrame(rows)


def _write_player_split(meta, split, prefix: str, seed: int, answers: dict):
    pool = build_player_pool(meta, split, composition="uniform", seed=seed)
    name = {p: f"{prefix}p_{i:04d}" for i, p in enumerate(pool.players)}
    lookup = _bulk_sgf(meta, [pool.gallery_rows[p] for p in pool.players]
                             + [pool.query_rows[p] for p in pool.players])

    cand_rows = []
    for p in pool.players:
        sub = meta.iloc[pool.gallery_rows[p]]
        for r, color in zip(pool.gallery_rows[p], sub["color"]):
            cand_rows.append({"game_id": f"{prefix}g_{len(cand_rows) + 1:06d}",
                              "player_id": name[p], "sgf_content": lookup[int(r)],
                              "color": color})

    test_rows = []
    for i, p in enumerate(pool.players, start=1):
        qid = f"{prefix}q_{i:04d}"
        sub = meta.iloc[pool.query_rows[p]]
        sgfs = [lookup[int(r)] for r in pool.query_rows[p]]
        rec = {"question_id": qid, "num_games": len(sgfs)}
        for j in range(MAX_GAMES):
            rec[f"sgf_{j + 1}"] = sgfs[j] if j < len(sgfs) else ""
            rec[f"color_{j + 1}"] = sub["color"].iloc[j] if j < len(sgfs) else ""
        test_rows.append(rec)
        answers[qid] = name[p]
    return pd.DataFrame(test_rows), pd.DataFrame(cand_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-questions", type=int, default=400)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = load_metadata()
    # 用另一個 seed 的切分，確保模擬測試集的玩家與主驗證集也不重疊
    split = make_player_split(meta, val_frac=0.20, seed=42)
    answers: dict = {}

    for prefix, seed in (("pub_", 101), ("priv_", 202)):
        df = _write_rank_split(meta, split, prefix, seed, args.n_questions, answers)
        tag = "public" if prefix == "pub_" else "private"
        df.to_csv(f"{args.out}/rank_prediction_test_{tag}.csv", index=False)
        print(f"  rank_prediction_test_{tag}.csv  {len(df)} 題")

        t, c = _write_player_split(meta, split, prefix, seed, answers)
        t.to_csv(f"{args.out}/player_identification_test_{tag}.csv", index=False)
        c.to_csv(f"{args.out}/player_identification_candidates_{tag}.csv", index=False)
        print(f"  player_identification_test_{tag}.csv  {len(t)} 題"
              f" / candidates {len(c):,} 局")

    with open(f"{args.out}/_answers.json", "w") as f:
        json.dump(answers, f)
    print(f"\n正解已寫出（官方不會提供，僅供離線評分）：{args.out}/_answers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
