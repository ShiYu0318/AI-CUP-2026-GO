"""提交檔的產生與驗證。

官方規則中會讓整份提交作廢（0 分）的情況：
  - 缺 public 或 private 任一份的題目
  - 競賽一的 Top-5 有重複
  - 競賽一 pub_ 開頭的題目填了 priv_p_ 開頭的玩家（跨池汙染）
  - 填了不在該池候選庫裡的玩家

因為「最終分數採最後一次上傳」而非最佳分數，一次格式錯誤就可能毀掉整季，
所以這裡把檢查寫成會丟例外的程式碼，而不是靠人工核對。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ID_TO_RANK, LEVELS

N_PUBLIC = N_PRIVATE = 400
N_TOTAL = N_PUBLIC + N_PRIVATE
TOP_K = 5

RANK_COLUMNS = ["question_id", "pred_rank"]
PLAYER_COLUMNS = ["question_id"] + [f"top_{i}" for i in range(1, TOP_K + 1)]


class SubmissionError(ValueError):
    """提交檔不符合官方格式。"""


def _check_question_ids(qids: pd.Series, *, expect_counts: bool) -> None:
    if qids.duplicated().any():
        dup = qids[qids.duplicated()].tolist()[:5]
        raise SubmissionError(f"question_id 重複：{dup}")
    bad = qids[~qids.str.startswith(("pub_", "priv_"))]
    if len(bad):
        raise SubmissionError(f"question_id 前綴必須是 pub_ 或 priv_，例如：{bad.tolist()[:3]}")
    if expect_counts:
        n_pub = int(qids.str.startswith("pub_").sum())
        n_priv = int(qids.str.startswith("priv_").sum())
        if n_pub != N_PUBLIC or n_priv != N_PRIVATE:
            raise SubmissionError(
                f"public/private 題數必須各為 {N_PUBLIC}，實得 {n_pub}/{n_priv}。"
                " 缺任一份即整份提交無效。")


def build_rank_submission(question_ids, pred_rank_ids) -> pd.DataFrame:
    """由等級索引（0..9）產生競賽二的提交表。"""
    ids = np.asarray(pred_rank_ids)
    if ids.min() < 0 or ids.max() >= len(LEVELS):
        raise SubmissionError(f"等級索引超出 0..{len(LEVELS) - 1}")
    return pd.DataFrame({
        "question_id": list(question_ids),
        "pred_rank": [ID_TO_RANK[int(i)] for i in ids],
    })


def validate_rank_submission(df: pd.DataFrame, *, strict_counts: bool = True) -> None:
    if list(df.columns) != RANK_COLUMNS:
        raise SubmissionError(f"欄位必須是 {RANK_COLUMNS}，實得 {list(df.columns)}")
    if df["pred_rank"].isna().any():
        raise SubmissionError("pred_rank 有缺值")
    bad = set(df["pred_rank"]) - set(LEVELS)
    if bad:
        raise SubmissionError(f"pred_rank 出現不合法等級：{sorted(bad)}")
    if strict_counts and len(df) != N_TOTAL:
        raise SubmissionError(f"列數必須是 {N_TOTAL}（public {N_PUBLIC} + private {N_PRIVATE}），實得 {len(df)}")
    _check_question_ids(df["question_id"], expect_counts=strict_counts)


def build_player_submission(question_ids, top5) -> pd.DataFrame:
    """由 Top-5 名單產生競賽一的提交表。top5 形狀為 (n, 5)。"""
    arr = np.asarray(top5, dtype=object)
    if arr.ndim != 2 or arr.shape[1] != TOP_K:
        raise SubmissionError(f"top5 形狀必須是 (n, {TOP_K})，實得 {arr.shape}")
    out = {"question_id": list(question_ids)}
    for i in range(TOP_K):
        out[f"top_{i + 1}"] = arr[:, i].tolist()
    return pd.DataFrame(out)


def validate_player_submission(df: pd.DataFrame, *,
                               public_candidates: set | None = None,
                               private_candidates: set | None = None,
                               strict_counts: bool = True) -> None:
    if list(df.columns) != PLAYER_COLUMNS:
        raise SubmissionError(f"欄位必須是 {PLAYER_COLUMNS}，實得 {list(df.columns)}")
    if strict_counts and len(df) != N_TOTAL:
        raise SubmissionError(f"列數必須是 {N_TOTAL}（public {N_PUBLIC} + private {N_PRIVATE}），實得 {len(df)}")
    _check_question_ids(df["question_id"], expect_counts=strict_counts)

    top_cols = PLAYER_COLUMNS[1:]
    if df[top_cols].isna().any().any():
        raise SubmissionError("Top-5 欄位有缺值")

    picks = df[top_cols].to_numpy(dtype=object)
    for qid, row in zip(df["question_id"], picks):
        if len(set(row.tolist())) != TOP_K:
            raise SubmissionError(f"{qid} 的 Top-5 有重複：{row.tolist()}")
        prefix = "pub_p_" if qid.startswith("pub_") else "priv_p_"
        wrong = [x for x in row if not str(x).startswith(prefix)]
        if wrong:
            raise SubmissionError(f"{qid} 跨池汙染，應為 {prefix} 開頭，實得 {wrong}")

    if public_candidates is not None or private_candidates is not None:
        for qid, row in zip(df["question_id"], picks):
            pool = public_candidates if qid.startswith("pub_") else private_candidates
            if pool is None:
                continue
            missing = [x for x in row if x not in pool]
            if missing:
                raise SubmissionError(f"{qid} 填入不在候選庫中的玩家：{missing}")


def write_submission(df: pd.DataFrame, path: str, kind: str, **kwargs) -> str:
    """驗證通過才寫檔。kind 為 'rank' 或 'player'。"""
    if kind == "rank":
        validate_rank_submission(df, **kwargs)
    elif kind == "player":
        validate_player_submission(df, **kwargs)
    else:
        raise ValueError(kind)
    df.to_csv(path, index=False)
    return path
