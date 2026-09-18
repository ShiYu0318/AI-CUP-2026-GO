"""提交檔驗證的測試——每一條都對應一種會讓提交作廢的官方規則。"""
import numpy as np
import pandas as pd
import pytest

from aicup.submission import (
    N_PUBLIC, N_PRIVATE, N_TOTAL, TOP_K, SubmissionError,
    build_rank_submission, validate_rank_submission,
    build_player_submission, validate_player_submission, write_submission,
)


def _qids():
    return ([f"pub_q_{i:04d}" for i in range(N_PUBLIC)]
            + [f"priv_q_{i:04d}" for i in range(N_PRIVATE)])


def _rank_df():
    return build_rank_submission(_qids(), np.arange(N_TOTAL) % 10)


def _player_df():
    rows = []
    for q in _qids():
        p = "pub_p_" if q.startswith("pub_") else "priv_p_"
        rows.append([f"{p}{i:04d}" for i in range(TOP_K)])
    return build_player_submission(_qids(), rows)


# ---- 競賽二 ----

def test_valid_rank_submission_passes():
    validate_rank_submission(_rank_df())


def test_rank_submission_maps_indices_to_labels():
    df = build_rank_submission(["pub_q_0001"], [9])
    assert df["pred_rank"].iloc[0] == "6D"


def test_rank_rejects_wrong_row_count():
    df = _rank_df().iloc[:-1]
    with pytest.raises(SubmissionError, match="列數"):
        validate_rank_submission(df)


def test_rank_rejects_missing_private_half():
    """總列數對、但 public/private 比例錯（500+300）——官方要求各 400。"""
    qids = ([f"pub_q_{i:04d}" for i in range(500)]
            + [f"priv_q_{i:04d}" for i in range(300)])
    df = build_rank_submission(qids, np.arange(800) % 10)
    with pytest.raises(SubmissionError, match="題數"):
        validate_rank_submission(df)


def test_rank_rejects_only_public_half():
    """只交 public 那 400 題——缺 private 即整份無效。"""
    df = _rank_df()
    df = df[df["question_id"].str.startswith("pub_")]
    with pytest.raises(SubmissionError, match="列數"):
        validate_rank_submission(df)


def test_rank_rejects_bad_label():
    df = _rank_df()
    df.loc[0, "pred_rank"] = "7D"
    with pytest.raises(SubmissionError, match="不合法等級"):
        validate_rank_submission(df)


def test_rank_rejects_duplicate_question_id():
    df = _rank_df()
    df.loc[1, "question_id"] = df.loc[0, "question_id"]
    with pytest.raises(SubmissionError, match="重複"):
        validate_rank_submission(df)


def test_rank_rejects_bad_columns():
    df = _rank_df().rename(columns={"pred_rank": "rank"})
    with pytest.raises(SubmissionError, match="欄位"):
        validate_rank_submission(df)


def test_rank_rejects_out_of_range_index():
    with pytest.raises(SubmissionError, match="超出"):
        build_rank_submission(["pub_q_0001"], [10])


# ---- 競賽一 ----

def test_valid_player_submission_passes():
    validate_player_submission(_player_df())


def test_player_rejects_duplicate_in_top5():
    df = _player_df()
    df.loc[0, "top_2"] = df.loc[0, "top_1"]
    with pytest.raises(SubmissionError, match="Top-5 有重複"):
        validate_player_submission(df)


def test_player_rejects_cross_pool_contamination():
    """pub_ 題目填 priv_p_ 玩家——官方明文判為無效提交。"""
    df = _player_df()
    df.loc[0, "top_1"] = "priv_p_0000"
    with pytest.raises(SubmissionError, match="跨池汙染"):
        validate_player_submission(df)


def test_player_rejects_unknown_candidate():
    df = _player_df()
    pub = {f"pub_p_{i:04d}" for i in range(TOP_K)}
    priv = {f"priv_p_{i:04d}" for i in range(TOP_K)}
    df.loc[0, "top_1"] = "pub_p_9999"
    with pytest.raises(SubmissionError, match="不在候選庫"):
        validate_player_submission(df, public_candidates=pub, private_candidates=priv)


def test_player_accepts_valid_candidates():
    pub = {f"pub_p_{i:04d}" for i in range(TOP_K)}
    priv = {f"priv_p_{i:04d}" for i in range(TOP_K)}
    validate_player_submission(_player_df(), public_candidates=pub, private_candidates=priv)


def test_player_rejects_wrong_shape():
    with pytest.raises(SubmissionError, match="形狀"):
        build_player_submission(["pub_q_0001"], [["a", "b", "c"]])


def test_player_rejects_nan():
    df = _player_df()
    df.loc[0, "top_3"] = np.nan
    with pytest.raises(SubmissionError, match="缺值"):
        validate_player_submission(df)


# ---- 寫檔 ----

def test_write_submission_refuses_invalid(tmp_path):
    df = _rank_df().iloc[:-1]
    with pytest.raises(SubmissionError):
        write_submission(df, str(tmp_path / "s.csv"), "rank")
    assert not (tmp_path / "s.csv").exists()      # 驗證失敗就不該留下檔案


def test_write_submission_roundtrip(tmp_path):
    p = write_submission(_rank_df(), str(tmp_path / "r.csv"), "rank")
    back = pd.read_csv(p)
    validate_rank_submission(back)
    assert len(back) == N_TOTAL
