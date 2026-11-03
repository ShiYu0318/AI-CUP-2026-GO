import os
"""測試題目檔的解析——這是 11/04 當天最可能出錯的地方。

官方題目檔有 sgf_1..sgf_20 共 20 欄，但只有前 num_games 欄有值，
其餘是空字串。若把空欄當成棋局，特徵矩陣會混入垃圾列而毫無警訊。
"""
import importlib.util
import pathlib
import sys

from aicup.data import TRAINING_DIR
import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "predict", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "predict.py")
predict = importlib.util.module_from_spec(_spec)
sys.modules["predict"] = predict
_spec.loader.exec_module(predict)


@pytest.fixture(scope="module")
def real_sgfs():
    d = pd.read_csv(os.path.join(TRAINING_DIR, "train_5D.csv"), nrows=6, usecols=["sgf_content", "color"])
    return d["sgf_content"].tolist(), d["color"].tolist()


def _make_question_df(rows):
    """rows: [(qid, [(sgf, color), ...]), ...]"""
    recs = []
    for qid, games in rows:
        rec = {"question_id": qid, "num_games": len(games)}
        for j in range(20):
            rec[f"sgf_{j + 1}"] = games[j][0] if j < len(games) else ""
            rec[f"color_{j + 1}"] = games[j][1] if j < len(games) else ""
        recs.append(rec)
    return pd.DataFrame(recs)


def test_ignores_empty_sgf_columns(real_sgfs):
    sgfs, colors = real_sgfs
    df = _make_question_df([("pub_q_0001", list(zip(sgfs[:3], colors[:3])))])
    feats = predict._question_features(df)
    assert len(feats) == 1
    assert feats[0].shape[0] == 3          # 只有 3 局，不是 20


def test_variable_question_sizes(real_sgfs):
    sgfs, colors = real_sgfs
    df = _make_question_df([
        ("pub_q_0001", list(zip(sgfs[:2], colors[:2]))),
        ("pub_q_0002", list(zip(sgfs[:5], colors[:5]))),
    ])
    feats = predict._question_features(df)
    assert [f.shape[0] for f in feats] == [2, 5]


def test_handles_nan_instead_of_empty_string(real_sgfs):
    """pandas 讀空欄位時給的是 NaN 而非空字串，兩種都要能處理。"""
    sgfs, colors = real_sgfs
    df = _make_question_df([("pub_q_0001", list(zip(sgfs[:2], colors[:2])))])
    for j in range(3, 21):                       # 直接寫入 NaN，避免 replace 的降型警告
        df[f"sgf_{j}"] = np.nan
        df[f"color_{j}"] = np.nan
    feats = predict._question_features(df)
    assert feats[0].shape[0] == 2


def test_question_with_all_unparseable_games_gives_empty_block():
    df = _make_question_df([("pub_q_0001", [("garbage", "B"), ("(;SZ[19])", "W")])])
    feats = predict._question_features(df)
    assert feats[0].shape[0] == 0          # 呼叫端必須能處理空題目


def test_feature_width_is_consistent(real_sgfs):
    from aicup.features import N_FEATURES
    sgfs, colors = real_sgfs
    df = _make_question_df([("pub_q_0001", list(zip(sgfs[:4], colors[:4])))])
    assert predict._question_features(df)[0].shape[1] == N_FEATURES
