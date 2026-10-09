"""KataGo HumanSL 管線中不需要引擎的部分。

座標或視角一旦弄反，KataGo 不會報錯——它只會乖乖地對錯的點、錯的一方打分，
特徵安靜地變成雜訊。這些測試把那幾個轉換釘死。
"""
import json

import numpy as np
import pytest

from aicup.humansl import (FEATURE_NAMES, N_FEATURES, N_PROFILES, HumanSLStore, features,
                           game_key, plan_game, queries, question_loglik)

SGF = "(;SZ[19];B[pd];W[dp];B[aa];W[sa];B[ss];W[as]" + "".join(
    f";{'B' if i % 2 == 0 else 'W'}[{chr(97 + (i * 3) % 19)}{chr(97 + (i * 7 + 5) % 19)}]"
    for i in range(60)) + ")"


def test_coordinates_follow_gtp_convention():
    p = plan_game(SGF, "B")
    # SGF 'pd' = 第 15 欄、第 3 列（從上數）→ GTP 'Q16'（跳過 I）
    assert p["moves"][0] == ["B", "Q16"]
    assert p["moves"][1] == ["W", "D4"]
    assert p["moves"][2] == ["B", "A19"]          # 左上角
    assert p["moves"][3] == ["W", "T19"]          # 右上角
    assert p["moves"][4] == ["B", "T1"]           # 右下角
    assert p["moves"][5] == ["W", "A1"]           # 左下角


def test_policy_index_is_row_major_from_top_left():
    p = plan_game(SGF, "B")
    assert p["target"][0] == 3 * 19 + 15          # pd
    assert p["target"][2] == 0                     # aa = A19
    assert p["target"][4] == 18 * 19 + 18         # ss = T1


def test_only_own_moves_are_scored():
    for color, parity in (("B", 0), ("W", 1)):
        p = plan_game(SGF, color, n_moves=10)
        assert p["turns"] and all(t % 2 == parity for t in p["turns"])
        assert len(p["turns"]) <= 10


def test_profile0_also_asks_for_the_position_after_each_move():
    p = plan_game(SGF, "B", n_moves=5)
    qs = queries("k", p)
    assert len(qs) == N_PROFILES
    assert set(qs[0]["analyzeTurns"]) == set(p["turns"]) | {t + 1 for t in p["turns"]}
    assert all(q["analyzeTurns"] == p["turns"] for q in qs[1:])
    assert all(q["maxVisits"] == 1 and q["includePolicy"] for q in qs)


def test_score_loss_is_from_our_perspective():
    """scoreLead 是黑方視角。白方下了一手讓黑方領先變多 = 白方虧了 → 損失為正。"""
    for color, before, after, want in (("B", 5.0, 2.0, 3.0), ("W", 5.0, 8.0, 3.0)):
        p = plan_game(SGF, color, n_moves=1)
        t = p["turns"][0]
        f = features(p, np.zeros((1, N_PROFILES)), np.zeros((1, N_PROFILES)), {t: before, t + 1: after})
        assert f[FEATURE_NAMES.index("loss_mean")] == pytest.approx(want)


def test_feature_vector_shape_and_counts():
    p = plan_game(SGF, "W", n_moves=7)
    f = features(p, np.full((len(p["turns"]), N_PROFILES), -2.0), np.ones((len(p["turns"]), N_PROFILES)), {})
    assert f.shape == (N_FEATURES,)
    assert f[FEATURE_NAMES.index("n_used")] == len(p["turns"])
    assert f[FEATURE_NAMES.index("n_own")] == p["n_own"]


def test_question_loglik_weights_by_moves_and_skips_missing():
    a = np.zeros(N_FEATURES, dtype=np.float32); a[:N_PROFILES] = -1.0; a[FEATURE_NAMES.index("n_used")] = 10
    b = a.copy(); b[:N_PROFILES] = -2.0; b[FEATURE_NAMES.index("n_used")] = 5
    bad = a.copy(); bad[0] = np.nan
    assert np.allclose(question_loglik([a, b, None, bad]), -10 * 1 - 5 * 2)
    assert np.allclose(question_loglik([None]), 0)


def test_key_depends_on_colour():
    assert game_key(SGF, "B") != game_key(SGF, "W")
    assert game_key(SGF, "b") == game_key(SGF, "B")


def test_store_resumes_and_ignores_a_truncated_last_line(tmp_path):
    path = tmp_path / "s.jsonl"
    s = HumanSLStore(str(path))
    with open(path, "a") as fh:
        s.append("x", np.arange(N_FEATURES, dtype=np.float32), fh)
        fh.write('{"k": "y", "f": [1, 2')            # 中斷時寫到一半
    s2 = HumanSLStore(str(path))
    assert "x" in s2 and "y" not in s2 and len(s2) == 1
    assert np.allclose(s2.get("x"), np.arange(N_FEATURES))
