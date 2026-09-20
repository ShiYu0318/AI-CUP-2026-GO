"""題目層級後驗快取的測試。

快取是效率工具，但也是最容易產生「安靜地用到錯的東西」的地方——
本專案已經被這類 bug 咬過五次。所以這裡的重點全部放在**失效條件**：
任何一項依賴變了，就必須重算，不可以給出舊結果。
"""
import os

import numpy as np
import pytest

from aicup.oofcache import (CACHE_VERSION, OofCache, file_fingerprint,
                            question_fingerprint)


class _Q:
    """最小的 RankQuestions 替身。"""

    def __init__(self, rows, y=None, pid=None):
        self.game_rows = rows
        n = len(rows)
        self.true_rank_id = np.arange(n) % 10 if y is None else y
        self.player_id = np.array([f"p{i}" for i in range(n)], dtype=object) if pid is None else pid


def _rows(n=3):
    return [np.arange(i, i + 4) for i in range(n)]


def _counter():
    calls = []

    def f():
        calls.append(1)
        return np.log(np.full((len(calls) and 3 or 3, 10), 0.1))
    return f, calls


def test_second_call_hits_cache(tmp_path):
    c = OofCache(str(tmp_path), verbose=False)
    f, calls = _counter()
    q = _Q(_rows())
    a, _, _ = c.get("t", q, f)
    b, _, _ = c.get("t", q, f)
    assert len(calls) == 1 and np.array_equal(a, b)
    assert (c.hits, c.misses) == (1, 1)


def test_labels_and_players_round_trip(tmp_path):
    c = OofCache(str(tmp_path), verbose=False)
    f, _ = _counter()
    q = _Q(_rows(), y=np.array([2, 5, 9]),
           pid=np.array(["x", "y", "z"], dtype=object))
    c.get("t", q, f)
    _, y, pid = c.get("t", q, f)
    assert y.tolist() == [2, 5, 9] and pid.tolist() == ["x", "y", "z"]


def test_different_questions_do_not_collide(tmp_path):
    """只用 seed 或題數當 key 是不夠的——題目內容變了就必須重算。"""
    c = OofCache(str(tmp_path), verbose=False)
    f, calls = _counter()
    c.get("t", _Q(_rows()), f)
    c.get("t", _Q([np.arange(i + 100, i + 104) for i in range(3)]), f)
    assert len(calls) == 2


def test_model_change_invalidates_even_with_same_path(tmp_path):
    """**最危險的情況**：重訓後檔名不變，但內容變了。必須重算。"""
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"version-one")
    c = OofCache(str(tmp_path / "oof"), verbose=False)
    f, calls = _counter()
    q = _Q(_rows())
    c.get("t", q, f, model_paths=[str(ckpt)])
    c.get("t", q, f, model_paths=[str(ckpt)])
    assert len(calls) == 1, "同一個模型不該重算"

    ckpt.write_bytes(b"version-two-retrained")     # 重訓，路徑不變
    c.get("t", q, f, model_paths=[str(ckpt)])
    assert len(calls) == 2, "模型內容變了卻用了舊快取——這正是要防的 bug"


def test_fingerprint_is_content_based_not_mtime(tmp_path):
    """指紋必須來自內容。只碰 mtime 不該改變指紋。"""
    p = tmp_path / "a.bin"
    p.write_bytes(b"same content")
    fp1 = file_fingerprint(str(p))
    os.utime(p, (1, 1))
    assert file_fingerprint(str(p)) == fp1
    p.write_bytes(b"different")
    assert file_fingerprint(str(p)) != fp1


def test_version_bump_invalidates(tmp_path, monkeypatch):
    """後驗語意改變時，升版要能讓所有舊快取失效。"""
    import aicup.oofcache as mod

    c = OofCache(str(tmp_path), verbose=False)
    f, calls = _counter()
    q = _Q(_rows())
    c.get("t", q, f)
    monkeypatch.setattr(mod, "CACHE_VERSION", CACHE_VERSION + 1)
    c.get("t", q, f)
    assert len(calls) == 2


def test_corrupt_cache_recomputes_instead_of_raising(tmp_path):
    """壞掉的快取檔不該讓整個實驗失敗。"""
    c = OofCache(str(tmp_path), verbose=False)
    f, calls = _counter()
    q = _Q(_rows())
    c.get("t", q, f)
    bad = [p for p in os.listdir(tmp_path) if p.endswith(".npz")][0]
    (tmp_path / bad).write_bytes(b"not an npz")
    c.get("t", q, f)
    assert len(calls) == 2


def test_wrong_shape_is_rejected(tmp_path):
    """後驗列數與題數不符一定是接錯了，必須當場拋錯而不是存進快取。"""
    c = OofCache(str(tmp_path), verbose=False)
    q = _Q(_rows(3))
    with pytest.raises(ValueError, match="題數"):
        c.get("t", q, lambda: np.zeros((5, 10)))
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".npz")]


def test_no_tmp_files_left_behind(tmp_path):
    c = OofCache(str(tmp_path), verbose=False)
    f, _ = _counter()
    c.get("t", _Q(_rows()), f)
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]


def test_question_fingerprint_is_order_sensitive():
    a = [np.array([1, 2]), np.array([3, 4])]
    b = [np.array([3, 4]), np.array([1, 2])]
    assert question_fingerprint(a) != question_fingerprint(b)


# ---------------------------------------------------------------------------
# 融合評估器的核心：分數計算路徑
# ---------------------------------------------------------------------------

def test_blend_order_matches_prediction_pipeline():
    """`eval_blend.score_blend` 的融合順序必須與 predict.py 一致，
    否則快取加速換來的是「快速得到錯誤答案」。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from eval_blend import score_blend

    from aicup.question_model import blend_log_posteriors
    from aicup.scoring import (apply_temperature, apply_thresholds,
                               expected_level, fit_temperature,
                               optimise_thresholds, rank_final_score)

    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 10, n)

    def post(strength):
        lg = rng.normal(size=(n, 10)) * 0.5
        lg[np.arange(n), y] += strength
        lg -= lg.max(1, keepdims=True)
        e = np.exp(lg)
        return np.log(e / e.sum(1, keepdims=True))

    parts = {"game": post(2.0), "question": post(1.2), "seq": post(1.6)}
    cal, sco = np.arange(n // 2), np.arange(n // 2, n)
    got = score_blend(parts, {"question": 0.2, "seq": 0.7}, y, cal, sco)

    lo = blend_log_posteriors(parts["game"], parts["question"], w=0.2)
    lo = blend_log_posteriors(lo, parts["seq"], w=0.7)
    T = fit_temperature(lo[cal], y[cal])
    ev = expected_level(apply_temperature(lo, T))
    th = optimise_thresholds(ev[cal], y[cal])
    want = rank_final_score(apply_thresholds(ev[sco], th), y[sco])
    assert got == pytest.approx(want)


def test_blend_with_zero_weight_ignores_component():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from eval_blend import score_blend

    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 10, n)
    lg = rng.normal(size=(n, 10))
    lg -= lg.max(1, keepdims=True)
    g = np.log(np.exp(lg) / np.exp(lg).sum(1, keepdims=True))
    junk = np.log(np.full((n, 10), 0.1))
    cal, sco = np.arange(n // 2), np.arange(n // 2, n)
    assert (score_blend({"game": g, "seq": junk}, {"seq": 0.0}, y, cal, sco)
            == score_blend({"game": g}, {}, y, cal, sco))
