"""學習式融合的測試。

Stacker 取代固定權重的幾何平均，實測 +0.0088（8/8 勝，t=5.00）。
它是在**校準題目上現場擬合**的，所以最大的風險是欄位錯位與洩漏。
"""
import numpy as np
import pytest

from aicup.stacker import Stacker, stack_features


def _parts(n=300, seed=0, levels=10):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, levels, n)

    def post(strength):
        lg = rng.normal(size=(n, levels)) * 0.5
        lg[np.arange(n), y] += strength
        lg -= lg.max(1, keepdims=True)
        e = np.exp(lg)
        return np.log(e / e.sum(1, keepdims=True))

    return {"game": post(2.0), "question": post(1.0), "seq": post(1.5)}, y, rng.integers(5, 21, n)


def test_feature_width():
    """每個成分 13 欄（10 log 後驗 + 期望等級 + 信心 + 熵）＋ 1 欄 log 局數。"""
    parts, _, sizes = _parts()
    assert stack_features(parts, sizes).shape == (300, 3 * 13 + 1)


def test_feature_column_order_is_deterministic():
    """訓練與推論必須用同一個欄位順序，否則係數會對到錯的欄位。
    dict 的插入順序不可靠，所以實作用 sorted(parts) 固定。"""
    parts, _, sizes = _parts()
    shuffled = {k: parts[k] for k in ["seq", "game", "question"]}
    assert np.array_equal(stack_features(parts, sizes), stack_features(shuffled, sizes))


def test_rejects_mismatched_lengths():
    parts, _, sizes = _parts()
    parts["seq"] = parts["seq"][:100]
    with pytest.raises(ValueError, match="題"):
        stack_features(parts, sizes)


def test_rejects_empty():
    with pytest.raises(ValueError):
        stack_features({}, [1, 2, 3])


def test_beats_best_single_component_on_holdout():
    """學習式融合必須勝過最強的單一成分，否則沒有存在意義。"""
    from aicup.scoring import expected_level

    parts, y, sizes = _parts(n=800, seed=1)
    X = stack_features(parts, sizes)
    tr, te = np.arange(400), np.arange(400, 800)
    ev = Stacker().fit(X[tr], y[tr]).expected_level(X[te])
    err = np.abs(ev - y[te]).mean()
    best_single = min(
        np.abs(expected_level(np.exp(parts[k]) /
                              np.exp(parts[k]).sum(1, keepdims=True))[te] - y[te]).mean()
        for k in parts)
    assert err < best_single


def test_missing_class_columns_are_zero_filled():
    """校準集若少了某個等級，predict_proba 不會有那一欄；
    直接拿去算期望等級會對錯欄位加權。"""
    parts, y, sizes = _parts(n=400, seed=2)
    y = np.where(y == 9, 8, y)                 # 校準集完全沒有等級 9
    X = stack_features(parts, sizes)
    st = Stacker().fit(X, y)
    ev = st.expected_level(X)
    assert len(st.model.classes_) == 9
    assert np.isfinite(ev).all() and 0 <= ev.min() and ev.max() <= 9


def test_predict_before_fit_raises():
    parts, _, sizes = _parts()
    with pytest.raises(RuntimeError, match="fit"):
        Stacker().expected_level(stack_features(parts, sizes))


def test_fit_rejects_length_mismatch():
    parts, y, sizes = _parts()
    with pytest.raises(ValueError):
        Stacker().fit(stack_features(parts, sizes), y[:10])


def test_expected_level_within_range():
    parts, y, sizes = _parts()
    X = stack_features(parts, sizes)
    ev = Stacker().fit(X, y).expected_level(X)
    assert ev.min() >= 0.0 and ev.max() <= 9.0


def test_deterministic():
    parts, y, sizes = _parts()
    X = stack_features(parts, sizes)
    a = Stacker().fit(X, y).expected_level(X)
    b = Stacker().fit(X, y).expected_level(X)
    assert np.allclose(a, b)


def test_part_mismatch_is_rejected():
    """推論時的成分集合與擬合時不同 → 特徵欄位整個錯位，必須拋錯。

    這正是本專案一再被咬的那類 bug：形狀可能剛好對得上，
    但每一欄的語意都跑掉了，而且不會有任何徵兆。
    """
    parts, y, sizes = _parts()
    st = Stacker().fit(stack_features(parts, sizes), y, parts=parts)
    st.check_parts(parts)                          # 一致 → 不該拋錯
    renamed = {"game": parts["game"], "question": parts["question"],
               "cnn": parts["seq"]}                # 同樣三個成分，但換了一個來源
    with pytest.raises(ValueError, match="成分"):
        st.check_parts(renamed)


def test_check_parts_is_noop_when_not_recorded():
    """舊的（未記錄成分的）stacker 不該因為驗證而壞掉。"""
    parts, y, sizes = _parts()
    Stacker().fit(stack_features(parts, sizes), y).check_parts({"anything": 1})
