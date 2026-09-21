"""度量學習的特徵欄位選擇，必須容得下**寬度不同**的額外特徵庫。

`identity_feature_columns` 的索引是針對 589 維手工特徵定義的。
舊版對每一份額外特徵庫套用同一組相對索引——遇到 48 維的棋形統計會越界。
越界會拋 IndexError（還算幸運），但若某天有一份庫剛好夠寬，
就會**靜默地取到錯的欄位**，所以用測試釘住。
"""
import numpy as np
import pytest

from aicup.feature_metric import identity_feature_columns


def select_columns(widths, ident):
    """複製 `train_feature_metric.py` 的欄位選擇邏輯。"""
    ident = np.asarray(ident)
    n_base = widths[0]
    pieces, off = [], 0
    for w in widths:
        pieces.append(ident + off if w == n_base else np.arange(w) + off)
        off += w
    return np.concatenate(pieces)


@pytest.fixture
def ident():
    return np.asarray(identity_feature_columns(drop_global_position=True))


def test_single_store_matches_identity_columns(ident):
    assert np.array_equal(select_columns([589], ident), ident)


def test_two_equal_stores_repeat_the_subset(ident):
    got = select_columns([589, 589], ident)
    assert np.array_equal(got, np.concatenate([ident, ident + 589]))


def test_narrow_extra_store_takes_all_of_its_columns(ident):
    got = select_columns([589, 589, 48, 48], ident)
    assert len(got) == 2 * len(ident) + 96
    assert got.max() < 589 + 589 + 48 + 48


@pytest.mark.parametrize("widths", [[589, 48], [589, 589, 48], [589, 12, 48, 589]])
def test_indices_always_in_range(widths, ident):
    got = select_columns(widths, ident)
    assert got.min() >= 0
    assert got.max() < sum(widths)
    assert len(set(got.tolist())) == len(got), "欄位不可重複"
