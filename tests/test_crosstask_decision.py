"""跨子題增強的決策邏輯。

**這是安全關鍵的判斷**：假設成立時套用值 +0.0586，不成立時套用是 −0.1993
（RESULTS.md §2.20）。比較符號寫反、門檻用錯方向，代價都很大而且不會有錯誤訊息，
只會在 Private LB 上出現。
"""
import pytest

from conftest_paths import scripts_on_path      # noqa: F401

from predict import crosstask_decision

THR = 0.85


def test_applies_when_bijection_is_high():
    ok, why = crosstask_decision(overlap=0.0, bijection=0.9075, threshold=THR)
    assert ok and "套用" in why


def test_refuses_when_bijection_is_low():
    ok, why = crosstask_decision(overlap=0.0, bijection=0.5117, threshold=THR)
    assert not ok and "不套用" in why


def test_refuses_at_partial_overlap():
    """實測 mock（兩子題各自抽樣、部分重疊）是 0.61——必須擋下來。

    這一格最重要：淨增益的零點在「真正匹配的比例 f=77%」，
    而 f=77% 對應的相異率約 0.82。任何低於門檻的值都該拒絕。
    """
    ok, _ = crosstask_decision(overlap=0.126, bijection=0.610, threshold=THR)
    assert not ok


def test_sgf_overlap_overrides_a_low_bijection():
    """棋譜字串直接重疊是 definitive 的證據，比 embedding 強。"""
    ok, why = crosstask_decision(overlap=0.92, bijection=0.30, threshold=THR)
    assert ok and "棋譜字串" in why


def test_small_sgf_overlap_does_not_override():
    ok, _ = crosstask_decision(overlap=0.30, bijection=0.30, threshold=THR)
    assert not ok


@pytest.mark.parametrize("bij,expected", [(0.849, False), (0.85, True), (0.851, True)])
def test_threshold_is_inclusive(bij, expected):
    assert crosstask_decision(0.0, bij, THR)[0] is expected


def test_reason_is_always_a_nonempty_explanation():
    for ov, bj in ((0.0, 0.9), (0.0, 0.4), (0.9, 0.4)):
        ok, why = crosstask_decision(ov, bj, THR)
        assert isinstance(why, str) and len(why) > 10
