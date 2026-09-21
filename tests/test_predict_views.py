"""`predict.py` 從 checkpoint 自己的 args 推導序列視角。

這條路徑錯了**不會報錯**——只會讓特徵欄錯位、分數悄悄變差，
所以要用測試釘住。之前 24 維的舊變體與 48 維的棋形變體不能並存，
就是因為 `predict.py` 對所有變體共用同一組序列庫。
"""
import pytest

from conftest_paths import scripts_on_path      # noqa: F401

from predict import SEQ_VIEWS, _store_paths, _views_of


def test_default_two_views():
    v = _views_of({"sequences": "cache/sequences.npy",
                   "extra_sequences": ["cache/sequences_opp.npy"]})
    assert len(v) == 2
    assert v[0] == ("move", False, 160)
    assert v[1] == ("move", True, 160)


def test_shape_variant_has_four_views():
    v = _views_of({"sequences": "cache/sequences.npy",
                   "extra_sequences": ["cache/sequences_opp.npy",
                                       "cache/seqshape.npy",
                                       "cache/seqshape_opp.npy"]})
    assert len(v) == 4
    assert [k for k, _, _ in v] == ["move", "move", "shape", "shape"]


def test_order_is_preserved():
    """順序**必須**與訓練時一致，否則特徵欄錯位。"""
    a = _views_of({"sequences": "cache/sequences.npy",
                   "extra_sequences": ["cache/seqshape.npy",
                                       "cache/sequences_opp.npy"]})
    b = _views_of({"sequences": "cache/sequences.npy",
                   "extra_sequences": ["cache/sequences_opp.npy",
                                       "cache/seqshape.npy"]})
    assert a != b


def test_240_variant_keeps_its_window():
    v = _views_of({"sequences": "cache/sequences240.npy",
                   "extra_sequences": ["cache/sequences240_opp.npy"]})
    assert all(mm == 240 for _, _, mm in v)


def test_unknown_store_raises_loudly():
    with pytest.raises(ValueError, match="不認得的序列庫"):
        _views_of({"sequences": "cache/something_new.npy", "extra_sequences": []})


def test_store_paths_roundtrip():
    for args in ({"sequences": "cache/sequences.npy",
                  "extra_sequences": ["cache/sequences_opp.npy"]},
                 {"sequences": "cache/sequences.npy",
                  "extra_sequences": ["cache/sequences_opp.npy",
                                      "cache/seqshape.npy",
                                      "cache/seqshape_opp.npy"]}):
        v = _views_of(args)
        assert _store_paths(v) == [args["sequences"]] + args["extra_sequences"]


def test_view_table_has_no_duplicate_specs():
    """兩個檔名對到同一個 spec 的話，`_store_paths` 的反查會對錯庫。"""
    assert len(set(SEQ_VIEWS.values())) == len(SEQ_VIEWS)


def test_quality_views_are_recognised():
    """品質視角是 **8 維**，不是 12——`_nfeat` 不能再假設每個視角同寬。"""
    v = _views_of({"sequences": "cache/sequences.npy",
                   "extra_sequences": ["cache/sequences_opp.npy",
                                       "cache/seqshape.npy",
                                       "cache/seqshape_opp.npy",
                                       "cache/seqqual.npy",
                                       "cache/seqqual_opp.npy"]})
    assert [k for k, _, _ in v] == ["move", "move", "shape", "shape",
                                    "quality", "quality"]
    assert _store_paths(v)[-1] == "cache/seqqual_opp.npy"
