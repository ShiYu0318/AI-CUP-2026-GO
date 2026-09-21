"""自駕迴圈的 state 機制。

釘住兩個**靜默失敗**型的 bug：

1. gate 版本變了卻沿用舊基準 → 下一個實驗拿新尺的分數比舊尺的基準，假性通過。
2. production 從單一浮點數改成分軌 dict 之後，
   判斷式若還是 `st["production"] is None`，空 dict 不是 None，
   基準量測會被**跳過**，然後在 gate 時 KeyError——
   2026-09-21 真的發生過，而且是因為一次字串替換靜默沒生效。
"""
import json

import pytest

from conftest_paths import scripts_on_path      # noqa: F401

import auto_loop


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    f = tmp_path / "state.json"
    monkeypatch.setattr(auto_loop, "STATE", str(f))
    return f


def test_fresh_state_has_dict_production(state_file):
    st = auto_loop.load_state()
    assert st["production"] == {}
    assert st["gate_version"] == auto_loop.GATE_VERSION


def test_stale_gate_version_invalidates_the_baseline(state_file):
    state_file.write_text(json.dumps(
        {"production": {"rank": 0.55}, "done": {}, "gate_version": auto_loop.GATE_VERSION - 1}))
    st = auto_loop.load_state()
    assert st["production"] == {}, "換尺之後舊基準必須作廢"
    assert st["gate_version"] == auto_loop.GATE_VERSION


def test_matching_gate_version_keeps_the_baseline(state_file):
    state_file.write_text(json.dumps(
        {"production": {"rank": 0.55}, "done": {}, "gate_version": auto_loop.GATE_VERSION}))
    assert auto_loop.load_state()["production"] == {"rank": 0.55}


def test_legacy_float_production_is_converted(state_file):
    """舊格式是單一浮點數。轉成 dict 而不是留著讓後面炸掉。"""
    state_file.write_text(json.dumps(
        {"production": 0.55, "done": {}, "gate_version": auto_loop.GATE_VERSION}))
    assert auto_loop.load_state()["production"] == {}


def test_empty_production_is_not_none(state_file):
    """這一格就是那個 bug：`{} is None` 是 False，
    所以用 `st["production"] is None` 當判斷式會**跳過**基準量測。"""
    st = auto_loop.load_state()
    assert st["production"] is not None
    assert st["production"].get("rank") is None, "要用 .get(track) 判斷，不是判斷整個 dict"


def test_every_experiment_declares_a_known_track():
    for e in auto_loop.EXPERIMENTS:
        assert e.get("track", "rank") in auto_loop.TRACKS, \
            f"{e['name']} 宣告了未知的 track"


def test_every_track_has_a_baseline_and_eval():
    for name, spec in auto_loop.TRACKS.items():
        assert spec["baseline"] > 0 and spec["eval"] and spec["json"], name
