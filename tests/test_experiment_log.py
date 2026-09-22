"""實驗記錄的測試：追加不覆寫、可回溯 commit、能展開巢狀欄位。"""
import json
import os

import pytest

from aicup.experiment_log import log_experiment, load_experiments, best_runs, FIELDS


@pytest.fixture
def log_path(tmp_path):
    return str(tmp_path / "exp.csv")


def test_creates_file_with_header(log_path):
    log_experiment("rank", "a", {"x": 1}, {"official": 0.4}, path=log_path)
    assert os.path.exists(log_path)
    df = load_experiments(log_path)
    assert len(df) == 1


def test_appends_never_overwrites(log_path):
    for i in range(4):
        log_experiment("rank", f"run{i}", {"i": i}, {"official": 0.1 * i}, path=log_path)
    df = load_experiments(log_path)
    assert len(df) == 4
    assert sorted(df["name"]) == ["run0", "run1", "run2", "run3"]


def test_records_git_commit(log_path):
    log_experiment("rank", "a", {}, {"official": 0.5}, path=log_path)
    df = load_experiments(log_path)
    assert df["commit"].iloc[0] != ""


def test_expands_config_and_metrics_columns(log_path):
    log_experiment("player", "a", {"top_n": 20, "pool": "set"},
                   {"official": 0.69, "top1": 0.65}, path=log_path)
    df = load_experiments(log_path)
    assert "c.top_n" in df.columns and "c.pool" in df.columns
    assert "m.official" in df.columns and "m.top1" in df.columns
    assert df["m.official"].iloc[0] == pytest.approx(0.69)


def test_best_runs_sorts_and_filters_by_task(log_path):
    log_experiment("rank", "lo", {}, {"official": 0.30}, path=log_path)
    log_experiment("rank", "hi", {}, {"official": 0.44}, path=log_path)
    log_experiment("player", "other", {}, {"official": 0.99}, path=log_path)
    best = best_runs("rank", "official", top=5, path=log_path)
    assert list(best["name"]) == ["hi", "lo"]        # 只有 rank，且由高到低


def test_handles_non_serialisable_values(log_path):
    import numpy as np
    log_experiment("rank", "a", {"arr": np.arange(3)}, {"official": np.float64(0.4)},
                   path=log_path)
    df = load_experiments(log_path)
    assert len(df) == 1


def test_missing_file_returns_empty_frame(tmp_path):
    df = load_experiments(str(tmp_path / "nope.csv"))
    assert df.empty
    assert list(df.columns) == FIELDS


def test_minutes_is_recorded(tmp_path):
    p = tmp_path / "e.csv"
    log_experiment("rank", "a", {"x": 1}, {"s": 0.5}, path=str(p), minutes=12.5)
    df = load_experiments(str(p))
    assert float(df.iloc[0]["minutes"]) == 12.5


def test_minutes_may_be_omitted(tmp_path):
    p = tmp_path / "e.csv"
    log_experiment("rank", "a", {"x": 1}, {"s": 0.5}, path=str(p))
    df = load_experiments(str(p))
    assert df.iloc[0]["minutes"] != df.iloc[0]["minutes"]      # NaN


def test_old_header_is_migrated_not_misaligned(tmp_path):
    """加欄位之後，舊檔案的表頭必須先遷移。

    不遷移的話 DictWriter 會照**新的** FIELDS 寫，而表頭是**舊的**——
    每一列都欄位錯位，而且 CSV 不會報錯。這是這次改動最危險的地方。
    """
    import csv
    p = tmp_path / "e.csv"
    old = ["timestamp", "task", "name", "commit", "dirty", "host",
           "config", "metrics", "note"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=old)
        w.writeheader()
        w.writerow({k: "old" for k in old} | {"config": "{}", "metrics": '{"s": 0.1}'})
    log_experiment("rank", "new", {}, {"s": 0.9}, path=str(p), minutes=3.0)
    df = load_experiments(str(p))
    assert list(df["name"]) == ["old", "new"]
    assert df.iloc[0]["task"] == "old", "舊列的欄位不能錯位"
    assert float(df.iloc[1]["minutes"]) == 3.0
    assert float(df.iloc[0]["m.s"]) == 0.1 and float(df.iloc[1]["m.s"]) == 0.9


def test_timer_measures_and_survives_exceptions():
    from aicup.experiment_log import timer
    import pytest as _pt
    with timer() as t:
        pass
    assert t.minutes >= 0
    with _pt.raises(ValueError):
        with timer() as t2:
            raise ValueError
    assert t2.minutes >= 0, "失敗的實驗也要有時間可查"


def test_cost_report_adds_gain_per_hour(tmp_path):
    from aicup.experiment_log import cost_report
    p = tmp_path / "e.csv"
    log_experiment("rank", "a", {}, {"gain": 0.01}, path=str(p), minutes=60.0)
    r = cost_report(str(p))
    assert abs(float(r.iloc[0]["增益/小時"]) - 0.01) < 1e-9
