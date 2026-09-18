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
