"""打包的 SGF 檔必須與 CSV 逐字相同，而且 CSV 一變就要失效。"""
import os
import pickle

import pandas as pd
import pytest

from aicup import sgfstore
from aicup.sgfstore import SgfStore, build, is_valid


@pytest.fixture
def csvs(tmp_path):
    rows = [["(;GM[1];B[aa];W[bb])", "(;GM[1];B[cc])", "(;GM[1]C[中文註解];B[dd])"],
            ["(;GM[1];W[ee])", ""]]
    paths = []
    for i, r in enumerate(rows):
        p = tmp_path / f"train_{i}.csv"
        pd.DataFrame({"sgf_content": r, "x": range(len(r))}).to_csv(p, index=False)
        paths.append(str(p))
    return paths, rows, tmp_path


def _paths(tmp):
    return str(tmp / "blob.bin"), str(tmp / "index.npz")


def test_roundtrip_matches_csv(csvs):
    paths, rows, tmp = csvs
    blob, index = _paths(tmp)
    st = build(paths, blob, index)
    assert st["games"] == 5
    s = SgfStore(blob, index)
    for f, r in enumerate(rows):
        want = pd.read_csv(paths[f], usecols=["sgf_content"])["sgf_content"]
        for i in range(len(r)):
            exp = want[i] if isinstance(want[i], str) else ""
            assert s.get(f, i) == exp


def test_non_ascii_survives(csvs):
    paths, _, tmp = csvs
    blob, index = _paths(tmp)
    build(paths, blob, index)
    assert "中文註解" in SgfStore(blob, index).get(0, 2)


def test_invalidates_when_csv_changes(csvs):
    paths, _, tmp = csvs
    blob, index = _paths(tmp)
    build(paths, blob, index)
    assert is_valid(paths, blob, index)
    with open(paths[1], "a") as fh:
        fh.write('"(;GM[1];B[ff])",9\n')
    assert not is_valid(paths, blob, index)


def test_invalidates_when_file_list_changes(csvs):
    paths, _, tmp = csvs
    blob, index = _paths(tmp)
    build(paths, blob, index)
    assert not is_valid(paths[:1], blob, index)


def test_pickle_does_not_carry_the_mapping(csvs):
    """傳給 spawn 的 worker 時不能把整個 memmap 複製過去。"""
    paths, _, tmp = csvs
    blob, index = _paths(tmp)
    build(paths, blob, index)
    s = SgfStore(blob, index)
    s.get(0, 0)
    assert s._mm is not None
    s2 = pickle.loads(pickle.dumps(s))
    assert s2._mm is None and s2.get(0, 1) == s.get(0, 1)


def test_missing_store_is_not_valid(tmp_path):
    assert not is_valid([], str(tmp_path / "nope.bin"), str(tmp_path / "nope.npz"))
