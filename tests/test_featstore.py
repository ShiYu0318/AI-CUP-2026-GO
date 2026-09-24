"""多份特徵庫串接的測試。

對手視角的特徵是獨立建置的第二份 2.2 GB 檔案。把兩份實體合併既浪費磁碟、
又讓「只用我方」的實驗得重建，所以改成取列時才串接。
"""
import numpy as np
import pytest

from aicup.featstore import MultiStore


def _parts():
    return [np.arange(40).reshape(10, 4).astype(np.float32),
            np.arange(100, 130).reshape(10, 3).astype(np.float32)]


def test_shape_and_len():
    m = MultiStore(_parts())
    assert m.shape == (10, 7) and len(m) == 10


def test_row_gather_matches_manual_concat():
    a, b = _parts()
    m = MultiStore([a, b])
    rows = np.array([0, 3, 9])
    assert np.array_equal(m[rows], np.concatenate([a[rows], b[rows]], axis=1))


def test_single_part_is_passthrough():
    a, _ = _parts()
    assert np.array_equal(MultiStore([a])[[2, 5]], a[[2, 5]])


def test_rejects_mismatched_row_counts():
    """兩份必須由同一份 metadata 以相同順序建置——列數不符一定是搞錯了。"""
    a, b = _parts()
    with pytest.raises(ValueError, match="列"):
        MultiStore([a, b[:5]])


def test_rejects_empty():
    with pytest.raises(ValueError):
        MultiStore([])


def test_refuses_whole_array_conversion():
    """整份轉陣列會把數 GB 讀進記憶體，必須擋掉。"""
    m = MultiStore(_parts())
    with pytest.raises(TypeError, match="記憶體"):
        np.asarray(m)


def test_ok_mask_is_intersected(tmp_path):
    """任一份抽取失敗的棋局都不能用，否則向量會有一半是零而不自知。"""
    from aicup.featstore import open_stores

    for name, ok in (("a", [True, True, False]), ("b", [True, False, True])):
        np.save(tmp_path / f"{name}.npy", np.zeros((3, 2), dtype=np.float32))
        np.save(tmp_path / f"{name}_ok.npy", np.array(ok))
    _, ok = open_stores([str(tmp_path / "a.npy"), str(tmp_path / "b.npy")])
    assert ok.tolist() == [True, False, False]


def test_take_columns_matches_naive_gather():
    """分塊版必須與 `np.asarray(X[rows])[:, cols]` 逐元素相同，含跨份欄位與非遞增順序。"""
    import numpy as np
    from aicup.featstore import MultiStore, take_columns
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1000, 7)).astype(np.float32)
    b = rng.normal(size=(1000, 5)).astype(np.float32)
    X = MultiStore([a, b])
    rows = np.sort(rng.choice(1000, 300, replace=False))
    cols = np.array([11, 0, 3, 7, 8, 2])
    want = np.asarray(X[rows])[:, cols]
    assert np.array_equal(take_columns(X, rows, cols, chunk=64), want)
    assert np.array_equal(X.take(rows, cols, chunk=7), want)
    assert np.array_equal(take_columns(a, rows, [6, 1], chunk=50), a[rows][:, [6, 1]])
