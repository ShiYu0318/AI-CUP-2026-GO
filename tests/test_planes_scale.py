"""板面張量的尺度。

`game_planes` 為了省記憶體回傳 uint8，其中「手數進度」量化成 0~255，
其餘 33 層都是 0/1。**讀取端若直接 `.float()`，那一層的尺度就是其他層的 255 倍。**

實測後果：rank 損失卡在 ln(10)=2.303、題目分數停在隨機水準（0.19），
而且沒有任何錯誤訊息——看起來只像「模型學不起來」。
這支測試釘住「必須經過 `planes_to_float`」。
"""
import numpy as np

from aicup.planes import game_planes, n_planes, planes_to_float, progress_plane

SGF = ("(;GM[1]SZ[19]" + "".join(
    f";{'B' if i % 2 == 0 else 'W'}[{chr(97 + (i * 3) % 19)}{chr(97 + (i * 5) % 19)}]"
    for i in range(120)) + ")")


def test_raw_planes_really_do_exceed_one():
    """如果哪天 game_planes 改成自己正規化，這條會失敗——
    那時就該把 planes_to_float 一併簡化，而不是讓它悄悄變成 no-op。"""
    t = np.asarray(game_planes(SGF, "B", n_snapshots=4))
    assert t.dtype == np.uint8
    assert t.max() > 1, "手數進度層應該是 0~255"


def test_normaliser_brings_every_plane_into_unit_range():
    t = game_planes(SGF, "B", n_snapshots=4)
    a = planes_to_float(t)
    assert a.dtype == np.float32
    assert a.max() <= 1.0 + 1e-6
    assert a.min() >= 0.0


def test_only_the_progress_plane_is_rescaled():
    t = np.asarray(game_planes(SGF, "B", n_snapshots=4))
    a = planes_to_float(t)
    idx = progress_plane()
    others = [i for i in range(t.shape[1]) if i != idx]
    assert np.array_equal(a[:, others], t[:, others].astype(np.float32))
    assert np.allclose(a[:, idx], t[:, idx].astype(np.float32) / 255.0)


def test_progress_plane_index_matches_the_layout():
    assert progress_plane() == n_planes() - 2


def test_normaliser_does_not_mutate_the_input():
    t = game_planes(SGF, "B", n_snapshots=4)
    before = np.asarray(t).copy()
    planes_to_float(t)
    assert np.array_equal(np.asarray(t), before)
