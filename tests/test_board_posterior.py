"""板面成分的集合成員定義。

這支測試鎖住一個曾經造成 **0.198 Spearman** 損失的不變式（RESULTS.md §2.27）：
一題的集合成員是「每一局的每一個 snapshot」，**不是「每一局的 snapshot 平均」**。

訓練時 `aicup.dataset.QuestionDataset` 用前者。這個邏輯先前被抄在三個地方
（dataset / train / eval_blend），修正只改到兩處，漏掉的那條正是做上線決策
用的那條——而且熵幾乎不變（1.437 vs 1.439），任何「輸出分布看起來合不合理」
的檢查都抓不到。現在只有 `aicup.board_posterior` 一份實作。
"""
import inspect

import numpy as np

from aicup.board_posterior import (groups_from_flat, groups_from_meta,
                                   question_members)
from aicup.planes import n_planes
from aicup.goboard import SIZE

SGF = ("(;GM[1]SZ[19]" + "".join(
    f";{'B' if i % 2 == 0 else 'W'}[{chr(97 + (i * 3) % 19)}{chr(97 + (i * 5) % 19)}]"
    for i in range(120)) + ")")


def test_each_snapshot_is_its_own_member():
    """3 局 x 4 snapshot = 12 個成員，不是 3 個。"""
    x = question_members([(SGF, "B"), (SGF, "W"), (SGF, "B")], n_snapshots=4)
    assert x.shape == (12, n_planes(), SIZE, SIZE), x.shape


def test_member_count_scales_with_snapshots():
    for n in (1, 2, 4, 8):
        assert len(question_members([(SGF, "B"), (SGF, "W")], n_snapshots=n)) == 2 * n


def test_members_are_not_all_identical():
    """若哪天又退回「先平均」，同一局的 snapshot 會塌成同一張。"""
    x = question_members([(SGF, "B")], n_snapshots=4)
    assert not np.allclose(x[0], x[-1]), "開局與終局的盤面不該相同"


def test_planes_are_normalised():
    """手數進度層是 0~255 量化的；必須已經除回去。"""
    x = question_members([(SGF, "B")], n_snapshots=4)
    assert x.dtype == np.float32 and x.max() <= 1.0 + 1e-6, x.max()


def test_unparseable_game_still_yields_one_member():
    """空集合會讓 forward_set 的 N=0 而炸掉；要有一個零成員兜底。"""
    x = question_members([("not an sgf", "B")], n_snapshots=4)
    assert x.shape == (1, n_planes(), SIZE, SIZE)


def test_groups_from_flat_respects_sizes():
    flat = [f"g{i}" for i in range(9)]
    cols = ["B"] * 9
    g = groups_from_flat(flat, cols, [2, 3, 4])
    assert [len(x) for x in g] == [2, 3, 4]
    assert g[1][0] == ("g2", "B") and g[2][-1] == ("g8", "B")


def test_no_averaging_left_in_the_source():
    """實作裡不該再出現把 snapshot 軸平均掉的寫法。"""
    import aicup.board_posterior as m
    src = inspect.getsource(m)
    body = src[src.index("def question_members"):src.index("def board_question")]
    assert ".mean(" not in body, "question_members 不該對 snapshot 軸取平均"


def test_groups_from_meta_reads_the_right_rows():
    class FakeMeta:
        def __init__(self, rows):
            self._rows = rows

        @property
        def iloc(self):
            return self._rows

    class FakeCache:
        def get(self, file_idx, row):
            return f"sgf-{file_idx}-{row}"

    meta = FakeMeta([{"file_idx": 0, "row": 5, "color": "B"},
                     {"file_idx": 1, "row": 7, "color": "W"}])
    g = groups_from_meta(meta, [[0, 1]], FakeCache())
    assert g == [[("sgf-0-5", "B"), ("sgf-1-7", "W")]]
