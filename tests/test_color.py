"""執色欄位的解析必須嚴格。

早期版本是 `BLACK if color == "B" else WHITE`：任何不等於 `"B"` 的值
都被當成白棋。`"b"`、`"black"`、`" B"` 會讓**所有黑棋局用錯誤的視角**
抽特徵，而且不拋例外、不改變形狀，只是分數安靜地變差。
測試集是比賽當天才釋出的，寧可當場炸掉也不要靜默算錯。
"""
import numpy as np
import pytest

from aicup.features import normalise_color, game_features
from aicup.goboard import BLACK, WHITE


@pytest.mark.parametrize("v", ["B", "b", " B ", "black", "BLACK", BLACK, 1])
def test_black_spellings(v):
    assert normalise_color(v) == BLACK


@pytest.mark.parametrize("v", ["W", "w", " W ", "white", "WHITE", WHITE, 2])
def test_white_spellings(v):
    assert normalise_color(v) == WHITE


@pytest.mark.parametrize("v", ["X", "", None, 7, -1, "0", "b w"])
def test_unknown_colour_raises(v):
    """關鍵：無法辨識時必須拋錯，不可以預設成白棋。"""
    with pytest.raises(ValueError):
        normalise_color(v)


def _sgf(n=40):
    cs = "abcdefghijklmnopqrs"
    mv = []
    for i in range(n):
        r, c = divmod(i * 7 % 361, 19)
        mv.append(f";{'B' if i % 2 == 0 else 'W'}[{cs[c]}{cs[r]}]")
    return "(;SZ[19]" + "".join(mv) + ")"


def test_case_insensitive_gives_same_features():
    """'b' 與 'B' 必須抽出完全相同的特徵——這正是舊版會靜默弄反的地方。"""
    sgf = _sgf()
    a = game_features(sgf, "B")
    b = game_features(sgf, "b")
    assert a is not None and b is not None
    assert np.array_equal(a, b)


def test_black_and_white_features_actually_differ():
    """反證：視角真的會改變特徵，所以上面的測試不是空轉。"""
    sgf = _sgf()
    assert not np.array_equal(game_features(sgf, "B"), game_features(sgf, "W"))


def test_game_planes_is_strict_too():
    """板面張量走的是另一條路徑，一樣不能用 `== "B"` 的寬鬆判斷。

    2026-09-22 之前 `planes.py` 還是舊寫法。在板面 CNN 進入 `predict.py`
    的推論路徑之前這沒差（訓練讀的是已正規化的 metadata），
    但現在它直接讀官方測試檔的執色欄位。
    """
    from aicup.planes import game_planes
    sgf = _sgf()
    a = game_planes(sgf, "B")
    b = game_planes(sgf, "b")
    assert a is not None and np.array_equal(a, b)
    assert not np.array_equal(a, game_planes(sgf, "W"))
    with pytest.raises(ValueError):
        game_planes(sgf, "藍")
