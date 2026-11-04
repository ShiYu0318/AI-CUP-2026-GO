"""官方測試檔的兩套命名都要找得到（課程投影片 vs tutorial repo）。"""
import os

import pytest

from aicup.submission import resolve_test_file


def _touch(d, name):
    open(os.path.join(d, name), "w").close()


def test_tutorial_naming(tmp_path):
    _touch(tmp_path, "rank_prediction_test_public.csv")
    assert resolve_test_file(str(tmp_path), "rank", "public").endswith("rank_prediction_test_public.csv")


def test_slide_deck_naming(tmp_path):
    """EP1 投影片寫的是 rank_test_* / player_test_* / *candidates*。"""
    for n in ("rank_test_private.csv", "player_test_private.csv", "player_candidates_private.csv"):
        _touch(tmp_path, n)
    d = str(tmp_path)
    assert resolve_test_file(d, "rank", "private").endswith("rank_test_private.csv")
    assert resolve_test_file(d, "player", "private").endswith("player_test_private.csv")
    assert resolve_test_file(d, "candidates", "private").endswith("player_candidates_private.csv")


def test_missing_returns_canonical_name_so_check_reports_it(tmp_path):
    p = resolve_test_file(str(tmp_path), "candidates", "public")
    assert p.endswith("player_identification_candidates_public.csv") and not os.path.exists(p)


def test_two_schemes_at_once_is_an_error(tmp_path):
    _touch(tmp_path, "rank_test_public.csv")
    _touch(tmp_path, "rank_prediction_test_public.csv")
    with pytest.raises(FileExistsError):
        resolve_test_file(str(tmp_path), "rank", "public")


def test_official_subfolders_are_found(tmp_path):
    """官方把兩個子題分放在 tests_rank/、tests_player/。"""
    (tmp_path / "tests_rank").mkdir(); (tmp_path / "tests_player").mkdir()
    _touch(tmp_path / "tests_rank", "rank_prediction_test_public.csv")
    _touch(tmp_path / "tests_player", "player_identification_candidates_public.csv")
    d = str(tmp_path)
    assert resolve_test_file(d, "rank", "public").endswith("tests_rank/rank_prediction_test_public.csv")
    assert resolve_test_file(d, "candidates", "public").endswith(
        "tests_player/player_identification_candidates_public.csv")
