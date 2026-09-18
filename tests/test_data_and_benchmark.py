"""驗證框架的測試：切分必須 player-disjoint，基準必須符合官方協定。

這些測試保護的是整個專案最關鍵的假設。若切分洩漏或基準偏離官方協定，
67 天沒有 leaderboard 的期間所有決策都會建立在錯的數字上。
"""
import numpy as np
import pytest

from aicup import LEVELS
from aicup.data import make_player_split, split_summary, PlayerSplit, load_sgf
from aicup.benchmark import (
    build_rank_questions, build_player_pool, pool_summary, MIN_GAMES, MAX_GAMES,
)


# ---------------- 切分 ----------------

def test_split_is_player_disjoint(split):
    assert split.train_players & split.val_players == set()
    split.check_disjoint()


def test_split_covers_every_rank(meta, split):
    s = split_summary(meta, split)
    assert len(s) == len(LEVELS)
    assert (s["val_players"] > 0).all()
    assert (s["val_games"] > 5000).all()      # 每級至少要有夠多驗證局


def test_split_is_deterministic(meta):
    a = make_player_split(meta, seed=7)
    b = make_player_split(meta, seed=7)
    assert a.val_players == b.val_players


def test_different_seeds_give_different_splits(meta):
    a = make_player_split(meta, seed=1)
    b = make_player_split(meta, seed=2)
    assert a.val_players != b.val_players


def test_split_fraction_is_roughly_honoured(meta, split):
    n_total = meta["player_id"].nunique()
    assert 0.17 < len(split.val_players) / n_total < 0.23


def test_no_game_belongs_to_both_sides(meta, split):
    m = split.mask(meta)
    train_players = set(meta.loc[~m, "player_id"].unique())
    val_players = set(meta.loc[m, "player_id"].unique())
    assert train_players & val_players == set()


def test_split_roundtrip_json(split, tmp_path):
    p = tmp_path / "split.json"
    split.to_json(str(p))
    back = PlayerSplit.from_json(str(p))
    assert back.val_players == split.val_players
    assert back.seed == split.seed


# ---------------- 競賽二題目 ----------------

def test_rank_questions_only_use_val_players(meta, split):
    q = build_rank_questions(meta, split, n_per_player=1, seed=0, max_questions=300)
    assert set(q.player_id) <= split.val_players


def test_rank_questions_game_count_matches_official_range(meta, split):
    q = build_rank_questions(meta, split, n_per_player=1, seed=0, max_questions=500)
    assert q.n_games.min() >= MIN_GAMES
    assert q.n_games.max() <= MAX_GAMES


def test_rank_questions_have_no_duplicate_games(meta, split):
    q = build_rank_questions(meta, split, n_per_player=1, seed=0, max_questions=300)
    for rows in q.game_rows:
        assert len(set(rows.tolist())) == len(rows)


def test_rank_questions_label_matches_games(meta, split):
    """題目的標籤必須等於其所有棋局的實際等級。"""
    q = build_rank_questions(meta, split, n_per_player=1, seed=0, max_questions=200)
    rank_id = meta["rank_id"].to_numpy()
    for rows, y in zip(q.game_rows, q.true_rank_id):
        assert (rank_id[rows] == y).all()


def test_rank_questions_all_games_same_player(meta, split):
    q = build_rank_questions(meta, split, n_per_player=1, seed=0, max_questions=200)
    pid = meta["player_id"].to_numpy()
    for rows, p in zip(q.game_rows, q.player_id):
        assert (pid[rows] == p).all()


# ---------------- 競賽一候選庫 ----------------

def test_pool_size_matches_official(meta, split):
    pool = build_player_pool(meta, split, composition="uniform", seed=0)
    assert len(pool) == 400


def test_pool_gallery_and_query_are_disjoint(meta, split):
    pool = build_player_pool(meta, split, seed=0)
    for pid in pool.players:
        g = set(pool.gallery_rows[pid].tolist())
        q = set(pool.query_rows[pid].tolist())
        assert g & q == set()


def test_pool_gallery_size_and_query_range(meta, split):
    pool = build_player_pool(meta, split, seed=0)
    for pid in pool.players:
        assert len(pool.gallery_rows[pid]) <= 100
        assert MIN_GAMES <= len(pool.query_rows[pid]) <= MAX_GAMES


def test_pool_players_are_unique_and_from_val(meta, split):
    pool = build_player_pool(meta, split, seed=0)
    assert len(set(pool.players)) == len(pool.players)
    assert set(pool.players) <= split.val_players


def test_pool_uniform_composition_is_balanced(meta, split):
    pool = build_player_pool(meta, split, composition="uniform", seed=0)
    counts = [sum(1 for p in pool.players if pool.rank_id[p] == i) for i in range(len(LEVELS))]
    assert max(counts) - min(counts) <= 1


def test_pool_dan_only_has_no_kyu_players(meta, split):
    pool = build_player_pool(meta, split, composition="dan_only", seed=0)
    assert all(pool.rank_id[p] >= 4 for p in pool.players)


def test_pool_kyu_only_has_no_dan_players(meta, split):
    pool = build_player_pool(meta, split, composition="kyu_only", seed=0)
    assert all(pool.rank_id[p] < 4 for p in pool.players)


def test_pool_different_seeds_differ(meta, split):
    a = build_player_pool(meta, split, seed=0)
    b = build_player_pool(meta, split, seed=1)
    assert set(a.players) != set(b.players)


def test_pool_summary_shape(meta, split):
    pool = build_player_pool(meta, split, seed=0)
    s = pool_summary(pool)
    assert s["n_players"].sum() == len(pool)


# ---------------- 原始棋譜載入 ----------------

def test_load_sgf_returns_matching_rows(meta, split):
    sub = meta[split.mask(meta)].head(50)
    sgf = load_sgf(sub)
    assert len(sgf) == 50
    assert sgf.notna().all()
    assert sgf.str.startswith("(;SZ[19]").all()


# ---------------- 資料排列的陷阱 ----------------

def test_csv_rows_are_sorted_by_color():
    """官方 CSV 依執色排序：前 50,000 列全 B、後 50,000 列全 W。

    這是個沉默的陷阱——用 nrows=N (N<=50000) 做快速實驗會拿到 100% 執黑樣本。
    任何抽樣都必須先 shuffle 或跨越第 50,000 列的分界。
    """
    import pandas as pd
    head = pd.read_csv("training/train_3D.csv", nrows=5, usecols=["color"])
    tail = pd.read_csv("training/train_3D.csv", skiprows=range(1, 50_001),
                       nrows=5, usecols=["color"])
    assert set(head["color"]) == {"B"}
    assert set(tail["color"]) == {"W"}


def test_split_and_questions_are_not_color_biased(meta, split):
    """驗證框架本身必須不受上述排列影響——抽樣後黑白應接近各半。"""
    import numpy as np
    q = build_rank_questions(meta, split, n_per_player=1, seed=0, max_questions=400)
    colors = meta["color"].to_numpy()
    all_rows = np.concatenate(q.game_rows)
    frac_black = float((colors[all_rows] == "B").mean())
    assert 0.40 < frac_black < 0.60
