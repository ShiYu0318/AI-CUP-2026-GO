"""題目層級的離線基準——完整複製官方測試協定。

在 2026/11/04 測試集釋出之前有 67 天沒有 leaderboard，
這份基準是這段期間唯一的儀表板（見 docs/02 §2）。

兩個子題的官方協定：
  競賽二：每題 = 同一位玩家的 5~20 局 -> 預測 1 個等級，共 400 題
  競賽一：候選庫 400 人 x 100 局；每題 = 同一位玩家的 5~20 局 -> 交出 Top-5

刻意保留的設計：
  - 局數一律在 5~20 均勻抽樣（實測局數是最強的分數驅動因素，
    5 局 0.157 vs 20 局 0.458，訓練與評估的集合大小分布必須對齊）
  - 候選庫支援多種等級組成（實測各等級辨識難度差 1.6 倍，
    真實候選庫的組成未知，方法必須在各種組成下都穩定）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from . import LEVELS
from .data import PlayerSplit

MIN_GAMES, MAX_GAMES = 5, 20
N_CANDIDATES, N_GALLERY = 400, 100

Composition = Literal["uniform", "natural", "dan_only", "kyu_only", "skewed"]


@dataclass
class RankQuestions:
    """競賽二的模擬題目集。"""
    game_rows: list[np.ndarray]          # 每題所用的 meta 列索引
    true_rank_id: np.ndarray
    player_id: np.ndarray

    def __len__(self) -> int:
        return len(self.true_rank_id)

    @property
    def n_games(self) -> np.ndarray:
        return np.array([len(g) for g in self.game_rows])


@dataclass
class PlayerPool:
    """競賽一的模擬候選庫與題目。"""
    players: list[str]
    gallery_rows: dict[str, np.ndarray]
    query_rows: dict[str, np.ndarray]
    rank_id: dict[str, int]
    composition: str = "uniform"
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.players)


def _player_rows(meta: pd.DataFrame, players) -> dict[str, np.ndarray]:
    sub = meta[meta["player_id"].isin(players)]
    return {pid: g.index.to_numpy() for pid, g in sub.groupby("player_id")}


def build_rank_questions(meta: pd.DataFrame, split: PlayerSplit, *,
                         n_per_player: int = 3, seed: int = 0,
                         max_questions: int | None = None) -> RankQuestions:
    """從驗證玩家建立競賽二的模擬題目。每位玩家出 n_per_player 題。"""
    rng = np.random.default_rng(seed)
    rows = _player_rows(meta, split.val_players)
    rank_of = meta.groupby("player_id")["rank_id"].first()

    game_rows, true_rank, pids = [], [], []
    for pid, idx in rows.items():
        if len(idx) < MIN_GAMES:
            continue
        for _ in range(n_per_player):
            k = int(rng.integers(MIN_GAMES, min(MAX_GAMES, len(idx)) + 1))
            game_rows.append(rng.choice(idx, size=k, replace=False))
            true_rank.append(int(rank_of[pid]))
            pids.append(pid)

    order = rng.permutation(len(game_rows))
    if max_questions is not None:
        order = order[:max_questions]
    return RankQuestions(
        game_rows=[game_rows[i] for i in order],
        true_rank_id=np.array([true_rank[i] for i in order]),
        player_id=np.array([pids[i] for i in order], dtype=object),
    )


def _choose_candidates(eligible_by_rank: dict[int, list[str]], composition: Composition,
                       n_candidates: int, rng) -> list[str]:
    """依指定的等級組成挑選候選玩家。"""
    if composition == "dan_only":
        pool_ranks = list(range(4, 10))
    elif composition == "kyu_only":
        pool_ranks = list(range(0, 4))
    else:
        pool_ranks = list(range(len(LEVELS)))

    if composition == "natural":
        flat = [p for r in pool_ranks for p in eligible_by_rank.get(r, [])]
        rng.shuffle(flat)
        return flat[:n_candidates]

    if composition == "skewed":
        # 權重隨等級線性遞增，模擬候選庫偏向段位的情況
        w = np.array([r + 1 for r in pool_ranks], dtype=float)
    else:                                   # uniform / dan_only / kyu_only 皆為均分
        w = np.ones(len(pool_ranks), dtype=float)
    w = w / w.sum()

    chosen: list[str] = []
    quotas = np.floor(w * n_candidates).astype(int)
    quotas[-1] += n_candidates - quotas.sum()
    for r, q in zip(pool_ranks, quotas):
        avail = list(eligible_by_rank.get(r, []))
        rng.shuffle(avail)
        chosen.extend(avail[:q])
    # 若某些等級人數不足，從其餘等級補滿
    if len(chosen) < n_candidates:
        rest = [p for r in pool_ranks for p in eligible_by_rank.get(r, []) if p not in set(chosen)]
        rng.shuffle(rest)
        chosen.extend(rest[: n_candidates - len(chosen)])
    return chosen[:n_candidates]


def build_player_pool(meta: pd.DataFrame, split: PlayerSplit, *,
                      n_candidates: int = N_CANDIDATES, n_gallery: int = N_GALLERY,
                      composition: Composition = "uniform", seed: int = 0) -> PlayerPool:
    """建立競賽一的模擬候選庫。gallery 與 query 的棋局互不重疊。"""
    rng = np.random.default_rng(seed)
    rows = _player_rows(meta, split.val_players)
    rank_of = meta.groupby("player_id")["rank_id"].first()

    need = n_gallery + MIN_GAMES
    eligible_by_rank: dict[int, list[str]] = {}
    for pid, idx in rows.items():
        if len(idx) >= need:
            eligible_by_rank.setdefault(int(rank_of[pid]), []).append(pid)

    chosen = _choose_candidates(eligible_by_rank, composition, n_candidates, rng)

    gallery, query, rid = {}, {}, {}
    for pid in chosen:
        idx = rows[pid].copy()
        rng.shuffle(idx)
        g = min(n_gallery, len(idx) - MIN_GAMES)
        k = int(rng.integers(MIN_GAMES, min(MAX_GAMES, len(idx) - g) + 1))
        gallery[pid] = idx[:g]
        query[pid] = idx[g:g + k]
        rid[pid] = int(rank_of[pid])

    return PlayerPool(
        players=chosen, gallery_rows=gallery, query_rows=query, rank_id=rid,
        composition=composition,
        meta={"seed": seed, "n_gallery": n_gallery,
              "available_by_rank": {k: len(v) for k, v in eligible_by_rank.items()}},
    )


def pool_summary(pool: PlayerPool) -> pd.DataFrame:
    """候選庫的等級組成與局數分布，用來確認基準本身沒有偏掉。"""
    rows = []
    for lv_id, lv in enumerate(LEVELS):
        pids = [p for p in pool.players if pool.rank_id[p] == lv_id]
        if not pids:
            continue
        rows.append({
            "rank": lv,
            "n_players": len(pids),
            "gallery_games_mean": float(np.mean([len(pool.gallery_rows[p]) for p in pids])),
            "query_games_mean": float(np.mean([len(pool.query_rows[p]) for p in pids])),
        })
    return pd.DataFrame(rows)
