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

# ⚠️ 官方規格：查詢棋譜集是「**5, 10, 15, 20 盤各 100 組**」，
# 不是 5~20 的均勻隨機。差別不小——官方有 **25% 的題目只有 5 盤**（最難的情況），
# 而均勻隨機只有 1/16 = 6.25%。
# 用錯的分布校準切點，會讓驗證分數偏樂觀，而且切點是配在錯的題型組成上。
OFFICIAL_SIZES = (5, 10, 15, 20)


def _draw_size(rng, available: int, sizes, target: int | None = None) -> int:
    """決定一個題目的局數，並夾在該玩家實際可用的局數內。

    target 是「希望」的局數（來自全域輪替，讓四種題型各佔 25%）；
    玩家局數不足時退到他供得起的最大值。
    """
    if sizes == "uniform":
        return int(rng.integers(MIN_GAMES, min(MAX_GAMES, available) + 1))
    allowed = [k for k in OFFICIAL_SIZES if k <= available]
    if not allowed:                       # 連 5 盤都不夠的玩家在上游就被濾掉了
        return MIN_GAMES
    if target is not None:
        return target if target <= allowed[-1] else allowed[-1]
    return int(rng.choice(allowed))
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


# 玩家 -> 列索引 的全表索引快取。
# 建立候選庫時每次都對 100 萬列做一次 isin + groupby 相當慢，
# 而同一份 metadata 在一輪實驗中會被重複使用數十次（5 種組成 x 多個 seed）。
_ROWS_CACHE: dict[int, dict[str, np.ndarray]] = {}


def player_row_index(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """建立（並快取）整份 metadata 的 player_id -> 列索引對照表。"""
    key = id(meta)
    cached = _ROWS_CACHE.get(key)
    if cached is None:
        cached = {pid: g.index.to_numpy()
                  for pid, g in meta.groupby("player_id", sort=False)}
        _ROWS_CACHE.clear()          # 只保留最近一份，避免在長流程中累積記憶體
        _ROWS_CACHE[key] = cached
    return cached


def _player_rows(meta: pd.DataFrame, players) -> dict[str, np.ndarray]:
    """回傳 player_id -> 列索引，**鍵一律排序**。

    ⚠️ 這個 sorted() 不是裝飾。`PlayerSplit.val_players` 是 frozenset，
    而 Python 的字串雜湊每個 process 隨機化（除非設 PYTHONHASHSEED），
    所以 frozenset 的迭代順序**跨 process 不穩定**。
    若直接照 set 的順序建題目，同一個 seed 在不同次執行會產生不同的題目集，
    量到的分數就無法重現——實測同一支腳本跑兩次得到 0.4550 與 0.4637。
    """
    index = player_row_index(meta)
    return {pid: index[pid] for pid in sorted(players) if pid in index}


def build_rank_questions(meta: pd.DataFrame, split: PlayerSplit, *,
                         n_per_player: int = 3, seed: int = 0,
                         max_questions: int | None = None,
                         sizes: str = "official") -> RankQuestions:
    """從驗證玩家建立競賽二的模擬題目。每位玩家出 n_per_player 題。

    sizes="official"（預設）：局數從 {5, 10, 15, 20} 等機率抽，與官方規格一致。
    sizes="uniform"：5~20 均勻隨機（2026-09-20 之前的舊行為，偏樂觀）。
    """
    rng = np.random.default_rng(seed)
    rows = _player_rows(meta, split.val_players)
    rank_of = meta.groupby("player_id")["rank_id"].first()

    # 全域輪替四種題型，讓 5/10/15/20 各佔約 25%（官方是各 100 組）。
    # 逐玩家獨立抽會讓局數少的玩家把分布拉向小題型（實測 30.5/27/21.8/20.7）。
    eligible = [(pid, idx) for pid, idx in rows.items() if len(idx) >= MIN_GAMES]
    n_slots = len(eligible) * n_per_player
    targets = np.array(list(OFFICIAL_SIZES) * (n_slots // len(OFFICIAL_SIZES) + 1))[:n_slots]
    rng.shuffle(targets)

    game_rows, true_rank, pids = [], [], []
    slot = 0
    for pid, idx in eligible:
        for _ in range(n_per_player):
            tgt = int(targets[slot]) if sizes != "uniform" else None
            slot += 1
            k = _draw_size(rng, len(idx), sizes, target=tgt)
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
        taken = set(chosen)
        rest = [p for r in pool_ranks for p in eligible_by_rank.get(r, []) if p not in taken]
        rng.shuffle(rest)
        chosen.extend(rest[: n_candidates - len(chosen)])
    return chosen[:n_candidates]


def build_player_pool(meta: pd.DataFrame, split: PlayerSplit, *,
                      n_candidates: int = N_CANDIDATES, n_gallery: int = N_GALLERY,
                      composition: Composition = "uniform", seed: int = 0, sizes: str = "official") -> PlayerPool:
    """建立競賽一的模擬候選庫。gallery 與 query 的棋局互不重疊。"""
    rng = np.random.default_rng(seed)
    rows = _player_rows(meta, split.val_players)
    rank_of = meta.groupby("player_id")["rank_id"].first()

    need = n_gallery + MIN_GAMES
    eligible_by_rank: dict[int, list[str]] = {}
    for pid, idx in rows.items():          # rows 的鍵已排序，順序可重現
        if len(idx) >= need:
            eligible_by_rank.setdefault(int(rank_of[pid]), []).append(pid)

    chosen = _choose_candidates(eligible_by_rank, composition, n_candidates, rng)

    gallery, query, rid = {}, {}, {}
    for pid in chosen:
        idx = rows[pid].copy()
        rng.shuffle(idx)
        g = min(n_gallery, len(idx) - MIN_GAMES)
        k = _draw_size(rng, len(idx) - g, sizes)
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


def rank_prior(name: str, meta, n_levels: int = None) -> np.ndarray | None:
    """回傳一個目標等級先驗。`"natural"` 回傳 None（代表「照資料本來的分布」）。

    `"p100"` 是訓練資料中**至少有 100 譜**的玩家的等級分布。之所以特別看這個，
    是因為官方競賽一的候選庫要求「400 人**各 100 譜**」，測試玩家必然出自
    譜數足夠的那一群；而那一群的分布（7.4%~12.3%）比全體玩家
    （5.4%~19.8%）平均得多。詳見 RESULTS.md §2.7。
    """
    n = len(LEVELS) if n_levels is None else n_levels
    if name == "natural":
        return None
    if name == "uniform":
        return np.full(n, 1.0 / n)
    if name == "p100":
        cnt = meta.groupby("player_id").size()
        rk = meta.groupby("player_id")["rank_id"].first()
        p = np.bincount(rk[cnt >= 100].to_numpy(), minlength=n).astype(float)
        return p / p.sum()
    raise ValueError(f"未知的先驗 {name!r}，可用：natural / uniform / p100")


def resample_to_prior(idx, y, prior, rng) -> np.ndarray:
    """把索引集重抽成目標先驗，**題數保持不變**。

    以 w_i ∝ prior[y_i] / 目前頻率[y_i] 的機率有放回地抽。
    題數保持不變是刻意的：取子集會讓最稀有的等級把整體題數壓到協定門檻以下
    （自然分布下 rank 4 只佔 5.4%，取子集只剩約一半的題目）。
    """
    idx = np.asarray(idx)
    if prior is None or len(idx) == 0:
        return idx
    y = np.asarray(y)
    cnt = np.bincount(y[idx], minlength=len(prior)).astype(float)
    w = np.where(cnt[y[idx]] > 0, np.asarray(prior)[y[idx]] / np.maximum(cnt[y[idx]], 1), 0.0)
    if w.sum() <= 0:
        return idx
    return rng.choice(idx, size=len(idx), replace=True, p=w / w.sum())
