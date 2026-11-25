"""訓練資料的載入、metadata 快取，與 player-disjoint 切分。

為什麼要 player-disjoint：官方 tutorial 的驗證集是把 100 萬局 shuffle 後取前 10 萬局，
實測 7,463 位驗證集玩家有 100.00% 同時出現在訓練集裡。
模型可以靠「記住這個人」猜等級，而真實測試集的玩家是完全未見的。
所有切分一律以 player_id 為單位。
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import LEVELS, RANK_TO_ID

META_COLS = ["player_id", "game_id", "rank", "color"]

# 資料路徑集中在這裡：程式與測試都引用這些常數，搬動資料夾時只改這一處。
DATA_DIR = "data"
TRAINING_DIR = os.path.join(DATA_DIR, "training")          # 官方訓練集（v1.0，見 repro/README.md）
MOCK_TEST_DIR = os.path.join(DATA_DIR, "mock_tests")       # scripts/make_mock_testset.py 產生
OFFICIAL_TEST_DIR = DATA_DIR   # 官方測試集（2026-11-04）放在 data/tests_rank/ 與 data/tests_player/


def training_files(training_dir: str = TRAINING_DIR) -> list[str]:
    files = sorted(glob.glob(os.path.join(training_dir, "train_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"在 {training_dir}/ 找不到 train_*.csv。"
            " 訓練集需自官方 Releases 下載："
            " https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases"
        )
    return files


def _rank_of(path: str) -> str:
    return os.path.basename(path).split("train_")[1][: -len(".csv")]


def load_metadata(training_dir: str = TRAINING_DIR, cache: str = "cache/metadata.npz",
                  rebuild: bool = False) -> pd.DataFrame:
    """載入全部 100 萬局的 metadata（不含 sgf_content），並快取。

    回傳欄位：player_id, rank, rank_id, color, file_idx, row
      file_idx / row 讓後續能精準回到原始 CSV 取 sgf_content，不必整份載入記憶體。
    """
    files = training_files(training_dir)
    if cache and os.path.exists(cache) and not rebuild:
        z = np.load(cache, allow_pickle=True)
        return pd.DataFrame({k: z[k] for k in z.files})

    frames = []
    for fi, path in enumerate(files):
        d = pd.read_csv(path, usecols=META_COLS)
        d["rank"] = _rank_of(path)
        d["rank_id"] = RANK_TO_ID[_rank_of(path)]
        d["file_idx"] = fi
        d["row"] = np.arange(len(d), dtype=np.int32)
        frames.append(d.drop(columns=["game_id"]))
    meta = pd.concat(frames, ignore_index=True)

    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez_compressed(cache, **{c: meta[c].to_numpy() for c in meta.columns})
    return meta


def load_sgf(meta_subset: pd.DataFrame, training_dir: str = TRAINING_DIR) -> pd.Series:
    """依 metadata 子集取回對應的 sgf_content（每個檔案只讀一次）。

    回傳的 Series 索引與 meta_subset 對齊。
    """
    files = training_files(training_dir)
    out = pd.Series(index=meta_subset.index, dtype=object)
    for fi, g in meta_subset.groupby("file_idx"):
        col = pd.read_csv(files[int(fi)], usecols=["sgf_content"])["sgf_content"]
        out.loc[g.index] = col.to_numpy()[g["row"].to_numpy()]
    return out


@dataclass(frozen=True)
class PlayerSplit:
    """一次 player-disjoint 切分的結果。"""
    train_players: frozenset
    val_players: frozenset
    seed: int
    val_frac: float

    def mask(self, meta: pd.DataFrame) -> np.ndarray:
        """回傳 meta 中屬於驗證玩家的布林遮罩。"""
        return meta["player_id"].isin(self.val_players).to_numpy()

    def check_disjoint(self) -> None:
        overlap = self.train_players & self.val_players
        if overlap:
            raise AssertionError(f"切分不互斥，有 {len(overlap)} 位玩家重疊")

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"seed": self.seed, "val_frac": self.val_frac,
                 "val_players": sorted(self.val_players),
                 "train_players": sorted(self.train_players)}, f)

    @staticmethod
    def from_json(path: str) -> "PlayerSplit":
        with open(path) as f:
            d = json.load(f)
        return PlayerSplit(frozenset(d["train_players"]), frozenset(d["val_players"]),
                           int(d["seed"]), float(d["val_frac"]))


def make_player_split(meta: pd.DataFrame, val_frac: float = 0.20, seed: int = 42) -> PlayerSplit:
    """依等級分層，抽出 val_frac 比例的「玩家」作為驗證集。

    分層是必要的：各等級的玩家數差異極大（實測 1D 僅 433 人、A 有 1,582 人），
    不分層會讓某些等級在驗證集裡樣本過少。
    """
    rng = np.random.default_rng(seed)
    player_rank = meta.groupby("player_id")["rank_id"].first()
    val = []
    for _, grp in player_rank.groupby(player_rank):
        ids = grp.index.to_numpy()
        rng.shuffle(ids)
        n = max(1, int(round(len(ids) * val_frac)))
        val.extend(ids[:n].tolist())
    val_set = frozenset(val)
    train_set = frozenset(player_rank.index) - val_set
    split = PlayerSplit(train_set, val_set, seed, val_frac)
    split.check_disjoint()
    return split


def split_summary(meta: pd.DataFrame, split: PlayerSplit) -> pd.DataFrame:
    """逐等級回報切分後的玩家數與局數，用來確認驗證集夠大、分布沒歪。"""
    m = split.mask(meta)
    rows = []
    for lv in LEVELS:
        sel = meta["rank"].to_numpy() == lv
        rows.append({
            "rank": lv,
            "train_players": meta.loc[sel & ~m, "player_id"].nunique(),
            "val_players": meta.loc[sel & m, "player_id"].nunique(),
            "train_games": int((sel & ~m).sum()),
            "val_games": int((sel & m).sum()),
        })
    return pd.DataFrame(rows)
