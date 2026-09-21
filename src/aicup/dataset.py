"""PyTorch 資料集：從 CSV 串流出盤面平面。

兩個刻意的設計決定：

1. **P x K 取樣**（PKPlayerSampler）
   競賽一的 batch 必須是「P 位玩家 x 每人 K 局」，batch-hard mining 才有東西可挑。
   官方 baseline 均勻隨機挑兩位玩家，絕大多數 triplet 早已滿足 margin、梯度為 0。
   這個結構對齊 Behavioral Stylometry (NeurIPS 2021) 的 40 players x 20 games。

2. **題目層級資料集的局數隨機 5~20**（QuestionDataset）
   官方測試每題 5~20 局，若訓練時固定局數，聚合層會偏向大集合、
   在 5 局的題目上失準——而實測 5 局的題目分數只有 20 局的三分之一，
   那正是決定名次的地方。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .data import training_files
from .goboard import SIZE
from .planes import game_planes, n_planes, planes_to_float


class SgfCache:
    """依 (file_idx, row) 取 sgf_content。整個檔案只在第一次用到時載入。"""

    def __init__(self, training_dir: str = "training", max_files: int = 10):
        self.files = training_files(training_dir)
        self.max_files = max_files
        self._cache: dict[int, np.ndarray] = {}

    def get(self, file_idx: int, row: int) -> str:
        arr = self._cache.get(file_idx)
        if arr is None:
            if len(self._cache) >= self.max_files:
                self._cache.pop(next(iter(self._cache)))
            arr = pd.read_csv(self.files[file_idx], usecols=["sgf_content"])[
                "sgf_content"].to_numpy()
            self._cache[file_idx] = arr
        return arr[row]


def _dihedral(x: np.ndarray, k: int) -> np.ndarray:
    """棋盤的 8 種對稱等價之一。k 取 0..7。

    實測（docs/02 §10.2）：對子題一施加對稱不變性有益
    （折疊 0.4697 vs 不折疊 0.3989），落子方位不帶身分訊號，
    所以這個增強對兩個子題都安全。
    """
    if k >= 4:
        x = np.flip(x, axis=-1)
    return np.rot90(x, k % 4, axes=(-2, -1)).copy()


class GameDataset(Dataset):
    """單局資料集：回傳 (planes, rank_id, player_label)。"""

    def __init__(self, meta: pd.DataFrame, rows: np.ndarray, *,
                 training_dir: str = "training", n_snapshots: int = 8,
                 augment: bool = True, player_to_label: dict | None = None,
                 quality_store: np.ndarray | None = None,
                 quality_cols: list[int] | None = None):
        self.meta = meta
        self.rows = np.asarray(rows)
        self.cache = SgfCache(training_dir)
        self.n_snapshots = n_snapshots
        self.augment = augment
        self.player_to_label = player_to_label or {}
        # 棋形品質的輔助回歸目標，直接取自已算好的特徵庫。
        # 棋力標籤一局只有 1 個、訊號稀疏；這些連續量提供密集的監督訊號。
        self.quality_store = quality_store
        self.quality_cols = quality_cols or []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = int(self.rows[i])
        rec = self.meta.iloc[r]
        sgf = self.cache.get(int(rec["file_idx"]), int(rec["row"]))
        t = game_planes(sgf, rec["color"], n_snapshots=self.n_snapshots)
        if t is None:
            # ⚠️ snapshot 維度要與正常情況一致。原本寫死 1，
            # 一旦真的有棋局解析失敗，批次裡就會混進形狀不同的張量。
            # （目前訓練集沒有這種棋局，所以一直沒被觸發——
            #   但那是運氣，不是設計。）
            t = np.zeros((self.n_snapshots, n_planes(), SIZE, SIZE), dtype=np.uint8)
        if self.augment:
            t = _dihedral(t, int(np.random.randint(8)))
        # ⚠️ 一律走 planes_to_float，不要自己 astype——
        # 手數進度那一層是 0~255，直接轉型會讓它的尺度是其他層的 255 倍。
        item = (torch.from_numpy(planes_to_float(t)),
                int(rec["rank_id"]),
                int(self.player_to_label.get(rec["player_id"], -1)))
        if self.quality_store is None:
            return item
        qual = np.asarray(self.quality_store[r][self.quality_cols], dtype=np.float32)
        return item + (torch.from_numpy(qual),)


class PKPlayerSampler(Sampler):
    """每個 batch 取 P 位玩家 x 每人 K 局，讓 batch-hard mining 有素材。"""

    def __init__(self, meta: pd.DataFrame, rows: np.ndarray, p: int = 40, k: int = 8,
                 n_batches: int = 1000, seed: int = 0):
        self.p, self.k, self.n_batches = p, k, n_batches
        self.rng = np.random.default_rng(seed)
        sub = meta.iloc[rows]
        pos = {}
        for local, (_, pid) in enumerate(sub["player_id"].items()):
            pos.setdefault(pid, []).append(local)
        self.by_player = {pid: np.array(v) for pid, v in pos.items() if len(v) >= 2}
        self.players = list(self.by_player)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            chosen = self.rng.choice(self.players, size=min(self.p, len(self.players)),
                                     replace=False)
            batch = []
            for pid in chosen:
                idx = self.by_player[pid]
                take = self.rng.choice(idx, size=min(self.k, len(idx)),
                                       replace=len(idx) < self.k)
                batch.extend(int(x) for x in take)
            yield batch


class QuestionDataset(Dataset):
    """題目層級資料集：一題 = 同一位玩家的 5~20 局，對齊官方測試協定。"""

    def __init__(self, meta: pd.DataFrame, questions, *, training_dir: str = "training",
                 n_snapshots: int = 4, max_games: int = 20, augment: bool = False):
        self.meta = meta
        self.questions = questions
        self.cache = SgfCache(training_dir)
        self.n_snapshots = n_snapshots
        self.max_games = max_games
        self.augment = augment

    def __len__(self) -> int:
        return len(self.questions.true_rank_id)

    def __getitem__(self, i: int):
        rows = self.questions.game_rows[i][: self.max_games]
        tensors = []
        for r in rows:
            rec = self.meta.iloc[int(r)]
            sgf = self.cache.get(int(rec["file_idx"]), int(rec["row"]))
            t = game_planes(sgf, rec["color"], n_snapshots=self.n_snapshots)
            if t is not None:
                # ⚠️ **每個 snapshot 各自是集合的一個成員，不要先平均。**
                # 訓練時 trunk 看到的是單一時點的盤面；若評估時餵
                # 「S 個 snapshot 的平均」，那是一張模糊的「平均局面」——
                # 一種 trunk 從來沒見過的輸入分布。
                # 實測後果：逐局 rank 損失明明在下降（2.29 → 2.14），
                # 題目分數卻一路惡化（0.2684 → 0.1435）。
                tensors.extend(planes_to_float(t))
        if not tensors:
            tensors = [np.zeros((n_planes(), SIZE, SIZE), dtype=np.float32)]
        x = np.stack(tensors)
        if self.augment:
            x = _dihedral(x, int(np.random.randint(8)))
        return torch.from_numpy(x), int(self.questions.true_rank_id[i])


def collate_questions(batch):
    """把長度不一的題目 padding 成 (B, N, C, 19, 19) 並回傳有效遮罩。"""
    xs, ys = zip(*batch)
    n = max(x.shape[0] for x in xs)
    B, C, H, W = len(xs), xs[0].shape[1], xs[0].shape[2], xs[0].shape[3]
    out = torch.zeros(B, n, C, H, W)
    mask = torch.zeros(B, n, dtype=torch.bool)
    for i, x in enumerate(xs):
        out[i, : x.shape[0]] = x
        mask[i, : x.shape[0]] = True
    return out, mask, torch.tensor(ys, dtype=torch.long)
