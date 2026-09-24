"""把全部 SGF 打包成一個 memory-mapped 檔，讓任意多個程序共用。

**為什麼**：`SgfCache` 原本是每個實例各自把 CSV 讀進記憶體——10 檔約 1.3 GB 文字，
變成 pandas 字串物件後約 3 GB。於是：

- DataLoader 每多一個 worker 就多 3 GB，開 6 個就 18 GB，機器直接換頁；
  所以訓練只能 `--workers 0`。
- 訓練與評估／探針同時跑也會互相擠爆記憶體（RESULTS.md §2.16b、§2.28）。

實測（batch 48 局 x 4 snapshot）：workers 0 / 4 / 8 = 69 / 18 / 15 ms/批次。
訓練一步約 576 ms，所以資料載入只佔約 12%——主要的收穫是**記憶體**
（每個程序省約 3 GB）與啟動時間，不是訓練速度。

打包後是一份位元組串＋偏移量。`np.memmap` 走 OS 的 page cache，
所有程序、所有 worker 讀的是**同一份實體記憶體**，額外成本趨近零；
啟動也從「解析 1.3 GB CSV」變成「開一個檔」。

**自我失效**：索引裡記下每個 CSV 的檔名、大小與修改時間，
任何一個對不上就不使用（`SgfCache` 會退回讀 CSV），不會讀到過期的內容。
"""

from __future__ import annotations

import os

import numpy as np

BLOB = "cache/sgf_blob.bin"
INDEX = "cache/sgf_index.npz"


def _signature(paths) -> np.ndarray:
    """每個 CSV 的 (大小, 修改時間)，用來判斷打包檔是否過期。"""
    return np.array([[os.path.getsize(p), int(os.path.getmtime(p))] for p in paths],
                    dtype=np.int64)


def build(files, blob: str = BLOB, index: str = INDEX) -> dict:
    """把每個 CSV 的 sgf_content 依序寫進 blob，回傳統計。"""
    import pandas as pd

    os.makedirs(os.path.dirname(blob) or ".", exist_ok=True)
    offsets, pos = {}, 0
    tmp = blob + ".tmp"
    with open(tmp, "wb") as fh:
        for f, path in enumerate(files):
            col = pd.read_csv(path, usecols=["sgf_content"])["sgf_content"]
            off = np.empty(len(col) + 1, dtype=np.int64)
            off[0] = pos
            for i, s in enumerate(col):
                b = ("" if not isinstance(s, str) else s).encode("utf-8")
                fh.write(b)
                pos += len(b)
                off[i + 1] = pos
            offsets[f"off_{f}"] = off
    os.replace(tmp, blob)                     # 寫完才換名，中斷不會留下半個檔
    np.savez(index, names=np.array([os.path.basename(p) for p in files]),
             signature=_signature(files), **offsets)
    return {"files": len(files), "games": sum(len(v) - 1 for v in offsets.values()),
            "bytes": pos}


def is_valid(files, blob: str = BLOB, index: str = INDEX) -> bool:
    if not (os.path.exists(blob) and os.path.exists(index)):
        return False
    with np.load(index) as z:
        names = [str(x) for x in z["names"]]
        if names != [os.path.basename(p) for p in files]:
            return False
        return bool(np.array_equal(z["signature"], _signature(files)))


class SgfStore:
    """(file_idx, row) → SGF 字串。memmap 在第一次用到時才開（對 fork/spawn 都安全）。"""

    def __init__(self, blob: str = BLOB, index: str = INDEX):
        self.blob_path, self.index_path = blob, index
        with np.load(index) as z:
            self.offsets = [z[f"off_{f}"] for f in range(len(z["names"]))]
        self._mm = None

    def __getstate__(self):
        # 傳給 spawn 的 worker 時不要把整個 memmap 序列化（那會複製 1.2 GB）
        d = dict(self.__dict__); d["_mm"] = None
        return d

    def get(self, file_idx: int, row: int) -> str:
        if self._mm is None:
            self._mm = np.memmap(self.blob_path, dtype=np.uint8, mode="r")
        off = self.offsets[file_idx]
        return bytes(self._mm[off[row]:off[row + 1]]).decode("utf-8")

    def __len__(self) -> int:
        return sum(len(o) - 1 for o in self.offsets)
