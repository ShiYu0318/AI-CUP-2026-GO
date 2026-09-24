"""把多份特徵庫當成一份使用。

動機：對手視角的特徵（`cache/features_opp.npy`）與我方的（`cache/features.npy`）
是兩個分開建置的檔案，各 2.2 GB。把它們實體合併成一個 4.4 GB 的陣列既浪費磁碟、
又讓「只想用我方」的實驗得重建一次。

這個包裝在**取列的時候**才把兩份接起來，所以：
  - 磁碟上維持兩個獨立檔案，可以各自重建
  - memmap 的惰性讀取特性保留（不會把 4.4 GB 讀進記憶體）
  - 呼叫端只需要 `X[rows]`、`X.shape`、`len(X)`，與 np.ndarray 一致
"""
from __future__ import annotations

import numpy as np


class MultiStore:
    """多份同列數的特徵庫，沿著特徵軸串接。"""

    def __init__(self, parts: list[np.ndarray]):
        if not parts:
            raise ValueError("至少要有一份特徵庫")
        n = len(parts[0])
        for i, p in enumerate(parts):
            if len(p) != n:
                raise ValueError(
                    f"第 {i} 份特徵庫有 {len(p):,} 列，與第 0 份的 {n:,} 列不符——"
                    "兩份必須由同一份 metadata 以相同順序建置")
        self.parts = parts
        self._n = n
        self._dim = sum(p.shape[1] for p in parts)

    def __len__(self) -> int:
        return self._n

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n, self._dim)

    @property
    def dtype(self):
        return self.parts[0].dtype

    def __getitem__(self, rows):
        if len(self.parts) == 1:
            return self.parts[0][rows]
        return np.concatenate([np.asarray(p[rows]) for p in self.parts], axis=-1)

    def take(self, rows, cols, chunk: int = 50_000) -> np.ndarray:
        """等同 `np.asarray(X[rows])[:, cols]`，但**不經過全寬的中間陣列**。

        舊寫法會先把所有欄位讀出來（78 萬局 x 1274 維 = 4.0 GB），
        再挑欄位複製一次（3.6 GB），峰值約 7.6 GB。2026-09-24 在記憶體
        只剩約 7 GB 時，度量模型訓練因此在 mmap 上換頁爬行，20 分鐘沒前進。
        這裡預先配置輸出、分塊讀取、每塊只留需要的欄位，峰值 ≈ 輸出本身。
        """
        return take_columns(self, rows, cols, chunk)

    def __array__(self, dtype=None):
        raise TypeError(
            "MultiStore 不支援整份轉成陣列（會把數 GB 讀進記憶體）。"
            "請用 X[rows] 取需要的列。")


def open_stores(paths, mask_paths=None):
    """開啟一或多份特徵庫，回傳 (MultiStore, 共同的 ok 遮罩)。

    ok 遮罩取**交集**：任何一份抽取失敗的棋局都不能用，
    否則串接出來的向量會有一半是零而不自知。
    """
    paths = [paths] if isinstance(paths, str) else list(paths)
    parts = [np.load(p, mmap_mode="r") for p in paths]
    if mask_paths is None:
        mask_paths = [p.replace(".npy", "_ok.npy") for p in paths]
    mask_paths = [mask_paths] if isinstance(mask_paths, str) else list(mask_paths)
    ok = None
    for mp in mask_paths:
        m = np.load(mp)
        ok = m if ok is None else (ok & m)
    return MultiStore(parts), ok


def take_columns(X, rows, cols, chunk: int = 50_000) -> np.ndarray:
    """`np.asarray(X[rows])[:, cols]` 的低記憶體版本；X 可以是 MultiStore 或陣列。"""
    rows = np.asarray(rows)
    cols = np.asarray(cols, dtype=np.int64)
    parts = X.parts if isinstance(X, MultiStore) else [X]
    out = np.empty((len(rows), len(cols)), dtype=parts[0].dtype)
    # 每一份特徵庫負責哪些輸出欄位
    plan, off = [], 0
    for p in parts:
        w = int(p.shape[1])
        sel = np.flatnonzero((cols >= off) & (cols < off + w))
        if len(sel):
            plan.append((p, sel, cols[sel] - off))
        off += w
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        for p, sel, local in plan:
            out[a:a + len(r), sel] = np.asarray(p[r])[:, local]
    return out
