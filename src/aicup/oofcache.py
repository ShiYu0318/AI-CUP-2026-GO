"""題目層級後驗的快取層。

**為什麼需要**：每做一次融合實驗，都要對 4776 題 × 8 個 seed 重跑模型推論，
一輪 10~20 分鐘。但融合實驗真正在變的只有**權重與組合**——
後驗本身完全沒變。把後驗算一次存起來，之後掃權重、換組合、比較解碼器
就只是讀檔加矩陣運算，秒級完成。

這是能堆到十幾個成分的前提：沒有這層，光試一次組合就要等半小時。

**⚠️ 快取必須能自我失效，否則會變成最危險的那種 bug。**
本專案已經被「安靜地用到錯的東西」咬過五次
（推論時 dropout 沒關、checkpoint 猜錯 pool 型別…），
所以這裡把三件事都寫進指紋，任何一項變了就重算，不會安靜地給你舊結果：

1. **模型指紋** —— checkpoint 檔案內容的 SHA-256（不是路徑、不是 mtime）
2. **題目集指紋** —— 該 seed 產生的 game_rows 內容雜湊
3. **格式版本** —— 後驗的語意若改變（例如改了聚合方式），手動升版即可全部失效

用法：

    from aicup.oofcache import OofCache
    cache = OofCache()
    lo, y, pid = cache.get("rank_mlp_opp_big", q, model_path="cache/...pt",
                           compute=lambda: (log_post, y, player_id))
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

CACHE_VERSION = 1              # 後驗語意改變時 +1，讓舊快取全部失效
DEFAULT_DIR = "artifacts/oof"


def file_fingerprint(path: str, _memo: dict = {}) -> str:
    """checkpoint 檔案內容的 SHA-256（前 16 碼）。

    刻意不用 mtime/size：換了模型但檔名不變是最容易發生的情況，
    而那正是快取會安靜地給出錯誤結果的時候。
    同一個 process 內對同一路徑只算一次（42 MB 的 sha256 約 0.1 秒）。
    """
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key in _memo:
        return _memo[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    _memo[key] = h.hexdigest()[:16]
    return _memo[key]


def question_fingerprint(game_rows) -> str:
    """題目集的內容雜湊。

    只用題數或 seed 當 key 是不夠的：`build_rank_questions` 的輸出
    取決於 meta、split、n_per_player 等一整串上游設定，
    其中任何一個變了都會產生不同的題目，而 seed 看起來還是一樣。
    """
    h = hashlib.sha256()
    h.update(str(len(game_rows)).encode())
    for r in game_rows:
        a = np.ascontiguousarray(np.asarray(r, dtype=np.int64))
        h.update(a.tobytes())
    return h.hexdigest()[:16]


class OofCache:
    """題目層級後驗的磁碟快取。"""

    def __init__(self, directory: str = DEFAULT_DIR, verbose: bool = True):
        self.dir = directory
        self.verbose = verbose
        os.makedirs(self.dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, tag: str, qfp: str) -> str:
        return os.path.join(self.dir, f"{tag}__q{qfp}.npz")

    def get(self, tag: str, questions, compute, model_paths=None):
        """取得（或計算並存下）某個成分在某組題目上的 log 後驗。

        tag:         成分名稱，例如 "game_1178"、"seq_dual"
        questions:   RankQuestions（需要 .game_rows / .true_rank_id / .player_id）
        compute:     () -> log_post，形狀 (n_questions, n_levels)
        model_paths: 這個成分依賴的 checkpoint 路徑清單；任何一個檔案內容變了就重算
        回傳:        (log_post, true_rank_id, player_id)
        """
        qfp = question_fingerprint(questions.game_rows)
        mfp = [file_fingerprint(p) for p in (model_paths or [])]
        path = self._path(tag, qfp)

        if os.path.exists(path):
            try:
                z = np.load(path, allow_pickle=True)
                meta = json.loads(str(z["meta"]))
                if (meta.get("version") == CACHE_VERSION
                        and meta.get("model_fingerprints") == mfp
                        and meta.get("question_fingerprint") == qfp):
                    self.hits += 1
                    if self.verbose:
                        print(f"    [快取命中] {tag} q{qfp}", flush=True)
                    return z["log_post"], z["y"], z["pid"]
                if self.verbose:
                    print(f"    [快取失效] {tag} q{qfp}——"
                          f"{'模型變了' if meta.get('model_fingerprints') != mfp else '版本或題目變了'}"
                          f"，重算", flush=True)
            except Exception as e:                      # 壞掉的快取不該讓實驗失敗
                if self.verbose:
                    print(f"    [快取損毀] {tag}: {type(e).__name__}，重算", flush=True)

        t0 = time.time()
        log_post = np.asarray(compute(), dtype=np.float64)
        self.misses += 1
        if log_post.ndim != 2 or len(log_post) != len(questions.game_rows):
            raise ValueError(
                f"{tag} 的後驗形狀 {log_post.shape} 與題數 {len(questions.game_rows)} 不符")

        y = np.asarray(questions.true_rank_id)
        pid = np.asarray(questions.player_id, dtype=object)
        meta = {"version": CACHE_VERSION, "tag": tag,
                "model_fingerprints": mfp, "question_fingerprint": qfp,
                "n_questions": int(len(y)), "seconds": round(time.time() - t0, 1)}
        # 原子寫入：先寫暫存檔再 rename，中途被中斷不會留下半個壞掉的快取。
        # ⚠️ 要傳**檔案物件**給 np.savez——傳路徑的話它會自作主張補上 .npz，
        #    暫存檔名就對不上後面的 os.replace。
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, log_post=log_post, y=y, pid=pid, meta=json.dumps(meta))
        os.replace(tmp, path)
        if self.verbose:
            print(f"    [已快取] {tag} q{qfp}（{meta['seconds']:.1f} 秒）", flush=True)
        return log_post, y, pid

    def summary(self) -> str:
        n = self.hits + self.misses
        return (f"快取 {self.hits}/{n} 命中"
                if n else "快取未使用")
