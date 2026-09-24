"""把訓練集的 SGF 打包成共用的 memory-mapped 檔（`aicup.sgfstore`）。

用法：
    python3 scripts/build_sgf_store.py            # 約 1 分鐘，產生 cache/sgf_blob.bin
"""
from __future__ import annotations

import argparse
import time

from aicup.data import training_files
from aicup.sgfstore import BLOB, INDEX, build, is_valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", default="training")
    ap.add_argument("--force", action="store_true", help="即使現有的打包檔仍有效也重建")
    args = ap.parse_args()
    files = training_files(args.training_dir)
    if is_valid(files) and not args.force:
        print(f"{BLOB} 已是最新，略過（--force 可強制重建）")
        return 0
    t0 = time.time()
    st = build(files)
    print(f"已寫出 {BLOB} 與 {INDEX}：{st['files']} 檔、{st['games']:,} 局、"
          f"{st['bytes'] / 2**30:.2f} GB（{time.time() - t0:.0f} 秒）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
