"""產生 `repro/` 底下的重現性紀錄：環境、套件、模型雜湊、訓練參數、資料指紋、commit 對照。

    python3 scripts/build_repro.py                     # 全部重建
    python3 scripts/build_repro.py --test-dir data --test-dir data/mock_tests --test-dir data/v2/tests

產物（全部進 repo，給賽後的報告與論文用）：

| 檔案 | 內容 |
|---|---|
| `python.txt` | Python 版本與平台 |
| `requirements.txt` | 原始碼實際 import 的第三方套件（AST 掃描 src/、scripts/），版本釘死 |
| `models.tsv` | 上線用的每個 checkpoint 與外部模型的 sha256、大小、用途、來源 |
| `train_args/*.json` | 每個 checkpoint 內存的訓練參數（不靠記憶或文件，直接從權重檔取出） |
| `data_manifest.json` | 訓練集與測試檔的 sha256，以及整體指紋 |
| `commit_map.tsv` | 2026-09-29 作者改名改寫了所有 commit 雜湊；帳本引用舊雜湊，這裡給新舊對照 |
"""
from __future__ import annotations

from aicup.data import TRAINING_DIR  # noqa: E402
import argparse
import ast
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata

REPRO = "repro"

#: 上線用的 checkpoint（與 `predict.py` 的預設一致）。用途寫完整，兩個月後要能單獨讀懂。
MODELS = [
    ("cache/rank_mlp_opp_big/models.pt", "競賽二 單局 MLP（5 seed，1178 維：我方＋對手手工特徵）"),
    ("cache/rank_question_opp/models.pt", "競賽二 題目層級 MLP（3 seed，聚合特徵）"),
    ("cache/rank_seq_opp_full/models.pt", "競賽二 逐手序列 1D-CNN（我方＋對手 24 維/手，全訓練集）"),
    ("cache/rank_seq_shape/models.pt", "競賽二 逐手序列 1D-CNN（＋棋形，48 維/手）"),
    ("cache/board/best.pt", "競賽二 板面 CNN（34 層盤面快照、Set Transformer 集合池化，160k 步）"),
    ("cache/feature_metric_h1024s12k/models.pt", "競賽一 度量學習（8 seed、隱藏層 1024、12000 步、ArcFace）"),
    ("cache/skill/best.pt", "（未上線）以棋力為條件的走法模型，RESULTS.md §2.33"),
    ("cache/seq_identity/step12000.pt", "競賽一 v2 逐手序列身分模型（sequences＋seqshape，seed 0，第 12k 步）"),
    ("cache/seq_identity_30k_s2/step14000.pt", "競賽一 v2 逐手序列身分模型（sequences＋seqshape，seed 2，第 14k 步）"),
    ("cache/seq_identity_30k_s3q/step12000.pt", "競賽一 v2 逐手序列身分模型（＋seqqual，seed 3，第 12k 步）"),
    ("cache/seq_identity_patch/step12000.pt", "競賽一 v2 逐手序列身分模型（＋5x5 局部盤面，50 萬盤，第 12k 步）"),
    ("cache/seq_identity_patch_s6/step12000.pt", "競賽一 v2 逐手序列身分模型（＋5x5 局部盤面，seed 6，第 12k 步）"),
    ("cache/seq_identity_patch7/step16000.pt", "競賽一 v2 逐手序列身分模型（＋7x7 局部盤面，全部 78 萬盤，第 16k 步）"),
    ("cache/seq_identity_all256/step12000.pt", "競賽一 v2 逐手序列身分模型（全部 132 維逐手特徵、寬 256，第 12k 步）"),
]
#: 外部模型：來源與授權要寫進最終報告（官方 Q&A Q5）
EXTERNAL = [
    ("cache/katago/b18c384nbt-humanv0.bin.gz",
     "競賽二 KataGo HumanSL（10 種棋力 profile 的人類走法機率）",
     "https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz",
     "KataGo network license"),
    (None, "競賽二 KataGo 主模型（HumanSL 查詢時一併載入、算目數損失）",
     "Homebrew katago 1.18.2 附帶：kata1-b18c384nbt-s9996604416-d4316597426.bin.gz",
     "KataGo network license"),
]
#: import 名稱 → 發行套件名稱（兩者不同的）
DIST = {"sklearn": "scikit-learn", "yaml": "PyYAML", "PIL": "Pillow"}


def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def write(name: str, text: str) -> None:
    with open(os.path.join(REPRO, name), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  寫出 repro/{name}")


def third_party_imports() -> list[str]:
    stdlib = set(sys.stdlib_module_names)
    mods = set()
    for p in glob.glob("src/**/*.py", recursive=True) + glob.glob("scripts/*.py"):
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
    local = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob("scripts/*.py")}
    return sorted(m for m in mods if m not in stdlib and m not in local
                  and m not in ("aicup", "__future__"))


def build_requirements() -> None:
    lines, missing = [], []
    for m in third_party_imports():
        dist = DIST.get(m, m)
        try:
            lines.append(f"{dist}=={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            missing.append(m)
    head = "# 由 scripts/build_repro.py 掃描 src/、scripts/ 的 import 產生；版本為產生當下的安裝版本\n"
    if missing:
        head += f"# 未安裝、未列入（僅在選用路徑中 import）：{', '.join(missing)}\n"
    write("requirements.txt", head + "\n".join(lines) + "\n")


def build_models() -> None:
    import torch
    os.makedirs(os.path.join(REPRO, "train_args"), exist_ok=True)
    rows = ["path\tsha256\tbytes\t用途\t來源"]
    for path, use in MODELS:
        if not os.path.exists(path):
            rows.append(f"{path}\t（缺）\t\t{use}\t本專案訓練"); continue
        rows.append(f"{path}\t{sha256(path)}\t{os.path.getsize(path)}\t{use}\t本專案訓練")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        info = {"checkpoint": path, "sha256": sha256(path)}
        for k in ("args", "config", "step", "score", "blend_weight", "v2_score", "stores", "n_feat", "width", "dim", "in_scale"):
            if isinstance(ck, dict) and k in ck:
                info[k] = ck[k]
        name = path.replace("cache/", "").replace("/", "__").replace(".pt", "") + ".json"
        with open(os.path.join(REPRO, "train_args", name), "w", encoding="utf-8") as fh:
            json.dump(info, fh, ensure_ascii=False, indent=1, default=str)
    for path, use, src, lic in EXTERNAL:
        if path is None:
            hits = sorted(glob.glob("/opt/homebrew/Cellar/katago/*/share/katago/kata1-b18c384nbt-*.bin.gz"))
            path = hits[-1] if hits else None
        if path and os.path.exists(path):
            rows.append(f"{path}\t{sha256(path)}\t{os.path.getsize(path)}\t{use}\t{src}（{lic}）")
        else:
            rows.append(f"（缺）\t\t\t{use}\t{src}（{lic}）")
    write("models.tsv", "\n".join(rows) + "\n")
    print(f"  寫出 repro/train_args/（{len(MODELS)} 份）")


def build_data_manifest(test_dirs) -> None:
    files = {}
    for p in sorted(glob.glob(os.path.join(TRAINING_DIR, "train_*.csv"))):
        files[p] = sha256(p)
    for d in test_dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.csv")) + glob.glob(os.path.join(d, "tests_*", "*.csv"))):
            files[p] = sha256(p)
    whole = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in files.items()).encode()).hexdigest()
    with open(os.path.join(REPRO, "data_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"all_files": len(files), "all_sha256": whole, "files": files,
                   "note": "訓練集為官方 v1.0（v1.1 只補上 HD[n]，刻意不換：HumanSL 儲存檔以 SGF 原文雜湊為鍵）"},
                  fh, ensure_ascii=False, indent=1)
    print(f"  寫出 repro/data_manifest.json（{len(files)} 個檔，指紋 {whole[:16]}）")


def build_commit_map() -> None:
    """舊雜湊（改名前，tag backup/remote-before-author-rename）→ 新雜湊，依主旨＋作者時間配對。"""
    def log(ref):
        out = subprocess.run(["git", "log", "--format=%h%x09%at%x09%s", ref],
                             capture_output=True, text=True).stdout
        return [ln.split("\t", 2) for ln in out.splitlines() if ln.count("\t") >= 2]
    old = log("backup/remote-before-author-rename")
    if not old:
        print("  （找不到改名前的 tag，略過 commit_map）"); return
    new = {(t, s): h for h, t, s in log("main")}
    rows = ["# 2026-09-29 作者改名（士育→ShiYu）改寫了全部 commit 雜湊；內容（tree）不變。",
            "# 帳本（docs/RESULTS.md）引用的是左欄的舊雜湊，在遠端請查右欄。", "old\tnew\tsubject"]
    for h, t, s in old:
        rows.append(f"{h}\t{new.get((t, s), '（無對應）')}\t{s}")
    write("commit_map.tsv", "\n".join(rows) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", action="append", default=[],
                    help="一併記錄指紋的測試檔目錄（可重複），例如 data（含 tests_rank/、tests_player/）、data/mock_tests")
    args = ap.parse_args()
    os.makedirs(REPRO, exist_ok=True)
    write("python.txt", f"python {platform.python_version()} {platform.machine()} "
                        f"{platform.system()} {platform.release()}\n")
    build_requirements()
    build_models()
    build_data_manifest(args.test_dir)
    build_commit_map()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
