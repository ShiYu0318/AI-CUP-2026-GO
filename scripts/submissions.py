"""提交紀錄：每份提交檔的出處、本地分數、以及上傳後的 Public LB 分數。

    python3 scripts/submissions.py add cache/submissions/submission_rank.csv --task rank --variant default \\
        --cmd "python3 scripts/predict.py --test-dir data"
    python3 scripts/submissions.py lb R001 0.6123 --rank 12/80 --at "2026-11-04 11:42:10"
    python3 scripts/submissions.py note R001 "第一次上傳"
    python3 scripts/submissions.py show

兩個子題分開上傳、分開計分，所以編號分開：**R**（棋力預測）與 **P**（玩家辨識）。
檔案分放 `submissions/rank/`、`submissions/player/`，檔名用短編號（`r001.csv`）：PV 競賽踩過 AIdea 有未公開的檔名規則，
長檔名被拒收、改成短編號就被接受。

為什麼要記
----------
1. **最終成績取「最後一次上傳」**——必須知道每個子題最後送出的是哪一份。
2. **Public LB 是第一個來自官方評分的數字**：每份同時有本地分數與 LB，
   累積幾筆就能看出兩把尺的偏移；但 Public 只有 400 題（標準誤約 0.025），
   **LB 只用來校準，不用來挑模型**。
3. **可重現**：commit ＋ 完整指令 ＋ 模型與資料的雜湊，賽後寫報告與論文時能原樣重產
   （配方寫在 `repro/recipes/<編號>.json`）。

事實來源是 `docs/submissions.csv`；`docs/SUBMISSIONS.md` 由本程式**自動產生**，不要手改。
兩者與實驗帳本一樣只留本機，不進 repo。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime

LOG = os.path.join("docs", "submissions.csv")
DOC = os.path.join("docs", "SUBMISSIONS.md")
SUB_DIR = "submissions"
RECIPES = os.path.join("repro", "recipes")
RANKS = {"D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"}

#: (子題, 變體) → **絕對**的方法描述（完整元件鏈）。兩個月後要能單獨讀懂，不寫「同上，但……」。
_RANK = ("特徵：589 維手工特徵 x 我方／對手雙視角（1178 維）｜"
         "成分：單局 MLP（5 seed，軟標籤＋CORN）、題目層級 MLP（3 seed）、"
         "逐手序列 1D-CNN 兩變體（opp_full、shape）、板面 CNN（34 層盤面快照、Set Transformer 集合池化、160k 步）｜"
         "融合：學習式 stacker（邏輯迴歸 C=1，各成分 log 後驗＋期望等級＋信心＋熵＋log 盤數）｜"
         "解碼：DP 最佳化切點")
METHOD = {
    ("rank", "default"): _RANK + "（全驗證集 1,602 位玩家、4,776 題校準）",
    ("rank", "humansl"): (_RANK.replace("、板面 CNN", "、KataGo HumanSL（b18c384nbt-humanv0，10 種棋力 profile、"
                                         "每盤 24 手、每手平均對數似然）、板面 CNN")
                          + "（HumanSL 棋池 800 位玩家、4,800 題校準）"),
    ("rank", "humansl_uniform"): (_RANK.replace("、板面 CNN", "、KataGo HumanSL（b18c384nbt-humanv0，10 種棋力 profile、"
                                                 "每盤 24 手、每手平均對數似然）、板面 CNN")
                                  + "（HumanSL 棋池 800 位玩家、4,800 題校準；校準集重抽成**均勻**等級先驗）"),
    ("rank", "humansl_p5"): _RANK + "＋**KataGo HumanSL 5 個 profile（8k,2k,2d,4d,6d）x 每盤全部我方手數**（mean，棋池校準 800 人／4,800 題）",
    ("rank", "humansl_opp"): _RANK + "＋KataGo HumanSL 5 profile x 全部手數，**我方＋對手落子兩個成分**（mean，棋池校準）",
    ("rank", "uniform_thresholds"): _RANK + "（stacker 以全驗證集 4,776 題擬合；**切點改在均勻等級先驗下校準**）",
    ("player", "pool_classifier"): ("特徵：度量學習 MLP（h1024、12000 步、8 seed）的 embedding；"
                                    "**用候選池本身（每人 100 盤）現場訓練 400 類邏輯迴歸**（L-BFGS），"
                                    "每題各盤 log 機率平均 → Sinkhorn（eps 0.05）。官方 Q&A Q10 允許候選池用於校正"),
    ("player", "v2_template"): ("**更新版測試集（1,600 題）**：原測試集 pub_/priv_ 沿用 P003；"
                                "新增測試集 pub2_/priv2_（候選 600 人 x 10 盤、題目 1/2/3/5 盤、非一對一）"
                                "用度量學習 MLP 模板餘弦、**不做 Sinkhorn**"),
    ("player", "v2_classifier"): ("**更新版測試集**：舊題沿用 P003；新題用候選池分類器（600 類邏輯迴歸，每人 10 盤，L2 1e-4），不做 Sinkhorn"),
    ("player", "v2_seq"): ("**更新版測試集**：舊題沿用 P003；新題用候選池分類器（L2 1e-4），"
                           "embedding ＝度量學習 MLP 8 段＋**逐手序列身分模型**（1D-CNN＋ArcFace）第 9 段"),
    ("player", "v2_zfuse"): ("**更新版測試集**：舊題沿用 P003；新題 = z(逐手序列身分模型的模板餘弦) + 0.4·z(度量 MLP 的候選池分類器)"),
    ("player", "no_sinkhorn"): ("特徵：手工特徵 1164 維（我方＋對手＋棋形雙視角，去掉全域落子位置）｜"
                                "度量學習：MLP（隱藏層 1024）＋Sub-center ArcFace＋batch-hard triplet，8 seed、12000 步｜"
                                "玩家模板：L2→平均→L2｜指派：**不做全域指派**，每題各自取餘弦最近的 Top-5（消融）"),
    ("player", "default"): ("特徵：手工特徵 1164 維（我方＋對手＋棋形雙視角，去掉全域落子位置）｜"
                            "度量學習：MLP（隱藏層 1024）＋Sub-center ArcFace＋batch-hard triplet，8 seed、12000 步｜"
                            "玩家模板：L2→平均→L2｜指派：Sinkhorn 全域一對一（ε=0.002）→ Top-5"),
}
METHOD_SHORT = {("rank", "default"): "生產集成（不含 HumanSL）",
                ("rank", "humansl"): "生產集成＋KataGo HumanSL",
                ("rank", "humansl_uniform"): "＋HumanSL、均勻等級先驗",
                ("rank", "uniform_thresholds"): "生產集成、切點用均勻先驗",
                ("rank", "humansl_p5"): "生產集成＋HumanSL（5 profile、全部手數）",
                ("rank", "humansl_opp"): "生產集成＋HumanSL 我方＋對手",
                ("player", "default"): "度量學習＋Sinkhorn",
                ("player", "pool_classifier"): "候選池分類器＋Sinkhorn",
                ("player", "v2_template"): "更新版：舊題沿用 P003、新題模板餘弦",
                ("player", "v2_classifier"): "更新版：舊題沿用 P003、新題候選池分類器",
                ("player", "v2_seq"): "更新版：新題分類器＋序列身分 embedding",
                ("player", "v2_zfuse"): "更新版：新題 序列模板＋度量分類器（z 分數融合）",
                ("player", "no_sinkhorn"): "度量學習、無 Sinkhorn（消融）"}
#: 變體 → 本地分數（建立紀錄時的預設；可用 --cv / --e2e 覆寫）。協定寫在 cv_source。
LOCAL = {
    ("rank", "default"): ("0.6226", "8 seed x 5 切分、player-disjoint、官方局數分布（GATE 13）", "0.6216"),
    ("rank", "humansl"): ("0.6034", "HumanSL 棋池 800 位玩家、8 seed x 5 切分；同協定的 default 為 0.5846", "0.6366"),
    ("player", "default"): ("0.9901", "8 評估 seed x 5 種候選庫組成，取最壞組成（平均 0.9944）", "0.9934"),
    ("rank", "humansl_uniform"): ("", "未以 8 seed 量；先驗之間最壞情況差約 0.010（RESULTS.md §2.7）", ""),
    ("rank", "humansl_opp"): ("0.6252", "棋池前 200 位、8 seed x 5 切分（同條件只加我方 0.5876、無 HumanSL 0.5519）", ""),
    ("rank", "humansl_p5"): ("0.6140", "棋池協定 8 seed x 5 切分（舊版 HumanSL 0.6034、無 HumanSL 0.5846）", ""),
    ("rank", "uniform_thresholds"): ("0.5970", "每級等量的模擬測試集（8 seed x 5 切分）；同條件下現行解碼 0.5908", ""),
    ("player", "pool_classifier"): ("0.9969", "4 seed x 5 組成、20 池逐池配對（現行 0.9945）；模擬測試集端到端 0.9992（現行 0.9934）", ""),
    ("player", "v2_template"): ("0.4145", "新增測試集本地模擬（600 候選 x 10 盤、400 題、4 seed）；只計新題", ""),
    ("player", "v2_zfuse"): ("0.5632", "新增測試集本地模擬 4 seed（序列模型第 12,000 步；c=0.3 0.5632、c=0.5 0.5616；P005 方法 0.4354）；只計新題", ""),
    ("player", "v2_seq"): ("0.4511", "新增測試集本地模擬 4 seed（序列模型第 3,000 步；分類器度量 0.4354）；只計新題", ""),
    ("player", "v2_classifier"): ("0.4354", "新增測試集本地模擬（同上，4/4 勝模板）；只計新題", ""),
    ("player", "no_sinkhorn"): ("", "現行模型未量；舊模型上 Sinkhorn 值 +0.0400（15/15，RESULTS.md §1.1）", ""),
}
#: 變體 → 依賴的模型（與 repro/models.tsv 對照，記進配方）
MODELS = {
    "rank": ["cache/rank_mlp_opp_big/models.pt", "cache/rank_question_opp/models.pt",
             "cache/rank_seq_opp_full/models.pt", "cache/rank_seq_shape/models.pt", "cache/board/best.pt"],
    "player": ["cache/feature_metric_h1024s12k/models.pt"],
}
EXTRA_MODELS = {("rank", "humansl"): ["cache/katago/b18c384nbt-humanv0.bin.gz", "cache/humansl/pool.npz"],
                ("rank", "humansl_uniform"): ["cache/katago/b18c384nbt-humanv0.bin.gz", "cache/humansl/pool.npz"]}

FIELDS = ["id", "task", "created_at", "file", "sha256_8", "n_rows", "variant", "method", "cmd",
          "git_commit", "git_dirty", "data_fingerprint", "models_fingerprint",
          "cv_score", "cv_source", "e2e_mock", "format_check", "env",
          "submitted_at", "public_lb", "lb_rank", "note"]


def _rows():
    if not os.path.exists(LOG):
        return []
    with open(LOG, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(rows):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    render(rows)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_dirty():
    """`src/`、`scripts/` 有未提交的修改 → 記下的 commit 無法完整描述實際跑的程式碼。"""
    return "dirty" if _git("status", "--porcelain", "--", "src", "scripts") else "clean"


def data_fingerprint():
    try:
        with open(os.path.join("repro", "data_manifest.json"), encoding="utf-8") as f:
            return json.load(f)["all_sha256"][:16]
    except (OSError, ValueError, KeyError):
        return ""


def env_info():
    try:
        import torch
        tv = torch.__version__
    except Exception:  # noqa: BLE001
        tv = "?"
    kv = ""
    try:
        kv = subprocess.run(["katago", "version"], capture_output=True, text=True).stdout.split("\n")[0]
    except OSError:
        pass
    return f"python {platform.python_version()}｜torch {tv}｜{kv or 'katago ?'}｜{platform.machine()}"


def format_check(path, task):
    """官方格式的最低檢查：列數、欄位、無空值；棋力標籤合法；Top-5 互不重複。

    競賽一自 2026-11-11 更新版測試集起為 1,600 列（pub_/priv_/pub2_/priv2_），舊版提交為 800 列。
    """
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    want = (1600, 800) if task == "player" else (800,)
    if len(rows) not in want:
        return f"✗ {len(rows)} 列（應為 {' 或 '.join(map(str, want))}）"
    n = f"✓（{len(rows)} 列）"
    if task == "rank":
        bad = [r for r in rows if r.get("pred_rank") not in RANKS]
        return f"✗ {len(bad)} 列棋力標籤不合法" if bad else n
    cols = [f"top_{i}" for i in range(1, 6)]
    bad = [r for r in rows if len({r.get(c) for c in cols}) != 5 or not all(r.get(c) for c in cols)]
    return f"✗ {len(bad)} 列 Top-5 有空值或重複" if bad else n


def next_id(rows, task):
    prefix = "R" if task == "rank" else "P"
    n = max([int(r["id"][1:]) for r in rows if r["id"].startswith(prefix)] or [0]) + 1
    return f"{prefix}{n:03d}"


def add(args):
    rows = _rows()
    rid = next_id(rows, args.task)
    sub = os.path.join(SUB_DIR, args.task)                # submissions/rank/、submissions/player/
    os.makedirs(sub, exist_ok=True)
    dst = os.path.join(sub, f"{rid.lower()}.csv")
    shutil.copyfile(args.file, dst)
    key = (args.task, args.variant)
    cv, cv_src, e2e = LOCAL.get(key, ("", "", ""))
    models = MODELS[args.task] + EXTRA_MODELS.get(key, [])
    model_sha = {m: (_sha(m) if os.path.exists(m) else "（缺）") for m in models}
    mfp = hashlib.sha256(json.dumps(model_sha, sort_keys=True).encode()).hexdigest()[:16]
    row = {
        "id": rid, "task": args.task, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "file": dst, "sha256_8": _sha(dst)[:8], "n_rows": str(sum(1 for _ in open(dst)) - 1),
        "variant": args.variant, "method": METHOD.get(key, args.method or ""), "cmd": args.cmd,
        "git_commit": _git("rev-parse", "--short", "HEAD"), "git_dirty": git_dirty(),
        "data_fingerprint": data_fingerprint(), "models_fingerprint": mfp,
        "cv_score": args.cv or cv, "cv_source": args.cv_source or cv_src,
        "e2e_mock": args.e2e or e2e, "format_check": format_check(dst, args.task), "env": env_info(),
        "submitted_at": "", "public_lb": "", "lb_rank": "", "note": args.note or "",
    }
    rows.append(row)
    _save(rows)
    os.makedirs(RECIPES, exist_ok=True)
    with open(os.path.join(RECIPES, f"{rid.lower()}.json"), "w", encoding="utf-8") as fh:
        json.dump({"id": rid, "task": args.task, "variant": args.variant, "cmd": args.cmd,
                   "git_commit": row["git_commit"], "git_dirty": row["git_dirty"],
                   "data_fingerprint": row["data_fingerprint"], "models_sha256": model_sha,
                   "output": {"source": args.file, "sha256": _sha(dst)}},
                  fh, ensure_ascii=False, indent=1)
    print(f"已登記 {rid} → {dst}（{row['format_check']}，sha256 {row['sha256_8']}，"
          f"commit {row['git_commit']} {row['git_dirty']}）")
    if row["git_dirty"] == "dirty":
        print("⚠️ 工作區有未提交的修改：這份提交無法由 commit 單獨重現，建議先 commit 再重產")


def lb(args):
    rows = _rows()
    hit = [r for r in rows if r["id"] == args.id.upper()]
    if not hit:
        raise SystemExit(f"找不到 {args.id}")
    r = hit[0]
    r["public_lb"] = args.score
    r["submitted_at"] = args.at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.rank:
        r["lb_rank"] = args.rank
    if args.note:
        r["note"] = (r["note"] + "；" if r["note"] else "") + args.note
    _save(rows)
    print(f"{r['id']}：Public LB {args.score}（{r['submitted_at']}）")


def note(args):
    rows = _rows()
    for r in rows:
        if r["id"] == args.id.upper():
            r["note"] = (r["note"] + "；" if r["note"] else "") + args.text
    _save(rows)


def render(rows=None):
    rows = _rows() if rows is None else rows
    g = lambda r, k: r.get(k) or "—"  # noqa: E731
    L = ["# 提交紀錄", "",
         "> ⚠️ 本檔由 `scripts/submissions.py` **自動產生**，不要手改。事實來源是 `docs/submissions.csv`。", ""]
    for task, title in (("rank", "競賽二 棋力預測（R）"), ("player", "競賽一 玩家辨識（P）")):
        tr = [r for r in rows if r["task"] == task]
        L += [f"## {title}", "",
              "| 編號 | 檔名 | 上傳時間 | **Public LB** | 排名 | 本地分數 | 模擬測試集 | 方法 |",
              "|---|---|---|---:|---|---:|---:|---|"]
        for r in tr:
            L.append(f"| **{r['id']}** | `{os.path.basename(r['file'])}` | {g(r, 'submitted_at')} | "
                     f"**{g(r, 'public_lb')}** | {g(r, 'lb_rank')} | {g(r, 'cv_score')} | {g(r, 'e2e_mock')} | "
                     f"{METHOD_SHORT.get((task, r['variant']), r['variant'])} |")
        up = [r for r in tr if r.get("public_lb")]
        if up:
            last = max(up, key=lambda r: r["submitted_at"])
            L += ["", f"- 已上傳 {len(up)} 份；**最後一次送出：{last['id']}**——⚠️ 最終成績取每個子題的最後一次上傳"]
            sc = [r for r in up if r.get("cv_score")]
            if sc:
                off = sum(float(r["public_lb"]) - float(r["cv_score"]) for r in sc) / len(sc)
                L.append(f"- Public LB − 本地分數 平均偏移 {off:+.4f}（{len(sc)} 份）")
        elif tr:
            L += ["", f"- 已登記 {len(tr)} 份，尚未上傳"]
        L.append("")
    L += ["## 各編號", ""]
    for r in rows:
        L += [f"### {r['id']} — `{os.path.basename(r['file'])}`（{r['task']}）", "",
              f"變體 `{r['variant']}`，產生於 {r['created_at']}", "", "**方法**", ""]
        L += [f"- {p}" for p in (r.get("method") or "（未記錄）").split("｜")]
        L += ["", "**分數**", "",
              f"- Public LB：**{g(r, 'public_lb')}**（排名 {g(r, 'lb_rank')}、上傳 {g(r, 'submitted_at')}）",
              f"- 本地：{g(r, 'cv_score')}（{g(r, 'cv_source')}）",
              f"- 模擬測試集端到端：{g(r, 'e2e_mock')}",
              "", "**出處**", "",
              f"- 指令：`{g(r, 'cmd')}`",
              f"- commit `{g(r, 'git_commit')}`（工作區 {g(r, 'git_dirty')}）、資料指紋 `{g(r, 'data_fingerprint')}`、"
              f"模型指紋 `{g(r, 'models_fingerprint')}`、檔案 sha256 `{g(r, 'sha256_8')}`",
              f"- 格式檢查 {g(r, 'format_check')}｜{g(r, 'env')}",
              f"- 配方：`repro/recipes/{r['id'].lower()}.json`"]
        if r.get("note"):
            L += ["", f"**備註**：{r['note']}"]
        L.append("")
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd_name", required=True)
    a = sp.add_parser("add", help="登記一份新的提交檔（複製到 submissions/ 並寫配方）")
    a.add_argument("file")
    a.add_argument("--task", choices=["rank", "player"], required=True)
    a.add_argument("--variant", required=True, help="例如 default、humansl")
    a.add_argument("--cmd", required=True, help="產生這份檔案的完整指令")
    a.add_argument("--method", default="", help="METHOD 表沒有的變體才需要")
    a.add_argument("--cv", default="")
    a.add_argument("--cv-source", default="")
    a.add_argument("--e2e", default="")
    a.add_argument("--note", default="")
    b = sp.add_parser("lb", help="記錄上傳後的 Public LB 分數")
    b.add_argument("id")
    b.add_argument("score")
    b.add_argument("--rank", default="")
    b.add_argument("--at", default="", help="上傳時間（預設現在）")
    b.add_argument("--note", default="")
    n = sp.add_parser("note", help="追加備註")
    n.add_argument("id")
    n.add_argument("text")
    sp.add_parser("show", help="重新產生 SUBMISSIONS.md 並印出摘要")
    args = ap.parse_args()
    if args.cmd_name == "add":
        add(args)
    elif args.cmd_name == "lb":
        lb(args)
    elif args.cmd_name == "note":
        note(args)
    else:
        render()
        for r in _rows():
            print(f"{r['id']:<5} {r['task']:<6} {r['variant']:<8} 本地 {r['cv_score'] or '—':<7} "
                  f"LB {r['public_lb'] or '—':<8} {r['submitted_at'] or '未上傳'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
