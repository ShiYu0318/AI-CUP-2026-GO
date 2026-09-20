"""無人值守的實驗迴圈：跑 → 用誠實協定 gate → 記錄 → commit → 跑下一個。

**為什麼要有這支**：這個專案的瓶頸不是方法，是**迭代速度**。
每個實驗都要人在旁邊等通知、看結果、決定下一步，一晚上只能推進兩三個。
這支把「跑實驗 → 評估 → 記錄」變成一個佇列，整夜自己推進。

## 設計上的三個刻意選擇

**1. Gate 用的是 RESULTS.md〇章的協定，不是訓練腳本自己報的分數。**
訓練腳本內建的評估是同集校準（樂觀）。這裡一律走 `scripts/eval_blend.py`：
8 組 player-disjoint、校準與評分玩家不重疊、DP 切點。

⚠️ **gate 的標準必須與實際會上線的管線一致，而且 `production` 要跟它同一把尺。**
真實踩過：`predict.py` 改用 stacker 之後，`eval_blend.py` 也跟著改，
但 state 裡的 `production` 還是固定權重時代的 0.5639——
下一個實驗會拿 stacker 分數（約 0.5715）去比 0.5639，**假性通過 +0.008**，
而那 0.008 只是換了評分標準，不是模型變好。
**換 gate 的定義時，一定要同時重設 `production`。**

**2. FLOOR：改進小於門檻就不算贏。**
本專案的 seed 間標準差約 ±0.010，所以低於 FLOOR 的「增益」多半是雜訊。
這是把「能寫死的就寫死」制度化——不讓迴圈去追雜訊。

**3. 平行化靠的是「裝置互補」，不是單純開更多工作。**
實測同一台機器上（batch 與模型都用實際訓練的設定）：

    逐手序列 1D-CNN（batch 256, 240 手）   MPS 435 ms   CPU 1849 ms   → MPS 快 4.25×
    單局 MLP（1178 維, batch 2048）        MPS 107 ms   CPU   43 ms   → CPU 快 2.5×

卷積在 MPS 上大贏，但**密集層的 MLP 在 MPS 上反而慢**——
每步要把 9.6 MB 搬上 GPU，而矩陣乘在 10 核 CPU 上已經夠快，搬運開銷吃掉全部好處。

MPS 只有一張 GPU，**兩個 MPS 工作並行不會有 2× 吞吐**，它們搶同一個裝置。
所以排程器限制**同時最多 1 個 MPS 工作**，其餘名額留給 CPU 工作——
這樣才是真的並行，而不是互相拖慢。

（順帶：這台是 Apple Silicon，**CUDA 永遠不可用**。`pick_device()` 的順序是
 cuda → mps → cpu，所以租到 NVIDIA GPU 時同一份程式不用改就會走 CUDA。）

**4. 它不會自己改 `predict.py` 的預設。**
通過 gate 的實驗會被記錄、commit，並把建議的一行改動印出來，
但換預設仍是人（或下一個 session）看過帳本後的決定。
自動改推論預設是不可逆的外顯行為，而且本專案已經被「安靜地換掉東西」
咬過五次——多等一次人工確認，比半夜自動上線一個沒人看過的設定安全。

用法：
    python3 scripts/auto_loop.py --dry-run      # 只印佇列，不執行
    nohup python3 scripts/auto_loop.py > artifacts/auto_loop.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = "artifacts/auto_loop_state.json"
LEDGER = "RESULTS.md"
FLOOR = 0.005          # 小於這個增益不算贏（seed 間標準差約 ±0.010）
BASELINE = 0.4631      # 官方 baseline（競賽二）


# ── 實驗佇列 ────────────────────────────────────────────────────────────────
# 每一項：
#   name      識別名，會寫進帳本
#   why       這個實驗在問什麼（寫不出來的實驗不該進佇列）
#   train     產生模型的指令（None = 只評估，不訓練）
#   produces  訓練會產出的檔案；已存在就跳過訓練
#   eval      評估指令，必須輸出 --json-out 指定的 json
#   json      評估結果的 json 路徑
#   device    "mps" 或 "cpu"。實測（見下方）兩類模型的最佳裝置**相反**，
#             排程器據此決定誰能跟誰並行。
#   mem_gb    訓練時的常駐記憶體估計（訓練張量 + 模型 + 工作空間）。
#             排程器用它決定能同時跑幾個——26 GB 的機器上盲目平行會直接爆掉，
#             而 swap 一旦發生，兩個工作都會慢十倍以上，比序列執行還糟。
EXPERIMENTS = [
    {
        "name": "seq_5seed",
        "device": "mps",
        "mem_gb": 6.5,
        "why": "序列模型目前只有 3 個 seed。加到 5 個純粹降變異，"
               "是最便宜的增益，也能檢驗現有 0.5639 有多少是 seed 運氣",
        "train": ["python3", "scripts/train_rank_seq.py",
                  "--extra-sequences", "cache/sequences_opp.npy",
                  "--seeds", "5", "--steps", "12000", "--train-games", "400000",
                  "--out", "cache/rank_seq_opp5"],
        "produces": "cache/rank_seq_opp5/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8",
                 "--json-out", "artifacts/blend_seq5.json"],
        "eval_env": {"AICUP_SEQ": "cache/rank_seq_opp5/models.pt"},
        "json": "artifacts/blend_seq5.json",
    },
    {
        "name": "seq_full_data",
        "device": "mps",
        "mem_gb": 7.5,
        "why": "序列模型只訓練 40 萬局，單局 MLP 是 78 萬局。"
               "序列模型自己就比整條手工特徵管線強，值得餵滿",
        "train": ["python3", "scripts/train_rank_seq.py",
                  "--extra-sequences", "cache/sequences_opp.npy",
                  "--seeds", "3", "--steps", "16000", "--train-games", "0",
                  "--out", "cache/rank_seq_opp_full"],
        "produces": "cache/rank_seq_opp_full/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8",
                 "--json-out", "artifacts/blend_seqfull.json"],
        "eval_env": {"AICUP_SEQ": "cache/rank_seq_opp_full/models.pt"},
        "json": "artifacts/blend_seqfull.json",
    },
    {
        "name": "seq_240moves",
        "device": "mps",
        "mem_gb": 5.0,
        "why": "序列只看前 160 手，但棋局手數中位數是 222——**平均丟掉 25%，"
               "而且只有 16.7% 的棋局被完整涵蓋**。丟掉的還是尾盤，"
               "而官子填空、終局打吃正是最強的棋力訊號所在。240 手可涵蓋 96%",
        "train": ["python3", "scripts/train_rank_seq.py",
                  "--sequences", "cache/sequences240.npy",
                  "--extra-sequences", "cache/sequences240_opp.npy",
                  "--seeds", "3", "--steps", "12000", "--train-games", "300000",
                  "--out", "cache/rank_seq240"],
        "produces": "cache/rank_seq240/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8",
                 "--json-out", "artifacts/blend_seq240.json"],
        "eval_env": {"AICUP_SEQ": "cache/rank_seq240/models.pt",
                     "AICUP_SEQ_STORES": "cache/sequences240.npy,cache/sequences240_opp.npy"},
        "json": "artifacts/blend_seq240.json",
    },
    {
        "name": "mlp_8seed",
        "device": "cpu",
        "mem_gb": 5.0,
        "why": "單局 MLP 目前 5 個 seed。加到 8 個降變異。"
               "**走 CPU**——實測 MLP 在 MPS 上反而慢 2.5 倍，"
               "而且這樣能與序列模型的 MPS 訓練真正並行",
        "train": ["python3", "scripts/train_rank_mlp.py",
                  "--extra-features", "cache/features_opp.npy",
                  "--seeds", "8", "--steps", "20000", "--hidden", "1024",
                  "--train-games", "0", "--device", "cpu",
                  "--out", "cache/rank_mlp_opp8"],
        "produces": "cache/rank_mlp_opp8/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8",
                 "--json-out", "artifacts/blend_mlp8.json"],
        "eval_env": {"AICUP_GAME": "cache/rank_mlp_opp8/models.pt"},
        "json": "artifacts/blend_mlp8.json",
    },
    {
        "name": "seq_wider",
        "device": "mps",
        "mem_gb": 4.5,
        "why": "序列模型 width=128。加寬到 192 測容量是否還是瓶頸——"
               "若無增益，代表瓶頸在每手 24 維的特徵，而不在模型",
        "train": ["python3", "scripts/train_rank_seq.py",
                  "--extra-sequences", "cache/sequences_opp.npy",
                  "--seeds", "3", "--steps", "12000", "--train-games", "400000",
                  "--width", "192", "--out", "cache/rank_seq_w192"],
        "produces": "cache/rank_seq_w192/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8",
                 "--json-out", "artifacts/blend_seqw192.json"],
        "eval_env": {"AICUP_SEQ": "cache/rank_seq_w192/models.pt"},
        "json": "artifacts/blend_seqw192.json",
    },
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_state() -> dict:
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"production": None, "done": {}}


def save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)


def run(cmd, env_extra=None, timeout=None) -> tuple[int, str]:
    env = os.environ.copy()
    env.update(env_extra or {})
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def total_memory_gb() -> float:
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                  capture_output=True, text=True).stdout) / 1e9
    except Exception:
        return 16.0


def spawn(cmd, log_path, env_extra=None):
    env = os.environ.copy()
    env.update(env_extra or {})
    fh = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh,
                            stderr=subprocess.STDOUT), fh


def record(exp: dict, score: float, prod: float, verdict: str, detail: str) -> None:
    """把結果寫進 RESULTS.md 的無效方法表或主線表——**成功失敗都要記**。"""
    line = (f"| {datetime.now():%m-%d} | **{exp['name']}** | {score:.4f} | "
            f"相對現況 {score - prod:+.4f}（門檻 {FLOOR:+.3f}）→ **{verdict}**。"
            f"{exp['why']}。{detail} |\n")
    path = os.path.join(ROOT, LEDGER)
    with open(path, encoding="utf-8") as f:
        s = f.read()
    anchor = "## 四、修掉的靜默 bug"
    if anchor not in s:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + line)
        return
    block = "\n### 自動迴圈的實驗紀錄\n\n| 日期 | 實驗 | 分數 | 結論 |\n|---|---|---:|---|\n"
    if "### 自動迴圈的實驗紀錄" not in s:
        s = s.replace(anchor, block + line + "\n---\n\n" + anchor)
    else:
        i = s.index("### 自動迴圈的實驗紀錄")
        j = s.index("\n\n", s.index("|---|---|---:|---|", i))
        s = s[:j + 1] + line + s[j + 1:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def git_commit(msg: str) -> None:
    run(["git", "add", "-A", "RESULTS.md", "artifacts/auto_loop_state.json"])
    run(["git", "commit", "-q", "-m", msg])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定的實驗名稱")
    ap.add_argument("--jobs", type=int, default=1,
                    help="同時訓練幾個實驗。排程器會依各實驗宣告的 mem_gb "
                         "與機器實體記憶體決定實際併發數——"
                         "⚠️ MPS 只有一張 GPU，平行主要是填滿資料載入與 CPU 階段的空檔，"
                         "不會線性加速；而記憶體超賣導致 swap 會讓所有工作慢十倍以上")
    ap.add_argument("--mem-budget", type=float, default=None,
                    help="可用於訓練的記憶體上限（GB）。預設取實體記憶體的 65%%")
    args = ap.parse_args()

    st = load_state()
    queue = [e for e in EXPERIMENTS
             if (args.only is None or e["name"] in args.only)
             and e["name"] not in st["done"]]

    if args.dry_run:
        log(f"佇列 {len(queue)} 項，門檻 {args.floor:+.3f}，"
            f"目前生產分數 {st['production']}")
        for e in queue:
            log(f"  - {e['name']}：{e['why']}")
        return 0

    if st["production"] is None:
        log("尚無生產分數，先跑一次目前的預設組合建立基準…")
        rc, out = run(["python3", "scripts/eval_blend.py", "--seeds", "8",
                       "--json-out", "artifacts/blend_default.json"])
        if rc != 0:
            log(f"基準評估失敗：\n{out[-1500:]}")
            return 1
        with open(os.path.join(ROOT, "artifacts/blend_default.json")) as f:
            st["production"] = json.load(f)["best_score"]
        save_state(st)
        log(f"生產分數 = {st['production']:.4f}")

    budget = args.mem_budget or total_memory_gb() * 0.65
    log(f"記憶體預算 {budget:.1f} GB（實體 {total_memory_gb():.0f} GB），"
        f"最多同時訓練 {args.jobs} 個")

    pending = list(queue)
    running = []        # [(exp, Popen, fh, log_path, t0)]
    to_gate = []        # 訓練完成、等待 gate 的實驗

    def used_mem():
        return sum(e.get("mem_gb", 4.0) for e, *_ in running)

    def mps_running():
        return sum(1 for e, *_ in running if e.get("device", "mps") == "mps")

    def can_start(exp):
        if len(running) >= args.jobs:
            return False
        # MPS 只有一張 GPU：兩個 MPS 工作並行只會互相拖慢，不會提高吞吐。
        if exp.get("device", "mps") == "mps" and mps_running() >= 1:
            return False
        return used_mem() + exp.get("mem_gb", 4.0) <= budget or not running

    while pending or running or to_gate:
        # 1) 在預算內盡量塞新的訓練
        while True:
            idx = next((i for i, e in enumerate(pending) if can_start(e)), None)
            if idx is None:
                break
            exp = pending.pop(idx)
            if not exp.get("train") or os.path.exists(os.path.join(ROOT, exp["produces"])):
                log(f"[{exp['name']}] 產物已存在，直接排入 gate")
                to_gate.append(exp)
                continue
            lp = os.path.join(ROOT, "artifacts", f"train_{exp['name']}.log")
            proc, fh = spawn(exp["train"], lp)
            running.append((exp, proc, fh, lp, time.time()))
            log(f"[{exp['name']}] 開始訓練於 {exp.get('device','mps').upper()}"
                f"（併發 {len(running)}/{args.jobs}，"
                f"預估佔用 {used_mem():.1f}/{budget:.1f} GB）")
            log(f"    {exp['why']}")

        # 2) 收割訓練完成的
        for item in list(running):
            exp, proc, fh, lp, t0 = item
            if proc.poll() is None:
                continue
            running.remove(item); fh.close()
            mins = (time.time() - t0) / 60
            if proc.returncode != 0:
                log(f"[{exp['name']}] 訓練失敗（{mins:.1f} 分），見 {lp}")
                st["done"][exp["name"]] = {"status": "train_failed"}
                save_state(st)
            else:
                log(f"[{exp['name']}] 訓練完成（{mins:.1f} 分）→ 排入 gate")
                to_gate.append(exp)

        # 3) Gate 一律**序列**執行。
        #    它要跑 8 組題目的推論，同樣吃 MPS；與訓練搶資源會讓兩邊都變慢，
        #    而且 gate 很快（有快取時幾秒），沒有平行的必要。
        if to_gate and (not running or len(running) < args.jobs or not pending):
            exp = to_gate.pop(0)
            log(f"[{exp['name']}] 用 8 組 player-disjoint 協定 gate…")
            rc, out = run(exp["eval"], env_extra=exp.get("eval_env"))
            if rc != 0:
                log(f"[{exp['name']}] 評估失敗：\n{out[-1200:]}")
                st["done"][exp["name"]] = {"status": "eval_failed"}
                save_state(st); continue
            with open(os.path.join(ROOT, exp["json"])) as f:
                res = json.load(f)
            score, prod = res["best_score"], st["production"]
            gain = score - prod
            won = gain >= args.floor
            verdict = "採用" if won else ("增益低於門檻" if gain > 0 else "無增益")
            log(f"[{exp['name']}] 分數 {score:.4f}（現況 {prod:.4f}，"
                f"差 {gain:+.4f}）→ {verdict}")
            detail = (f"建議把 `predict.py` 的 `--rank-seq` 預設改成 "
                      f"`{exp.get('eval_env', {}).get('AICUP_SEQ', '?')}`"
                      if won else "維持現況")
            record(exp, score, prod, verdict, detail)
            st["done"][exp["name"]] = {"score": score, "gain": gain, "won": won}
            if won:
                st["production"] = score
            save_state(st)
            git_commit(f"exp: {exp['name']} scores {score:.4f} ({verdict})")
            continue

        if running or pending or to_gate:
            time.sleep(20)

    log("=" * 62)
    log(f"佇列跑完。生產分數 {st['production']:.4f}"
        f"（官方 baseline {BASELINE}，{st['production'] / BASELINE:.2f}×）")
    won = [k for k, v in st["done"].items() if v.get("won")]
    log(f"通過 gate 的實驗：{won or '無'}")
    if won:
        log("⚠️ 這支不會自己改 predict.py 的預設——請看 RESULTS.md 的建議後手動確認")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
