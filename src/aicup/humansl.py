"""KataGo HumanSL：以棋力為條件的人類走法模型，對每一手棋打「在各棋力下的機率」。

這是競賽二的 SOTA 路線（Policies of Multiple Skill Levels for Better Strength
Estimation, arXiv:2505.00279）。一位棋力 r 的玩家，下出的棋在 profile r 下的機率最高
（Maia 的峰值特性）；把一組棋所有手數的 log 機率加總，就是這組棋在各棋力下的對數似然。

探針（RESULTS.md §2.32）：每盤只看 24 手，單盤 Spearman 0.677（我們的單局 MLP 0.606），
而且錯誤與我們所有成分幾乎無關（組內殘差相關 0.07）。

**外部依賴**（規則允許，報告須申報來源與貢獻）：
- `katago` 執行檔：`brew install katago`（v1.18.2，Metal 後端）
- 主模型：Homebrew 附帶的 kata1-b18c384nbt（只用來算目數損失）
- HumanSL 模型：https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz
  放在 `cache/katago/`

**一盤一筆特徵**，以 SGF＋執色的雜湊為鍵——訓練集與測試集共用同一個儲存檔，
不必關心那盤棋來自哪裡。
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import queue
import subprocess
import threading

import numpy as np

from .goboard import BLACK, normalise_color, parse_moves

# 我們的 10 個等級 → HumanSL profile。級位組（12k~10k、9k~7k、6k~4k、3k~1k）取中點。
PROFILES = ("rank_11k", "rank_8k", "rank_5k", "rank_2k",
            "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d", "rank_6d")
N_PROFILES = len(PROFILES)
COLS = "ABCDEFGHJKLMNOPQRST"
# 特徵欄位：每個 profile 的平均 log 機率、首選一致率；目數損失的 4 個統計量；兩個計數。
FEATURE_NAMES = ([f"logp_{p}" for p in PROFILES] + [f"top1_{p}" for p in PROFILES]
                 + ["loss_mean", "loss_median", "loss_std", "loss_gt3", "n_used", "n_own"])
N_FEATURES = len(FEATURE_NAMES)

HUMAN_MODEL = "cache/katago/b18c384nbt-humanv0.bin.gz"
CONFIG = os.path.join(os.path.dirname(__file__), "katago_analysis.cfg")


def main_model_path() -> str:
    """Homebrew 附帶的主模型。可用 AICUP_KATAGO_MODEL 覆寫。"""
    env = os.environ.get("AICUP_KATAGO_MODEL")
    if env:
        return env
    hits = sorted(glob.glob("/opt/homebrew/Cellar/katago/*/share/katago/kata1-b18c384nbt-*.bin.gz"))
    if not hits:
        raise FileNotFoundError("找不到 KataGo 主模型；請 `brew install katago` 或設定 AICUP_KATAGO_MODEL")
    return hits[-1]


def game_key(sgf: str, color) -> str:
    """一盤棋（含執色）的穩定鍵。同一份 SGF 換執色是不同的一筆。"""
    c = "B" if normalise_color(color) == BLACK else "W"
    return hashlib.sha1(f"{c}|{sgf}".encode()).hexdigest()[:20]


def plan_game(sgf: str, color, n_moves: int = 24) -> dict:
    """要送給 KataGo 的內容：完整手順、要評分的我方手數（均勻抽 n_moves 手）、各手的目標點。"""
    mv = parse_moves(sgf)
    my = normalise_color(color)
    own = [t for t, (col, _, _) in enumerate(mv) if col == my]
    turns = ([own[j] for j in np.unique(np.linspace(0, len(own) - 1, n_moves).astype(int))]
             if own else [])
    return {"moves": [["B" if c == BLACK else "W", f"{COLS[cc]}{19 - rr}"] for c, rr, cc in mv],
            "turns": turns, "target": {t: mv[t][1] * 19 + mv[t][2] for t in turns},
            "my_black": my == BLACK, "n": len(mv), "n_own": len(own)}


def queries(key: str, plan: dict) -> list[dict]:
    """一盤棋 → 每個 profile 一個查詢。profile 0 另外分析「我方落子後」的局面，用來算目數損失。"""
    out = []
    for pi, prof in enumerate(PROFILES):
        turns = plan["turns"] + ([t + 1 for t in plan["turns"] if t + 1 <= plan["n"]] if pi == 0 else [])
        out.append({"id": f"{key}:{pi}", "moves": plan["moves"], "rules": "japanese", "komi": 6.5,
                    "boardXSize": 19, "boardYSize": 19, "analyzeTurns": sorted(set(turns)),
                    "maxVisits": 1, "includePolicy": True,
                    "overrideSettings": {"humanSLProfile": prof}})
    return out


def features(plan: dict, logp: np.ndarray, top1: np.ndarray, lead: dict) -> np.ndarray:
    """收齊的回應 → 一盤的特徵向量（長度 N_FEATURES）。

    logp / top1：(評分手數, N_PROFILES)；lead：turn → 黑方視角的 scoreLead。
    目數損失 = 我方落子前的領先 − 落子後的領先（都換成我方視角），越大代表這手虧越多。
    """
    sign = 1.0 if plan["my_black"] else -1.0
    loss = np.array([sign * (lead[t] - lead[t + 1]) for t in plan["turns"]
                     if t in lead and t + 1 in lead]) if lead else np.zeros(0)
    if len(loss) == 0:
        loss = np.zeros(1)
    lp = np.nanmean(logp, axis=0) if len(logp) else np.full(N_PROFILES, np.nan)
    t1 = top1.mean(axis=0) if len(top1) else np.full(N_PROFILES, np.nan)
    return np.concatenate([lp, t1, [loss.mean(), np.median(loss), loss.std(), (loss > 3).mean(),
                                    len(plan["turns"]), plan["n_own"]]]).astype(np.float32)


class HumanSLEngine:
    """KataGo 分析引擎的串流客戶端。`run()` 吃 (key, sgf, color)，逐盤產出 (key, 特徵)。

    在途的盤數有上限，避免一次把幾萬盤的手順全塞進管線。
    """

    def __init__(self, n_moves: int = 24, in_flight: int = 48, katago: str = "katago"):
        self.n_moves, self.in_flight = n_moves, in_flight
        self.cmd = [katago, "analysis", "-config", CONFIG, "-model", main_model_path(),
                    "-human-model", HUMAN_MODEL]

    def run(self, items):
        proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        slots = threading.Semaphore(self.in_flight)
        pending: dict[str, dict] = {}
        lock = threading.Lock()
        done_q: queue.Queue = queue.Queue()

        def writer():
            for key, sgf, color in items:
                plan = plan_game(sgf, color, self.n_moves)
                if not plan["turns"]:
                    done_q.put((key, None)); continue
                qs = queries(key, plan)
                slots.acquire()
                with lock:
                    pending[key] = {"plan": plan, "left": sum(len(q["analyzeTurns"]) for q in qs),
                                    "logp": np.full((len(plan["turns"]), N_PROFILES), np.nan),
                                    "top1": np.zeros((len(plan["turns"]), N_PROFILES)), "lead": {}}
                for q in qs:
                    proc.stdin.write(json.dumps(q) + "\n")
                proc.stdin.flush()
            done_q.put(("__end__", None))

        def reader():
            for line in proc.stdout:
                d = json.loads(line)
                if ":" not in str(d.get("id", "")):
                    continue
                key, pi = d["id"].rsplit(":", 1); pi = int(pi)
                with lock:
                    st = pending.get(key)
                    if st is None:
                        continue
                    if "error" in d:
                        st["left"] = 0
                    else:
                        t, plan = d["turnNumber"], st["plan"]
                        if pi == 0:
                            st["lead"][t] = d["rootInfo"]["scoreLead"]
                        if t in plan["target"]:
                            k = plan["turns"].index(t)
                            hp = d["humanPolicy"]; v = hp[plan["target"][t]]
                            if v >= 0:
                                st["logp"][k, pi] = np.log(max(v, 1e-9))
                                st["top1"][k, pi] = float(v >= max(hp))
                        st["left"] -= 1
                    if st["left"] <= 0:
                        del pending[key]
                        done_q.put((key, features(st["plan"], st["logp"], st["top1"], st["lead"])))
                        slots.release()

        threading.Thread(target=writer, daemon=True).start()
        threading.Thread(target=reader, daemon=True).start()
        try:
            ended = False
            while True:
                key, f = done_q.get()
                if key == "__end__":
                    ended = True
                elif f is not None:
                    yield key, f
                if ended:
                    with lock:
                        if not pending:
                            break
        finally:
            proc.terminate()


class HumanSLStore:
    """只追加的 JSONL 儲存：一行一盤 {"k": 鍵, "f": 特徵}。中斷後重跑會自動跳過已完成的盤。"""

    def __init__(self, path: str = "cache/humansl/games.jsonl"):
        self.path = path
        self.data: dict[str, np.ndarray] = {}
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue                      # 中斷時寫到一半的最後一行
                    f = np.asarray(r["f"], dtype=np.float32)
                    if len(f) == N_FEATURES:
                        self.data[r["k"]] = f

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def get(self, key: str):
        return self.data.get(key)

    def append(self, key: str, f: np.ndarray, fh) -> None:
        self.data[key] = f
        fh.write(json.dumps({"k": key, "f": [round(float(x), 5) for x in f]}) + "\n")
        fh.flush()


def question_loglik(feats: list) -> np.ndarray:
    """一題（多盤）的 HumanSL 對數似然向量：Σ 各盤（平均 log 機率 x 評分手數）。

    缺特徵的盤略過；全缺時回傳全 0（均勻）。給 stacker 當成一個成分的 log 後驗用。
    """
    acc = np.zeros(N_PROFILES)
    n_used = FEATURE_NAMES.index("n_used")
    for f in feats:
        if f is None or np.isnan(f[:N_PROFILES]).any():
            continue
        acc += f[:N_PROFILES] * f[n_used]
    return acc
