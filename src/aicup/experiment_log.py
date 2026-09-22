"""統一的實驗記錄：每次跑完就把「配置 → 分數」追加到一份 CSV。

為什麼需要：67 天沒有 leaderboard，會累積上百次實驗。靠記憶或散落的 JSON
必然會出現「這個分數是哪個設定跑的？」的問題，而且 12/06 要交的報告
需要完整的實驗紀錄當素材。

刻意的設計：
  - **追加**而非覆寫，永遠不刪資料
  - 自動記錄 git commit，讓每個分數都能回溯到當時的程式碼
  - 分數欄位自由（不同實驗量的東西不同），用 JSON 存在一欄裡
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

DEFAULT_PATH = "cache/experiments.csv"
# `minutes` 是後加的。**加欄位時一定要走 `_ensure_header`**——
# 舊檔案的表頭沒有新欄位，直接用新的 FIELDS 追加會讓所有新列**欄位錯位**，
# 而 CSV 不會報錯，只會在幾週後讀出一堆對不上的資料。
FIELDS = ["timestamp", "task", "name", "commit", "dirty", "host",
          "minutes", "config", "metrics", "note"]


def _git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    timeout=5).stdout.strip())
        return sha or "unknown", dirty
    except Exception:
        return "unknown", False


def _ensure_header(path: str) -> None:
    """舊檔案的表頭若與現在的 FIELDS 不同，就地遷移（缺的欄位補空）。

    不這樣做的話，加欄位之後的每一列都會**欄位錯位**——
    而 CSV 不會報錯，只會在幾週後讀出一堆對不上的資料。
    """
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        f.seek(0)
        old = next(csv.reader(f), [])
    if old == FIELDS:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    os.replace(tmp, path)


def log_experiment(task: str, name: str, config: dict, metrics: dict,
                   note: str = "", path: str = DEFAULT_PATH,
                   minutes: float | None = None) -> str:
    """追加一筆實驗紀錄。

    task    'rank' / 'player' / 'both'
    name    這次實驗的簡短識別（例如 'gbdt_ensemble_5fold'）
    config  超參與設定
    metrics 分數。請用**題目層級的官方分數**，不要只記 loss 或單局 accuracy
    minutes 這次實驗花了幾分鐘。**請一律填**——
            「這個增益值不值得」是要把時間算進去才答得出來的問題，
            而事後從散落的 log 裡撈時間非常痛苦。用 `timer()` 最省事。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _ensure_header(path)
    sha, dirty = _git_state()
    row = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "task": task, "name": name, "commit": sha, "dirty": dirty,
        "host": platform.node(),
        "minutes": "" if minutes is None else round(float(minutes), 2),
        "config": json.dumps(config, ensure_ascii=False, sort_keys=True, default=str),
        "metrics": json.dumps(metrics, ensure_ascii=False, sort_keys=True, default=str),
        "note": note,
    }
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
    return path


@contextlib.contextmanager
def timer():
    """量一段程式跑了幾分鐘。

        with timer() as t:
            ...訓練...
        log_experiment(..., minutes=t.minutes)

    **即使中途拋例外也會記到那一刻為止**，所以失敗的實驗也有時間可查
    ——「這個方向試了多久才放棄」同樣是要記的東西。
    """
    class _T:
        start = time.time()
        minutes = 0.0
    t = _T()
    try:
        yield t
    finally:
        t.minutes = (time.time() - t.start) / 60.0


def load_experiments(path: str = DEFAULT_PATH):
    """讀回所有實驗紀錄，config / metrics 展開成欄位。"""
    import pandas as pd
    if not os.path.exists(path):
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(path)
    for col in ("config", "metrics"):
        exp = pd.json_normalize(df[col].map(json.loads)).add_prefix(col[0] + ".")
        df = pd.concat([df.drop(columns=[col]), exp], axis=1)
    return df


def best_runs(task: str, metric: str, top: int = 10, path: str = DEFAULT_PATH):
    """列出某個任務下該指標最好的幾次實驗。"""
    df = load_experiments(path)
    if df.empty:
        return df
    col = metric if metric in df.columns else f"m.{metric}"
    df = df[df["task"] == task].dropna(subset=[col])
    return df.nlargest(top, col)[["timestamp", "name", "commit", col]]


def cost_report(path: str = DEFAULT_PATH, task: str | None = None):
    """每個實驗的**分數增益 vs 機器時間**。

    「這個增益值不值得」不能只看增益——要除以成本。
    這張表把兩者放在一起，讓「下一步該投資哪裡」有數字可依據。
    """
    df = load_experiments(path)
    if df.empty:
        return df
    if task:
        df = df[df["task"] == task]
    cols = {"timestamp": "時間", "name": "實驗", "minutes": "分鐘"}
    for c in ("m.gain", "m.score", "m.verdict"):
        if c in df.columns:
            cols[c] = c.split(".", 1)[1]
    out = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    if "gain" in out.columns and "分鐘" in out.columns:
        hrs = out["分鐘"] / 60.0
        out["增益/小時"] = (out["gain"] / hrs.where(hrs > 0)).round(5)
    return out.sort_values("分鐘", ascending=False, na_position="last")


if __name__ == "__main__":                     # python3 -m aicup.experiment_log
    import sys as _sys
    _t = _sys.argv[1] if len(_sys.argv) > 1 else None
    _df = cost_report(task=_t)
    if _df.empty:
        print("還沒有任何紀錄。")
    else:
        import pandas as _pd
        with _pd.option_context("display.width", 200,
                                "display.max_colwidth", 28):
            print(_df.to_string(index=False))
        _m = _df["分鐘"].sum()
        print(f"\n合計 {_m:.0f} 分鐘（{_m / 60:.1f} 小時），"
              f"{int(_df['分鐘'].notna().sum())} 筆有時間紀錄")
