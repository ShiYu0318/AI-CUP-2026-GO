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

import csv
import json
import os
import platform
import subprocess
from datetime import datetime, timezone

DEFAULT_PATH = "cache/experiments.csv"
FIELDS = ["timestamp", "task", "name", "commit", "dirty", "host", "config", "metrics", "note"]


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


def log_experiment(task: str, name: str, config: dict, metrics: dict,
                   note: str = "", path: str = DEFAULT_PATH) -> str:
    """追加一筆實驗紀錄。

    task    'rank' / 'player' / 'both'
    name    這次實驗的簡短識別（例如 'gbdt_ensemble_5fold'）
    config  超參與設定
    metrics 分數。請用**題目層級的官方分數**，不要只記 loss 或單局 accuracy
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sha, dirty = _git_state()
    row = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "task": task, "name": name, "commit": sha, "dirty": dirty,
        "host": platform.node(),
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
