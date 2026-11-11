"""競賽一更新版測試集（2026-11-11）：原測試集（pub_/priv_）＋新增測試集（pub2_/priv2_），合併成 1,600 列。

新增測試集與原測試集的差別（官方說明）：每位候選人只有 10 盤（黑白都有），每題 1/2/3/5 盤，
候選人 600 位、題目 400 題，**不是一對一**（同一位玩家可能對應多題，也有不對應任何題目的候選人）
→ Sinkhorn 的一對一假設不成立，新增測試集預設不做全域指派。

原測試集四個檔與舊版逐位元相同，直接沿用既有的提交（`--old`，例如 P003）的 pub_/priv_ 列。

    python3 scripts/predict_player_v2.py --test-dir data/v2/tests --old submissions/player/p003.csv \\
        --out cache/submissions/player_v2/submission_player.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _predict_module():
    spec = importlib.util.spec_from_file_location("predict", os.path.join(HERE, "predict.py"))
    pr = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["predict.py"]
    try:
        spec.loader.exec_module(pr)
    finally:
        sys.argv = argv
    return pr


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def template(E: np.ndarray, n_models: int) -> np.ndarray:
    """與 predict.py 相同：每個模型各自平均再 L2，串接後再 L2。"""
    d = E.shape[1] // n_models
    return l2(np.concatenate([l2(E[:, k * d:(k + 1) * d].mean(0)) for k in range(n_models)]))


def scores_v2(pr, metric, tag: str, test_dir: str, method: str, l2_wd: float = 1e-4) -> tuple:
    """新增測試集一個切分 → (題號, 候選人名單, 分數矩陣)。"""
    from aicup.feature_metric import embed_features
    models, scaler, dev, cols, stores = metric

    def emb(A):
        A = np.asarray(A)
        if cols is not None:
            A = A[:, cols]
        return np.concatenate([embed_features(m, scaler, A, dev) for m in models], axis=1)

    cand = pd.read_csv(os.path.join(test_dir, f"player_identification_candidates2_{tag}.csv"))
    q = pd.read_csv(os.path.join(test_dir, f"player_identification_test2_{tag}.csv"))
    Xc, okc = pr._metric_features(cand["sgf_content"].tolist(), cand["color"].tolist(), stores)
    Ec = emb(Xc[okc]); pid = cand["player_id"].to_numpy()[okc]
    names = sorted(set(cand["player_id"]))
    feats = pr._metric_question_features(q, stores)
    Eq = [emb(f) if len(f) else np.zeros((0, Ec.shape[1])) for f in feats]
    D = len(models)
    if method == "template":
        G = np.stack([template(Ec[pid == n], D) for n in names])
        Q = np.stack([template(e, D) if len(e) else np.zeros(G.shape[1]) for e in Eq])
        S = Q @ G.T
    elif method == "classifier":
        from aicup.rerank import pool_classifier_scores
        idx = {n: i for i, n in enumerate(names)}
        S = pool_classifier_scores(Ec, np.array([idx[p] for p in pid]), Eq, len(names), device=dev, wd=l2_wd)
    else:
        raise ValueError(method)
    print(f"  {tag}：候選 {len(names)} 人／{len(Ec):,} 盤，題目 {len(q)}（{method}）", flush=True)
    return q["question_id"].tolist(), names, S


def top5(qids, names, S) -> pd.DataFrame:
    order = np.argsort(-S, axis=1)[:, :5]
    return pd.DataFrame({"question_id": qids,
                         **{f"top_{k + 1}": [names[j] for j in order[:, k]] for k in range(5)}})


def validate(sub: pd.DataFrame, test_dir: str) -> None:
    """官方規則：1,600 題、無缺無重、Top-5 不重複、player_id 前綴與 question_id 一致且在對應候選庫。"""
    want = []
    for f in ("test_public", "test_private", "test2_public", "test2_private"):
        want += pd.read_csv(os.path.join(test_dir, f"player_identification_{f}.csv"),
                            usecols=["question_id"])["question_id"].tolist()
    assert len(sub) == len(want) == 1600, (len(sub), len(want))
    assert set(sub["question_id"]) == set(want) and sub["question_id"].is_unique
    cand = {}
    for pre, f in (("pub", "candidates_public"), ("priv", "candidates_private"),
                   ("pub2", "candidates2_public"), ("priv2", "candidates2_private")):
        cand[pre] = set(pd.read_csv(os.path.join(test_dir, f"player_identification_{f}.csv"),
                                    usecols=["player_id"])["player_id"])
    T = sub[[f"top_{k}" for k in range(1, 6)]].to_numpy()
    for qid, row in zip(sub["question_id"], T):
        pre = qid.split("_q_")[0]
        assert len(set(row)) == 5, qid
        assert all(p.split("_p_")[0] == pre and p in cand[pre] for p in row), qid
    print("  格式驗證通過：1,600 題、前綴一致、Top-5 不重複且都在對應候選庫", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="data/v2/tests")
    ap.add_argument("--old", required=True, help="原測試集（pub_/priv_）沿用的提交檔，例如 submissions/player/p003.csv")
    ap.add_argument("--method", choices=["template", "classifier"], default="template")
    ap.add_argument("--l2", type=float, default=1e-4, help="classifier 的 L2 強度")
    ap.add_argument("--metric", default="cache/feature_metric_h1024s12k/models.pt")
    ap.add_argument("--out", default="cache/submissions/player_v2/submission_player.csv")
    args = ap.parse_args()

    pr = _predict_module()
    metric = pr._load_metric(args.metric)
    old = pd.read_csv(args.old)
    old = old[old["question_id"].str.match(r"^(pub|priv)_q_")]
    parts = [old]
    for tag in ("public", "private"):
        parts.append(top5(*scores_v2(pr, metric, tag, args.test_dir, args.method, args.l2)))
    sub = pd.concat(parts, ignore_index=True)
    validate(sub, args.test_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"已寫出 {args.out}（{len(sub)} 列）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
