"""從官方格式的測試檔產生兩個子題的提交檔，並（若有正解）離線評分。

這支腳本是 2026/11/04 的預演：測試集當天釋出後，理想情況是只需要換掉
`--test-dir`，其餘完全不動。現在就用 `scripts/make_mock_testset.py` 產生的
模擬測試集把整條路徑跑通，避免在有時間壓力時才發現格式或效能問題。

用法：
    python3 scripts/make_mock_testset.py          # 先產生模擬測試集
    python3 scripts/predict.py --test-dir mock_tests

正式使用：
    python3 scripts/predict.py --test-dir tests --out-dir .
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aicup import RANK_TO_ID, ID_TO_RANK                       # noqa: E402
from aicup.data import load_metadata, make_player_split        # noqa: E402
from aicup.features import batch_features, parallel_batch_features  # noqa: E402
from aicup.baselines import RankBaseline, PlayerBaseline       # noqa: E402
from aicup.scoring import (apply_temperature, decode_expected_score,   # noqa: E402
                           rank_final_score, top_k_from_similarity,
                           player_final_score, fit_temperature,
                           optimise_thresholds, apply_thresholds, expected_level)
from aicup.submission import (build_rank_submission, build_player_submission,  # noqa: E402
                              write_submission)

MAX_GAMES = 20


def _question_features(df: pd.DataFrame, with_opponent: bool = False):
    """把一份題目檔轉成 [每題的特徵矩陣]。

    with_opponent=True 會同時抽**對手視角**（同一盤棋、執色反轉）並沿特徵軸串接，
    順序必須與訓練時的 `--extra-features cache/features_opp.npy` 一致。
    ⚠️ 有效遮罩取**交集**：任一視角抽取失敗就整局不用，
    否則串接出來的向量會有一半是零而不自知。
    """
    sgf_cols = [f"sgf_{i}" for i in range(1, MAX_GAMES + 1)]
    col_cols = [f"color_{i}" for i in range(1, MAX_GAMES + 1)]
    flat_sgf, flat_col, sizes = [], [], []
    for _, r in df.iterrows():
        n = 0
        for sc, cc in zip(sgf_cols, col_cols):
            v = r.get(sc)
            if pd.notna(v) and str(v).strip():
                flat_sgf.append(v); flat_col.append(r[cc]); n += 1
        sizes.append(n)
    X, ok = parallel_batch_features(flat_sgf, flat_col)
    if with_opponent:
        from aicup.features import normalise_color
        from aicup.goboard import BLACK
        flip = ["W" if normalise_color(c) == BLACK else "B" for c in flat_col]
        Xo, oko = parallel_batch_features(flat_sgf, flip)
        X = np.concatenate([X, Xo], axis=1)
        ok = ok & oko
    out, pos = [], 0
    for n in sizes:
        block = X[pos:pos + n][ok[pos:pos + n]]
        out.append(block)
        pos += n
    return out


# 序列庫檔名 → (抽取方式, max_moves)。
# `predict.py` 直接從 checkpoint 自己記錄的 `args["sequences"]` /
# `args["extra_sequences"]` 推導出該模型需要哪些視角，**不另外開旗標**——
# 旗標會跟 checkpoint 不同步，而 checkpoint 自己記的不會。
SEQ_VIEWS = {
    "sequences.npy":       ("move", False, 160),
    "sequences_opp.npy":   ("move", True, 160),
    "seqshape.npy":        ("shape", False, 160),
    "seqshape_opp.npy":    ("shape", True, 160),
    "sequences240.npy":    ("move", False, 240),
    "sequences240_opp.npy": ("move", True, 240),
}


def _views_of(ck_args: dict) -> tuple:
    """由 checkpoint 的 args 推出它訓練時用了哪些視角，順序照舊。"""
    paths = [ck_args.get("sequences", "cache/sequences.npy")] + \
        list(ck_args.get("extra_sequences", []))
    out = []
    for p_ in paths:
        key = os.path.basename(p_)
        if key not in SEQ_VIEWS:
            raise ValueError(
                f"不認得的序列庫 {p_}——請在 predict.py 的 SEQ_VIEWS 補上它的抽取方式。"
                f"目前支援：{sorted(SEQ_VIEWS)}")
        out.append(SEQ_VIEWS[key])
    return tuple(out)


def _store_paths(views) -> list:
    """視角 → 對應的序列庫路徑（校準時走預先算好的庫，不重新解析 SGF）。"""
    rev = {v: k for k, v in SEQ_VIEWS.items()}
    return [f"cache/{rev[v]}" for v in views]


def _question_sequences(df: pd.DataFrame, views) -> list:
    """把一份題目檔轉成 [每題的逐手序列張量]，視角沿**特徵軸**串接。

    與 `_question_features` 平行：那個抽整局的統計摘要，這個抽逐手序列。
    有效判斷兩邊一致（手數 < 30 或我方手數 < 15 就算失敗）。

    views 是 `_views_of()` 的輸出，順序**必須與訓練時完全一致**，
    否則特徵欄會錯位——而錯位不會報錯，只會讓分數悄悄變差。
    """
    from aicup.features import normalise_color
    from aicup.goboard import BLACK
    from aicup.moveseq import batch_move_sequences
    from aicup.moveshape import batch_shape_sequences

    sgf_cols = [f"sgf_{i}" for i in range(1, MAX_GAMES + 1)]
    col_cols = [f"color_{i}" for i in range(1, MAX_GAMES + 1)]
    flat_sgf, flat_col, sizes = [], [], []
    for _, r in df.iterrows():
        n = 0
        for sc, cc in zip(sgf_cols, col_cols):
            v = r.get(sc)
            if pd.notna(v) and str(v).strip():
                flat_sgf.append(v); flat_col.append(r[cc]); n += 1
        sizes.append(n)

    flip = ["W" if normalise_color(c) == BLACK else "B" for c in flat_col]
    parts, ok = [], None
    for kind, opp, mm in views:
        fn = batch_shape_sequences if kind == "shape" else batch_move_sequences
        A, m = fn(flat_sgf, flip if opp else flat_col, max_moves=mm)
        parts.append(A)
        ok = m if ok is None else (ok & m)        # 任一視角失敗就整局不用
    S = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)

    out, pos = [], 0
    for n in sizes:
        blk = S[pos:pos + n][ok[pos:pos + n]]
        out.append(np.asarray(blk, dtype=np.float32))
        pos += n
    return out


def _seq_posteriors_from_blocks(seq_nets, blocks, device):
    """由已抽好的逐題序列算題目層級後驗（幾何平均，與單局模型一致）。"""
    import torch as _t
    from aicup.losses import corn_to_proba as _c2p
    from aicup.moveseq import MAX_MOVES

    out = []
    with _t.no_grad():
        for A in blocks:
            if len(A) == 0:
                out.append(np.full(10, 0.1)); continue
            lens = (np.abs(A).sum(-1) > 0).sum(-1)
            x = _t.from_numpy(A).to(device)
            mk = _t.from_numpy(np.arange(MAX_MOVES)[None, :] < lens[:, None]).to(device)
            ps = []
            for n in seq_nets:
                lc, lo = n(x, mk)
                pc = _c2p(lo).cpu().numpy()
                ps.append(0.5 * _t.softmax(lc, -1).cpu().numpy()
                          + 0.5 * pc / np.maximum(pc.sum(1, keepdims=True), 1e-12))
            p = np.mean(ps, axis=0)
            g = np.exp(np.log(np.clip(p, 1e-12, None)).mean(axis=0))
            out.append(g / g.sum())
    return np.stack(out)


def _question_posteriors_from_features(rank_question, Xq):
    """由聚合特徵算題目層級後驗。"""
    import torch
    from aicup.losses import corn_to_proba
    qnets, qsc, _, _, _ = rank_question
    dev = next(qnets[0].parameters()).device
    with torch.no_grad():
        A = torch.from_numpy(qsc.transform(Xq)).to(dev)
        ps = []
        for n in qnets:
            lc, lo = n(A)
            p = torch.softmax(lc, dim=-1).cpu().numpy()
            pc = corn_to_proba(lo).cpu().numpy()
            pc = pc / np.maximum(pc.sum(axis=1, keepdims=True), 1e-12)
            ps.append(0.5 * p + 0.5 * pc)
    return np.mean(ps, axis=0)


def _question_log_posteriors(rank_question, X, ok, game_rows):
    """只回傳**題目層級成分**的 log 後驗（不融合）。stacker 需要個別成分。"""
    _, _, _, build_q, _ = rank_question
    Xq = build_q(X, ok, game_rows)
    return np.log(np.clip(_question_posteriors_from_features(rank_question, Xq),
                          1e-12, None))


def _question_log_posteriors_from_feats(rank_question, feats):
    """同上，但吃已抽好的逐題特徵（推論路徑用）。"""
    from aicup.question_model import aggregate_question
    dim, rows = None, []
    for A in feats:
        v = aggregate_question(np.asarray(A)) if len(A) else None
        rows.append(v)
        if v is not None and dim is None:
            dim = len(v)
    Xq = np.stack([v if v is not None else np.zeros(dim, np.float32) for v in rows])
    return np.log(np.clip(_question_posteriors_from_features(rank_question, Xq),
                          1e-12, None))


def _blend_question(rank_question, X, ok, game_rows, log_game):
    """從 feature store 取聚合特徵後融合。"""
    _, _, w, build_q, blend = rank_question
    Xq = build_q(X, ok, game_rows)
    lq = np.log(np.clip(_question_posteriors_from_features(rank_question, Xq), 1e-12, None))
    return blend(log_game, lq, w)


def _blend_question_from_feats(rank_question, feats, log_game):
    """從已抽好的逐題特徵清單聚合後融合（推論路徑用）。"""
    from aicup.question_model import aggregate_question
    _, _, w, _, blend = rank_question
    dim = None
    rows = []
    for A in feats:
        v = aggregate_question(np.asarray(A)) if len(A) else None
        rows.append(v)
        if v is not None and dim is None:
            dim = len(v)
    Xq = np.stack([v if v is not None else np.zeros(dim, np.float32) for v in rows])
    lq = np.log(np.clip(_question_posteriors_from_features(rank_question, Xq), 1e-12, None))
    return blend(log_game, lq, w)


def _fit_rank_model(cache_path: str, train_per_rank: int):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    X = np.load("cache/features.npy", mmap_mode="r")
    ok = np.load("cache/features_ok.npy")
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    vm = split.mask(meta)
    rank_id = meta["rank_id"].to_numpy()
    rng = np.random.default_rng(0)
    if train_per_rank and train_per_rank > 0:
        rows = np.concatenate([
            rng.choice(np.flatnonzero((~vm) & ok & (rank_id == lv)),
                       size=min(train_per_rank,
                                int(((~vm) & ok & (rank_id == lv)).sum())),
                       replace=False)
            for lv in range(10)])
    else:
        rows = np.flatnonzero((~vm) & ok)          # 全量訓練玩家的棋局
    print(f"  棋力模型訓練局數 {len(rows):,}", flush=True)
    model = RankBaseline().fit(np.asarray(X[rows]), rank_id[rows])
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(model, f)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="mock_tests")
    ap.add_argument("--out-dir", default="cache/submissions")
    ap.add_argument("--train-per-rank", type=int, default=0,
                    help="每個等級取幾局訓練棋力模型。0 = 用全部訓練玩家的棋局"
                         "（實測全量 0.3973 vs 每級 2 萬局 0.3866，值得多花訓練時間）")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="題目後驗的溫度；<1 為銳化。實測最佳 0.5~0.6")
    ap.add_argument("--aggregation", default="logmean", choices=["mean", "logmean"])
    ap.add_argument("--rank-question", default="cache/rank_question_opp/models.pt",
                    help="競賽二的題目層級模型，與單局模型幾何融合（權重存在 "
                         "checkpoint 裡，目前 0.2）。8 組切分掃過 w=0~0.4，"
                         "0.2 就是最佳且變異最小（0.4895±0.0090）。"
                         "它單獨只有 0.4495、比單局模型還差，但看的是 5~20 局"
                         "之間的變異，所以融合有增益——弱但互補")
    ap.add_argument("--rank-mlp", default="cache/rank_mlp_opp_big/models.pt",
                    help="競賽二的單局 MLP 集成。8 組 player-disjoint 切分實測："
                         "1178 維（含對手視角）0.5402±0.0085 勝過 589 維的 0.4895±0.0090"
                         "（+0.0507，8/8 勝，t=14.84）。"
                         "設為空字串則退回 GBDT（0.4351，不建議）")
    ap.add_argument("--feature-metric", default="cache/feature_metric/models.pt",
                    help="競賽一的度量學習模型（目前增益最大：0.6630 -> 0.8125）。"
                         "設為空字串則退回「標準化+L2」的無學習版本")
    ap.add_argument("--stage2", default="",
                    help="Stage-2 重排器。⚠️ 它與第一階段綁定："
                         "對手工餘弦有 +0.0211，但對度量學習的 embedding 無效甚至有害"
                         "（實測平均 -0.003），所以預設停用。"
                         "只有在 --feature-metric 留空時才建議啟用 cache/stage2/model.pkl")
    ap.add_argument("--extra-features", nargs="*",
                    default=["cache/features_opp.npy"],
                    help="競賽二額外的特徵庫，必須與訓練時完全一致。"
                         "`cache/features_opp.npy` 是對手視角（同盤棋、執色反轉）："
                         "伺服器把你配對給實力相近的人，對手的表現反映配對分級，"
                         "而分級反映你的穩定實力。實測 +0.0491±0.0232（6/6 勝，t=5.19）。"
                         "⚠️ 給了這個就會在測試集上多抽一次特徵，時間加倍")
    ap.add_argument("--calib-prior", default="natural",
                    choices=["natural", "uniform", "p100"],
                    help="切點校準時假設的測試集等級先驗。natural = 校準集自身的分布"
                         "（預設，維持現況）。官方測試集是 400 位玩家各一題，"
                         "其等級分布未知；實測不同假設之間最壞情況差 0.010"
                         "（RESULTS.md §2.7）。11/04 公開測試集釋出後再定案")
    ap.add_argument("--stacker", type=float, default=1.0,
                    help="學習式融合的正則化強度 C（0 = 關閉，退回固定權重）。"
                         "固定權重融合 0.5639 → stacker 0.5727"
                         "（+0.0088，8/8 勝，t=5.00）。C 從 0.3 到 3.0 是平台。"
                         "stacker 在**校準題目**上現場擬合，套用到測試集")
    ap.add_argument("--rank-seq", nargs="*",
                    default=["cache/rank_seq_opp/models.pt",
                             "cache/rank_seq_opp_full/models.pt",
                             "cache/rank_seq_w192/models.pt"],
                    help="競賽二的逐手序列模型（dilated 1D-CNN）。"
                         "589 維是整局的統計摘要，看不到『愚形出現在第 30 手還是第 150 手』。"
                         "雙視角（我方 12 維 + 對手 12 維 = 24 維/手）。"
                         "**預設是三個變體的集成**：基準（40 萬局 w128）、"
                         "全量（78 萬局 w128）、加寬（40 萬局 w192）。"
                         "三者個別相對單一模型只有 +0.004~0.006（都低於 0.005 門檻），"
                         "但誤差來源不同（資料量 vs 容量），合起來 "
                         "0.5703 → 0.5782（+0.0079，8/8 勝，t=9.15）。"
                         "給空清單則停用序列成分")
    ap.add_argument("--extra-sequences", nargs="*",
                    default=["cache/sequences_opp.npy"],
                    help="序列模型的額外視角，必須與訓練時一致。"
                         "給了這個就會在測試集上多抽一次序列（時間加倍）")
    ap.add_argument("--seq-weight", type=float, default=0.7,
                    help="序列模型的融合權重。8 組 player-disjoint 掃過 0.35~0.9，"
                         "峰值在 0.7（0.5639±0.0104，+0.0237，8/8 勝，t=15.09），"
                         "0.7~0.8 是平台（0.5629~0.5639），確切值不敏感。"
                         "⚠️ 權重比單局模型還高——序列模型單獨（0.5483）"
                         "已經比整條對手視角管線（0.5402）強")
    ap.add_argument("--sinkhorn", type=float, default=0.002,
                    help="競賽一的全域指派強度（0 = 關閉）。"
                         "官方規格已確認**一一對應**："
                         "「查詢棋譜集 400 組是 400 位不同玩家」"
                         "「候選玩家集 400 個，包括查詢棋譜集的玩家」。"
                         "15 次評估（官方局數分布）：獨立 argmax 0.9087 → "
                         "eps=0.002 的 0.9599（+0.0512，15/15 勝）。"
                         "eps 0.001~0.003 是平台；≤0.003 時勝過匈牙利算法（0.9543）。"
                         "⚠️ 若官方規格與上述不符（例如候選庫含干擾項），"
                         "退回 0.0075——那是在假設被違反時仍為正的保守值")
    ap.add_argument("--decode", default="threshold",
                    choices=["threshold", "expected", "argmax"],
                    help="解碼方式。實測（5 次獨立切分）：最佳化切點 0.4336±0.0032 > "
                         "期望分數解碼 0.3921±0.0099 > argmax 0.3886±0.0086")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    answers = {"rank": {}, "player": {}}
    ans_path = os.path.join(args.test_dir, "_answers.json")
    if os.path.exists(ans_path):
        with open(ans_path) as f:
            answers = json.load(f)

    # ---------------- 競賽二 ----------------
    print("=" * 60, flush=True)
    print("競賽二：棋力預測", flush=True)
    print("=" * 60, flush=True)
    rank_mlp = None
    rank_question = None
    if args.rank_mlp and os.path.exists(args.rank_mlp):
        import torch
        from aicup.feature_metric import FeatureScaler
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_rank_mlp import RankMLP, question_posteriors
        rdev = torch.device("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
        ck = torch.load(args.rank_mlp, map_location=rdev, weights_only=False)
        rsc = FeatureScaler().load_state_dict(ck["scaler"])
        nets = []
        for sd in ck["models"]:
            n = RankMLP(len(rsc.mu), hidden=ck["args"]["hidden"])
            n.load_state_dict(sd); n.eval().to(rdev)
            nets.append(n)
        rank_mlp = (nets, rsc, rdev, question_posteriors)
        print(f"  競賽二使用 MLP 集成（{len(nets)} 個模型，device={rdev}）", flush=True)

        if args.rank_question and os.path.exists(args.rank_question):
            from aicup.question_model import build_question_features, blend_log_posteriors
            qck = torch.load(args.rank_question, map_location=rdev, weights_only=False)
            qsc = FeatureScaler().load_state_dict(qck["scaler"])
            qnets = []
            for sd in qck["models"]:
                n = RankMLP(len(qsc.mu), hidden=qck["args"]["hidden"])
                n.load_state_dict(sd); n.eval().to(rdev)
                qnets.append(n)
            rank_question = (qnets, qsc, float(qck["blend_weight"]),
                             build_question_features, blend_log_posteriors)
            print(f"  ＋題目層級模型（{len(qnets)} 個，融合權重 "
                  f"{qck['blend_weight']:.1f}）", flush=True)

    # 序列成分：可以是多個 checkpoint 的集成。每個 checkpoint 成為 stacker 的
    # 一個獨立成分（不是先平均），因為它們的誤差來源不同——
    # 讓 stacker 自己學怎麼配權重，比事先平均好。
    seq_models, seq_dev = [], None
    paths = [p_ for p_ in (args.rank_seq or []) if os.path.exists(p_)]
    if paths and args.seq_weight > 0:
        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_rank_seq import build_seq_net
        seq_dev = torch.device("cuda" if torch.cuda.is_available() else
                               "mps" if torch.backends.mps.is_available() else "cpu")
        for path in paths:
            sck = torch.load(path, map_location=seq_dev, weights_only=False)
            nets = []
            _views = _views_of(sck["args"])
            _nfeat = 12 * len(_views)
            for sd in sck["models"]:
                n = build_seq_net(_nfeat, sck["args"])
                n.load_state_dict(sd); n.eval().to(seq_dev)
                nets.append(n)
            seq_models.append((os.path.basename(os.path.dirname(path)), nets, _views))
        print(f"  ＋逐手序列模型 {len(seq_models)} 個變體："
              f"{', '.join(f'{t}({len(n)} seed, {len(v)} 視角)' for t, n, v in seq_models)}",
              flush=True)
    seq_nets = seq_models[0][1] if seq_models else None      # 固定權重路徑用第一個

    model = None if rank_mlp else _fit_rank_model("cache/rank_model.pkl",
                                                  args.train_per_rank)

    # 切點與溫度都在「held-out 玩家的模擬題目」上校準，與測試集完全無關
    calib = None
    if args.decode == "threshold":
        from aicup.data import make_player_split as _mps
        from aicup.benchmark import build_rank_questions as _brq
        from aicup.benchmark import rank_prior as _rank_prior
        from aicup.benchmark import resample_to_prior as _resample_to_prior
        from aicup.featstore import open_stores
        _X, _ok = open_stores(["cache/features.npy"] + list(args.extra_features))
        _meta = load_metadata()
        _sp = _mps(_meta, val_frac=0.20, seed=42)
        _q = _brq(_meta, _sp, n_per_player=3, seed=0)
        # 校準題目的等級分布決定了切點的位置。預設 natural（校準集自身的分布）。
        # 若 11/04 拿到公開測試集後判定官方的分布更接近平衡，改這個旗標即可，
        # 不必動程式碼——實測固定先驗之間最壞情況差 0.010（RESULTS.md §2.7）。
        _prior = _rank_prior(args.calib_prior, _meta)
        if rank_mlp:
            nets, rsc, rdev, qpost = rank_mlp
            _P = _Pg = qpost(nets, rsc, _X, _ok, _q.game_rows, rdev)
            if rank_question:
                _P = np.exp(_blend_question(rank_question, _X, _ok,
                                            _q.game_rows, np.log(np.clip(_P, 1e-12, None))))
        else:
            _P = _Pg = model.question_proba(
                [np.asarray(_X[r[_ok[r]]]) for r in _q.game_rows], args.aggregation)
        _parts = {"game": np.log(np.clip(_Pg, 1e-12, None))}
        if rank_question:
            _parts["question"] = _question_log_posteriors(rank_question, _X, _ok,
                                                          _q.game_rows)
        _lo = np.log(np.clip(_P, 1e-12, None))
        if seq_models:
            # 每個變體用**自己的**序列庫。24 維的舊變體與 48 維的棋形變體
            # 不能共用同一組 blocks——共用會讓其中一邊的輸入維度對不上。
            from aicup.question_model import blend_log_posteriors as _blp
            _bcache = {}
            for _tag, _nets, _views in seq_models:
                _sp = _store_paths(_views)
                key = tuple(_sp)
                if key not in _bcache:
                    _Ss = [np.load(x, mmap_mode="r") for x in _sp]
                    _sok = None
                    for x in _sp:
                        _m = np.load(x.replace(".npy", "_ok.npy"))
                        _sok = _m if _sok is None else (_sok & _m)
                    _bcache[key] = [np.concatenate(
                        [np.asarray(z[r[_sok[r]]], dtype=np.float32) for z in _Ss],
                        axis=-1) for r in _q.game_rows]
                _l = np.log(np.clip(
                    _seq_posteriors_from_blocks(_nets, _bcache[key], seq_dev),
                    1e-12, None))
                _parts[f"seq_{_tag}"] = _l
            _ls = _parts[f"seq_{seq_models[0][0]}"]
            _lo = _blp(_lo, _ls, w=args.seq_weight)   # 固定權重路徑只用第一個變體

        _sizes = np.array([max(int(_ok[r].sum()), 1) for r in _q.game_rows])
        _ci = _resample_to_prior(np.arange(len(_q.true_rank_id)), _q.true_rank_id,
                                 _prior, np.random.default_rng(42))
        if _prior is not None:
            print(f"  校準集已重抽成 {args.calib_prior} 先驗（{len(_ci)} 題）", flush=True)
        stacker = None
        if args.stacker > 0 and len(_parts) > 1:
            from aicup.stacker import Stacker, stack_features
            _F = stack_features(_parts, _sizes)
            stacker = Stacker(C=args.stacker).fit(_F[_ci], _q.true_rank_id[_ci],
                                                  parts=_parts)
            _ev, _T = stacker.expected_level(_F), None
            print(f"  學習式融合已擬合（{_F.shape[1]} 維特徵，C={args.stacker}）"
                  f"——取代固定權重", flush=True)
        else:
            _T = fit_temperature(_lo[_ci], _q.true_rank_id[_ci])
            _ev = expected_level(apply_temperature(_lo, _T))
        _th = optimise_thresholds(_ev[_ci], _q.true_rank_id[_ci])
        calib = (_T, _th, stacker)
        print(f"  切點校準完成（{len(_q.true_rank_id)} 題）："
              f"{'T=%.2f, ' % _T if _T is not None else ''}"
              f"切點={np.round(_th, 2).tolist()}", flush=True)

    qids, preds, truth = [], [], []
    for tag in ("public", "private"):
        path = os.path.join(args.test_dir, f"rank_prediction_test_{tag}.csv")
        df = pd.read_csv(path)
        t0 = time.time()
        feats = _question_features(df, with_opponent=bool(args.extra_features))
        if rank_mlp:
            nets, rsc, rdev, _ = rank_mlp
            for _n in nets:                 # 保險：推論一律 eval（dropout 關閉）
                _n.eval()
            import torch as _t
            from aicup.losses import corn_to_proba as _c2p
            P = []
            with _t.no_grad():
                for A in feats:
                    if len(A) == 0:
                        P.append(np.full(10, 0.1)); continue
                    t = _t.from_numpy(rsc.transform(A)).to(rdev)
                    ps = []
                    for n in nets:
                        lc, lo2 = n(t)
                        p = _t.softmax(lc, dim=-1).cpu().numpy()
                        pc = _c2p(lo2).cpu().numpy()
                        pc = pc / np.maximum(pc.sum(axis=1, keepdims=True), 1e-12)
                        ps.append(0.5 * p + 0.5 * pc)
                    p = np.mean(ps, axis=0)
                    g = np.exp(np.log(np.clip(p, 1e-12, None)).mean(axis=0))
                    P.append(g / g.sum())
            P = np.stack(P)
            parts = {"game": np.log(np.clip(P, 1e-12, None))}
            if rank_question:
                parts["question"] = _question_log_posteriors_from_feats(
                    rank_question, feats)
                P = np.exp(_blend_question_from_feats(
                    rank_question, feats, np.log(np.clip(P, 1e-12, None))))
        else:
            P = model.question_proba(feats, how=args.aggregation)
            parts = {"game": np.log(np.clip(P, 1e-12, None))}
        lo = np.log(np.clip(P, 1e-12, None))
        if seq_models:
            from aicup.question_model import blend_log_posteriors as _blp
            bcache = {}
            for tag_, nets_, views_ in seq_models:
                if views_ not in bcache:
                    bcache[views_] = _question_sequences(df, views_)
                parts[f"seq_{tag_}"] = np.log(np.clip(
                    _seq_posteriors_from_blocks(nets_, bcache[views_], seq_dev),
                    1e-12, None))
            lo = _blp(lo, parts[f"seq_{seq_models[0][0]}"], w=args.seq_weight)
        if args.decode == "threshold":
            T, th, stacker = calib
            if stacker is not None:
                from aicup.stacker import stack_features
                stacker.check_parts(parts)      # 成分集合必須與校準時一致
                sizes = np.array([max(len(f), 1) for f in feats])
                ev = stacker.expected_level(stack_features(parts, sizes))
            else:
                ev = expected_level(apply_temperature(lo, T))
            pred = apply_thresholds(ev, th)
        elif args.decode == "expected":
            pred = decode_expected_score(apply_temperature(lo, args.temperature))
        else:
            pred = apply_temperature(lo, args.temperature).argmax(axis=-1)
        qids.extend(df["question_id"].tolist()); preds.extend(pred.tolist())
        truth.extend([answers["rank"].get(q) for q in df["question_id"]])
        print(f"  {tag:<8} {len(df)} 題，{time.time() - t0:.0f} 秒", flush=True)

    sub = build_rank_submission(qids, preds)
    path = write_submission(sub, os.path.join(args.out_dir, "submission_rank.csv"), "rank")
    print(f"  已寫出 {path}（{len(sub)} 列，格式驗證通過）", flush=True)
    if truth and all(t is not None for t in truth):
        y = np.array([RANK_TO_ID[t] for t in truth])
        print(f"  離線分數 {rank_final_score(np.array(preds), y):.4f}", flush=True)

    # ---------------- 競賽一 ----------------
    print("\n" + "=" * 60, flush=True)
    print("競賽一：玩家辨識", flush=True)
    print("=" * 60, flush=True)
    metric = None
    if args.feature_metric and os.path.exists(args.feature_metric):
        import torch
        from aicup.feature_metric import FeatureEmbedder, FeatureScaler
        mdev = torch.device("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
        ck = torch.load(args.feature_metric, map_location=mdev, weights_only=False)
        a = ck["args"]
        scaler = FeatureScaler().load_state_dict(ck["scaler"])
        cols = ck.get("cols")            # 訓練時用的特徵子集，推論必須一致
        models = []
        for sd in ck["models"]:
            m = FeatureEmbedder(len(scaler.mu), embed_dim=a["embed_dim"], hidden=a["hidden"])
            m.load_state_dict(sd); m.eval().to(mdev)
            models.append(m)
        metric = (models, scaler, mdev, cols)
        print(f"  使用度量學習 embedding（{len(models)} 個模型集成，"
              f"{len(scaler.mu)} 維輸入，device={mdev}）", flush=True)

    all_qids, all_top5, all_truth = [], [], []
    for tag in ("public", "private"):
        cand = pd.read_csv(os.path.join(
            args.test_dir, f"player_identification_candidates_{tag}.csv"))
        t0 = time.time()
        Xc, okc = parallel_batch_features(cand["sgf_content"].tolist(),
                                          cand["color"].tolist())
        names = sorted(cand["player_id"].unique())
        pid = cand["player_id"].to_numpy()

        if metric is not None:
            from aicup.feature_metric import embed_features, player_template
            models, scaler, mdev, cols = metric

            def make_tpl(A):
                if cols is not None:
                    A = np.asarray(A)[:, cols]
                v = np.concatenate([player_template(embed_features(m, scaler, A, mdev))
                                    for m in models])
                return v / max(float(np.linalg.norm(v)), 1e-8)
        else:
            pb = PlayerBaseline().fit(Xc[okc])
            make_tpl = pb.template

        G = np.stack([make_tpl(Xc[okc & (pid == n)]) for n in names])
        print(f"  {tag:<8} 候選庫 {len(names)} 人 / {len(cand):,} 局，"
              f"{time.time() - t0:.0f} 秒")

        df = pd.read_csv(os.path.join(
            args.test_dir, f"player_identification_test_{tag}.csv"))
        feats = _question_features(df)
        Q = np.stack([make_tpl(f) if len(f) else np.zeros(G.shape[1]) for f in feats])
        sim = Q @ G.T
        if args.stage2 and os.path.exists(args.stage2):
            from aicup.rerank_stage2 import rerank_scores
            with open(args.stage2, "rb") as fh:
                st = pickle.load(fh)
            qn = np.array([len(f) for f in feats])
            gn = np.array([int((pid == n).sum()) for n in names])
            sim = rerank_scores(st["model"], sim, top_n=st["top_n"],
                                query_sizes=qn, gallery_sizes=gn)
            print(f"  已套用 Stage-2 重排（Top-{st['top_n']} → Top-5）", flush=True)
        if args.sinkhorn > 0:
            from aicup.rerank import sinkhorn_assignment, bijection_consistency
            before = bijection_consistency(sim)
            sim = sinkhorn_assignment(sim, eps=args.sinkhorn)
            print(f"  已套用 Sinkhorn 全域指派（eps={args.sinkhorn}）："
                  f"top-1 相異率 {before:.3f} → {bijection_consistency(sim):.3f}",
                  flush=True)
            if before < 0.60:
                print("  ⚠️ 套用前的相異率偏低，代表第一階段檢索很弱，"
                      "或『每位候選正好是一題的答案』不成立。"
                      "實測基礎 Top-1 < 0.4 時 Sinkhorn 會變成負增益——"
                      "請考慮 --sinkhorn 0 關閉。", flush=True)
        top5 = top_k_from_similarity(sim, names, k=5)
        all_qids.extend(df["question_id"].tolist()); all_top5.extend(top5.tolist())
        all_truth.extend([answers["player"].get(q) for q in df["question_id"]])

    sub = build_player_submission(all_qids, all_top5)
    pools = {}
    for tag, key in (("public", "pub"), ("private", "priv")):
        c = pd.read_csv(os.path.join(
            args.test_dir, f"player_identification_candidates_{tag}.csv"))
        pools[key] = set(c["player_id"].unique())
    path = write_submission(sub, os.path.join(args.out_dir, "submission_player.csv"),
                            "player", public_candidates=pools["pub"],
                            private_candidates=pools["priv"])
    print(f"  已寫出 {path}（{len(sub)} 列，格式驗證通過）", flush=True)
    if all_truth and all(t is not None for t in all_truth):
        ranks = np.array([row.index(t) if t in row else 99
                          for row, t in zip(all_top5, all_truth)])
        print(f"  離線分數 {player_final_score(ranks):.4f}"
              f"   Top-1 {(ranks == 0).mean():.4f}   Top-5 {(ranks < 5).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
