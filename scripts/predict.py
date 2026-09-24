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

from aicup import RANK_TO_ID, ID_TO_RANK                       # noqa: E402
from aicup.data import load_metadata, make_player_split        # noqa: E402
from aicup.features import batch_features, parallel_batch_features  # noqa: E402
from aicup.baselines import RankBaseline, PlayerBaseline       # noqa: E402
from aicup.scoring import (apply_temperature, decode_expected_score,   # noqa: E402
                           rank_final_score, top_k_from_similarity,
                           player_final_score, fit_temperature,
                           optimise_thresholds, apply_thresholds, expected_level)
from aicup.submission import (resolve_test_file, build_rank_submission, build_player_submission,  # noqa: E402
                              write_submission)

MAX_GAMES = 20
CROSSTASK_GALLERY = 100      # 官方候選庫是「400 人各 100 譜」


def _load_metric(path: str):
    """載入競賽一的度量學習模型。競賽一與跨子題增強共用同一份。"""
    if not path or not os.path.exists(path):
        return None
    import torch
    from aicup.feature_metric import FeatureEmbedder, FeatureScaler
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["args"]
    scaler = FeatureScaler().load_state_dict(ck["scaler"])
    models = []
    for sd in ck["models"]:
        m = FeatureEmbedder(len(scaler.mu), embed_dim=a["embed_dim"], hidden=a["hidden"])
        m.load_state_dict(sd); m.eval().to(dev)
        models.append(m)
    stores = ["cache/features.npy"] + list(a.get("extra_features", []))
    return models, scaler, dev, ck.get("cols"), stores


def crosstask_decision(overlap: float, bijection: float, threshold: float):
    """要不要套用跨子題增強？回傳 (要不要, 理由)。

    抽成純函式是因為**這是安全關鍵的四行**：比較符號寫反的代價是 −0.1993。
    兩個獨立判準：

    1. **棋譜字串直接重疊 ≥ 50%** —— definitive。主辦方若是從同一份測試資料
       衍生兩個子題，棋力題的棋譜會原封不動出現在候選庫裡。
    2. **Sinkhorn 之前的 top-1 相異率 ≥ threshold** —— 間接但夠用
       （實測假設成立 0.9075、不成立 0.5117）。
    """
    if overlap >= 0.5:
        return True, (f"棋譜字串有 {overlap:.1%} 直接重疊——兩個子題出自同一份"
                      f"測試資料，判準直接通過（embedding 相異率 "
                      f"{bijection:.3f} 僅供參考）")
    if bijection < threshold:
        return False, (f"**不套用**——Sinkhorn 前的 top-1 相異率 {bijection:.3f} "
                       f"< {threshold}。兩個子題看起來不是同一批玩家，"
                       f"硬套的話是把別人的棋局當成強證據（實測 −0.1993）")
    return True, (f"**套用**——Sinkhorn 前的 top-1 相異率 {bijection:.3f} "
                  f"≥ {threshold}")


# 特徵庫檔名 → 推論時怎麼算。與 `SEQ_VIEWS` 同一個道理：
# 從 checkpoint 自己記錄的 `args["extra_features"]` 推導，不另外開旗標。
METRIC_FEATURES = {
    "features.npy":           ("game", False),
    "features_opp.npy":       ("game", True),
    "features_shape.npy":     ("shape", False),
    "features_shape_opp.npy": ("shape", True),
}


def _metric_features(sgfs, colors, stores):
    """依 checkpoint 訓練時的**特徵庫順序**重算特徵矩陣。

    ⚠️ 順序必須與 `open_stores([...])` 串接的順序完全一致，
    否則欄位錯位——而錯位不會報錯，只會讓分數悄悄變差。
    棋形統計用的是 `aicup.moveshape.aggregate_shape`，
    與建庫時**同一份實作**（`build_shape_features.py` 也是 import 它）。
    """
    from aicup.features import normalise_color
    from aicup.goboard import BLACK
    from aicup.moveshape import aggregate_shape, batch_shape_sequences

    flip = ["W" if normalise_color(c) == BLACK else "B" for c in colors]
    parts, ok = [], None
    for path in stores:
        key = os.path.basename(path)
        if key not in METRIC_FEATURES:
            raise ValueError(
                f"不認得的特徵庫 {path}——請在 predict.py 的 METRIC_FEATURES "
                f"補上它的算法。目前支援：{sorted(METRIC_FEATURES)}")
        kind, opp = METRIC_FEATURES[key]
        cc = flip if opp else list(colors)
        if kind == "game":
            A, m = parallel_batch_features(sgfs, cc)
        else:
            S, m = batch_shape_sequences(sgfs, cc)
            A = aggregate_shape(S)
        parts.append(np.asarray(A, dtype=np.float32))
        ok = m if ok is None else (ok & m)       # 任一視角失敗就整局不用
    return (parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)), ok


def _metric_question_features(df: pd.DataFrame, stores):
    """題目檔 → [每題的度量學習特徵矩陣]。"""
    flat_sgf, flat_col, sizes = _flatten_questions(df)
    X, ok = _metric_features(flat_sgf, flat_col, stores)
    out, pos = [], 0
    for n in sizes:
        out.append(X[pos:pos + n][ok[pos:pos + n]])
        pos += n
    return out


def _crosstask_games(tag: str, feats, args, metric):
    """把每道棋力題配對到競賽一的候選人，回傳那位候選人的棋局（扁平清單）。

    ⚠️ **判準不過就回傳 None**，而且判準看的是 **Sinkhorn 之前**的 top-1 相異率。
    Sinkhorn 會強制產生近乎完美的一一對應，**不管真正的答案在不在候選庫裡**——
    套用之後的信心對「全部配對」與「全部配錯」長得一模一樣
    （實測全錯時 margin 中位數 0.64、p90 0.9956）。
    用被污染的訊號當安全門檻，比沒有門檻更危險。

    實測（RESULTS.md §2.20）：
        假設成立   Sinkhorn 前相異率 0.9075，增強 **+0.0586**
        假設不成立 Sinkhorn 前相異率 0.5117，增強 **−0.1993**
    """
    if metric is None:
        print("  跨子題增強：沒有度量學習模型，略過", flush=True)
        return None
    path = resolve_test_file(args.test_dir, "candidates", tag)
    if not os.path.exists(path):
        print(f"  跨子題增強：找不到 {path}，略過", flush=True)
        return None

    import numpy as _np
    from aicup.feature_metric import embed_features, player_template
    from aicup.rerank import bijection_consistency, sinkhorn_assignment

    models, scaler, mdev, cols, mstores = metric
    cand = pd.read_csv(path)
    Xc, okc = _metric_features(cand["sgf_content"].tolist(),
                               cand["color"].tolist(), mstores)

    def tpl(A):
        A = _np.asarray(A)
        if cols is not None:
            A = A[:, cols]
        v = _np.concatenate([player_template(embed_features(m, scaler, A, mdev))
                             for m in models])
        return v / max(float(_np.linalg.norm(v)), 1e-8)

    names = sorted(cand["player_id"].unique())
    pid = cand["player_id"].to_numpy()
    rows_by_name = {n: _np.flatnonzero(okc & (pid == n)) for n in names}

    # 獨立且**definitive** 的判準：棋譜字串直接重疊。
    # 若主辦方是從同一份測試資料衍生兩個子題，棋力題的棋譜會原封不動
    # 出現在候選庫裡。這個訊號比 embedding 強得多，而且零成本。
    _rp = resolve_test_file(args.test_dir, "rank", tag)
    overlap = 0.0
    if os.path.exists(_rp):
        _rk = pd.read_csv(_rp)
        _rs = set()
        for c in [c for c in _rk.columns if c.startswith("sgf_")]:
            _rs |= set(_rk[c].dropna().astype(str))
        _rs.discard("")
        if _rs:
            overlap = len(_rs & set(cand["sgf_content"].astype(str))) / len(_rs)
    G = _np.stack([tpl(Xc[rows_by_name[n]]) for n in names])
    Q = _np.stack([tpl(f) if len(f) else _np.zeros(G.shape[1]) for f in feats])
    raw = Q @ G.T
    bij = bijection_consistency(raw)
    thr = args.crosstask_min_bijection
    apply, reason = crosstask_decision(overlap, bij, thr)
    print(f"  跨子題增強：{reason}", flush=True)
    if not apply:
        return None

    sim = (sinkhorn_assignment(raw, eps=args.sinkhorn) if args.sinkhorn > 0 else raw)
    top1 = sim.argmax(axis=1)
    sgfs = cand["sgf_content"].to_numpy()
    cols_ = cand["color"].to_numpy()
    fsgf, fcol, sizes = [], [], []
    for j in top1:
        idx = rows_by_name[names[int(j)]][:CROSSTASK_GALLERY]
        fsgf.extend(sgfs[idx].tolist())
        fcol.extend(cols_[idx].tolist())
        sizes.append(len(idx))
    print(f"  跨子題增強：**套用**——Sinkhorn 前的 top-1 相異率 {bij:.3f} ≥ {thr}，"
          f"每題改用配對到的候選人的 {int(_np.mean(sizes))} 盤"
          f"（混合權重 {args.crosstask_weight}）", flush=True)
    return fsgf, fcol, sizes


def _question_features(df: pd.DataFrame, with_opponent: bool = False):
    """把一份題目檔轉成 [每題的特徵矩陣]。

    with_opponent=True 會同時抽**對手視角**（同一盤棋、執色反轉）並沿特徵軸串接，
    順序必須與訓練時的 `--extra-features cache/features_opp.npy` 一致。
    ⚠️ 有效遮罩取**交集**：任一視角抽取失敗就整局不用，
    否則串接出來的向量會有一半是零而不自知。
    """
    flat_sgf, flat_col, sizes = _flatten_questions(df)
    return _features_from_flat(flat_sgf, flat_col, sizes, with_opponent)


def _flatten_questions(df: pd.DataFrame):
    """題目檔 → (扁平的 sgf 清單, 扁平的 color 清單, 每題幾盤)。"""
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
    return flat_sgf, flat_col, sizes


def _features_from_flat(flat_sgf, flat_col, sizes, with_opponent: bool = False):
    """扁平清單 → [每題的特徵矩陣]。

    跨子題增強要對「配對到的候選人的 100 盤」算同樣的東西，
    而那 100 盤不是從題目檔的 sgf_1..sgf_20 欄位來的——
    所以底層要吃扁平清單，不是 DataFrame。
    """
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
    "seqqual.npy":         ("quality", False, 160),
    "seqqual_opp.npy":     ("quality", True, 160),
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

    flat_sgf, flat_col, sizes = _flatten_questions(df)
    return _sequences_from_flat(flat_sgf, flat_col, sizes, views)


def _sequences_from_flat(flat_sgf, flat_col, sizes, views) -> list:
    """扁平清單 → [每題的逐手序列張量]（與 `_question_sequences` 同一套邏輯）。"""
    from aicup.features import normalise_color
    from aicup.goboard import BLACK
    from aicup.moveseq import batch_move_sequences
    from aicup.moveshape import batch_shape_sequences
    from aicup.movequality import batch_quality_sequences

    _FN = {"shape": batch_shape_sequences, "quality": batch_quality_sequences}
    flip = ["W" if normalise_color(c) == BLACK else "B" for c in flat_col]
    parts, ok = [], None
    for kind, opp, mm in views:
        fn = _FN.get(kind, batch_move_sequences)
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


def check_artifacts(args) -> int:
    """把「靜默降級」變成明確的報告。

    `predict.py` 對缺檔案的反應是**安靜地少用一個成分**——
    在比賽當天，那代表你會交出一份分數較低的提交檔而完全不知道。
    這個模式在推論前先把所有依賴列出來，缺什麼一目了然。

    回傳 0 = 全部就緒；1 = 有東西缺。
    """
    from aicup.planes import n_planes                      # noqa: F401  (確認可 import)

    rows: list[tuple[str, str, bool, str]] = []

    def add(group, path, why):
        rows.append((group, path, os.path.exists(path), why))

    add("競賽二 單局", args.rank_mlp, "缺了就退回無學習基準，分數掉一大截")
    add("競賽二 題目層級", args.rank_question, "缺了就少一個 stacker 成分")
    for i, sp in enumerate(args.rank_seq or []):
        add("競賽二 序列", sp, f"第 {i + 1} 個變體；少一個就少 13 維 stacker 特徵")
        ck_dir = os.path.dirname(sp)
        if os.path.exists(sp):
            try:
                import torch
                a = torch.load(sp, map_location="cpu", weights_only=False)["args"]
                for st in _store_paths(_views_of(a)):
                    add("　└ 它需要的序列庫", st, f"{os.path.basename(ck_dir)} 的輸入")
            except Exception as e:                          # noqa: BLE001
                rows.append(("　└ 讀取失敗", sp, False, f"{type(e).__name__}: {e}"))
    if args.board:
        add("競賽二 板面 CNN", args.board,
            "缺了就少 stacker 最強的一個成分（−0.0269）")
    for _be in args.board_extra:
        add("競賽二 板面 CNN（附加）", _be, "附加的集成成員")
    add("競賽一 度量學習", args.feature_metric, "缺了就退回無學習的 PlayerBaseline")
    if os.path.exists(args.feature_metric):
        m = _load_metric(args.feature_metric)
        if m is not None:
            for st in m[4]:
                add("　└ 它需要的特徵庫", st, "推論時會即時重算，但維度要對得上")
    for tg in ("public", "private"):
        for kind in ("rank", "player", "candidates"):
            add("測試集", resolve_test_file(args.test_dir, kind, tg), "官方檔案")

    seen, uniq = set(), []
    for r in rows:
        if r[1] in seen:
            continue
        seen.add(r[1]); uniq.append(r)

    miss = [r for r in uniq if not r[2]]
    w = max(len(r[1]) for r in uniq) + 2
    print("=" * 60)
    print("依賴檢查")
    print("=" * 60)
    for group, path, ok, why in uniq:
        print(f"  {'✅' if ok else '❌'} {group:<16}{path:<{w}}" + ("" if ok else why))
    print("=" * 60)
    if miss:
        print(f"**缺少 {len(miss)} 項。** `predict.py` 不會報錯——"
              f"它會安靜地少用那些成分，交出一份分數較低的檔案。")
        print("產生方式見 docs/RUNBOOK.md 的「事前必須就緒的東西」（本機文件）。")
        return 1
    print(f"全部 {len(uniq)} 項就緒，可以跑 `python3 scripts/predict.py`。")
    return 0


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
    # 8 seed、隱藏層 1024：3 seed/512 → 8 seed/512 最壞組成 0.9719 → 0.9774；
    # 512 → 1024 再到 0.9822（每 seed 最壞組成 7/8，t=3.16）。RESULTS.md §2.30
    ap.add_argument("--feature-metric", default="cache/feature_metric_h1024/models.pt",
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
    ap.add_argument("--check", action="store_true",
                    help="只檢查依賴是否齊全就結束，不做推論。"
                         "比賽當天第一個該跑的東西——`predict.py` 對缺檔案是"
                         "**靜默降級**，這個模式把它變成明確的清單")
    ap.add_argument("--crosstask", action="store_true",
                    help="**跨子題證據轉移**：先用競賽一認出每道棋力題是哪位候選人，"
                         "再用那位候選人的 100 譜增強棋力預測。"
                         "實測若假設成立 +0.0586，不成立 −0.1993（RESULTS.md §2.20）"
                         "——所以內建偵測器，判準不過就自動不套用。預設關閉")
    ap.add_argument("--crosstask-weight", type=float, default=1.0,
                    help="增強的混合權重（0 = 只用題目自己的盤，1 = 全用配對到的）。"
                         "實測（RESULTS.md §2.20）：\n"
                         "  w=0.75  假設成立 +0.0543 ／ 不成立 −0.1498\n"
                         "  w=1.0   假設成立 +0.0586 ／ 不成立 −0.1993\n"
                         "兩者的取捨點在「偵測器判斷錯誤的機率」約 8%%——"
                         "高於 8%% 就該用 0.75。門檻訂在 0.85 之後"
                         "（相異率實測 0.9075 vs 0.5117，兩側各約 0.2 邊際），"
                         "判錯的機率遠低於 8%%，所以預設 1.0。"
                         "若當天的相異率只是**勉強**過 0.85，手動改成 0.75")
    ap.add_argument("--crosstask-min-bijection", type=float, default=0.85,
                    help="套用的門檻：**Sinkhorn 之前**的 top-1 相異率。"
                         "實測假設成立 0.9075、完全不成立 0.5117。"
                         "0.85 不是取中間值——設 f 為查詢玩家真的在候選庫裡的比例，"
                         "由兩個量測點線性內插得 相異率 ≈ 0.512 + 0.396f；"
                         "而淨增益 f×0.0586 + (1−f)×(−0.1993) 的零點在 f=77%%。"
                         "門檻 0.70 對應 f=47.5%%，會在 f∈[48%%, 77%%] 這段**放行但虧損**。"
                         "0.85 對應 f≈85%%，留了安全邊際。"
                         "⚠️ 一定要看 Sinkhorn 之前的——Sinkhorn 會強制一一對應，"
                         "之後的信心對兩種情境長得一模一樣")
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
    ap.add_argument("--board", default="cache/board/best.pt",
                    help="競賽二的板面 CNN（34x19x19 的盤面快照 → 集合池化）。"
                         "空字串 = 不使用。這是集成裡唯一直接看盤面的成分，"
                         "與其他成分的輸入完全不重疊：加進來 "
                         "**+0.0269 ±0.0050（8/8 勝，t=15.32）**，"
                         "是專案至今最大的單一增益（RESULTS.md 2.27）。"
                         "代價：推論要即時把每盤棋重播成張量，"
                         "每 4776 題約 160 秒，比其他成分貴一個數量級。"
                         "⚠️ 這是**凍結的上線副本**；訓練輸出在 cache/cnn_mps/，"
                         "續訓會原地覆寫那裡，所以絕不能直接指向訓練目錄")
    ap.add_argument("--board-extra", nargs="*", default=[],
                    help="額外的板面 CNN checkpoint，各自成為一個 stacker 成分"
                         "（成分名 board_<目錄名>）。預設不用")
    ap.add_argument("--board-tta", type=int, default=1,
                    help="板面成分的棋盤對稱 TTA 數（1 = 不做，8 = 全部）")
    ap.add_argument("--rank-seq", nargs="*",
                    # 板面 CNN 上線後，五個變體彼此冗餘：只留 opp_full + shape
                    # 與全部五個相比 +0.0003，16 seed 95% CI [−0.0009, +0.0015]
                    #（RESULTS.md §2.29）。少 3 顆 checkpoint、少 4 份序列庫。
                    default=["cache/rank_seq_opp_full/models.pt",
                             "cache/rank_seq_shape/models.pt"],
                    help="競賽二的逐手序列模型（dilated 1D-CNN）。"
                         "589 維是整局的統計摘要，看不到『愚形出現在第 30 手還是第 150 手』。"
                         "**預設是四個變體的集成**，每個變體用**自己的**序列庫"
                         "（由它 checkpoint 裡的 args 自動推導，見 SEQ_VIEWS）：\n"
                         "  base / full / w192  雙視角 24 維/手"
                         "（我方 12 + 對手 12），差在資料量與容量；\n"
                         "  shape               四視角 48 維/手（再加上棋形 12×2）；\n"
                         "  qual                六視角 64 維/手（再加上局面判斷品質 8×2）。\n"
                         "前三個個別相對單一模型只有 +0.004~0.006（都低於門檻），"
                         "但誤差來源不同，合起來 0.5703 → 0.5782（+0.0079，8/8，t=9.15）。"
                         "加上 shape 之後 **0.5782 → 0.5889**（+0.0063，7/8，t=3.73）；"
                         "再加上 qual **0.5892 → 0.5928**（+0.0036，**15/16**，t=5.03，16 seed）。"
                         "兩個都是**帶新資訊**的成員——換架構（GRU）、換寬度、"
                         "換視窗全都加不了東西（RESULTS.md §2.19）。"
                         "給空清單則停用序列成分")
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

    if args.check:
        return check_artifacts(args)

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
        from aicup.rank_nets import RankMLP
        from aicup.rank_nets import mlp_question_posteriors as question_posteriors
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
        from aicup.rank_nets import build_seq_net
        seq_dev = torch.device("cuda" if torch.cuda.is_available() else
                               "mps" if torch.backends.mps.is_available() else "cpu")
        for path in paths:
            sck = torch.load(path, map_location=seq_dev, weights_only=False)
            nets = []
            _views = _views_of(sck["args"])
            # ⚠️ 不能再假設「每個視角 12 維」——品質視角是 8 維。
            # 由實際的序列庫去問，錯了 load_state_dict 會直接拋錯。
            _nfeat = sum(int(np.load(x, mmap_mode="r").shape[-1])
                         for x in _store_paths(_views))
            for sd in sck["models"]:
                n = build_seq_net(_nfeat, sck["args"])
                n.load_state_dict(sd); n.eval().to(seq_dev)
                nets.append(n)
            seq_models.append((os.path.basename(os.path.dirname(path)), nets, _views))
        print(f"  ＋逐手序列模型 {len(seq_models)} 個變體："
              f"{', '.join(f'{t}({len(n)} seed, {len(v)} 視角)' for t, n, v in seq_models)}",
              flush=True)
    seq_nets = seq_models[0][1] if seq_models else None      # 固定權重路徑用第一個

    # 板面成分。與序列成分不同，它沒有預先算好的特徵庫——推論時要即時
    # 重播 SGF。所以只在 checkpoint 真的存在時才載入，缺檔就安靜略過
    # （stacker 會少一個成分，但不會整支掛掉）。
    # 每顆板面模型一個 stacker 成分：(成分名, net, snapshot 數, device)。
    # 第一顆的成分名維持 "board"，與舊版的校準／快取完全相容。
    boards = []
    _bpaths = ([("board", args.board)] if args.board else []) + [
        (f"board_{os.path.basename(os.path.dirname(p_))}", p_) for p_ in args.board_extra]
    for _bname, _bpath in _bpaths:
        if not os.path.exists(_bpath):
            continue
        import torch
        from aicup.board_posterior import load_board_net
        _bdev = torch.device("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
        _bnet, _bsnap = load_board_net(_bpath, _bdev)
        boards.append((_bname, _bnet, _bsnap, _bdev))
        print(f"  ＋板面 CNN {_bname}（{_bpath}，{_bsnap} 個 snapshot/局，{_bdev}）",
              flush=True)
    board = boards[0] if boards else None

    model = None if rank_mlp else _fit_rank_model("cache/rank_model.pkl",
                                                  args.train_per_rank)

    # 切點與溫度都在「held-out 玩家的模擬題目」上校準，與測試集完全無關
    # ⚠️ 跨子題的判定必須在**校準之前**做完。
    # 若校準做了增強、推論卻因為判準不過而沒做，切點就配在錯的分布上——
    # 那比完全不做增強還糟，而且不會有任何錯誤訊息。
    metric_ct = _load_metric(args.feature_metric) if args.crosstask else None
    ct_plan: dict = {}
    if args.crosstask:
        for _tg in ("public", "private"):
            _pth = resolve_test_file(args.test_dir, "rank", _tg)
            if not os.path.exists(_pth):
                continue
            _dft = pd.read_csv(_pth)
            # 度量學習模型可能用多份特徵庫訓練（例如加了棋形統計），
            # 所以這裡要照它 checkpoint 記的組合重算，不能只給我方 589 維。
            _qf = _metric_question_features(_dft, metric_ct[4])
            ct_plan[_tg] = _crosstask_games(_tg, _qf, args, metric_ct)
        if not ct_plan or any(v is None for v in ct_plan.values()):
            print("  跨子題增強：至少一個測試池沒通過判準 → **兩池都不套用**"
                  "（校準只有一份，兩池必須用同一種證據結構）", flush=True)
            ct_plan = {}
    args._crosstask_active = bool(ct_plan)

    calib = None
    if args.decode == "threshold":
        from aicup.data import make_player_split as _mps
        from aicup.benchmark import build_rank_questions as _brq
        from aicup.benchmark import rank_prior as _rank_prior
        from aicup.benchmark import resample_to_prior as _resample_to_prior
        from aicup.featstore import open_stores
        _X, _ok = open_stores(["cache/features.npy"] + list(args.extra_features))
        _meta = load_metadata()
        _split = _mps(_meta, val_frac=0.20, seed=42)
        _q = _brq(_meta, _split, n_per_player=3, seed=0)
        # 校準題目的等級分布決定了切點的位置。預設 natural（校準集自身的分布）。
        # 若 11/04 拿到公開測試集後判定官方的分布更接近平衡，改這個旗標即可，
        # 不必動程式碼——實測固定先驗之間最壞情況差 0.010（RESULTS.md §2.7）。
        _prior = _rank_prior(args.calib_prior, _meta)
        from aicup.question_model import blend_log_posteriors as _blp
        _bcache = {}
        # 板面成分要從 meta 的列索引回頭讀 SGF；SgfCache 每個檔只載一次。
        _bcache_sgf = None
        if board is not None:
            from aicup.dataset import SgfCache
            _bcache_sgf = SgfCache()

        def _calib_parts(game_rows, tag="q"):
            """對一組題目算出所有 stacker 成分的 log 後驗，外加固定權重路徑的 _lo。

            抽成函式是為了讓**跨子題增強**能用完全相同的邏輯算第二次
            （一次是題目自己的 5~20 盤，一次是配對到的候選人的 100 盤）——
            兩邊若用不同的程式碼，混合出來的東西就不可比。
            """
            if rank_mlp:
                nets, rsc, rdev, qpost = rank_mlp
                P = Pg = qpost(nets, rsc, _X, _ok, game_rows, rdev)
                if rank_question:
                    P = np.exp(_blend_question(rank_question, _X, _ok, game_rows,
                                               np.log(np.clip(P, 1e-12, None))))
            else:
                P = Pg = model.question_proba(
                    [np.asarray(_X[r[_ok[r]]]) for r in game_rows], args.aggregation)
            parts = {"game": np.log(np.clip(Pg, 1e-12, None))}
            if rank_question:
                parts["question"] = _question_log_posteriors(rank_question, _X, _ok,
                                                             game_rows)
            if boards:
                from aicup.board_posterior import (board_question_posteriors,
                                                   groups_from_meta)
                _grp = groups_from_meta(_meta, game_rows, _bcache_sgf)
                for _bname, _bn, _bs, _bd in boards:
                    parts[_bname] = np.log(np.clip(board_question_posteriors(
                        _bn, _bs, _grp, _bd, tta=args.board_tta), 1e-12, None))
            lo = np.log(np.clip(P, 1e-12, None))
            if seq_models:
                # 每個變體用**自己的**序列庫。24 維的舊變體與 48 維的棋形變體
                # 不能共用同一組 blocks——共用會讓其中一邊的輸入維度對不上。
                for tag_, nets_, views_ in seq_models:
                    paths = _store_paths(views_)
                    # ⚠️ 快取的 key 用呼叫端給的 tag，**不要用 id(game_rows)**：
                    # Python 的 id() 在物件被回收之後會重用，
                    # 兩個不同的清單可能拿到同一個 id → 取到錯的 blocks 而且不報錯。
                    key = (tuple(paths), tag)
                    if key not in _bcache:
                        Ss = [np.load(x, mmap_mode="r") for x in paths]
                        sok = None
                        for x in paths:
                            m_ = np.load(x.replace(".npy", "_ok.npy"))
                            sok = m_ if sok is None else (sok & m_)
                        _bcache[key] = [np.concatenate(
                            [np.asarray(z[r[sok[r]]], dtype=np.float32) for z in Ss],
                            axis=-1) for r in game_rows]
                    parts[f"seq_{tag_}"] = np.log(np.clip(
                        _seq_posteriors_from_blocks(nets_, _bcache[key], seq_dev),
                        1e-12, None))
                lo = _blp(lo, parts[f"seq_{seq_models[0][0]}"], w=args.seq_weight)
            return parts, lo

        _parts, _lo = _calib_parts(_q.game_rows, tag="question")
        if args._crosstask_active:
            # 校準也必須用**增強後**的證據，否則切點是配在 5~20 盤的期望等級
            # 分布上，而推論時的分布會因為多了 100 盤而變銳——切點就錯位了。
            # 這裡假設配對正確（實測 top-1 0.949），用同一位玩家的其他棋局。
            _prow = _meta.groupby("player_id").indices
            _aux = []
            for _pid, _rr in zip(_q.player_id, _q.game_rows):
                _all = np.setdiff1d(_prow[_pid], _rr, assume_unique=False)
                _aux.append(_all[:CROSSTASK_GALLERY] if len(_all) else _rr)
            _pa, _la = _calib_parts(_aux, tag="crosstask_gallery")
            w_ = args.crosstask_weight
            _parts = {k: (1 - w_) * v + w_ * _pa[k] for k, v in _parts.items()}
            _lo = (1 - w_) * _lo + w_ * _la
            print(f"  跨子題增強：校準也用同一位玩家的另外 "
                  f"{CROSSTASK_GALLERY} 盤（權重 {w_}）", flush=True)

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
        path = resolve_test_file(args.test_dir, "rank", tag)
        df = pd.read_csv(path)
        t0 = time.time()
        from aicup.question_model import blend_log_posteriors as _blp

        def _infer_parts(fsgf, fcol, fsizes):
            """一組棋局 → 所有 stacker 成分的 log 後驗（與校準端 `_calib_parts` 對稱）。

            吃扁平清單而不是 DataFrame，因為跨子題增強要對
            「配對到的候選人的 100 盤」算同樣的東西，
            而那 100 盤不在題目檔的 sgf_1..sgf_20 欄位裡。
            """
            ft = _features_from_flat(fsgf, fcol, fsizes,
                                     with_opponent=bool(args.extra_features))
            if rank_mlp:
                nets, rsc, rdev, _ = rank_mlp
                for n_ in nets:             # 保險：推論一律 eval（dropout 關閉）
                    n_.eval()
                import torch as _t
                from aicup.losses import corn_to_proba as _c2p
                Pl = []
                with _t.no_grad():
                    for A in ft:
                        if len(A) == 0:
                            Pl.append(np.full(10, 0.1)); continue
                        t = _t.from_numpy(rsc.transform(A)).to(rdev)
                        ps = []
                        for n in nets:
                            lc, lo2 = n(t)
                            pp = _t.softmax(lc, dim=-1).cpu().numpy()
                            pc = _c2p(lo2).cpu().numpy()
                            pc = pc / np.maximum(pc.sum(axis=1, keepdims=True), 1e-12)
                            ps.append(0.5 * pp + 0.5 * pc)
                        pp = np.mean(ps, axis=0)
                        g = np.exp(np.log(np.clip(pp, 1e-12, None)).mean(axis=0))
                        Pl.append(g / g.sum())
                Pm = np.stack(Pl)
                pr = {"game": np.log(np.clip(Pm, 1e-12, None))}
                if rank_question:
                    pr["question"] = _question_log_posteriors_from_feats(
                        rank_question, ft)
                    Pm = np.exp(_blend_question_from_feats(
                        rank_question, ft, np.log(np.clip(Pm, 1e-12, None))))
            else:
                Pm = model.question_proba(ft, how=args.aggregation)
                pr = {"game": np.log(np.clip(Pm, 1e-12, None))}
            if boards:
                from aicup.board_posterior import (board_question_posteriors,
                                                   groups_from_flat)
                _grp = groups_from_flat(fsgf, fcol, fsizes)
                for _bname, _bn, _bs, _bd in boards:
                    pr[_bname] = np.log(np.clip(board_question_posteriors(
                        _bn, _bs, _grp, _bd, tta=args.board_tta), 1e-12, None))
            lo_ = np.log(np.clip(Pm, 1e-12, None))
            if seq_models:
                bc = {}
                for tag_, nets_, views_ in seq_models:
                    if views_ not in bc:
                        bc[views_] = _sequences_from_flat(fsgf, fcol, fsizes, views_)
                    pr[f"seq_{tag_}"] = np.log(np.clip(
                        _seq_posteriors_from_blocks(nets_, bc[views_], seq_dev),
                        1e-12, None))
                lo_ = _blp(lo_, pr[f"seq_{seq_models[0][0]}"], w=args.seq_weight)
            return pr, lo_, ft

        _fs, _fc, _fz = _flatten_questions(df)
        parts, lo, feats = _infer_parts(_fs, _fc, _fz)

        if args._crosstask_active:
            aug = ct_plan.get(tag)
            if aug is not None:
                pa, la, _ = _infer_parts(*aug)
                w_ = args.crosstask_weight
                parts = {k: (1 - w_) * v + w_ * pa[k] for k, v in parts.items()}
                lo = (1 - w_) * lo + w_ * la
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
    metric = metric_ct if metric_ct is not None else _load_metric(args.feature_metric)
    if metric is not None:
        print(f"  使用度量學習 embedding（{len(metric[0])} 個模型集成，"
              f"{len(metric[1].mu)} 維輸入，device={metric[2]}）", flush=True)

    all_qids, all_top5, all_truth = [], [], []
    for tag in ("public", "private"):
        cand = pd.read_csv(resolve_test_file(args.test_dir, "candidates", tag))
        t0 = time.time()
        _mstores = metric[4] if metric is not None else ["cache/features.npy"]
        Xc, okc = _metric_features(cand["sgf_content"].tolist(),
                                   cand["color"].tolist(), _mstores)
        names = sorted(cand["player_id"].unique())
        pid = cand["player_id"].to_numpy()

        if metric is not None:
            from aicup.feature_metric import embed_features, player_template
            models, scaler, mdev, cols, _ = metric

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

        df = pd.read_csv(resolve_test_file(args.test_dir, "player", tag))
        feats = _metric_question_features(df, _mstores)
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
        c = pd.read_csv(resolve_test_file(args.test_dir, "candidates", tag))
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
