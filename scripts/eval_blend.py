"""競賽二：融合實驗的統一入口（吃 OOF 快取）。

**為什麼要有這支**：每做一次融合實驗都重跑模型推論，一輪 10~20 分鐘，
但融合真正在變的只有權重與組合——後驗本身沒變。
這支把每個成分的題目層級後驗算一次存進 `artifacts/oof/`，
之後掃權重、換組合、比較解碼器全部是讀檔加矩陣運算，秒級完成。

快取會自我失效：checkpoint 內容變了（即使檔名不變）、題目集變了、
或後驗語意升版，都會自動重算（見 `aicup.oofcache`）。

用法：
    # 只算後驗並存快取（第一次會慢）
    python3 scripts/eval_blend.py --warm

    # 掃序列模型的融合權重（有快取的話幾秒鐘）
    python3 scripts/eval_blend.py --sweep seq --grid 0.5 0.6 0.7 0.8

    # 評估目前的預設組合
    python3 scripts/eval_blend.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aicup.benchmark import build_rank_questions                 # noqa: E402
from aicup.data import load_metadata, make_player_split          # noqa: E402
from aicup.feature_metric import FeatureScaler                   # noqa: E402
from aicup.featstore import open_stores                          # noqa: E402
from aicup.losses import corn_to_proba                           # noqa: E402
from aicup.moveseq import MAX_MOVES                              # noqa: E402
from aicup.oofcache import OofCache                              # noqa: E402
from aicup.question_model import (blend_log_posteriors,          # noqa: E402
                                  build_question_features)
from aicup.scoring import (apply_temperature, apply_thresholds,  # noqa: E402
                           expected_level, fit_temperature,
                           optimise_thresholds, rank_final_score)

# 三個成分的 checkpoint。可用環境變數覆寫，讓 `scripts/auto_loop.py` 能在
# 不改原始碼的情況下把評估指向另一個候選模型。
# ⚠️ 快取的 key 帶有 checkpoint 的**內容指紋**，所以換模型不會拿到舊後驗。
GAME = os.environ.get("AICUP_GAME", "cache/rank_mlp_opp_big/models.pt")
# 板面 CNN。空字串 = 不使用（預設）。它是唯一測到誤差與單局模型**負相關**
# 的成分（−0.14 ~ −0.22），但推論要即時重播 SGF 產生 34×19×19 的張量，
# 比其他成分貴一個數量級——所以只在明確指定時才算，而且結果會進 OOF 快取。
BOARD = os.environ.get("AICUP_BOARD", "")
QUESTION = os.environ.get("AICUP_QUESTION", "cache/rank_question_opp/models.pt")
# ⚠️ 預設**就是上線中的三變體集成**，不是單一模型。
# 實測 seq_full（+0.0053）與 seq_w192（+0.0061）各自都沒過 0.005 門檻，
# 合起來卻是 +0.0079（8/8，t=9.15）——用「單獨比 base 好不好」當 gate
# 會系統性地拒絕好的集成成員。要問的是「**加進現有集成有沒有幫助**」。
SEQ = [p for p in os.environ.get(
    "AICUP_SEQ",
    "cache/rank_seq_opp/models.pt,cache/rank_seq_opp_full/models.pt,"
    "cache/rank_seq_w192/models.pt,cache/rank_seq_shape/models.pt").split(",") if p]
# 候選模型用 **AICUP_SEQ_EXTRA 附加**，不要用 AICUP_SEQ 取代。
# 取代的話等於拿單一模型去比三變體集成的分數——那是不公平的比較，
# 而且會把「它加進去有沒有幫助」這個真正要問的問題換成另一個問題。
_extra = [p for p in os.environ.get("AICUP_SEQ_EXTRA", "").split(",") if p]
FEATS = os.environ.get(
    "AICUP_FEATS", "cache/features.npy,cache/features_opp.npy").split(",")
# ⚠️ 單局模型與題目層級模型讀**同一份**特徵庫，所以換其中一個的輸入維度
# 會讓另一個的輸入層對不上。`AICUP_GAME_FEATS` 只換單局模型那一邊。
GAME_FEATS = [p for p in os.environ.get("AICUP_GAME_FEATS", "").split(",") if p]
# 每個 checkpoint 一組序列庫，用 ";" 分隔；只給一組就套用到全部。
# ⚠️ 每個變體一組序列庫（用 ";" 分隔）。shape 變體是 48 維/手、其餘是 24 維/手，
# 套用同一組庫的話其中一邊的輸入維度會對不上、`load_state_dict` 直接拋錯。
_DEF_2V = "cache/sequences.npy,cache/sequences_opp.npy"
_DEF_4V = _DEF_2V + ",cache/seqshape.npy,cache/seqshape_opp.npy"
_ss = [g.split(",") for g in os.environ.get(
    "AICUP_SEQ_STORES",
    ";".join([_DEF_2V, _DEF_2V, _DEF_2V, _DEF_4V])).split(";") if g]
SEQS = _ss if len(_ss) == len(SEQ) else [_ss[0]] * len(SEQ)
# 附加的候選可以有自己的序列庫（例如棋形庫、240 手庫）
_es = [g.split(",") for g in os.environ.get("AICUP_SEQ_EXTRA_STORES", "").split(";") if g]
if _extra:
    SEQ = SEQ + _extra
    SEQS = SEQS + (_es if len(_es) == len(_extra) else [SEQS[0]] * len(_extra))


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def _log(p):
    return np.log(np.clip(p, 1e-12, None))


class Components:
    """三個成分的後驗計算器。只在快取失效時才真的跑模型。"""

    def __init__(self, device):
        self.dev = device
        from aicup.rank_nets import (RankMLP, SeqStore, build_seq_net,
                                     mlp_question_posteriors as question_posteriors,
                                     seq_question_posteriors as seq_posteriors)
        self._RankMLP, self._qp = RankMLP, question_posteriors
        self._SeqNet, self._SeqStore, self._sp = build_seq_net, SeqStore, seq_posteriors
        self._meta = None
        self.X, self.ok = open_stores(FEATS)
        if GAME_FEATS:
            self.Xg, okg = open_stores(GAME_FEATS)
            self.okg = self.ok & okg      # 兩份都有效才用，遮罩才對得起來
        else:
            self.Xg, self.okg = self.X, self.ok
        self._loaded = {}

    def _mlp(self, path):
        if path in self._loaded:
            return self._loaded[path]
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        sc = FeatureScaler().load_state_dict(ck["scaler"])
        nets = []
        for sd in ck["models"]:
            n = self._RankMLP(len(sc.mu), hidden=ck["args"]["hidden"])
            n.load_state_dict(sd); n.eval().to(self.dev)
            nets.append(n)
        self._loaded[path] = (nets, sc, ck)
        return self._loaded[path]

    def game(self, q):
        nets, sc, _ = self._mlp(GAME)
        return _log(self._qp(nets, sc, self.Xg, self.okg, q.game_rows, self.dev))

    def question(self, q):
        nets, sc, _ = self._mlp(QUESTION)
        Xq = build_question_features(self.X, self.ok, q.game_rows)
        with torch.no_grad():
            A = torch.from_numpy(sc.transform(Xq)).to(self.dev)
            ps = []
            for n in nets:
                lc, lo = n(A)
                pc = corn_to_proba(lo).cpu().numpy()
                ps.append(0.5 * torch.softmax(lc, -1).cpu().numpy()
                          + 0.5 * pc / np.maximum(pc.sum(1, keepdims=True), 1e-12))
        return _log(np.mean(ps, axis=0))

    def board(self, q, ckpt=None):
        """板面 CNN 的題目層級後驗。

        用 `forward_set`——模型本來就是用集合池化在**題目層級**訓練的，
        比逐局後驗再幾何平均更貼近它學到的東西。

        ⚠️ 一定要用 `load_gonet`（strict=True），不要用 `strict=False`。
        `pool` / `global_pool_bias` 曾經漏存，strict=False 會**靜默地**
        載入一個隨機初始化的 pooling head，讓整個評估失去意義
        （RESULTS.md 四章的靜默 bug 之一）。
        """
        ckpt = ckpt or BOARD
        key = f"board::{ckpt}"
        if key not in self._loaded:
            from aicup.dataset import SgfCache
            from aicup.nets import load_gonet
            from aicup.planes import game_planes, planes_to_float
            ck = torch.load(ckpt, map_location=self.dev, weights_only=False)
            net = load_gonet(ck["model"], ck.get("config"))
            net.eval().to(self.dev)
            nsnap = int(ck["args"].get("snapshots", 4))
            self._loaded[key] = (net, SgfCache(), game_planes, nsnap,
                                 planes_to_float)
        net, cache, game_planes, nsnap, planes_to_float = self._loaded[key]
        if self._meta is None:
            self._meta = load_metadata()
        meta = self._meta

        out = []
        with torch.no_grad():
            for rows in q.game_rows:
                planes = []
                for r in rows:
                    rec = meta.iloc[int(r)]
                    t = game_planes(cache.get(int(rec["file_idx"]), int(rec["row"])),
                                    rec["color"], n_snapshots=nsnap)
                    if t is not None:
                        planes.append(planes_to_float(t).mean(axis=0))
                if not planes:
                    out.append(np.full(10, 0.1)); continue
                x = torch.from_numpy(np.stack(planes)).float().to(self.dev)[None]
                o = net.forward_set(x)
                pc = corn_to_proba(o["corn_logits"]).cpu().numpy()
                pc = pc / np.maximum(pc.sum(axis=1, keepdims=True), 1e-12)
                p = (0.5 * torch.softmax(o["rank_logits"], -1).cpu().numpy()
                     + 0.5 * pc)[0]
                out.append(p / p.sum())
        return _log(np.stack(out))

    def seq(self, q, ckpt=None, stores=None):
        """一個序列 checkpoint 的題目層級 log 後驗。

        多變體時**每個 checkpoint 各自成為一個 stacker 成分**，不先平均——
        先平均會把各變體的信心與熵抹平，而那正是 stacker 要用的訊號。
        （與 `predict.py` 的做法一致。）
        """
        ckpt = ckpt or SEQ[0]
        stores = stores or SEQS[0]
        key = f"seq::{ckpt}"
        if key not in self._loaded:
            ck = torch.load(ckpt, map_location=self.dev, weights_only=False)
            nfeat = sum(np.load(p, mmap_mode="r").shape[-1] for p in stores)
            nets = []
            for sd in ck["models"]:
                n = self._SeqNet(nfeat, ck["args"])
                n.load_state_dict(sd); n.eval().to(self.dev)
                nets.append(n)
            S = self._SeqStore([np.load(p, mmap_mode="r") for p in stores])
            ok = None
            for p in stores:
                m = np.load(p.replace(".npy", "_ok.npy"))
                ok = m if ok is None else (ok & m)
            self._loaded[key] = (nets, S, ok)
        nets, S, ok = self._loaded[key]
        used = np.unique(np.concatenate([r for r in q.game_rows]))
        lens = np.zeros(len(S), dtype=np.int64)
        lens[used] = (np.abs(np.asarray(S[used], dtype=np.float32)).sum(-1) > 0).sum(-1)
        return _log(self._sp(nets, S, ok, lens, q.game_rows, self.dev))
        # 註：`train_rank_seq.question_posteriors` 的 mask 長度由 S.shape[1] 推得，
        #     所以換成 240 手的序列庫不需要改任何東西。


def score_blend(parts, weights, y, cal, sco, stacker_c=0.0, sizes=None):
    """parts: {name: log_post}；weights: {name: w}。回傳題目層級官方分數。

    融合順序固定為「單局 → 題目層級 → 序列」，與 `predict.py` 一致。

    stacker_c > 0 時改用**學習式融合**（在 cal 上擬合、在 sco 上套用），
    與 `predict.py --stacker` 的行為一致。
    ⚠️ gate 用的標準必須與實際會上線的管線一致，
    否則會用固定權重的分數去決定一個跑 stacker 的系統該不該換模型。
    """
    if stacker_c > 0 and len(parts) > 1:
        from aicup.stacker import Stacker, stack_features
        if sizes is None:
            sizes = np.full(len(y), 12.0)
        F = stack_features(parts, sizes)
        ev = Stacker(C=stacker_c).fit(F[cal], y[cal], parts=parts).expected_level(F)
    else:
        lo = parts["game"]
        # 固定權重路徑只吃第一個序列變體（與 predict.py 的固定權重分支一致）；
        # 多變體集成是 stacker 專屬的能力。
        seq_keys = sorted(k for k in parts if k.startswith("seq"))
        for name, w in (("question", weights.get("question", 0.0)),
                        (seq_keys[0] if seq_keys else "seq", weights.get("seq", 0.0))):
            if w > 0 and name in parts:
                lo = blend_log_posteriors(lo, parts[name], w=w)
        T = fit_temperature(lo[cal], y[cal])
        ev = expected_level(apply_temperature(lo, T))
    th = optimise_thresholds(ev[cal], y[cal])
    return rank_final_score(apply_thresholds(ev[sco], th), y[sco])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8, help="題目抽樣的 seed 數")
    ap.add_argument("--splits", type=int, default=1,
                    help="每組題目用幾種**校準/評分玩家切分**重複量測。"
                         "後驗是共用且快取的，所以多切分幾乎免費，"
                         "但它移掉了一個真實的雜訊來源——實測 seed 層級標準差"
                         "從 ±0.0102 降到 ±0.0060（5 個切分）。"
                         "⚠️ 改這個會改變分數的絕對值，換 gate 時記得 bump GATE_VERSION")
    ap.add_argument("--warm", action="store_true", help="只算後驗並存快取，不評估")
    ap.add_argument("--sweep", choices=["seq", "question"], default=None)
    ap.add_argument("--grid", type=float, nargs="*", default=None)
    ap.add_argument("--w-question", type=float, default=0.2)
    ap.add_argument("--w-seq", type=float, default=0.7)
    ap.add_argument("--stacker", type=float, default=1.0,
                    help="學習式融合的 C（0 = 關閉，退回固定權重）。"
                         "預設與 predict.py 一致，讓 gate 的標準等同會上線的管線")
    ap.add_argument("--cache-dir", default="artifacts/oof")
    ap.add_argument("--json-out", default=None, help="把結果寫成 json（給自動迴圈用）")
    args = ap.parse_args()

    dev = pick_device()
    print(f"device = {dev}", flush=True)
    print(f"  單局 {GAME}\n  題目 {QUESTION}", flush=True)
    for _p in SEQ:
        print(f"  序列 {_p}", flush=True)
    comp = Components(dev)
    cache = OofCache(args.cache_dir)
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)

    t0 = time.time()
    rows = {}
    for seed in range(args.seeds):
        q = build_rank_questions(meta, split, n_per_player=3, seed=seed)
        parts, y = {}, None
        comps = [("game", comp.game, [GAME]),
                 ("question", comp.question, [QUESTION])]
        if BOARD:
            comps.append(("board", comp.board, [BOARD]))
        for _ck, _st in zip(SEQ, SEQS):
            _tag = f"seq_{os.path.basename(os.path.dirname(_ck))}"
            comps.append((_tag,
                          (lambda qq, c=_ck, t=_st: comp.seq(qq, c, t)), [_ck]))
        for name, fn, paths in comps:
            # tag 帶上 checkpoint 的目錄名，不同候選模型才不會共用同一個快取項。
            # （內容指紋本來就會擋掉錯誤命中，但分開存可以保留多個候選、互相比較。）
            tag = f"{name}__{os.path.basename(os.path.dirname(paths[0]))}"
            lo, y, pid = cache.get(tag, q, lambda f=fn, qq=q: f(qq),
                                   model_paths=paths)
            parts[name] = lo
        if args.warm:
            print(f"  seed {seed} 後驗已就緒", flush=True)
            continue

        up = np.array(sorted(set(pid)))
        sizes = np.array([len(r) for r in q.game_rows], dtype=np.float64)
        splits = []
        for sp in range(max(args.splits, 1)):
            rr = (np.random.default_rng(1000 + seed) if args.splits <= 1
                  else np.random.default_rng(90000 + 31 * seed + sp))
            u = up.copy(); rr.shuffle(u)
            cs = set(u[:len(u) // 2].tolist())
            inc = np.array([p in cs for p in pid])
            splits.append((np.flatnonzero(inc), np.flatnonzero(~inc)))

        def _avg(fn):
            """多個切分的平均。切分之間共用同一組題目與後驗，
            所以這只移掉「誰去校準」的雜訊，不是憑空多出獨立樣本。"""
            return float(np.mean([fn(c, s_) for c, s_ in splits]))

        cal, sco = splits[0]
        if args.sweep:
            grid = args.grid or [0.0, 0.2, 0.4, 0.6, 0.8]
            for w in grid:
                ws = {"question": args.w_question, "seq": args.w_seq}
                ws[args.sweep] = w
                rows.setdefault(f"{args.sweep} w={w}", []).append(
                    _avg(lambda c, s_, w_=ws: score_blend(parts, w_, y, c, s_)))
        else:
            for label, ws in (("只有單局", {}),
                              ("固定權重融合", {"question": args.w_question,
                                           "seq": args.w_seq})):
                rows.setdefault(label, []).append(
                    _avg(lambda c, s_, w_=ws: score_blend(parts, w_, y, c, s_)))
            if args.stacker > 0:
                rows.setdefault("學習式融合（預設）", []).append(
                    _avg(lambda c, s_: score_blend(parts, {}, y, c, s_,
                                                   stacker_c=args.stacker, sizes=sizes)))
        print(f"  seed {seed}  " + "  ".join(f"{k}:{v[-1]:.4f}" for k, v in rows.items()),
              flush=True)

    print(f"\n{cache.summary()}，總耗時 {time.time() - t0:.0f} 秒", flush=True)
    if args.warm:
        return 0

    print(f"=== {args.seeds} seed（校準/評分玩家不重疊）  官方 baseline 0.4631 ===")
    out = {}
    for k, v in rows.items():
        a = np.array(v)
        out[k] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                  "above_baseline": int((a > 0.4631).sum()), "n": len(a),
                  "scores": [float(x) for x in a]}
        print(f"  {k:26s} {a.mean():.4f} ±{a.std(ddof=1):.4f}  "
              f"超過 baseline {out[k]['above_baseline']}/{len(a)}")
    best = max(out, key=lambda k: out[k]["mean"])
    print(f"\n最佳：{best} = {out[best]['mean']:.4f}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"results": out, "best": best,
                       "best_score": out[best]["mean"]}, f, indent=2, ensure_ascii=False)
        print(f"已寫出 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
