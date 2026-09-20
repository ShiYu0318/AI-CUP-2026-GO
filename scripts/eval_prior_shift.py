"""競賽二：**等級先驗不匹配**的暴露評估。

## 為什麼要有這支

官方測試集是 400 位玩家、每人一題，所以它的等級分布是**玩家層級**的。
訓練資料每個等級剛好 100,000 **局**（明顯是刻意平衡的），
但**玩家數**從 433（rank 4）到 1582（rank 3），相差 3.65 倍。

我們的驗證題目對每一位合格（≥5 譜）的驗證玩家出題，所以驗證集的先驗是
**自然玩家分布**：rank 3 佔 19.8%、rank 4 佔 5.4%。

而官方競賽一的候選庫要求「400 人**各 100 譜**」。把玩家限制在 ≥100 譜之後，
等級分布變成 7.4%~12.3%——**接近均勻**。若競賽二的 400 位查詢玩家
出自同一個池子，真實測試先驗就比我們校準時用的分布平均得多。

切點是在校準集上用 DP 挑的，**切點的位置直接取決於校準集的等級分布**。
先驗不匹配 = 切點系統性偏向 rank 3。這種錯誤在自家驗證上完全看不出來，
只會在 Private LB 上出現——正是最該事先量掉的風險。

## 做法

用重抽樣（不是取子集）把題目集調成目標先驗：
取子集會讓題數掉到 450（rank 4 只有約 45 題），低於協定的 800 題門檻；
帶權重重抽到原本的題數則能保住統計力，代價是每個 seed 要多抽幾次平均掉雜訊。

四種組合都要量，因為**改校準也可能有代價**：

    A 自然校準 / 自然測試    ← 現況
    B 自然校準 / 平衡測試    ← 若官方是平衡的，我們實際會拿到的分數
    C 平衡校準 / 平衡測試    ← 修正後
    D 平衡校準 / 自然測試    ← 修正的代價（萬一官方其實是自然分布）

有意義的比較是 B↔C（同一個測試先驗）與 A↔D（同一個測試先驗）。
A 與 B 之間不可直接比——它們是不同的測試分布。

用法：
    python3 scripts/eval_prior_shift.py --seeds 8 --draws 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aicup.benchmark import build_rank_questions                 # noqa: E402
from aicup.data import load_metadata, make_player_split          # noqa: E402
from aicup.oofcache import OofCache                              # noqa: E402
from aicup.scoring import (apply_thresholds, estimate_prior_em,    # noqa: E402
                           optimise_thresholds, rank_score)
from eval_blend import (GAME, QUESTION, SEQ, Components,         # noqa: E402
                        pick_device)

N_LEVELS = 10


def per_question_scores(parts, y, cal, sco, stacker_c, sizes):
    """與 `eval_blend.score_blend` 相同的管線，但回傳**逐題**分數而不是平均。

    逐題分數才能做重要性加權——評分端用有放回重抽會讓稀有等級的
    同一題被複製幾十次，把變異低估掉；加權則是精確且無偏的。
    """
    from aicup.stacker import Stacker, stack_features
    F = stack_features(parts, sizes)
    ev = Stacker(C=stacker_c).fit(F[cal], y[cal], parts=parts).expected_level(F)
    th = optimise_thresholds(ev[cal], y[cal])
    return rank_score(apply_thresholds(ev[sco], th), y[sco])


def adaptive_scores(parts, y, cal, sco, stacker_c, sizes, w_test, rng,
                    shrink: float):
    """**不押注**先驗：先用測試端的後驗把測試先驗估出來，再照它重做校準。

    這是 transductive 的——只用到測試輸入的後驗，沒有用到任何標籤，
    官方測試集在推論當下就拿得到，所以線上可以照做。
    """
    from aicup.stacker import Stacker, stack_features
    F = stack_features(parts, sizes)
    st = Stacker(C=stacker_c).fit(F[cal], y[cal], parts=parts)
    src = np.bincount(y[cal], minlength=N_LEVELS).astype(float)
    src /= src.sum()
    est = estimate_prior_em(st.proba(F[sco]), source_prior=src,
                            weights=w_test, shrink=shrink)
    cal2 = resample_to_prior(cal, y, est, rng)
    return per_question_scores(parts, y, cal2, sco, stacker_c, sizes), est


def prior_weights(y_sco, prior):
    """w_i ∝ prior[y_i] / 自然頻率[y_i]，平均值正規化為 1。"""
    cnt = np.bincount(y_sco, minlength=N_LEVELS).astype(float)
    w = np.where(cnt[y_sco] > 0, prior[y_sco] / np.maximum(cnt[y_sco], 1), 0.0)
    return w / max(w.mean(), 1e-12)


def resample_to_prior(idx, y, prior, rng):
    """把索引集重抽成目標先驗，**題數保持不變**。

    以 w_i ∝ prior[y_i] / natural[y_i] 的機率有放回地抽。
    有放回會讓稀有等級的同一題被重複抽到——這會低估該等級的多樣性，
    所以外層要多抽幾次（`--draws`）把這個雜訊平均掉。
    """
    cnt = np.bincount(y[idx], minlength=N_LEVELS).astype(float)
    w = np.where(cnt[y[idx]] > 0, prior[y[idx]] / np.maximum(cnt[y[idx]], 1), 0.0)
    if w.sum() <= 0:
        return idx
    return rng.choice(idx, size=len(idx), replace=True, p=w / w.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--draws", type=int, default=3,
                    help="每個 seed 重抽幾次（平均掉有放回抽樣的雜訊）")
    ap.add_argument("--stacker", type=float, default=1.0)
    ap.add_argument("--cache-dir", default="artifacts/oof")
    ap.add_argument("--alphas", type=float, nargs="*", default=[],
                    help="額外測試 natural^(1-α)·uniform^α 這族中間先驗")
    ap.add_argument("--shrink", type=float, nargs="*", default=[0.0, 0.3],
                    help="EM 自適應校準的收縮係數（往校準集自身的先驗拉回）")
    ap.add_argument("--test-priors", nargs="*", default=["自然", "p100", "均勻"],
                    help="最壞情況要涵蓋的**測試**先驗（校準先驗一律全測）")
    ap.add_argument("--json-out", default="artifacts/prior_shift.json")
    args = ap.parse_args()

    dev = pick_device()
    comp = Components(dev)
    cache = OofCache(args.cache_dir)
    meta = load_metadata()
    split = make_player_split(meta, val_frac=0.20, seed=42)
    uniform = np.full(N_LEVELS, 1.0 / N_LEVELS)

    # 三種可能的測試先驗。p100 不是對沖，是**估計**——
    # 官方候選庫明確要求「400 人各 100 譜」，所以測試玩家必定是 ≥100 譜的那群，
    # 而那群人的等級分布是可以直接從訓練資料算出來的。
    cnt_all = meta.groupby("player_id").size()
    rk_all = meta.groupby("player_id")["rank_id"].first()
    p100 = np.bincount(rk_all[cnt_all >= 100].to_numpy(), minlength=N_LEVELS).astype(float)
    p100 /= p100.sum()
    # 校準先驗不該賭在端點上。natural^(1-α)·uniform^α 是一族介於兩者之間的
    # 先驗；α=0 是現況、α=1 是完全平衡。在**整個合理範圍**內最壞情況最好的那個
    # 才是該用的，這比賭「官方一定是平衡的」或「一定是自然的」都穩。
    nat = np.bincount(
        build_rank_questions(meta, split, n_per_player=3, seed=0).true_rank_id,
        minlength=N_LEVELS).astype(float)
    nat /= nat.sum()
    PRIORS = {"自然": None, "均勻": uniform, "p100": p100}
    for a in args.alphas:
        m = nat ** (1 - a) * uniform ** a
        PRIORS[f"a{a:g}"] = m / m.sum()
    print("目標先驗 %：")
    for k, v in PRIORS.items():
        print(f"  {k:<4} " + (str(np.round(v * 100, 1).tolist()) if v is not None
                              else "（驗證集自身的分布）"))

    rows: dict[str, list[float]] = {}
    t0 = time.time()
    for seed in range(args.seeds):
        q = build_rank_questions(meta, split, n_per_player=3, seed=seed)
        parts, y, pid = {}, None, None
        for name, fn, paths in (("game", comp.game, [GAME]),
                                ("question", comp.question, [QUESTION]),
                                ("seq", comp.seq, [SEQ])):
            tag = f"{name}__{os.path.basename(os.path.dirname(paths[0]))}"
            lo, y, pid = cache.get(tag, q, lambda f=fn, qq=q: f(qq), model_paths=paths)
            parts[name] = lo

        # 校準與評分的**玩家**不重疊（與 eval_blend 完全相同的協定）
        up = np.array(sorted(set(pid)))
        rr = np.random.default_rng(1000 + seed); rr.shuffle(up)
        cs = set(up[:len(up) // 2].tolist())
        inc = np.array([p in cs for p in pid])
        cal, sco = np.flatnonzero(inc), np.flatnonzero(~inc)
        sizes = np.array([len(r) for r in q.game_rows], dtype=np.float64)
        if seed == 0:
            d = np.bincount(y, minlength=N_LEVELS) / len(y)
            print("\n驗證題目的等級分布 %：", np.round(d * 100, 1).tolist(), flush=True)

        # 評分端的權重只依賴 sco 與目標先驗，與校準方式無關 → 先算好
        wts = {k: (np.ones(len(sco)) if PRIORS[k] is None
                   else prior_weights(y[sco], PRIORS[k]))
               for k in args.test_priors}

        for test_name, w in wts.items():
            for sh in args.shrink:
                rng = np.random.default_rng(31000 + 97 * seed)
                ps, est = adaptive_scores(parts, y, cal, sco, args.stacker,
                                          sizes, w, rng, sh)
                rows.setdefault(f"校準=EM{sh:g} 測試={test_name}", []).append(
                    float(np.average(ps, weights=w)))
                if seed == 0:
                    print(f"    EM(shrink={sh:g}) 在「{test_name}」下估到的先驗 %：",
                          np.round(est * 100, 1).tolist(), flush=True)

        for cal_name, cal_prior in PRIORS.items():
            for d in range(args.draws if cal_prior is not None else 1):
                rng = np.random.default_rng(7000 + 97 * seed + d)
                c = cal if cal_prior is None else resample_to_prior(cal, y, cal_prior, rng)
                ps = per_question_scores(parts, y, c, sco, args.stacker, sizes)
                for test_name, w in wts.items():
                    rows.setdefault(f"校準={cal_name:<3} 測試={test_name}", []).append(
                        float(np.average(ps, weights=w)))
        print(f"  seed {seed} 完成", flush=True)

    print(f"\n{cache.summary()}，總耗時 {time.time() - t0:.0f} 秒")
    print(f"=== {args.seeds} seed（校準端重抽 {args.draws} 次，評分端精確加權）===")
    out = {}
    for k in sorted(rows):
        a = np.array(rows[k])
        out[k] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "n": len(a)}
        print(f"  {k}  {a.mean():.4f} ±{a.std(ddof=1):.4f}")

    print("\n每種校準方式在三種測試先驗下的**最壞情況**（不知道真實先驗時該看這個）：")
    best, best_worst = None, -1.0
    labels = ([f"{c:<3}" for c in PRIORS]
              + [f"EM{sh:g}" for sh in args.shrink])
    for cal_name in labels:
        vals = [out[f"校準={cal_name} 測試={t}"]["mean"] for t in args.test_priors]
        worst = min(vals)
        out[f"_worst__{cal_name.strip()}"] = worst
        print(f"  校準={cal_name:<6} 最壞 {worst:.4f}  平均 {np.mean(vals):.4f}")
        if worst > best_worst:
            best, best_worst = cal_name.strip(), worst
    print(f"\n最壞情況最佳的校準方式：{best}（{best_worst:.4f}）")
    out["_best_by_worst_case"] = best

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"已寫出 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
