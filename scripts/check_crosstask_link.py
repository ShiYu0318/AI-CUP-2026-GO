"""11/04 當天先跑這支：**競賽二的 400 位查詢玩家，是不是競賽一的 400 位候選人？**

## 為什麼這件事可能很大

官方競賽一的候選庫是「400 人**各 100 譜**」。
如果競賽二的 400 道棋力題出自同一批人，那我們可以：

    1. 用競賽一的模型把每道棋力題認出是哪位候選人（top-1 0.9475）
    2. 改用那位候選人的 **100 盤**去預測棋力，而不是題目裡的 5~20 盤

證據量多 5~20 倍。實測盤數對分數的影響很大
（5 盤 0.5398、15 盤 0.5928，見 RESULTS.md §2.10），
所以這是目前想得到**最大的單一結構性機會**。

⚠️ 但它建立在一個**官方沒有明說**的假設上，而且賭錯的代價很大：
若兩邊其實是不同的人，我們會把別人的 100 盤棋塞進去，分數會崩。
所以這支的工作是**用資料回答，不是用猜的**。

## 兩個獨立的判準

**1. 棋譜字串直接重疊。** 最definitive——如果主辦方是從同一份測試資料
   衍生兩個子題，棋力題的棋譜會**原封不動**出現在候選庫裡。零成本、零歧義。

**2. Embedding 的一一對應程度。** 即使棋譜不重疊（各自抽不同的局），
   只要是同一批人，400 道棋力題對 400 位候選人的 top-1 應該幾乎**相異**
   （競賽一實測 0.885）。若是不同的人，會大量撞在同幾個候選身上。
   這用的是 `bijection_consistency`——競賽一已經靠它確認過一一對應。

用法：
    python3 scripts/check_crosstask_link.py --test-dir tests
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aicup.rerank import bijection_consistency                  # noqa: E402
from aicup.submission import resolve_test_file                          # noqa: E402

# 判準門檻。兩個都很保守——這個增強一旦用錯是災難性的，
# 寧可放過機會也不要賭。
SGF_OVERLAP_STRONG = 0.50      # 一半以上的棋力題棋譜出現在候選庫 → 幾乎確定同源

# 相異率門檻 0.85 不是取中間值，是算出來的。
# 設 f = 查詢玩家真的在候選庫裡的比例，由兩個量測點
#（f=1 → 0.9075、f=0 → 0.5117）線性內插：
#     相異率 ≈ 0.512 + 0.396 f
# 而增強的淨增益是 f × 0.0586 + (1−f) × (−0.1993)，**零點在 f = 77%**。
# 門檻若訂 0.70，對應 f = 47.5%——會在 f∈[48%, 77%] 這一整段**放行但虧損**。
# 0.85 對應 f ≈ 85%，留了安全邊際；而真實同源的情況實測 0.8975~0.9075，仍然通得過。
BIJECTION_STRONG = 0.85


def _sgf_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("sgf_")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="tests")
    ap.add_argument("--tags", nargs="*", default=["public", "private"])
    ap.add_argument("--feature-metric", default="cache/feature_metric/models.pt")
    ap.add_argument("--skip-embedding", action="store_true",
                    help="只做棋譜字串比對（幾秒鐘），不跑模型")
    args = ap.parse_args()

    verdicts = {}
    for tag in args.tags:
        rp = resolve_test_file(args.test_dir, "rank", tag)
        cp = resolve_test_file(args.test_dir, "candidates", tag)
        if not (os.path.exists(rp) and os.path.exists(cp)):
            print(f"  {tag}: 找不到檔案，跳過（{rp} / {cp}）")
            continue

        rank = pd.read_csv(rp)
        cand = pd.read_csv(cp, usecols=["player_id", "sgf_content"])
        rs = set()
        for c in _sgf_columns(rank):
            rs |= set(rank[c].dropna().astype(str))
        rs.discard("")
        cs = set(cand["sgf_content"].astype(str))
        inter = rs & cs
        frac = len(inter) / max(len(rs), 1)
        print(f"\n=== {tag} ===")
        print(f"  棋力題不重複棋譜 {len(rs):,}，候選庫 {len(cs):,}，"
              f"交集 {len(inter):,}（{frac:.3f}）")

        # 判準 1：棋譜直接重疊
        if frac >= SGF_OVERLAP_STRONG:
            # 每道題的棋譜對到哪位候選人——若一題的多盤都指向同一人，更強的證據
            owner = dict(zip(cand["sgf_content"].astype(str), cand["player_id"]))
            agree, n = 0, 0
            for _, row in rank.iterrows():
                ids = {owner[s] for c in _sgf_columns(rank)
                       if isinstance(row[c], str) and row[c] in owner
                       for s in [row[c]]}
                if ids:
                    n += 1
                    agree += (len(ids) == 1)
            print(f"  ✅ 棋譜直接重疊。{n} 道題能對到候選人，"
                  f"其中 {agree} 道的所有棋譜都指向**同一位**（{agree/max(n,1):.3f}）")
            verdicts[tag] = "sgf_overlap"
            continue

        if args.skip_embedding:
            print("  棋譜沒有明顯重疊；--skip-embedding 已指定，不做 embedding 檢查")
            verdicts[tag] = "unknown"
            continue

        # 判準 2：embedding 的一一對應程度
        print("  棋譜沒有明顯重疊 → 改用 embedding 檢查是否為同一批人…")
        import torch
        from aicup.feature_metric import (FeatureEmbedder, FeatureScaler,
                                          embed_features, player_template)
        from aicup.features import parallel_batch_features
        dev = torch.device("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
        ck = torch.load(args.feature_metric, map_location=dev, weights_only=False)
        sc = FeatureScaler().load_state_dict(ck["scaler"])
        cols = ck.get("cols")
        models = []
        for sd in ck["models"]:
            m = FeatureEmbedder(len(sc.mu), embed_dim=ck["args"]["embed_dim"],
                                hidden=ck["args"]["hidden"])
            m.load_state_dict(sd); m.eval().to(dev)
            models.append(m)

        def tpl(A):
            if cols is not None:
                A = np.asarray(A)[:, cols]
            v = np.concatenate([player_template(embed_features(m, sc, A, dev))
                                for m in models])
            return v / max(float(np.linalg.norm(v)), 1e-8)

        cand_full = pd.read_csv(cp)
        Xc, okc = parallel_batch_features(cand_full["sgf_content"].tolist(),
                                          cand_full["color"].tolist())
        names = sorted(cand_full["player_id"].unique())
        pidc = cand_full["player_id"].to_numpy()
        G = np.stack([tpl(Xc[okc & (pidc == n)]) for n in names])

        Q = []
        for _, row in rank.iterrows():
            sg = [row[c] for c in _sgf_columns(rank) if isinstance(row[c], str) and row[c]]
            co = [row[c.replace("sgf_", "color_")] for c in _sgf_columns(rank)
                  if isinstance(row[c], str) and row[c]]
            Xq, okq = parallel_batch_features(sg, co)
            Q.append(tpl(Xq[okq]) if okq.any() else np.zeros(G.shape[1]))
        sim = np.stack(Q) @ G.T
        bij = bijection_consistency(sim)
        print(f"  top-1 相異率 {bij:.3f}（競賽一實測同一批人時是 0.885）")
        verdicts[tag] = "bijection" if bij >= BIJECTION_STRONG else "no"

    print("\n" + "=" * 60)
    ok = [t for t, v in verdicts.items() if v in ("sgf_overlap", "bijection")]
    if len(ok) == len(verdicts) and verdicts:
        print("結論：**兩個子題是同一批玩家** → 可以用候選庫的 100 譜增強棋力預測")
        print("      跑 `python3 scripts/predict.py --crosstask`")
    else:
        print("結論：**沒有足夠證據** → 不要開 --crosstask。")
        print("      賭錯的代價是把別人的 100 盤棋塞進題目裡，分數會崩；")
        print("      而放過這個機會只是回到原本的分數。不對稱，所以保守。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
