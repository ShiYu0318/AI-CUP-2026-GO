"""無人值守的實驗迴圈：跑 → 用誠實協定 gate → 記錄 → commit → 跑下一個。

**為什麼要有這支**：這個專案的瓶頸不是方法，是**迭代速度**。
每個實驗都要人在旁邊等通知、看結果、決定下一步，一晚上只能推進兩三個。
這支把「跑實驗 → 評估 → 記錄」變成一個佇列，整夜自己推進。

## 設計上的三個刻意選擇

**0. Gate 問的是「加進現有集成有沒有幫助」，不是「單獨比 base 好不好」。**
實測 `seq_full`（+0.0053）與 `seq_w192`（+0.0061）各自都沒過 0.005 門檻，
但兩個加上 base 組成三變體集成後是 **+0.0079（8/8，t=9.15）**。
用「單獨比較」當 gate 會**系統性地拒絕好的集成成員**——
集成要的是分歧，不是每個成分都更強。
所以 `eval_blend.py` 的預設就是上線中的整個集成，
候選模型以 **`AICUP_SEQ_EXTRA`** 附加在後面，成為額外的一個 stacker 成分。
⚠️ 不要用 `AICUP_SEQ`——那是**取代**整個清單，等於拿單一模型去比集成的分數。

**1. Gate 用的是 RESULTS.md〇章的協定，不是訓練腳本自己報的分數。**
訓練腳本內建的評估是同集校準（樂觀）。這裡一律走 `scripts/eval_blend.py`：
8 組 player-disjoint、校準與評分玩家不重疊、DP 切點。

⚠️ **gate 的標準必須與實際會上線的管線一致，而且 `production` 要跟它同一把尺。**
真實踩過：`predict.py` 改用 stacker 之後，`eval_blend.py` 也跟著改，
但 state 裡的 `production` 還是固定權重時代的 0.5639——
下一個實驗會拿 stacker 分數（約 0.5715）去比 0.5639，**假性通過 +0.008**，
而那 0.008 只是換了評分標準，不是模型變好。
**換 gate 的定義時，一定要同時重設 `production`。**
這件事現在由 `GATE_VERSION` 強制執行：版本不符就把 production 作廢、
重新量一次。文件擋不住這種錯，機制可以。

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

⚠️ **不要手動在迴圈外啟動訓練或評估。** 排程器只看得到自己排的工作，
手動啟動的會繞過記憶體預算。2026-09-21 犯了兩次，兩次都一樣：
機器上同時有兩個 6.14 GB 的訓練張量加上一個評估，
RSS 掉到 0.6 GB——**張量被換出到磁碟，每個 batch 都在 page fault**。
症狀是「還在跑、CPU 也有在動，但慢五倍」，
而且從日誌上看不出來（只有 seed 之間的時間會突然從 34 分變成 80 分）。
要並行就用 `--jobs`，讓排程器按 `mem_gb` 決定。

**連 `pytest` 都算。** 完整測試套件只跑 2 分鐘，但其中有測試會載入特徵庫，
而那會把訓練張量擠出 page cache。實測在兩個訓練跑的時候跑一次完整測試，
單局 MLP 的一個 seed 從 15 分變成 **70 分**——
測試本身兩分鐘就結束了，代價卻是一小時。訓練期間只用 `tail` 看日誌。

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
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = "artifacts/auto_loop_state.json"
# 實驗紀錄放在 `docs/`（gitignore），**不會進 GitHub**——
# 它記的是什麼有效、什麼無效，那是競賽策略。
LEDGER = "docs/RESULTS.md"
FLOOR = 0.005          # 小於這個增益不算贏（seed 間標準差約 ±0.010）

# 兩個子題各有自己的生產基準與 baseline。
# ⚠️ 0.9389（競賽一）與 0.5782（競賽二）不能放在同一個欄位比較——
# 混在一起的話，一個競賽一的實驗會拿 0.94 去比 0.58 而「大勝」。
TRACKS = {
    "rank":   {"baseline": 0.4631, "name": "競賽二 棋力",
               "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8",
                        "--splits", "5", "--json-out", "artifacts/blend_default.json"],
               "json": "artifacts/blend_default.json"},
    # ⚠️ 競賽一的預設已經換成棋形版（predict.py 的 --feature-metric），
    # 所以基準也必須用棋形版量，否則候選會拿去比一個已經下線的設定。
    "player": {"baseline": 0.2845, "name": "競賽一 身分",
               "eval": ["python3", "scripts/eval_player.py", "--seeds", "8",
                        "--json-out", "artifacts/player_baseline.json"],
               "env": {"AICUP_METRIC": "cache/feature_metric_shape8/models.pt",
                       "AICUP_PLAYER_FEATS":
                           "cache/features.npy,cache/features_opp.npy,"
                           "cache/features_shape.npy,cache/features_shape_opp.npy"},
               "json": "artifacts/player_baseline.json"},
}

# gate 的「尺」的版本。**改動 gate 的定義時一定要 +1。**
# 這不是裝飾：state 裡的 production 是用舊尺量的，換尺卻沿用舊數字，
# 下一個實驗就會拿新尺的分數去比舊尺的基準，**假性通過**。
# 真實踩過一次（見下方模組說明）。版本不符時會強制重新量一次基準，
# 讓這個錯誤在機制上不可能發生，而不是靠人記得。
#   1  固定權重融合
#   2  學習式融合（stacker）取代固定權重
#   3  gate 改為「候選 + 現有三變體集成」——單獨比 base 會系統性拒絕好的集成成員
#   4  每組題目改用 5 種校準/評分切分取平均——seed 層級標準差 ±0.0102 → ±0.0060，
#      統計力接近翻倍而成本幾乎為零（後驗是共用且快取的）
#   5  production 改成**分軌**（競賽一/競賽二各一個），迴圈不再只能改進競賽二
#   6  競賽二的預設集成加入 seq_shape（三變體 → 四變體）——
#      基準的組成變了，舊的 production 就不能用了
#   7  競賽一的預設換成棋形版度量模型（0.9389 → 0.9653，最壞組成）
#   8  競賽二的預設集成補上 seq_qual（它早就上線了，但 eval_blend 的預設漏了，
#      gate 因此一直在跟一個比生產少一個成分的系統比）＋加入板面 CNN
#   9  競賽一的 gate 從 3 個評估 seed 改成 8 個（量測協定要求 ≥8），
#      預設度量模型換成 8 seed 集成
#  10  競賽二的序列變體從 5 個精簡成 opp_full + shape（板面上線後其餘冗餘）
GATE_VERSION = 10
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
    # ── 2026-09-21 第三輪：**換資訊、換模型類別**，不再換大小 ──────────────
    # 前兩輪已經把「放大既有東西」測死了：
    #   加 seed +0.0010 / 加資料 +0.0041 / 加寬 +0.0049 / 加視窗 −0.0009 / 加 MLP 寬度 −0.0031
    # 但三個「個別不夠」的變體集成起來 +0.0079（8/8，t=9.15）。
    # 結論：集成要的是**分歧**。所以這一輪每一項都改動「資訊」或「歸納偏置」，
    # 不改容量。
    #
    # 另外兩個已經證偽、不要再排的方向：
    #   · 「尾盤沒被看到」——max_moves 數的是我方的手，只有 1.46% 的棋局被截斷
    #   · EM 自適應先驗估計——序數任務的混淆矩陣近奇異，每一格都輸給固定先驗
    # 順序：MPS 的先跑（卷積在 MPS 上快 4 倍），CPU 的排後面與之並行。
    # seq_gru 放最後是因為它此刻正在手動訓練中——
    # 排前面的話 `produces` 還不存在，迴圈會重複訓練同一個模型。
    {
        "name": "seq_shape",
        "device": "mps",
        "mem_gb": 10.0,
        "why": "**換資訊**：加上逐手棋形（接／尖／跳／飛／象步、貼／尖沖、愚形、厚度）。"
               "現有 12 維只有鄰居的**數量**，沒有**排列**——而 尖(1,1) 與 跳(0,2) "
               "的曼哈頓距離都是 2，兩種完全不同的手在特徵上是同一個數字。"
               "愚形率 0.1325 與 features.py 既有的整局愚形率 0.13 吻合（交叉驗證）",
        "train": ["python3", "scripts/train_rank_seq.py",
                  "--extra-sequences", "cache/sequences_opp.npy",
                  "cache/seqshape.npy", "cache/seqshape_opp.npy",
                  "--seeds", "3", "--steps", "12000", "--train-games", "400000",
                  "--out", "cache/rank_seq_shape"],
        "produces": "cache/rank_seq_shape/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8", "--splits", "5",
                 "--json-out", "artifacts/blend_seqshape.json"],
        "eval_env": {"AICUP_SEQ_EXTRA": "cache/rank_seq_shape/models.pt",
                     "AICUP_SEQ_EXTRA_STORES": "cache/sequences.npy,cache/sequences_opp.npy,"
                                               "cache/seqshape.npy,cache/seqshape_opp.npy"},
        "json": "artifacts/blend_seqshape.json",
    },
    {
        "name": "mlp_shape",
        "device": "cpu",
        "mem_gb": 7.0,
        "why": "**把棋形接到最強的成分上**。stacker 裡貢獻最大的是單局 MLP，"
               "但它讀的 1178 維手工特徵沒有手法類別。線性探針："
               "手工 0.2826、棋形統計 96 維單獨 0.2609、合併 0.2987（+0.0161）——"
               "**96 維就達到 1178 維九成的水準**，代表手工特徵嚴重低估了棋形",
        "train": ["python3", "scripts/train_rank_mlp.py",
                  "--extra-features", "cache/features_opp.npy",
                  "cache/features_shape.npy", "cache/features_shape_opp.npy",
                  "--seeds", "5", "--steps", "20000", "--hidden", "1024",
                  "--train-games", "0", "--device", "cpu",
                  "--out", "cache/rank_mlp_shape"],
        "produces": "cache/rank_mlp_shape/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8", "--splits", "5",
                 "--json-out", "artifacts/blend_mlpshape.json"],
        # 單局模型是**取代**不是附加——它在 stacker 裡只有一個成分，
        # 而且新舊兩版讀的特徵維度不同，不可能同時存在。
        "eval_env": {"AICUP_GAME": "cache/rank_mlp_shape/models.pt",
                     "AICUP_GAME_FEATS": "cache/features.npy,cache/features_opp.npy,"
                                         "cache/features_shape.npy,"
                                         "cache/features_shape_opp.npy"},
        "json": "artifacts/blend_mlpshape.json",
    },
    # ⛔ 2026-09-21：跑失敗且**不打算修**，留著是為了記錄原因。
    #    失敗本身是維度不合：`--game-model rank_mlp_opp_big` 是用 1178 維訓練的，
    #    這裡卻餵 1274 維 → ValueError: shapes (15,1274) (1178,)。
    #    修法是改用 `rank_mlp_shape`，但那會讓題目層級模型依賴一個
    #    **我們不會上線的**單局模型（mlp_shape 只有 +0.0011，沒過門檻）。
    #    而且期望值本來就低：棋形對單局 MLP 只值 +0.0011，
    #    因為 stacker 已經透過 seq_shape 拿到同樣的資訊了（RESULTS.md §2.21）。
    #    題目層級模型讀的是同一份特徵，幾乎必然同樣冗餘。
    {
        "name": "question_shape",
        "device": "cpu",
        "mem_gb": 6.0,
        "skip": True,
        "why": "（已跳過）題目層級模型讀同一份手工特徵，"
               "而棋形對單局 MLP 只值 +0.0011——資訊已被 seq_shape 抓走，"
               "這裡幾乎必然同樣冗餘。詳見上方註解",
        "train": ["python3", "scripts/train_rank_question.py",
                  "--extra-features", "cache/features_opp.npy",
                  "cache/features_shape.npy", "cache/features_shape_opp.npy",
                  "--game-model", "cache/rank_mlp_opp_big/models.pt",
                  "--seeds", "3", "--device", "cpu",
                  "--out", "cache/rank_question_shape"],
        "produces": "cache/rank_question_shape/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8", "--splits", "5",
                 "--json-out", "artifacts/blend_qshape.json"],
        # AICUP_FEATS 換題目層級那一邊；AICUP_GAME_FEATS 明確釘回原本的，
        # 免得單局模型也跟著換掉（那樣就分不清增益是誰的）。
        "eval_env": {"AICUP_QUESTION": "cache/rank_question_shape/models.pt",
                     "AICUP_FEATS": "cache/features.npy,cache/features_opp.npy,"
                                    "cache/features_shape.npy,"
                                    "cache/features_shape_opp.npy",
                     "AICUP_GAME_FEATS": "cache/features.npy,cache/features_opp.npy"},
        "json": "artifacts/blend_qshape.json",
    },
    {
        "name": "seq_rankloss",
        "device": "mps",
        "mem_gb": 6.5,
        "why": "**換損失**：加上成對排序損失。診斷顯示切點已到頂"
               "（oracle 切點只剩 +0.0172），分數幾乎全由期望等級的排序決定"
               "（Spearman 0.9037），但 CE 與 CORN 都是分類損失、不直接管順序。"
               "這是唯一一個直接對著瓶頸下手的實驗",
        "train": ["python3", "scripts/train_rank_seq.py",
                  "--extra-sequences", "cache/sequences_opp.npy",
                  "--rank-loss", "0.3", "--seeds", "3", "--steps", "12000",
                  "--train-games", "400000", "--out", "cache/rank_seq_rankloss"],
        "produces": "cache/rank_seq_rankloss/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8", "--splits", "5",
                 "--json-out", "artifacts/blend_seqrankloss.json"],
        "eval_env": {"AICUP_SEQ_EXTRA": "cache/rank_seq_rankloss/models.pt"},
        "json": "artifacts/blend_seqrankloss.json",
    },
    {
        "name": "seq_gru_shape",
        "device": "cpu",
        "mem_gb": 8.0,
        "why": "兩個新方向疊起來：GRU 讀棋形序列。若 seq_gru 與 seq_shape 各自"
               "帶來的分歧來自不同地方，這一個應該與兩者都不一樣——"
               "集成成員之間**兩兩都不像**才是最理想的",
        "train": ["python3", "scripts/train_rank_seq.py",
                  "--extra-sequences", "cache/sequences_opp.npy",
                  "cache/seqshape.npy", "cache/seqshape_opp.npy",
                  "--arch", "gru", "--seeds", "3", "--steps", "12000",
                  "--train-games", "400000", "--device", "cpu",
                  "--out", "cache/rank_seq_gru_shape"],
        "produces": "cache/rank_seq_gru_shape/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8", "--splits", "5",
                 "--json-out", "artifacts/blend_seqgrushape.json"],
        "eval_env": {"AICUP_SEQ_EXTRA": "cache/rank_seq_gru_shape/models.pt",
                     "AICUP_SEQ_EXTRA_STORES": "cache/sequences.npy,cache/sequences_opp.npy,"
                                               "cache/seqshape.npy,cache/seqshape_opp.npy"},
        "json": "artifacts/blend_seqgrushape.json",
    },
    {
        "name": "metric_shape",
        "track": "player",
        "device": "cpu",
        "mem_gb": 6.0,
        "why": "**競賽一也可能吃得到棋形**。手法偏好（接／尖／跳／飛／象步）"
               "本質上是**個人簽名**——帳本裡最強的身分特徵『開局順序族』就是同一類東西。"
               "而競賽一目前 0.9599、最壞組成 dan_only 0.9389，"
               "愈接近天花板、每 0.01 能拉開的名次愈多。"
               "⚠️ 這是迴圈第一個競賽一的實驗，gate 走 eval_player.py（最壞組成）",
        "train": ["python3", "scripts/train_feature_metric.py",
                  "--extra-features", "cache/features_opp.npy",
                  "cache/features_shape.npy", "cache/features_shape_opp.npy",
                  "--seeds", "3", "--device", "cpu",
                  "--out", "cache/feature_metric_shape"],
        "produces": "cache/feature_metric_shape/models.pt",
        "eval": ["python3", "scripts/eval_player.py", "--seeds", "3",
                 "--json-out", "artifacts/player_shape.json"],
        "eval_env": {"AICUP_METRIC": "cache/feature_metric_shape/models.pt",
                     "AICUP_PLAYER_FEATS": "cache/features.npy,cache/features_opp.npy,"
                                           "cache/features_shape.npy,"
                                           "cache/features_shape_opp.npy"},
        "json": "artifacts/player_shape.json",
    },
    {
        "name": "metric_trans",
        "track": "player",
        "device": "cpu",
        "mem_gb": 6.0,
        "why": "棋形**頻率**讓競賽一從 0.9599 跳到 0.9745，最大增益在段位（+0.0264）"
               "——因為風格不像品質那樣會隨棋力收斂。"
               "把同一個邏輯再推一步：手法的**轉移**（接→跳、尖→飛）"
               "比手法的頻率更能刻畫節奏。帳本裡最強的身分特徵"
               "「開局轉移族」就是同一類東西（順序 one-hot 0.6603 → 加轉移 0.7221）。"
               "7×7=49 維／視角，從棋形庫用純 numpy 算，建庫只花 8 秒",
        "train": ["python3", "scripts/train_feature_metric.py",
                  "--extra-features", "cache/features_opp.npy",
                  "cache/features_shape.npy", "cache/features_shape_opp.npy",
                  "cache/features_shapetrans.npy",
                  "cache/features_shapetrans_opp.npy",
                  "--seeds", "3", "--device", "cpu",
                  "--out", "cache/feature_metric_trans"],
        "produces": "cache/feature_metric_trans/models.pt",
        "eval": ["python3", "scripts/eval_player.py", "--seeds", "3",
                 "--json-out", "artifacts/player_trans.json"],
        "eval_env": {"AICUP_METRIC": "cache/feature_metric_trans/models.pt",
                     "AICUP_PLAYER_FEATS": "cache/features.npy,cache/features_opp.npy,cache/features_shape.npy,cache/features_shape_opp.npy,cache/features_shapetrans.npy,cache/features_shapetrans_opp.npy"},
        "json": "artifacts/player_trans.json",
    },
    {
        "name": "seq_quality",
        "device": "mps",
        "mem_gb": 12.0,
        "why": "**補上序列模型看不到的最強訊號**。features.py 裡相關係數最高的是"
               "territory_rate（−0.959）與 invade_rate（+0.953），"
               "而序列模型完全看不到它們；愚形（−0.936）已經靠棋形補上、"
               "提子（+0.654）本來就有。再加上無法從既有 24 維推導的"
               "打吃、連接、減氣與半徑 3 的厚薄——8 維／視角。"
               "這是「統計摘要丟掉了什麼時候發生」的同一個論證："
               "在自己地裡補棋，發生在布局是厚實、發生在收官是浪費",
        "train": ["python3", "scripts/train_rank_seq.py",
                  "--extra-sequences", "cache/sequences_opp.npy",
                  "cache/seqshape.npy", "cache/seqshape_opp.npy",
                  "cache/seqqual.npy", "cache/seqqual_opp.npy",
                  "--seeds", "3", "--steps", "12000", "--train-games", "400000",
                  "--out", "cache/rank_seq_qual"],
        "produces": "cache/rank_seq_qual/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8", "--splits", "5",
                 "--json-out", "artifacts/blend_seqqual.json"],
        "eval_env": {"AICUP_SEQ_EXTRA": "cache/rank_seq_qual/models.pt",
                     "AICUP_SEQ_EXTRA_STORES": "cache/sequences.npy,cache/sequences_opp.npy,cache/seqshape.npy,cache/seqshape_opp.npy,cache/seqqual.npy,cache/seqqual_opp.npy"},
        "json": "artifacts/blend_seqqual.json",
    },
    {
        "name": "seq_gru",
        "device": "cpu",
        "mem_gb": 5.0,
        "why": "**換 encoder 類別**：雙向 GRU 取代 dilated CNN。目前三個集成成員"
               "都是同一個 CNN，只差寬度與資料量，分歧其實很小卻已值 +0.0079。"
               "遞迴狀態是真正不同的歸納偏置。"
               "實測 GRU 在 CPU 上 246 ms/step、MPS 上 343 ms——與卷積**相反**，"
               "所以它能跟 MPS 上的 CNN 真正並行",
        "train": ["python3", "scripts/train_rank_seq.py",
                  "--extra-sequences", "cache/sequences_opp.npy",
                  "--arch", "gru", "--seeds", "3", "--steps", "12000",
                  "--train-games", "400000", "--device", "cpu",
                  "--out", "cache/rank_seq_gru"],
        "produces": "cache/rank_seq_gru/models.pt",
        "eval": ["python3", "scripts/eval_blend.py", "--seeds", "8", "--splits", "5",
                 "--json-out", "artifacts/blend_seqgru.json"],
        "eval_env": {"AICUP_SEQ_EXTRA": "cache/rank_seq_gru/models.pt"},
        "json": "artifacts/blend_seqgru.json",
    },
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_state() -> dict:
    if os.path.exists(STATE):
        with open(STATE) as f:
            st = json.load(f)
        if st.get("gate_version") != GATE_VERSION:
            log(f"gate 版本 {st.get('gate_version')} → {GATE_VERSION}，"
                f"舊的生產分數 {st.get('production')} 是用**舊的尺**量的，已作廢，"
                f"將重新量一次基準")
            st["production"] = {}
            st["gate_version"] = GATE_VERSION
        if not isinstance(st.get("production"), dict):
            st["production"] = {}          # 舊格式是單一浮點數
        return st
    return {"production": {}, "done": {}, "gate_version": GATE_VERSION}


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


def record(exp: dict, score: float, prod: float, verdict: str, detail: str,
           train_mins: float = 0.0, gate_mins: float = 0.0) -> None:
    """把結果寫進 RESULTS.md 的無效方法表或主線表——**成功失敗都要記**。

    時間也要記：「這個增益值不值得」是要把成本算進去才答得出來的問題，
    而事後從散落的 log 裡撈時間非常痛苦。
    """
    cost = f"訓練 {train_mins:.0f} 分＋gate {gate_mins:.0f} 分"
    line = (f"| {datetime.now():%m-%d} | **{exp['name']}** | {score:.4f} | {cost} | "
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
    block = ("\n### 自動迴圈的實驗紀錄\n\n"
             "| 日期 | 實驗 | 分數 | 耗時 | 結論 |\n|---|---|---:|---|---|\n")
    if "### 自動迴圈的實驗紀錄" not in s:
        s = s.replace(anchor, block + line + "\n---\n\n" + anchor)
    else:
        i = s.index("### 自動迴圈的實驗紀錄")
        hdr = "|---|---|---:|---|---|" if "|---|---|---:|---|---|" in s[i:i + 400] \
            else "|---|---|---:|---|"
        j = s.index("\n\n", s.index(hdr, i))
        s = s[:j + 1] + line + s[j + 1:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def git_commit(msg: str) -> None:
    """（已停用）迴圈不再自動 commit。

    2026-09-22 起 `docs/`（實驗帳本）與 `artifacts/`（迴圈日誌、分數 JSON）
    都在 `.gitignore` 裡——它們記的是什麼有效、什麼無效，屬於競賽策略，
    不進公開 repo。既然兩者都不再被追蹤，自動 commit 就沒有東西可 commit 了。
    程式碼的改動由人自己 commit。
    """


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
             and e["name"] not in st["done"]
             and not e.get("skip")]        # skip=True 的留在檔案裡只為了記錄原因

    if args.dry_run:
        log(f"佇列 {len(queue)} 項，門檻 {args.floor:+.3f}，"
            f"目前生產分數 {st['production']}")
        for e in queue:
            log(f"  - {e['name']}：{e['why']}")
        return 0

    # 只為佇列裡真的用得到的軌道建立基準（每個都要跑好幾分鐘，不要多算）。
    # ⚠️ 判斷要用 `.get(tr) is None`，不是 `st["production"] is None`——
    # 分軌之後 production 是 dict，空 dict 不是 None，
    # 寫錯的話基準量測會被**靜默跳過**，然後在 gate 時 KeyError。
    for tr in sorted({e.get("track", "rank") for e in queue}):
        if st["production"].get(tr) is not None:
            continue
        spec = TRACKS[tr]
        log(f"[{spec['name']}] 尚無生產分數，先跑一次目前的預設組合建立基準…")
        rc, out = run(spec["eval"], env_extra=spec.get("env"))
        if rc != 0:
            log(f"基準評估失敗：\n{out[-1500:]}")
            return 1
        with open(os.path.join(ROOT, spec["json"])) as f:
            st["production"][tr] = json.load(f)["best_score"]
        save_state(st)
        log(f"[{spec['name']}] 生產分數 = {st['production'][tr]:.4f}")

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
                # 失敗也要記時間——「這個方向試了多久才放棄」同樣是成本
                st["done"][exp["name"]] = {"status": "train_failed",
                                           "train_minutes": round(mins, 1)}
                save_state(st)
            else:
                log(f"[{exp['name']}] 訓練完成（{mins:.1f} 分）→ 排入 gate")
                to_gate.append((exp, mins))

        # 3) Gate 一律**序列**執行。
        #    它要跑 8 組題目的推論，同樣吃 MPS；與訓練搶資源會讓兩邊都變慢，
        #    而且 gate 很快（有快取時幾秒），沒有平行的必要。
        if to_gate and (not running or len(running) < args.jobs or not pending):
            exp, train_mins = to_gate.pop(0)
            log(f"[{exp['name']}] 用 8 組 player-disjoint 協定 gate…")
            _g0 = time.time()
            rc, out = run(exp["eval"], env_extra=exp.get("eval_env"))
            gate_mins = (time.time() - _g0) / 60
            if rc != 0:
                log(f"[{exp['name']}] 評估失敗：\n{out[-1200:]}")
                st["done"][exp["name"]] = {"status": "eval_failed",
                                           "train_minutes": round(train_mins, 1),
                                           "gate_minutes": round(gate_mins, 1)}
                save_state(st); continue
            with open(os.path.join(ROOT, exp["json"])) as f:
                res = json.load(f)
            tr = exp.get("track", "rank")
            score, prod = res["best_score"], st["production"][tr]
            gain = score - prod
            won = gain >= args.floor
            verdict = "採用" if won else ("增益低於門檻" if gain > 0 else "無增益")
            log(f"[{exp['name']}] 分數 {score:.4f}（現況 {prod:.4f}，"
                f"差 {gain:+.4f}）→ {verdict}")
            detail = (f"建議把 `predict.py` 的 `--rank-seq` 預設改成 "
                      f"`{exp.get('eval_env', {}).get('AICUP_SEQ', '?')}`"
                      if won else "維持現況")
            record(exp, score, prod, verdict, detail, train_mins, gate_mins)
            st["done"][exp["name"]] = {"score": score, "gain": gain, "won": won,
                                       "train_minutes": round(train_mins, 1),
                                       "gate_minutes": round(gate_mins, 1)}
            try:
                from aicup.experiment_log import log_experiment
                log_experiment(tr, exp["name"],
                               {"train": exp.get("train"),
                                "eval_env": exp.get("eval_env")},
                               {"score": score, "production": prod, "gain": gain,
                                "won": won, "gate_minutes": round(gate_mins, 2)},
                               note=exp["why"][:200],
                               minutes=train_mins + gate_mins)
            except Exception as e:                       # noqa: BLE001
                log(f"[{exp['name']}] 寫 experiments.csv 失敗（不影響結果）：{e}")
            if won:
                st["production"][tr] = score
            save_state(st)
            git_commit(f"exp: {exp['name']} scores {score:.4f} ({verdict})")
            continue

        if running or pending or to_gate:
            time.sleep(20)

    log("=" * 62)
    log("佇列跑完。")
    for tr, v in sorted(st["production"].items()):
        b = TRACKS[tr]["baseline"]
        log(f"  {TRACKS[tr]['name']}：{v:.4f}（官方 baseline {b}，{v / b:.2f}×）")
    won = [k for k, v in st["done"].items() if v.get("won")]
    log(f"通過 gate 的實驗：{won or '無'}")
    if won:
        log("⚠️ 這支不會自己改 predict.py 的預設——請看 RESULTS.md 的建議後手動確認")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
