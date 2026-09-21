# 比賽當天操作手冊（2026/11/04）

測試集釋出當天**不該做任何臨時決定**。所有選擇都已在 player-disjoint 的
離線驗證上做完並寫死，當天只要換掉 `--test-dir` 就好。

## 〇、官方時程與規則（2026-09-20 由官方頁面確認）

| 事項 | 時間 |
|---|---|
| 測試集開放下載、比賽開始 | **2026/11/04（三）11:00** |
| 上傳截止 | **2026/11/24（二）23:59:59** |
| Private Leaderboard 公布 | 2026/11/25（三）16:00 |
| 優勝隊伍交報告與程式碼 | 2026/11/26 – 2026/12/06 |
| 公布最終名次 | 2027/01/04 11:00 |

**三條會影響策略的規則：**

1. **每日上傳上限 5 次**，共 21 天 → 最多 105 次。
2. **最終分數取「最後一次上傳」**，不是最佳的那次。
   → 最後一次上傳必須是**離線驗證最強**的版本，不是 Public LB 最高的那版。
3. **Public Score 同分時以繳交時間排序，先上傳者在前。**
   → 沒有理由把上傳拖到最後一刻。

### 測試集的確切規格

| | 預測玩家棋力 | 辨識玩家 |
|---|---|---|
| Public | 查詢棋譜集 **5/10/15/20 盤各 100 組**，共 400 組 | 同左，**候選玩家集 400 人各 100 譜** |
| Private | 同上 | 同上 |

- **查詢棋譜集 400 組是 400 位不同玩家**
- **候選玩家集 400 個，包括查詢棋譜集的玩家** → 競賽一是**一一對應**
- 級位對應：**A: 1-3K、B: 4-6K、C: 7-9K、D: 10-12K**（D 最弱）

⚠️ 局數是 **5/10/15/20 四個離散值各 100 組**，不是 5~20 隨機。
`build_rank_questions` 與 `build_player_pool` 的預設已改為這個分布
（`sizes="official"`）；用錯的分布校準切點會讓驗證分數偏樂觀。

## 一、三行完成

```bash
# 1. 解壓官方測試集到 tests/（下面假設路徑是 tests/）
# 2. 產生兩份提交檔
python3 scripts/predict.py --test-dir tests --out-dir .
# 3. 上傳 submission_rank.csv 與 submission_player.csv
```

`predict.py` 會自己做完整的格式驗證（800 列、Top-5 相異、無跨池汙染），
驗證不過會直接拋錯，不會產生一份看起來正常但會被判無效的檔案。

## 二、事前必須就緒的東西

⚠️ 這張表**必須與 `scripts/predict.py` 的預設值一致**。
舊版列的是 `rank_mlp_big` / `rank_question`，但實際上線的是 `_opp` 版本——
照舊表在新機器上準備會缺檔案，而且缺的時候 `predict.py` 是**靜默降級**，
不是報錯。改動 `predict.py` 的預設時請一起改這裡。

### 資料層（先建，其他全部依賴它）

| 產出 | 指令 | 耗時 |
|---|---|---|
| `cache/features.npy`（2.2 GB） | `python3 scripts/build_feature_store.py --nproc 10` | 約 10 分 |
| `cache/features_opp.npy` | `python3 scripts/build_feature_store.py --nproc 10 --opponent` | 約 10 分 |
| `cache/sequences.npy`（3.8 GB） | `python3 scripts/build_sequence_store.py --nproc 10` | 約 8 分 |
| `cache/sequences_opp.npy`（3.8 GB） | `python3 scripts/build_sequence_store.py --nproc 10 --opponent` | 約 8 分 |
| `cache/seqshape.npy`（3.8 GB） | `python3 scripts/build_sequence_store.py --nproc 10 --shape` | 約 8 分 |
| `cache/seqshape_opp.npy`（3.8 GB） | `python3 scripts/build_sequence_store.py --nproc 10 --shape --opponent` | 約 8 分 |
| `cache/features_shape*.npy`（各 192 MB） | `python3 scripts/build_shape_features.py [--opponent]` | 各 20 秒 |

### 模型層（`predict.py` 的預設就是這些）

| 產出 | 指令 | 耗時 |
|---|---|---|
| `cache/rank_mlp_opp_big/models.pt` | `python3 scripts/train_rank_mlp.py --extra-features cache/features_opp.npy --train-games 0 --seeds 5 --out cache/rank_mlp_opp_big` | 約 1.5 小時（CPU 較快） |
| `cache/rank_question_opp/models.pt` | `python3 scripts/train_rank_question.py --extra-features cache/features_opp.npy --seeds 3 --out cache/rank_question_opp` | 約 20 分 |
| `cache/rank_seq_opp/models.pt` | `python3 scripts/train_rank_seq.py --extra-sequences cache/sequences_opp.npy --seeds 3` + `--out cache/rank_seq_opp` | 約 1.5 小時（MPS） |
| `cache/rank_seq_opp_full/models.pt` | 同上，`--train-games 0 --out cache/rank_seq_opp_full` | 約 2.5 小時 |
| `cache/rank_seq_w192/models.pt` | 同上，`--width 192 --out cache/rank_seq_w192` | 約 2 小時 |
| `cache/feature_metric/models.pt` | `python3 scripts/train_feature_metric.py --seeds 3` | 約 30 分 |

三個序列變體**都要**——它們是三個獨立的 stacker 成分，
少一個 stacker 的特徵維度就對不上（`Stacker.check_parts` 會擋下來並報錯）。

驗證全部就緒：`python3 scripts/predict.py`（會印出離線分數）。

## 二之二、租到 GPU 的話按這個順序做

本機是 Apple Silicon，**CUDA 永遠不可用**；`pick_device()` 的順序是
cuda → mps → cpu，所以同一份程式在 NVIDIA 機器上不用改就會走 CUDA。

```bash
# 0) 先把資料層建起來（純 CPU，與 GPU 無關，可與下面並行）
python3 scripts/build_feature_store.py   --nproc $(nproc)
python3 scripts/build_feature_store.py   --nproc $(nproc) --opponent
python3 scripts/build_sequence_store.py  --nproc $(nproc)
python3 scripts/build_sequence_store.py  --nproc $(nproc) --opponent
python3 scripts/build_sequence_store.py  --nproc $(nproc) --shape
python3 scripts/build_sequence_store.py  --nproc $(nproc) --shape --opponent
python3 scripts/build_shape_features.py
python3 scripts/build_shape_features.py --opponent

# 1) 板面 CNN —— GPU 上唯一值得排的大工作
#    這是目前**唯一**測到誤差與單局模型負相關的成分（−0.14 ~ −0.22），
#    比任何已知有效的成分都互補，但在 MPS 上跑不動。
#    單張 A100/4090 約 6~10 小時；有 --resume 與 --max-hours，被砍掉可接著跑。
python3 scripts/train.py --width 128 --blocks 10 --embed-dim 512 \
    --pool set_transformer --steps 60000 --p 16 --k 4 \
    --quality-weight 0.3 --balance-losses \
    --eval-questions 800 --save-every 1000 --max-hours 10 \
    --out cache/cnn_gpu
```

**先不要在 GPU 上重跑序列模型與 MLP。** 實測它們已經飽和：
加 seed +0.0010、加資料 +0.0041、加寬 +0.0049、換成 GRU −0.0001，
全部低於門檻（RESULTS.md §2.19）。GPU 的價值在**板面 CNN 這個新資訊源**，
不在把既有模型訓練得更久。

## 三、當天最可能出錯的三件事

### 1. 欄位寫法與訓練資料不同

執色欄位現在是**嚴格解析**的：`B`/`b`/`black`/`W`/`w`/`white` 都接受，
其餘一律拋 `ValueError`。這是故意的——早期版本把任何不等於 `"B"` 的值
都當成白棋，所有黑棋局會用錯誤的視角抽特徵，不拋例外、只是分數安靜變差。

**若當天拋這個錯**：看清楚官方用的是什麼寫法，加進
`src/aicup/features.py` 的 `_BLACK_TOKENS` / `_WHITE_TOKENS`，不要改成預設值。

### 2. 每題的棋局數超過 20

`scripts/predict.py` 的 `MAX_GAMES = 20` 是照官方規格寫死的。
若實際更多，改這個常數即可（特徵抽取本身沒有上限）。

### 3. 跑太久

競賽一要對 2 × 40,000 局候選庫抽特徵。實測 M 系列晶片上
每份約 25 秒（12 個 worker，fork）。競賽二每份 400 題約 10 秒。
整條路徑約 2 分鐘。**若遠超過這個數字，先確認 `parallel_batch_features`
真的在多進程跑**（macOS 上一定要用 `fork`，`spawn` 會慢到不可用）。

## 四、離線彩排

```bash
python3 scripts/make_mock_testset.py                 # 產生模擬測試集
python3 scripts/predict.py --test-dir mock_tests     # 會印出離線分數
```

模擬測試集用的是 held-out 玩家，所以離線分數有參考價值。
目前的端到端分數：競賽一 0.9172、競賽二 0.4720（官方 baseline 0.2845 / 0.4631）。

⚠️ **這一步不能省。** 內部驗證與端到端是兩條獨立的路徑，
曾經差到 0.4895 vs 0.3321——原因是推論時 dropout 沒關，
而所有「資料層」的檢查（特徵逐位元相同、分布吻合）全都通過。

## 五、不要在當天做的事

- **不要看 Public LB 調任何東西。** Public 與 Private 各 400 題，
  用 Public 調參等於用 400 題的雜訊做決定（標準誤約 0.025）。
- **不要臨時加超參數搜尋。** 實測過：多一個「在校準集上搜尋」的自由度
  （證據合併的 alpha），比直接寫死還差 0.0048，即使最佳值就在候選清單裡。
- **不要臨時換模型。** 目前的組合是在 8 組 player-disjoint 切分上選定的。
