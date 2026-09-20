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

| 檔案 | 產生方式 | 大約耗時 |
|---|---|---|
| `cache/features.npy`（2.2 GB） | `python3 scripts/build_feature_store.py` | 約 10 分 |
| `cache/rank_mlp_big/models.pt` | `python3 scripts/train_rank_mlp.py --train-games 0 --seeds 5 --out cache/rank_mlp_big` | 約 1.5 小時（MPS） |
| `cache/rank_question/models.pt` | `python3 scripts/train_rank_question.py --seeds 3` | 約 20 分 |
| `cache/feature_metric/models.pt` | `python3 scripts/train_feature_metric.py --seeds 3` | 約 30 分 |

四個都在就可以直接跑 `predict.py`。缺 `cache/features.npy` 時
其他訓練腳本都跑不了，先建它。

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
