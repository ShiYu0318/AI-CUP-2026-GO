# AI CUP 2026 秋季賽－圍棋棋力與風格預測競賽

- [競賽一：從圍棋棋譜集辨識玩家](https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688)
- [競賽二：從圍棋棋譜預測玩家棋力等級](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403?focus=intro)

兩個子題共用同一份訓練集：100 萬局匿名人類圍棋棋譜，10 個棋力等級各 10 萬局。

| | 競賽一 | 競賽二 |
| --- | --- | --- |
| 任務 | 開放集合檢索（從 400 人候選庫找出是誰，交 Top-5） | 10 類有序分類（D/C/B/A/1D–6D） |
| 評分 | `exp(-(r-1))`，r 為正解名次 | 完全正確 1、相鄰一級 `exp(-1)`、其餘 0 |
| 官方 baseline | 0.2845 | 0.4631 |

## 設計原則

這個 repo 的所有程式都圍繞三條原則，而三條都來自對官方 baseline 的實測缺陷：

1. **驗證以「玩家」為切分單位，不是「棋局」。**
   官方 tutorial 把 100 萬局 shuffle 後取前 10 萬局當驗證集，實測驗證集的
   7,463 位玩家有 **100.00%** 同時出現在訓練集裡。真實測試集的玩家是完全未見的，
   所以那個驗證分數系統性高估泛化能力。

2. **指標以「題目」為單位，不是「單局」。**
   官方每題給 5～20 局。實測單局 top-1 只有 15.5%，但題目層級聚合後可達 0.35——
   分數的主體來自聚合，用單局 accuracy 選模型會選錯。

3. **解碼對齊評分函數，不用 argmax。**
   競賽二的相鄰等級有 `exp(-1)` 部分分，`argmax` 完全不知道這件事。
   改用期望分數解碼 `argmax_c Σ_k p_k · S(c,k)` 是三行程式碼的免費增益。

## 快速開始

```bash
# 1. 下載官方訓練集（1.3 GB，走官方 Releases）
#    https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases
tar xf aicup2026_go_training.tar.xz        # 解出 training/

# 2. 一次算好全部 100 萬局的手工特徵（10 核約 9 分鐘，產出 644 MB）
python3 scripts/build_feature_store.py --nproc 10

# 3. 跑 baseline，取得 player-disjoint、題目層級、官方公式的分數
python3 scripts/run_baselines.py

# 4. 多任務深度模型（建議在有 GPU 的機器上）
python3 scripts/train.py --device cuda --width 128 --blocks 10 --steps 20000
python3 scripts/train.py --smoke            # 只驗證流程能跑通

# 測試
python3 -m pytest tests/ -q
```

## 程式庫結構

```
src/aicup/
├── __init__.py      LEVELS 等常數（等級順序決定「相鄰」的定義，很重要）
├── scoring.py       官方評分函數、期望分數解碼、溫度校準、檢索指標
├── goboard.py       SGF 解析 + 棋盤（提子、氣）。不依賴 sgfmill
├── data.py          metadata 快取、player-disjoint 切分
├── benchmark.py     題目層級基準（完整複製官方協定，支援多種候選庫等級組成）
├── features.py      手工特徵 161 維（行為統計 / 死活統計 / 位置分布）
├── planes.py        深度模型的 34 層盤面表徵
├── baselines.py     手工特徵的兩個子題 baseline
├── dataset.py       PyTorch 資料集、P×K 取樣器、8 相位對稱增強
├── losses.py        指數 soft label、CORN、sub-center ArcFace、batch-hard triplet
├── nets.py          共用 backbone 的多任務網路（GeM + BN-Neck + attention pooling）
└── submission.py    提交檔產生與官方格式驗證

scripts/
├── build_feature_store.py   一次算好 100 萬局特徵
├── run_baselines.py         離線評估（67 天沒有 leaderboard 期間的儀表板）
└── train.py                 多任務訓練
```

## 資料的兩個陷阱

**CSV 依執色排序。** 每個 `train_*.csv` 的第 0–49,999 列全是 `B`、
第 50,000–99,999 列全是 `W`，只有一個切換點。
`pd.read_csv(..., nrows=10000)` 會拿到 100% 執黑樣本而毫無警訊。
抽樣前務必先 shuffle，或明確跨越分界。
（回歸測試：`tests/test_data_and_benchmark.py::test_csv_rows_are_sorted_by_color`）

**`game_id` 只在單一等級檔內唯一。** 10 個檔案都各自從 `g_0000001` 起編，
跨檔會重複 10 次。任何以 `game_id` 當鍵的快取都必須加上等級前綴。

## 提交格式（錯一次就整份作廢）

兩個子題都要把 public 400 題與 private 400 題**串接成一份 800 列**的 CSV，
缺任一份即該次提交無效不計分。競賽一另有兩條死線：
Top-5 必須相異、且 `pub_` 題目只能填 `pub_p_` 開頭的玩家（跨池即無效）。

`aicup.submission` 把這些規則寫成會丟例外的驗證函數，驗證不過就不寫檔。

## 測試

```bash
python3 -m pytest tests/ -q
```

測試涵蓋官方評分公式的邊界（`D` 與 `6D` 各只有一個相鄰級）、
棋盤提子的不變量（盤上子數 + 被提子數 = 總手數，對真實棋譜驗證）、
切分的互斥性、基準是否符合官方協定、以及每一條會讓提交作廢的格式規則。

## 授權與致謝

官方 tutorial 與資料集由國立東華大學智慧科技中心提供：
[AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)、
[aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset)。
