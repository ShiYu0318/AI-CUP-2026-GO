# AI CUP 2026 秋季賽－圍棋棋力與風格預測競賽

- [競賽一：從圍棋棋譜集辨識玩家](https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688)
- [競賽二：從圍棋棋譜預測玩家棋力等級](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403?focus=intro)

兩個子題共用同一份訓練集：100 萬局匿名人類圍棋棋譜，10 個棋力等級各 10 萬局。

| | 競賽一 | 競賽二 |
| --- | --- | --- |
| 任務 | 開放集合檢索（從 400 人候選庫找出是誰，交 Top-5） | 10 類有序分類（D/C/B/A/1D–6D） |
| 評分 | `exp(-(r-1))`，r 為正解名次 | 完全正確 1、相鄰一級 `exp(-1)`、其餘 0 |
| 官方 baseline | 0.2845 | 0.4631 |
| **本專案（離線驗證）** | **0.9745**（3.43×） | **0.6158 ±0.0039**（1.33×） |

分數是用**完整複製官方協定**的離線基準量的：player-disjoint 切分、題目層級、
官方局數分布（5/10/15/20 各 25%）、官方評分公式。競賽二在 8~16 組獨立題目
抽樣上取平均，競賽一涵蓋 5 種候選庫等級組成。

> 📁 **實驗紀錄、比賽當天手冊、雲端訓練說明保留在本機 `docs/`，不在這個 repo 裡。**
> 程式碼註解裡出現的 `RESULTS.md §X` / `docs/02 §X` 指的就是那些文件。
> 這個 repo 公開的是**方法與實作**；哪些方向試過、各值多少分，屬於競賽過程紀錄。

## 設計原則

所有程式都圍繞三條原則，而三條都來自對官方 tutorial baseline 的實測缺陷：

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
# 0. 安裝（editable，之後 `import aicup` 從任何目錄都能用）
pip install -e ".[torch,dev]"

# 1. 下載官方訓練集（214 MB，解壓後約 1.3 GB，走官方 Releases）
#    https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases
tar xf aicup2026_go_training.tar.xz        # 解出 training/

# 2. 一次算好全部 100 萬局的手工特徵（10 核約 10 分鐘）
python3 scripts/build_feature_store.py --nproc 10

# 3. 跑 baseline，取得 player-disjoint、題目層級、官方公式的分數
python3 scripts/run_baselines.py

# 4. 產生提交檔（會先做完整格式驗證）
python3 scripts/predict.py --check --test-dir tests   # 先檢查依賴齊不齊
python3 scripts/predict.py --test-dir tests --out-dir .

# 測試
python3 -m pytest tests/ -q
```

## 程式庫結構

```
src/aicup/
├── scoring.py        官方評分函數、期望分數解碼、溫度校準、切點最佳化（DP）
├── goboard.py        SGF 解析 + 棋盤（提子、氣）。不依賴 sgfmill
├── data.py           metadata 快取、player-disjoint 切分
├── benchmark.py      題目層級基準（完整複製官方協定，支援多種候選庫組成）
├── features.py       整局手工特徵（行為／死活／局面品質／位置分布／開局順序）
├── moveseq.py        逐手序列特徵——保留「什麼時候發生」
├── moveshape.py      逐手棋形（手法分類、愚形、厚薄）
├── movequality.py    逐手局面判斷品質（地／侵／廢手／打吃／連接／減氣）
├── planes.py         深度模型的 34 層盤面表徵
├── featstore.py      多份特徵庫沿特徵軸串接，取列時才合併
├── oofcache.py       題目層級後驗快取（內容指紋自我失效）
├── stacker.py        學習式融合：把各成分的後驗組成一個線性模型
├── question_model.py 題目層級聚合特徵
├── feature_metric.py 手工特徵上的度量學習（競賽一）
├── rerank.py         Sinkhorn 全域指派、一一對應一致性
├── dataset.py        PyTorch 資料集、P×K 取樣器、8 相位對稱增強
├── losses.py         指數 soft label、CORN、成對排序、sub-center ArcFace
├── nets.py           板面 CNN：共用 backbone 的多任務網路（GeM + BN-Neck + Set Transformer）
├── rank_nets.py      競賽二的單局 MLP 與逐手序列模型（CNN / GRU）＋題目層級推論
├── board_posterior.py 板面 CNN 的題目層級後驗（校準、推論、評估共用）
├── sgfstore.py       全部 SGF 打包成共用的 memory-mapped 檔
├── experiment_log.py 統一的實驗紀錄（含機器時間）
└── submission.py     提交檔產生與官方格式驗證

scripts/                          # 32 支，以下是主要的
├── build_sgf_store.py            SGF 打包檔（8 秒；讓多程序／多 worker 共用記憶體）
├── build_feature_store.py        整局特徵庫
├── build_sequence_store.py       逐手序列庫（--shape / --quality / --opponent）
├── build_shape_features.py       逐手棋形聚合成整局統計
├── train_rank_mlp.py             競賽二：單局模型
├── train_rank_question.py        競賽二：題目層級模型
├── train_rank_seq.py             競賽二：逐手序列模型（CNN / GRU）
├── train_feature_metric.py       競賽一：度量學習
├── train.py                      板面 CNN（多任務）
├── eval_blend.py                 競賽二的融合評估（吃 OOF 快取）
├── eval_player.py                競賽一的評估（跨候選庫組成取最壞）
├── eval_ensemble.py              集成成分選擇（貪婪前向，最壞情況收斂）
├── auto_loop.py                  無人值守的實驗迴圈：跑 → gate → 記錄 → commit
└── predict.py                    產生兩份提交檔（--check 先驗依賴）
```

## 工程實踐

**分層**：`src/aicup/` 是可匯入的函式庫，定義「什麼是正確的」——評分公式、
棋盤規則、切分協定、模型結構；沒有 CLI，被 22 個測試模組直接匯入。
`scripts/` 是可執行的流程，每支做一件事，31 支裡 28 支有 `__main__`。
套件以 editable 模式安裝（`pip install -e .`），所以腳本直接 `import aicup`，
不需要 `sys.path` 修補。**依賴是單向的**：`src/` 從不 import `scripts/`，而
`tests/test_layering.py` 會擋住這條界線再度漂移（它曾經漂移過一次：
模型定義住在訓練腳本裡，害得產生提交檔的 `predict.py` 依賴訓練腳本）。

其餘的瓶頸不是想法，是**迭代速度**與**不被自己騙**。所以：

- **量測協定寫死在程式裡。** 題目層級、player-disjoint、官方局數分布、
  校準與評分的玩家不重疊——不照這個量的數字不進紀錄。
- **切點用動態規劃求全域最佳解**，不用數值最佳化。官方分數對切點是分段常數的，
  Nelder-Mead 在上面本質是隨機遊走（實測同一份資料重跑會在 0.20~0.48 之間跳）。
- **OOF 後驗快取**（`oofcache.py`）帶三重指紋：checkpoint 內容雜湊、題目集雜湊、
  語意版本。換模型不會拿到舊後驗，而融合實驗從 20 分鐘變成秒級。
- **自駕迴圈**（`auto_loop.py`）：跑實驗 → 用誠實協定 gate → 記錄 → commit → 下一個。
  gate 的「尺」有版本號，換定義時會自動作廢舊基準——文件擋不住這種錯，機制可以。
- **每個實驗都記機器時間**（`experiment_log.py`）。「這個增益值不值得」要把成本
  算進去才答得出來。
- **測試釘住的是「不會報錯的錯」**：評分公式的邊界、棋盤提子的不變量
  （盤上子數 + 被提子數 = 總手數，對真實棋譜驗證）、切分互斥性、
  訓練與推論的張量尺度一致、以及每一條會讓提交作廢的格式規則。

```bash
python3 -m pytest tests/ -q     # 約 400 個測試
```

## 官方資料集規格（摘自 [aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset)）

欄位 `player_id, game_id, rank, color, sgf_content`，10 檔各 100,000 列。

**最重要的一條規格**：

> `rank` 是該玩家**所有對局的眾數等級**，不是單局等級；同一玩家的每一列都相同。

這解釋了為什麼單局 top-1 只有 0.1687 —— 標籤在單局層級本來就帶噪聲，
一位標為 3D 的玩家，個別對局的實際強度會散佈在附近幾級。
**這也是為什麼「5–20 局聚合」才是分數的主體，而不是單局模型精度。**

其他官方保證（已對全部 100 萬局驗證）：

| 官方說法 | 實測 |
|---|---|
| 手數至少 100 | ✅ 最小值恰為 100 |
| 不含自殺、打劫、疊子，主分支可完整重播 | ✅ 提子不變量測試通過 |
| 保留屬性僅 `SZ` 與 `B`/`W`/`AB`/`AW`/`AE` | ⚠️ **實測只出現 `SZ`/`B`/`W`**，`AB`/`AW`/`AE` 零次（無讓子局） |
| 「`sgf_content` 內含逗號，請用標準 CSV 解析器」 | ⚠️ **實測 100 萬列零逗號**；仍建議用標準 parser |

## 提交格式（錯一次就整份作廢）

兩個子題都要把 public 400 題與 private 400 題**串接成一份 800 列**的 CSV，
缺任一份即該次提交無效不計分。競賽一另有兩條死線：
Top-5 必須相異、且 `pub_` 題目只能填 `pub_p_` 開頭的玩家（跨池即無效）。

`aicup.submission` 把這些規則寫成會丟例外的驗證函數，驗證不過就不寫檔。

## 換一台機器（例如租/借 GPU）

```bash
git clone <repo> && cd AI-CUP-2026-GO
pip install -e ".[torch,dev]"
python3 -m pytest -q                    # 先確認環境正常
```

之後照「快速開始」建特徵庫即可。裝置選擇是自動的
（`cuda` → `mps` → `cpu`），同一份程式不用改。

`train.py --resume` 可從斷線處接回（還原模型 / ArcFace head / optimizer / scheduler）。

⚠️ **`--workers` 請留 0。** 每個 DataLoader worker 會各自載入全部 10 份 CSV
（約 3 GB），多開幾個就把機器吃光——而且是**安靜地**停住：程序還活著、
沒有錯誤、日誌毫無異常，只是一步都不動。

## 授權與致謝

官方 tutorial 與資料集由國立東華大學智慧科技中心提供：
[AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)、
[aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset)。
