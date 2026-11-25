# AI CUP 2026 秋季賽：圍棋棋力預測與玩家辨識

從人類圍棋棋譜中做兩件事：

- **棋力預測**：給一位玩家的 5～20 盤棋，判斷他是 10 個等級（D、C、B、A、1D～6D）中的哪一級。
- **玩家辨識**：給一位玩家的 1～20 盤棋，從候選池中找出他是誰；候選人全是訓練時沒見過的新面孔。

競賽頁面：[競賽一：從圍棋棋譜集辨識玩家](https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688)、
[競賽二：從圍棋棋譜預測玩家棋力等級](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)。

## 方法一覽

**棋力預測**：七個成分各自輸出 10 個等級的後驗，由學習式 stacker 融合，再用動態規劃求出的全域最佳切點解碼。

| 成分 | 輸入 |
|---|---|
| 單局 MLP、題目層級 MLP | 整盤手工特徵 589 維，**我方與對手兩個視角** |
| 逐手序列 1D-CNN ×2 | 每手的局部描述、棋形、局面品質（160 手） |
| 板面 CNN | 34 層盤面快照，Set Transformer 集合池化 |
| KataGo HumanSL ×2 | 5 個棋力 profile 下，**我方**與**對手**每一手的人類走法機率 |

**玩家辨識**：學一個「同一人拉近、不同人推開」的 embedding，推論時丟掉分類頭。

| 題型 | 做法 |
|---|---|
| 候選 100 盤、題目 5～20 盤、一對一 | 手工特徵 → 度量學習 MLP（Sub-center ArcFace）→ 候選池分類器 → Sinkhorn 全域指派 |
| 候選 10 盤、題目 1～5 盤、非一對一 | 7 個逐手序列身分模型（含**落子座標＋局部盤面**）＋度量分類器，z 分數融合，加上同色模板與讓先習慣的似然 |

詳細設計與實驗見 [docs/](docs/README.md)。

## 幾個關鍵發現

1. **驗證要以「玩家」切分。** 官方 tutorial 打亂棋局後切驗證集，驗證玩家 100% 也在訓練集裡，分數系統性高估。
   本專案所有數字都是 player-disjoint、題目層級、照官方盤數分布出題。
2. **對手是比自己更好的棋力證據。** 伺服器依等級配對，對手的表現反映你的評分；只用對手的特徵甚至比只用自己的準。
3. **在自己資料上訓練的模型，都卡在玩家層級的同一個偏差。** 單盤更強的模型在題目層級增益趨近 0；
   能突破的是外部知識（KataGo HumanSL），而它評「對手」時的增益和評自己一樣大。
4. **最強的身分訊號在開局與「下在哪裡」。** 開局順序是最強的手工特徵；逐手模型加上落子座標與局部棋形後才真正變強。
5. **解碼要對齊評分。** 相鄰等級有部分分，切點用動態規劃求全域最佳解；數值最佳化在分段常數的目標上只是隨機遊走。
6. **全域約束要先量「假設錯了會怎樣」。** Sinkhorn 軟性指派在一對一成立時與匈牙利算法相當，不成立時也不會變差。

## 快速開始

```bash
pip install -e ".[torch,dev]"
python3 -m pytest -q                       # 約 480 個測試

# 資料請自行從競賽頁面與官方 aicup2026-go-dataset 下載，放到 data/（結構見 docs/05）
python3 scripts/build_feature_store.py --nproc 10
python3 scripts/run_baselines.py           # player-disjoint、題目層級、官方公式的基準分數
```

完整的建置、訓練與預測指令見 [docs/05 工程與重現](docs/05-工程與重現.md)。

## 專案結構

```
src/aicup/                 可匯入的函式庫（沒有 CLI）
├── scoring.py             官方評分、期望分數解碼、溫度校準、DP 切點
├── goboard.py             SGF 解析與棋盤規則（提子、氣），不依賴外部圍棋套件
├── data.py、benchmark.py  metadata、player-disjoint 切分、照官方協定出題
├── features.py            整盤手工特徵（行為、死活、局面品質、位置分布、開局順序）
├── moveseq.py、moveshape.py、movequality.py、movepatch.py
│                          逐手特徵：局部描述、棋形、局面品質、落子座標＋局部盤面
├── planes.py、nets.py     板面 CNN 的盤面表徵與網路（GeM＋BN-Neck＋Set Transformer）
├── rank_nets.py           棋力的單局 MLP 與逐手序列模型
├── feature_metric.py      手工特徵上的度量學習
├── humansl.py             KataGo HumanSL 查詢、特徵、棋池校準
├── se_strength.py         Strength Estimator（ICLR 2025）預訓練模型的強度特徵
├── stacker.py             學習式融合
├── rerank.py              Sinkhorn 全域指派、候選池分類器
├── losses.py              指數軟標籤、CORN、成對排序、Sub-center ArcFace
├── oofcache.py            帶內容指紋的題目層級後驗快取
└── submission.py          輸出檔產生與官方格式驗證

scripts/                   可執行的流程，每支做一件事
├── build_*.py             特徵庫、序列庫、SGF 打包、HumanSL／SE 特徵、重現紀錄
├── train_*.py、train.py   各模型的訓練
├── eval_*.py              融合、集成、玩家辨識的評估
├── predict.py             棋力預測（＋一對一題型的玩家辨識）
├── predict_player_v2.py   少樣本題型的玩家辨識
└── auto_loop.py           無人值守的實驗迴圈：跑 → gate → 記錄 → commit

tests/                     約 480 個測試：評分邊界、提子不變量、切分互斥、張量尺度、格式規則
repro/                     重現性紀錄：依賴版本、模型與資料雜湊、訓練參數、每份輸出的配方
docs/                      設計文件與完整實驗日誌
```

## 外部資源

| 資源 | 用途 | 授權 |
|---|---|---|
| [KataGo](https://github.com/lightvector/KataGo) v1.18.2 | 分析引擎 | MIT |
| KataGo HumanSL 模型 `b18c384nbt-humanv0` | 棋力條件的人類走法機率 | KataGo network license |
| [Strength Estimator](https://github.com/rlglab/strength-estimator) `se_go.pt` | 實驗用（未上線）；`aicup.se_strength` 依其公開的輸入規格自行實作，執行時載入使用者自行下載的權重 | GPL-3.0（權重與程式皆不隨本 repo 散布） |

沒有使用競賽資料以外的棋譜。

## 致謝與授權

競賽由國立東華大學智慧科技中心主辦；官方 tutorial 與資料集：
[AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)、
[aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset)。
競賽資料不包含在本 repo 中，請依主辦單位的規定取得與使用。

本專案的程式碼以 [MIT License](LICENSE) 授權。
