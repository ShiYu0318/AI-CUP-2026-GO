# 在 Colab / Kaggle 上訓練（沒有本機 GPU 時）

## 先確認一件事：你真的需要 GPU 嗎？

| 子題 | 目前分數 | 需要 GPU 嗎 |
|---|---:|---|
| 競賽一 玩家辨識 | **0.6726**（官方 baseline 的 2.36 倍） | **不需要**。手工特徵純 CPU 就大幅領先，深度 embedding 是錦上添花 |
| 競賽二 棋力預測 | 0.3973（官方 baseline 0.4631） | **需要**。手工特徵已在 0.38~0.40 觸頂，必須靠深度模型 |

所以 GPU 只為競賽二而租。競賽一可以完全在筆電上做完。

## 資料管線不是瓶頸（實測）

擔心 Colab 免費版只有 2 個 CPU 核心撐不起 dataloader？實測數字：

```
盤面平面抽取（34 層，2 snapshots）：單核 1,202 局/秒
一個訓練步 = 64 局 -> 0.05 秒 CPU
2 個核心 -> 約 37 步/秒 的供給量
```

而 T4 消化一個步（256 個 19×19×34 樣本）大約要 0.1~0.2 秒。
**GPU 才是瓶頸，CPU 綽綽有餘。** 不需要預先把平面存到磁碟
（那要 24.5 GB / 100 萬局 × 2 snapshots，bit-pack 後仍要 3 GB，不划算）。

## 平台選擇

| | Colab 免費 | Colab Pro | Kaggle Notebooks |
|---|---|---|---|
| GPU | T4 16 GB（看運氣） | T4 / L4，偶爾 A100 | P100 或 T4×2 |
| CPU | 2 核 | 2–8 核 | **4 核** |
| 單次時長 | 常在 4–6 小時被斷 | 較長，可背景執行 | **最長 12 小時** |
| 配額 | 浮動、不保證 | 依方案 | **每週 30 GPU 小時**，明確 |
| 磁碟 | ~100 GB（暫存） | 同左 | 20 GB + 可掛載 Dataset |

**建議先用 Kaggle Notebooks**：CPU 多一倍、時長明確、每週 30 小時的配額對這個競賽綽綽有餘。
Colab 免費版當備援。

## Colab 操作步驟

開新 notebook，執行階段 → 變更執行階段類型 → **T4 GPU**，然後：

```python
# 1) 取得程式碼
!git clone https://github.com/ShiYu0318/AI-CUP-2026-GO.git
%cd AI-CUP-2026-GO
!pip install -q webdataset sgfmill   # torch/numpy/pandas/sklearn Colab 已內建

# 2) 直接在雲端下載資料集（214 MB，不要從本機上傳）
!wget -q https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases/download/v1.0/aicup2026_go_training.tar.xz
!echo "959ebc602b9638d665cb063a6269e0d5aef73956b77913e725763eb335ced94b  aicup2026_go_training.tar.xz" | sha256sum -c
!tar xf aicup2026_go_training.tar.xz     # 解出 training/，約 1.3 GB

# 3) 建手工特徵庫（Colab 2 核約 45 分鐘；只有要跑 baseline 或輔助 head 才需要）
!python scripts/build_feature_store.py --nproc 2

# 4) 確認測試全過
!python -m pytest -q
```

### 訓練

```python
# 把 checkpoint 存到 Google Drive，session 斷了才不會全部白跑
from google.colab import drive; drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/aicup2026/ckpt'

!python scripts/train.py --device cuda \
    --width 128 --blocks 10 --embed-dim 512 --snapshots 4 \
    --p 32 --k 8 --steps 20000 \
    --eval-every 1000 --eval-questions 800 \
    --save-every 500 --workers 2 \
    --balance-losses --quality-weight 0.3 \
    --out $OUT
```

**斷線後接回去**：完全一樣的指令，最後加 `--resume`。

```python
!python scripts/train.py ... --steps 20000 --resume --out $OUT
```

`--resume` 會還原模型、ArcFace head、optimizer、scheduler 與最佳分數，
從上次的步數接著跑。`--save-every 500` 表示最多損失 500 步。

### Kaggle Notebooks 的差異

- 資料集：可以先把 `aicup2026_go_training.tar.xz` 上傳成一個 Kaggle Dataset，
  之後每次 notebook 掛載即可，省下每次重新下載的時間
- checkpoint：寫到 `/kaggle/working/`（session 結束會保留為 output），
  或存成 Dataset 版本以便跨 session 續訓
- `--workers 4`（CPU 核心多一倍）

## 規模與時間估計

模型 `--width 128 --blocks 10` 約 550 萬參數，輸入 34×19×19。

| 設定 | 一步的樣本數 | T4 估計 | 20,000 步估計 |
|---|---:|---:|---:|
| `--p 16 --k 4 --snapshots 2` | 128 | ~0.08 秒 | 約 27 分鐘 |
| `--p 32 --k 8 --snapshots 4` | 1024 | ~0.5 秒 | 約 2.8 小時 |

20,000 步 × 64 局（p=16,k=4）≈ 128 萬個樣本 ≈ 訓練集的 1.6 個 epoch。
**建議先用小設定跑通 + 看曲線，確認分數會往上再開大。**

⚠️ 這些是依本機 MPS 實測外推的估計值，不是在 T4 上量到的。

## 判斷有沒有成功的標準

不要看 loss，看 **題目層級的官方分數**（訓練腳本每 `--eval-every` 步會印）。

| 目標 | 分數 |
|---|---:|
| 手工特徵地板（必須先贏過這個） | 0.3973 |
| 官方 CNN baseline | 0.4631 |

**若深度模型贏不過 0.3973，是實作或訓練量的問題，不是資料難。**
本機 MPS 上跑 3,000 步的小模型只到 0.2548，那是訓練量嚴重不足，不代表方法錯。

## 產生提交檔

訓練完把 `best.pt` 載回本機，或直接在雲端跑：

```python
!python scripts/predict.py --test-dir tests --out-dir submissions
```

競賽一的手工特徵版本不需要 GPU，本機跑 90 秒就能產生完整的 800 列提交檔。
