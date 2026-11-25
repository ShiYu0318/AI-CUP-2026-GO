# 重現指南

這個資料夾讓本專案的**每一份輸出**與上線模型都能在另一台機器上重新產生出來。除 `recipes/` 與 `commit_map.tsv` 外都由 `scripts/build_repro.py` 產生，不要手改。

| 檔案 | 內容 |
|---|---|
| `python.txt` | Python 版本與平台 |
| `requirements.txt` | 原始碼實際 import 的第三方套件（AST 掃描 `src/`、`scripts/`），版本釘死 |
| `models.tsv` | 每個上線 checkpoint 與外部模型的 sha256、大小、用途、來源 |
| `train_args/*.json` | 每個 checkpoint **內存**的訓練參數（直接從權重檔取出，不靠記憶或文件） |
| `data_manifest.json` | 訓練集與測試檔的 sha256，以及整體指紋（提交紀錄裡的「資料指紋」就是它的前 16 碼） |
| `commit_map.tsv` | 版控歷史曾被改寫（作者名稱統一、移除不應公開的資料檔）；實驗日誌引用的是舊雜湊，這裡給新舊對照 |
| `recipes/<編號>.json` | 每份輸出的完整指令、commit、模型與輸出雜湊（`scripts/submissions.py add` 自動寫入；r 開頭是棋力、p 開頭是玩家辨識） |

## 一、重建環境

```bash
python3.13 -m venv .venv-repro
.venv-repro/bin/pip install -r repro/requirements.txt
.venv-repro/bin/pip install -e . --no-deps
.venv-repro/bin/python -m pytest -q          # 應全數通過
brew install katago                          # KataGo HumanSL 成分需要（v1.18.2，Metal 後端）
```

## 二、驗證資料

```bash
.venv-repro/bin/python - <<'EOF'
import hashlib, json
m = json.load(open("repro/data_manifest.json"))
bad = [p for p, h in m["files"].items() if hashlib.sha256(open(p, "rb").read()).hexdigest() != h]
print("資料一致 ✓" if not bad else f"有 {len(bad)} 個檔案不同，例如 {bad[:3]}")
EOF
```

訓練集是官方 **v1.0**。v1.1（2026-11-02）只在每盤補上 `HD[n]` 標籤、其餘逐列相同；
所有解析路徑對 `HD[n]` 的輸出都不變，但 HumanSL 儲存檔以 SGF 原文雜湊為鍵，換成 v1.1 會讓鍵全部對不上。

## 三、重產一份輸出

每份輸出的配方在 `repro/recipes/<編號>.json`：

```bash
git checkout <配方裡的 git_commit>
# 1. 核對模型：配方的 models_sha256 應與現有檔案一致
# 2. 照配方的 cmd 重跑（例如）
.venv-repro/bin/python scripts/predict.py --humansl --test-dir data --out-dir /tmp/re
shasum -a 256 /tmp/re/submission_rank.csv    # 應與配方的 output.sha256 相同
```

`git_dirty` 為 `dirty` 的輸出在產生時工作區有未提交的修改，無法單靠 commit 重現。

## 四、重訓模型

每個上線模型的訓練參數在 `train_args/`，訓練指令對照如下（腳本會把參數存進 checkpoint）：

| 模型 | 腳本 |
|---|---|
| `rank_mlp_opp_big` | `scripts/train_rank_mlp.py` |
| `rank_question_opp` | `scripts/train_rank_question.py` |
| `rank_seq_opp_full`、`rank_seq_shape` | `scripts/train_rank_seq.py` |
| `board`（板面 CNN） | `scripts/train.py`（分段 `--resume` 訓練到 160k 步） |
| `feature_metric_h1024s12k` | `scripts/train_feature_metric.py` |
| HumanSL 棋池與測試集特徵 | `scripts/build_humansl_store.py --pool` ／ `--test-dir`（再加 `--opponent` 算對手的落子） |
| `seq_identity*`（玩家辨識的 7 個逐手序列身分模型） | `scripts/train_seq_identity.py`（各模型的特徵庫組合見 `models.tsv` 與 docs/05） |

特徵庫、序列庫、SGF 打包檔的建置順序與完整指令見 [docs/05-工程與重現.md](../docs/05-工程與重現.md)。

## 五、使用的外部資源

| 資源 | 版本／檔名 | 授權 | 用法 | 貢獻（消融） |
|---|---|---|---|---|
| KataGo HumanSL | `b18c384nbt-humanv0.bin.gz`（v1.15.0 release） | KataGo network license | 推論時抽特徵：5 種棋力 profile 下我方與對手每一手的走法機率 | 棋力：我方 +0.0294、對手再 +0.0376（本地 8 seed 配對，RESULTS.md §2.37、§2.38） |
| KataGo 主模型 | `kata1-b18c384nbt-s9996604416-d4316597426.bin.gz`（Homebrew katago 1.18.2 附帶） | KataGo network license | HumanSL 查詢時一併載入（分析引擎必需） | 含在上一列 |
| KataGo 程式 | v1.18.2（Homebrew） | MIT | 分析引擎 | — |

雜湊見 `models.tsv`。沒有使用外部棋譜資料。
