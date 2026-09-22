# 早期探索腳本（2026-09-18）

這三支寫在 `src/aicup/` 函式庫與 `scripts/` 結構出現**之前**，
各自重複實作了資料載入與評估，**不依賴函式庫**（只用 numpy/pandas）。
目前**沒有任何程式引用它們**，也不在任何管線上。

## 為什麼留著而不是刪掉

它們是幾個**現在還在生效**的決定的原始證據。最明顯的是
`src/aicup/dataset.py` 今天仍在用的 8 重對稱增強——
那個決定就出自 `generalization_audit.py` 的 E2。

結論已經進 `RESULTS.md`，但**能重跑的證據**只在這裡。

| 腳本 | 當初確立了什麼 | 結論現在在哪 |
|---|---|---|
| `playerid_floor.py` | 競賽一的離線驗證協定：400 人候選庫、每人 100 局、Top-5 官方公式。在沒有 leaderboard 的 67 天裡唯一的儀表板 | 已長成 `src/aicup/benchmark.py` 的 `build_player_pool`、`scripts/eval_player.py` |
| `gbdt_floor.py` | 不用深度學習能到多少分；「期望分數解碼」相對 argmax 的增益 | RESULTS.md §2.3 解碼器比較、§3.2 XGBoost vs MLP |
| `generalization_audit.py` | E1 時序漂移／E2 **8 重對稱折疊**／E3 各等級辨識難度差異／E4 特徵族消融 | E2 → `dataset.py` 的 `_dihedral`（**仍在用**）；E3 → RESULTS.md §1.3；E1/E4 → §1.4 |

## 要重跑的話

它們吃 `training/*.csv`，用位置參數控制每級取樣數，例如：

```bash
python3 scripts/legacy/playerid_floor.py 4000
```

⚠️ 它們的評估協定是**當時**的版本，與現在的不完全一致
（例如題目局數分布在 2026-09-20 才改成官方的 5/10/15/20）。
拿它們的絕對數字與現在的比較會誤導——它們的價值在於**方法論的證據**，不是分數。
