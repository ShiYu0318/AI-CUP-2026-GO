"""AI CUP 2026 圍棋競賽 —— 共用程式庫。

設計原則（詳見 docs/02-最佳解法方向分析.md）：
  1. 驗證一律以「玩家」為切分單位，不是「棋局」——官方隨機切分有 100% 身分洩漏
  2. 指標一律以「題目」為單位，不是「單局」——分數主體來自多局聚合
  3. 解碼一律對齊官方評分函數，不用 argmax
"""

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
RANK_TO_ID = {r: i for i, r in enumerate(LEVELS)}
ID_TO_RANK = {i: r for i, r in enumerate(LEVELS)}
N_LEVELS = len(LEVELS)

__all__ = ["LEVELS", "RANK_TO_ID", "ID_TO_RANK", "N_LEVELS"]
