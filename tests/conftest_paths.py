"""讓測試能 import `scripts/` 底下的訓練腳本。"""
import os
import sys

scripts_on_path = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts")
if scripts_on_path not in sys.path:
    sys.path.insert(0, scripts_on_path)
