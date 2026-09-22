"""`src/aicup/` 與 `scripts/` 的分層。

  src/aicup/  = 可匯入的**函式庫**——定義「什麼是正確的」
  scripts/    = 可執行的**流程**——編排「做一件事」

這條界線 2026-09-22 之前漂移過一次：單局 MLP 與序列模型的類別住在訓練腳本裡，
而 `scripts/predict.py`（比賽當天產生提交檔的路徑）得 `from train_rank_seq import ...`。
那讓提交路徑依賴兩個**訓練腳本**的 import——只要有人在訓練腳本頂層加一個重的
import 或副作用，提交路徑就跟著受影響。定義已移到 `aicup.rank_nets`。

這支測試守住它，因為漂移是漸進的，不會有人注意到。
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "aicup"
SCRIPTS = ROOT / "scripts"
SCRIPT_MODULES = {p.stem for p in SCRIPTS.glob("*.py")}


def _imported_names(path: pathlib.Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split(".")[0])
    return out


def test_library_never_imports_a_script():
    """**最重要的一條**：依賴必須是單向的。

    `src/` 匯入 `scripts/` 的話，函式庫就不能獨立使用、也不能獨立測試，
    而且會產生「執行腳本才能用函式庫」的循環。
    """
    bad = {}
    for f in sorted(SRC.glob("*.py")):
        hit = _imported_names(f) & SCRIPT_MODULES
        if hit:
            bad[f.name] = sorted(hit)
    assert not bad, f"src/aicup 不可以 import scripts：{bad}"


def test_submission_path_does_not_depend_on_training_scripts():
    """`predict.py` 是比賽當天的路徑，不該依賴訓練腳本。"""
    hit = _imported_names(SCRIPTS / "predict.py") & SCRIPT_MODULES
    trainers = {m for m in hit if m.startswith("train")}
    assert not trainers, f"predict.py 匯入了訓練腳本：{sorted(trainers)}"


def test_model_definitions_live_in_the_library():
    """模型的類別與推論函式要能從 `aicup` 匯入，不必先把 scripts 放進 sys.path。"""
    from aicup.rank_nets import (ARCHS, RankMLP, SeqRankGRU,  # noqa: F401
                                 SeqRankNet, SeqStore, build_seq_net,
                                 mlp_question_posteriors, pick_device,
                                 seq_question_posteriors)
    assert set(ARCHS) == {"cnn", "gru"}


def test_library_modules_have_no_cli():
    """函式庫不該有 argparse——那是 scripts/ 的事。

    唯一的例外是 `experiment_log`，它刻意提供 `python3 -m aicup.experiment_log`
    來看實驗紀錄；那是**讀取**工具，不是流程編排。
    """
    offenders = [f.name for f in sorted(SRC.glob("*.py"))
                 if "argparse" in _imported_names(f) and f.stem != "experiment_log"]
    assert not offenders, f"這些函式庫模組帶了 CLI：{offenders}"
