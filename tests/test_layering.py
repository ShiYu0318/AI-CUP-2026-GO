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
import sys

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


def test_scripts_do_not_patch_sys_path_for_the_library():
    """套件以 editable 安裝（`pip install -e .`），腳本直接 `import aicup`。

    2026-09-22 之前有 28 支腳本各自寫了
    `sys.path.insert(0, .../src)` 來繞過未安裝的問題。那會讓
    「程式能不能跑」取決於檔案的相對位置，而不是環境設定；
    搬動檔案或從別的地方匯入時就會壞。

    （匯入**同儕腳本**的 `sys.path.insert(scripts_dir)` 不在此限——
    那是腳本之間的事，與函式庫無關。）
    """
    needle, pkg_dir = "sys.path", '"' + "src" + '"'      # 拆開寫，才不會抓到自己
    offenders = []
    for f in sorted(SCRIPTS.glob("*.py")) + sorted((ROOT / "tests").glob("*.py")):
        if f.name == pathlib.Path(__file__).name:
            continue
        for line in f.read_text(encoding="utf-8").split("\n"):
            if needle in line and pkg_dir in line:
                offenders.append(f"{f.name}: {line.strip()}")
    assert not offenders, (
        "這些檔案還在手動把 src 加進 sys.path；改用 `pip install -e .`：\n"
        + "\n".join(offenders))


def test_library_is_importable_as_an_installed_package():
    """裝好之後，從任何工作目錄都該能 `import aicup`。"""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, "-c",
                            "import aicup.scoring; print(aicup.scoring.E_INV)"],
                           cwd=d, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (
        "從別的目錄 import aicup 失敗——套件沒裝？跑 `pip install -e .`\n"
        + (r.stderr or "")[-400:])
