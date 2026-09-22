"""每一支用 argparse 的腳本，`--help` 都要能跑。

看似瑣碎，但這一類 bug 只會在你最需要看說明的時候爆——而且**任何其他測試都抓不到**。

真實踩過：`predict.py` 的 `--crosstask-weight` 說明裡寫了「約 8%——」，
argparse 會對 help 字串做 `%` 格式化，於是 `predict.py --help` 直接
`ValueError: unsupported format character`。字面的 `%` 必須寫成 `%%`。
這個 bug 存在了一天，因為平常沒人跑 `--help`。
"""
import ast
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def _uses_argparse(p: pathlib.Path) -> bool:
    for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import) and any(a.name == "argparse" for a in node.names):
            return True
    return False


ARGPARSE_SCRIPTS = [p for p in SCRIPTS if _uses_argparse(p)]


def test_there_are_scripts_to_check():
    assert len(ARGPARSE_SCRIPTS) > 20, "腳本清單抓錯了"


@pytest.mark.parametrize("script", ARGPARSE_SCRIPTS, ids=lambda p: p.stem)
def test_help_does_not_crash(script):
    r = subprocess.run([sys.executable, str(script), "--help"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (
        f"{script.name} --help 失敗：\n{(r.stderr or r.stdout)[-600:]}\n"
        "常見原因：help 字串裡有字面的 `%`，要寫成 `%%`")
    assert r.stdout.strip(), f"{script.name} --help 沒有輸出"
