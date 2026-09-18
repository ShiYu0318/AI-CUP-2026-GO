import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest


@pytest.fixture(scope="session")
def meta():
    from aicup.data import load_metadata
    return load_metadata()


@pytest.fixture(scope="session")
def split(meta):
    from aicup.data import make_player_split
    return make_player_split(meta, val_frac=0.20, seed=42)
