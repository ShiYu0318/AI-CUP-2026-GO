"""推論函式不可以留下副作用，把模型的 train/eval 模式改掉。

這是真實踩過的坑：`question_posteriors` 結尾無條件呼叫 `n.train()`，
於是 predict.py 先用它做切點校準（eval 模式），接著跑測試集推論時
dropout 是**開著**的。校準與預測在不同模式下，期望等級的標準差
從 2.16 被壓到 0.25，切點完全遷移不過去——
內部驗證 0.48、端到端只有 0.33，而且特徵逐位元相同，極難追。
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_rank_mlp import RankMLP, question_posteriors   # noqa: E402


class _Scaler:
    def transform(self, A):
        return np.asarray(A, dtype=np.float32)


def _setup(dim=16, n_games=40):
    rng = np.random.default_rng(0)
    nets = [RankMLP(dim, hidden=32) for _ in range(2)]
    X = rng.normal(size=(n_games, dim)).astype(np.float32)
    ok = np.ones(n_games, dtype=bool)
    game_rows = [np.arange(0, 10), np.arange(10, 25), np.arange(25, 40)]
    return nets, _Scaler(), X, ok, game_rows


def test_question_posteriors_restores_eval_mode():
    nets, sc, X, ok, rows = _setup()
    for n in nets:
        n.eval()
    question_posteriors(nets, sc, X, ok, rows, torch.device("cpu"))
    assert all(not n.training for n in nets), "eval 模式被改成 train 了"


def test_question_posteriors_restores_train_mode():
    nets, sc, X, ok, rows = _setup()
    for n in nets:
        n.train()
    question_posteriors(nets, sc, X, ok, rows, torch.device("cpu"))
    assert all(n.training for n in nets), "train 模式沒有被還原"


def test_question_posteriors_is_deterministic_across_calls():
    """連續呼叫必須完全一致——若第一次把模型留在 train 模式，第二次會帶 dropout。"""
    nets, sc, X, ok, rows = _setup()
    for n in nets:
        n.eval()
    a = question_posteriors(nets, sc, X, ok, rows, torch.device("cpu"))
    b = question_posteriors(nets, sc, X, ok, rows, torch.device("cpu"))
    assert np.abs(a - b).max() == 0.0


def test_dropout_actually_changes_output():
    """反證：模型真的有 dropout，所以上面的測試不是空轉。"""
    nets, sc, X, ok, rows = _setup()
    for n in nets:
        n.eval()
    a = question_posteriors(nets, sc, X, ok, rows, torch.device("cpu"))
    for n in nets:
        n.train()
    torch.manual_seed(0)
    with torch.no_grad():
        t = torch.from_numpy(X[rows[0]])
        noisy = torch.softmax(nets[0](t)[0], dim=-1).numpy()
    nets[0].eval()
    with torch.no_grad():
        clean = torch.softmax(nets[0](t)[0], dim=-1).numpy()
    assert np.abs(noisy - clean).max() > 1e-6, "模型沒有 dropout，測試需要重新設計"
    assert a.shape == (3, 10)
