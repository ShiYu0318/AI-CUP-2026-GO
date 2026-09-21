"""序列後驗的整批計算必須與逐題計算給出相同答案。

`question_posteriors` 改成「先算完所有棋局、再逐題聚合」，
是因為逐題呼叫時每次只有 5~20 局，GRU 在 MPS 上的 kernel 啟動開銷
完全主導——實測一個 seed 要超過兩小時，整批版只要幾十秒。

這支測試釘住「加速不能改變數值」。batch 大小不同時結果必須一致，
否則 `artifacts/oof/` 裡用不同 batch 算出來的後驗會互相污染。
"""
import numpy as np
import pytest
import torch

from conftest_paths import scripts_on_path      # noqa: F401

from train_rank_seq import build_seq_net, question_posteriors


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(0)
    n, t, f = 300, 40, 12
    S = rng.random((n, t, f)).astype(np.float16)
    ok = np.ones(n, dtype=bool)
    ok[rng.choice(n, 20, replace=False)] = False
    lens = rng.integers(15, t, size=n)
    rows = [rng.choice(n, size=int(rng.integers(5, 21)), replace=False)
            for _ in range(40)]
    # 一題完全沒有有效棋局——這條路徑很容易在改寫時被漏掉
    rows.append(np.array([i for i in range(n) if not ok[i]][:3]))
    nets = [build_seq_net(f, {"width": 32, "arch": "cnn"})]
    for x in nets:
        x.eval()
    return nets, S, ok, lens, rows


@pytest.mark.parametrize("batch", [1, 7, 64, 4096])
def test_batch_size_does_not_change_posteriors(fixture, batch):
    nets, S, ok, lens, rows = fixture
    ref = question_posteriors(nets, S, ok, lens, rows, "cpu", batch=256)
    got = question_posteriors(nets, S, ok, lens, rows, "cpu", batch=batch)
    assert np.allclose(ref, got, atol=1e-9)


def test_empty_question_falls_back_to_uniform(fixture):
    nets, S, ok, lens, rows = fixture
    out = question_posteriors(nets, S, ok, lens, rows, "cpu")
    assert out.shape == (len(rows), 10)
    assert np.allclose(out[-1], 0.1), "沒有有效棋局的題目應該回傳均勻分布"
    assert np.allclose(out.sum(axis=1), 1.0)


def test_eval_mode_is_restored(fixture):
    """推論後必須還原 train/eval 狀態，否則後續會帶著 dropout 跑。"""
    nets, S, ok, lens, rows = fixture
    nets[0].train()
    question_posteriors(nets, S, ok, lens, rows, "cpu")
    assert nets[0].training is True
    nets[0].eval()
    question_posteriors(nets, S, ok, lens, rows, "cpu")
    assert nets[0].training is False
