"""序列 checkpoint 的序列庫要由它自己的 args 推導，順序照舊。"""
from aicup.rank_nets import seq_stores_of


def test_default_single_store():
    assert seq_stores_of({}) == ["cache/sequences.npy"]


def test_extra_stores_keep_order_and_normalise_dir():
    a = {"sequences": "/tmp/x/sequences.npy",
         "extra_sequences": ["cache/sequences_opp.npy", "elsewhere/seqshape.npy"]}
    assert seq_stores_of(a) == ["cache/sequences.npy", "cache/sequences_opp.npy",
                                "cache/seqshape.npy"]


def test_none_extra_is_fine():
    assert seq_stores_of({"extra_sequences": None}) == ["cache/sequences.npy"]
