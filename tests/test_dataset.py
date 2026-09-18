"""資料集與取樣器的測試：對稱增強、P x K 結構、題目 padding。"""
import numpy as np
import pytest
import torch

from aicup.dataset import _dihedral, GameDataset, PKPlayerSampler, QuestionDataset, collate_questions
from aicup.benchmark import build_rank_questions


def test_dihedral_has_eight_distinct_transforms():
    x = np.arange(361).reshape(1, 19, 19)
    outs = {_dihedral(x, k).tobytes() for k in range(8)}
    assert len(outs) == 8


def test_dihedral_identity_and_shape():
    x = np.random.rand(3, 19, 19)
    assert np.array_equal(_dihedral(x, 0), x)
    for k in range(8):
        assert _dihedral(x, k).shape == x.shape


def test_dihedral_preserves_stone_count():
    x = (np.random.rand(2, 19, 19) > 0.7).astype(np.uint8)
    for k in range(8):
        assert _dihedral(x, k).sum() == x.sum()


def test_game_dataset_returns_expected_types(meta, split):
    rows = np.flatnonzero(split.mask(meta))[:4]
    ds = GameDataset(meta, rows, n_snapshots=2, augment=True)
    x, y, p = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape[1:] == (34, 19, 19)
    assert 0 <= y < 10
    assert p == -1                       # 未提供 player_to_label 時為 -1


def test_game_dataset_player_label_mapping(meta, split):
    rows = np.flatnonzero(split.mask(meta))[:3]
    pid = meta.iloc[int(rows[0])]["player_id"]
    ds = GameDataset(meta, rows, n_snapshots=1, augment=False,
                     player_to_label={pid: 77})
    assert ds[0][2] == 77


def test_pk_sampler_batch_structure(meta, split):
    rows = np.flatnonzero(split.mask(meta))[:20000]
    s = PKPlayerSampler(meta, rows, p=8, k=4, n_batches=3, seed=0)
    batches = list(iter(s))
    assert len(batches) == 3
    for b in batches:
        assert len(b) == 8 * 4
        sub = meta.iloc[rows[np.array(b)]]
        counts = sub["player_id"].value_counts()
        assert len(counts) == 8              # 恰好 P 位玩家
        assert (counts == 4).all()           # 每人 K 局


def test_pk_sampler_excludes_single_game_players(meta, split):
    rows = np.flatnonzero(split.mask(meta))[:5000]
    s = PKPlayerSampler(meta, rows, p=4, k=2, n_batches=1)
    for pid, idx in s.by_player.items():
        assert len(idx) >= 2                 # 只有一局的玩家配不出正例


def test_question_dataset_and_collate(meta, split):
    q = build_rank_questions(meta, split, n_per_player=1, seed=0, max_questions=6)
    ds = QuestionDataset(meta, q, n_snapshots=2)
    batch = [ds[i] for i in range(4)]
    x, mask, y = collate_questions(batch)
    assert x.shape[0] == 4 and x.shape[2:] == (34, 19, 19)
    assert mask.shape == x.shape[:2]
    assert y.shape == (4,)


def test_collate_mask_marks_real_games_only(meta, split):
    q = build_rank_questions(meta, split, n_per_player=1, seed=1, max_questions=8)
    ds = QuestionDataset(meta, q, n_snapshots=1)
    batch = [ds[i] for i in range(5)]
    x, mask, _ = collate_questions(batch)
    for i, (xi, _) in enumerate(batch):
        assert int(mask[i].sum()) == xi.shape[0]
        assert torch.allclose(x[i, xi.shape[0]:], torch.zeros_like(x[i, xi.shape[0]:]))


def test_question_game_count_within_official_range(meta, split):
    q = build_rank_questions(meta, split, n_per_player=1, seed=2, max_questions=30)
    ds = QuestionDataset(meta, q, n_snapshots=1)
    for i in range(10):
        x, _ = ds[i]
        assert 5 <= x.shape[0] <= 20
