"""網路與損失的測試：形狀、梯度、數學性質。全部在 CPU 上跑，秒級完成。"""
import pytest
import torch

from aicup.losses import (
    exponential_soft_labels, SoftLabelCrossEntropy, CornLoss, corn_to_proba,
    SubCenterArcFace, batch_hard_triplet_loss,
)
from aicup.nets import GoTrunk, GeM, AttentionPool, MultiTaskGoNet, count_parameters


# ---- 損失：競賽二 ----

def test_soft_labels_sum_to_one():
    q = exponential_soft_labels(torch.tensor([0, 5, 9]), 10)
    assert torch.allclose(q.sum(1), torch.ones(3))


def test_soft_labels_match_official_score_shape():
    """decay=1 時，相鄰級與正確級的權重比必須恰為 exp(-1)。"""
    q = exponential_soft_labels(torch.tensor([5]), 10, decay=1.0)
    ratio = (q[0, 6] / q[0, 5]).item()
    assert ratio == pytest.approx(float(torch.exp(torch.tensor(-1.0))), rel=1e-5)


def test_soft_labels_peak_at_target():
    q = exponential_soft_labels(torch.tensor([7]), 10)
    assert int(q.argmax()) == 7


def test_soft_labels_boundary_targets():
    for y in (0, 9):
        q = exponential_soft_labels(torch.tensor([y]), 10)
        assert int(q.argmax()) == y
        assert torch.allclose(q.sum(), torch.tensor(1.0))


def test_soft_label_loss_is_finite_and_differentiable():
    logits = torch.randn(16, 10, requires_grad=True)
    loss = SoftLabelCrossEntropy(10)(logits, torch.randint(0, 10, (16,)))
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None


def test_corn_proba_sums_to_one_and_is_nonnegative():
    p = corn_to_proba(torch.randn(32, 9))
    assert torch.allclose(p.sum(1), torch.ones(32), atol=1e-5)
    assert (p >= 0).all()


def test_corn_loss_decreases_with_correct_direction():
    """把 logit 推向正確方向，CORN loss 應下降。"""
    y = torch.full((32,), 7)
    bad = torch.full((32, 9), -5.0)      # 全部預測 y<=0，對 y=7 是大錯
    good = torch.full((32, 9), 5.0)
    good[:, 7:] = -5.0                   # P(y>k) 高到 k=6，之後轉低 -> y=7
    lf = CornLoss(10)
    assert lf(good, y) < lf(bad, y)


def test_corn_loss_differentiable():
    logits = torch.randn(16, 9, requires_grad=True)
    CornLoss(10)(logits, torch.randint(0, 10, (16,))).backward()
    assert logits.grad is not None


# ---- 損失：競賽一 ----

def test_arcface_output_shape_and_scale():
    af = SubCenterArcFace(64, 100, k=3)
    e = torch.randn(8, 64)
    assert af(e).shape == (8, 100)
    assert af(e, torch.randint(0, 100, (8,))).shape == (8, 100)


def test_arcface_margin_lowers_target_logit():
    """加了 angular margin 之後，正確類別的 logit 必須變低（訓練才會更嚴格）。"""
    torch.manual_seed(0)
    af = SubCenterArcFace(32, 10, k=1, margin=0.4)
    e = torch.randn(4, 32)
    y = torch.randint(0, 10, (4,))
    without = af(e)[torch.arange(4), y]
    with_m = af(e, y)[torch.arange(4), y]
    assert (with_m < without + 1e-5).all()


def test_arcface_dynamic_margin_is_larger_for_frequent_classes():
    counts = torch.tensor([1, 10, 100, 1800])
    af = SubCenterArcFace(16, 4, class_counts=counts)
    m = af.margin
    assert m[0] > m[-1]          # 樣本少的類別用較大 margin
    assert (m > 0).all()


def test_arcface_is_differentiable():
    af = SubCenterArcFace(32, 20, k=2)
    e = torch.randn(8, 32, requires_grad=True)
    af(e, torch.randint(0, 20, (8,))).sum().backward()
    assert e.grad is not None


def test_batch_hard_triplet_zero_when_well_separated():
    """同類完全重合、異類相距很遠時，loss 應為 0。"""
    e = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    y = torch.tensor([0, 0, 1, 1])
    assert batch_hard_triplet_loss(e, y, margin=0.3).item() == pytest.approx(0.0)


def test_batch_hard_triplet_positive_when_mixed():
    e = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    y = torch.tensor([0, 0, 1, 1])       # 同類反而相距最遠
    assert batch_hard_triplet_loss(e, y, margin=0.3).item() > 0


def test_batch_hard_triplet_handles_all_same_label():
    e = torch.randn(4, 8)
    assert torch.isfinite(batch_hard_triplet_loss(e, torch.zeros(4, dtype=torch.long)))


# ---- 網路 ----

def test_gem_between_avg_and_max():
    x = torch.rand(2, 4, 5, 5)
    gem = GeM(p=3.0)(x)
    avg = x.mean(dim=(2, 3))
    mx = x.amax(dim=(2, 3))
    assert ((gem >= avg - 1e-5) & (gem <= mx + 1e-5)).all()


def test_trunk_output_shapes():
    t = GoTrunk(in_channels=34, width=32, blocks=2, embed_dim=64)
    t.eval()
    m, c = t(torch.randn(3, 34, 19, 19))
    assert m.shape == (3, 64) and c.shape == (3, 64)


def test_attention_pool_weights_sum_to_one():
    ap = AttentionPool(16)
    x = torch.randn(2, 7, 16)
    pooled, w = ap(x)
    assert pooled.shape == (2, 16)
    assert torch.allclose(w.sum(1), torch.ones(2), atol=1e-5)


def test_attention_pool_respects_mask():
    """被遮罩的棋局權重必須為 0——題目局數 5~20 不等，padding 不能參與。"""
    ap = AttentionPool(8)
    x = torch.randn(1, 5, 8)
    mask = torch.tensor([[True, True, False, False, False]])
    _, w = ap(x, mask)
    assert torch.allclose(w[0, 2:], torch.zeros(3), atol=1e-6)
    assert w[0, :2].sum().item() == pytest.approx(1.0, rel=1e-5)


def test_multitask_forward_shapes():
    net = MultiTaskGoNet(in_channels=34, width=32, blocks=2, embed_dim=64)
    net.eval()
    out = net(torch.randn(4, 34, 19, 19))
    assert out["embedding"].shape == (4, 64)
    assert out["rank_logits"].shape == (4, 10)
    assert out["corn_logits"].shape == (4, 9)


def test_multitask_forward_set_shapes():
    net = MultiTaskGoNet(in_channels=34, width=32, blocks=2, embed_dim=64)
    net.eval()
    out = net.forward_set(torch.randn(2, 6, 34, 19, 19))
    assert out["set_embedding"].shape == (2, 64)
    assert out["rank_logits"].shape == (2, 10)
    assert out["attention"].shape == (2, 6)


def test_multitask_backward():
    net = MultiTaskGoNet(in_channels=34, width=16, blocks=1, embed_dim=32)
    out = net(torch.randn(8, 34, 19, 19))
    (out["rank_logits"].sum() + out["embedding"].sum()).backward()
    assert all(p.grad is not None for p in net.rank_head.parameters())


def test_parameter_count_is_reported():
    net = MultiTaskGoNet(width=64, blocks=4, embed_dim=256)
    assert count_parameters(net) > 0


# ---- 棋形品質輔助 head ----

def test_quality_head_absent_by_default():
    net = MultiTaskGoNet(width=16, blocks=1, embed_dim=32)
    net.eval()
    assert "quality" not in net(torch.randn(2, 34, 19, 19))


def test_quality_head_shape_when_enabled():
    net = MultiTaskGoNet(width=16, blocks=1, embed_dim=32, n_quality=5)
    net.eval()
    out = net(torch.randn(3, 34, 19, 19))
    assert out["quality"].shape == (3, 5)


def test_quality_head_is_differentiable():
    net = MultiTaskGoNet(width=16, blocks=1, embed_dim=32, n_quality=4)
    out = net(torch.randn(4, 34, 19, 19))
    out["quality"].sum().backward()
    assert net.quality_head.weight.grad is not None


def test_quality_head_does_not_break_set_forward():
    net = MultiTaskGoNet(width=16, blocks=1, embed_dim=32, n_quality=3)
    net.eval()
    out = net.forward_set(torch.randn(2, 5, 34, 19, 19))
    assert out["rank_logits"].shape == (2, 10)
