import torch

from horizyn.generalization_residual import FrozenGeometryResidual, full_graph_contrastive_loss


def test_identity_initialization_preserves_retrieval():
    torch.manual_seed(8)
    enzyme = torch.nn.functional.normalize(torch.randn(7, 5), dim=1)
    reaction = torch.nn.functional.normalize(torch.randn(3, 5), dim=1)
    model = FrozenGeometryResidual(5, 9)
    torch.testing.assert_close(model.encode_reactions(reaction) @ model.encode_enzymes(enzyme).T,
                               reaction @ enzyme.T)


def test_full_graph_matches_explicit_positive_target_cross_entropy():
    logits = torch.tensor([[1., 2., -1.], [-2., 3., 2.]], requires_grad=True)
    r = torch.tensor([0, 0, 1, 1])
    e = torch.tensor([0, 1, 1, 2])
    actual, _, _ = full_graph_contrastive_loss(logits, r, e)
    explicit_r = -(torch.log_softmax(logits, 1) * torch.tensor([[.5, .5, 0.], [0., .5, .5]])).sum(1).mean()
    explicit_e = -(torch.log_softmax(logits, 0) * torch.tensor([[1., .5, 0.], [0., .5, 1.]])).sum(0).mean()
    torch.testing.assert_close(actual, (explicit_r + explicit_e) / 2)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_reaction_balanced_weighting_reduces_large_class_dominance():
    logits = torch.tensor([[1., 2., 3., 4.], [-1., -2., -3., 1.]])
    r, e = torch.tensor([0, 0, 0, 1]), torch.arange(4)
    actual, _, enzyme = full_graph_contrastive_loss(logits, r, e, "reaction_balanced")
    per_enzyme = -torch.log_softmax(logits, 0)[r, e]
    expected = (per_enzyme[:3].mean() + per_enzyme[3]) / 2
    torch.testing.assert_close(enzyme, expected)
    assert torch.isfinite(actual)
