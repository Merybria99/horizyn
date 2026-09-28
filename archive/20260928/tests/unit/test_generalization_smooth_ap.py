import torch

from horizyn.generalization_residual import sampled_smooth_ap_loss, full_graph_decoupled_loss


def test_smooth_ap_matches_explicit_ranks_and_finite_gradients():
    scores = torch.tensor([[2., 1., -1.], [-1., 0., 2.]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, False, True]])
    actual = sampled_smooth_ap_loss(scores, mask, torch.arange(2), temperature=0.3)
    values = []
    for row in range(2):
        ap = []
        for col in mask[row].nonzero().flatten():
            other = torch.arange(3) != col
            comp = torch.sigmoid((scores[row] - scores[row, col]) / 0.3)
            ap.append((1 + comp[other & mask[row]].sum()) / (1 + comp[other].sum()))
        values.append(torch.stack(ap).mean())
    torch.testing.assert_close(actual, 1 - torch.stack(values).mean())
    actual.backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[1, 2] < 0  # improve the positive
    assert scores.grad[1, 0] > 0  # suppress an unknown negative


def test_all_positive_bank_has_perfect_ap_and_permutation_invariance():
    scores = torch.randn(3, 5)
    anchors = torch.arange(3)
    assert sampled_smooth_ap_loss(scores, torch.ones_like(scores, dtype=torch.bool), anchors).item() == 0
    mask = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1], [0, 0, 0, 1, 0]], dtype=torch.bool)
    perm = torch.tensor([4, 0, 2, 1, 3])
    torch.testing.assert_close(sampled_smooth_ap_loss(scores, mask, anchors),
                               sampled_smooth_ap_loss(scores[:, perm], mask[:, perm], anchors))


def test_full_graph_decoupled_matches_training_objective():
    from horizyn.losses import DecoupledAllPositiveInfoNCELoss
    scores = torch.randn(4, 5, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1], [0, 0, 0, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    r, e = mask.nonzero(as_tuple=True)
    actual, _, _ = full_graph_decoupled_loss(scores, mask, r, e)
    objective = DecoupledAllPositiveInfoNCELoss()
    expected = (objective._decoupled_anchor_loss(-scores, mask, torch.tensor(1.)) +
                objective._decoupled_anchor_loss(-scores.T, mask.T, torch.tensor(1.))) / 2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, scores, retain_graph=True)[0],
                               torch.autograd.grad(expected, scores)[0])
