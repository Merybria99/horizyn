import torch

from scripts.generalization_hubness import fit_means


def test_means_use_only_training_ids_and_balance_opposite_anchors():
    queries = torch.tensor([[1., 0.], [0., 1.]])
    # The final protein is validation-only; its extreme value must not enter a mean.
    proteins = torch.tensor([[1., 0.], [0., 1.], [1000., 1000.]])
    edges = [[0, 0], [0, 1], [2, 1]]
    means = fit_means(queries, proteins, edges, torch.tensor([0, 2]), torch.tensor([0, 1]))
    assert torch.allclose(means["unique_id"][0], torch.tensor([.5, .5]))
    assert torch.allclose(means["unique_id"][1], torch.tensor([.5, .5]))
    assert torch.allclose(means["pair_frequency"][0], torch.tensor([2/3, 1/3]))
    assert torch.allclose(means["pair_frequency"][1], torch.tensor([1/3, 2/3]))
    assert torch.allclose(means["opposite_anchor_balanced"][0], torch.tensor([.75, .25]))
    assert torch.allclose(means["opposite_anchor_balanced"][1], torch.tensor([.25, .75]))


def test_additive_calibration_remains_an_independent_dual_encoder_dot_product():
    q, e = torch.tensor([[1., 2.], [3., 1.]]), torch.tensor([[2., 0.], [1., 3.]])
    mq, me = torch.tensor([.5, 1.]), torch.tensor([1., .5])
    alpha, beta = .25, .5
    score = q @ e.T - alpha * (mq @ e.T) - beta * (q @ me).unsqueeze(1)
    # Independent affine embeddings differ only by a global scalar.
    affine = (q - alpha * mq) @ (e - beta * me).T
    assert torch.allclose(score, affine - alpha * beta * (mq @ me))
