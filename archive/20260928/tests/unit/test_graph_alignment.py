import torch
from horizyn.generalization_graph_alignment import graph_statistics, fit_alignment, align_composed


def test_balanced_fit_ignores_duplicate_edges_and_unused_candidates():
    e = torch.eye(3, dtype=torch.float64)
    r = e.roll(1, 0)
    edges = torch.tensor([[0, 0], [1, 1], [2, 2]])
    a = graph_statistics(e, r, edges)
    b = graph_statistics(torch.cat((e, torch.full((1, 3), 99.))), r,
                         torch.cat((edges, edges[:1].repeat(7, 1))))
    for name in a:
        assert torch.equal(a[name]['gram'], b[name]['gram'])
        assert torch.equal(a[name]['cross'], b[name]['cross'])


def test_positive_partner_fit_recovers_a_known_permutation():
    e = torch.eye(4, dtype=torch.float64)
    r = e.roll(1, 0)
    edges = torch.stack((torch.arange(4), torch.arange(4)), 1)
    fitted = fit_alignment(graph_statistics(e, r, edges), 1e-6)
    assert torch.allclose(e + e @ fitted['enzyme'], r, atol=2e-6)
    assert torch.allclose(r + r @ fitted['reaction'], e, atol=2e-6)


def test_identity_and_semantic_coordinates_remain_exact():
    x = torch.randn(7, 9)
    delta = torch.randn(4, 4, dtype=torch.float64)
    assert align_composed(x, delta, 0, .25, 4) is x
    transformed = align_composed(x, delta, .25, .25, 4)
    assert torch.equal(transformed[:, 4:], x[:, 4:])
    assert torch.isfinite(transformed).all()
