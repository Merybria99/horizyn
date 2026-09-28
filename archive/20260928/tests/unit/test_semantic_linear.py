import torch

from horizyn.semantic_linear import apply_normalizer, encode_bridge, fit_bridges, fit_normalizer, paired_statistics


def test_normalization_ignores_validation_and_missing_vectors():
    vectors = torch.tensor([[1., 0.], [0., 1.], [-100., 1.], [8., 0.]])
    mask = torch.tensor([True, True, True, False])
    mean = fit_normalizer(vectors, torch.tensor([0, 1, 3]), mask)
    torch.testing.assert_close(mean, torch.tensor([.5, .5]))
    result = apply_normalizer(vectors, mean, mask)
    assert torch.equal(result[3], torch.zeros(2))
    torch.testing.assert_close(result[:3].norm(dim=1), torch.ones(3))


def test_weighted_covariance_matches_explicit_edge_distribution():
    x = torch.tensor([[1., 2.], [3., 1.], [0., 4.]])
    y = torch.tensor([[0., 1.], [2., 1.], [3., 0.], [1., 3.]])
    r, e = torch.tensor([0, 0, 1, 2, 2]), torch.tensor([0, 1, 2, 1, 3])
    s = paired_statistics(x, y, r, e)
    weights = torch.tensor([1/6, 1/6, 1/3, 1/6, 1/6], dtype=torch.float64)
    xx, yy = x[r].double(), y[e].double()
    xm, ym = weights @ xx, weights @ yy
    torch.testing.assert_close(s["x_mean"], xm)
    torch.testing.assert_close(s["y_mean"], ym)
    torch.testing.assert_close(s["xy"], (xx-xm).T @ ((yy-ym)*weights[:, None]))
    torch.testing.assert_close(s["yy"], (yy-ym).T @ ((yy-ym)*weights[:, None]))
    for state in fit_bridges(s, .1, cca_rank=2).values():
        q, t = encode_bridge(state, x, y)
        assert q.shape[1] == t.shape[1]
        assert torch.isfinite(q @ t.T).all()
        torch.testing.assert_close(q, encode_bridge(state, queries=x)[0])
        torch.testing.assert_close(t, encode_bridge(state, enzymes=y)[1])
