import pytest
import torch

from scripts.generalization_compose import compose_embeddings, ensemble_embeddings, mean_summaries


@pytest.mark.parametrize("alpha", [0., .1, .5, 1.])
def test_independent_concatenation_matches_direct_raw_anchor_mixture(alpha):
    torch.manual_seed(7)
    qg, eg = torch.randn(3, 4), torch.randn(7, 4)
    qa, ea = torch.randn(3, 6), torch.randn(7, 6)
    actual = compose_embeddings(qg, qa, alpha) @ compose_embeddings(eg, ea, alpha).T
    torch.testing.assert_close(actual, (1-alpha)*(qg@eg.T)+alpha*(qa@ea.T))


def test_seed_summary_weights_seeds_equally():
    values = [{"d":{"all":{"num_queries":10,"reactzyme_mrr":x}}} for x in (.3,.4,.8)]
    actual = mean_summaries(values)
    assert actual["d"]["all"]["reactzyme_mrr"] == pytest.approx(.5)
    assert actual["d"]["all"]["num_queries"] == 10


def test_fixed_ensemble_encodes_seed_heads_without_mixing_cross_seed_scores():
    torch.manual_seed(11)
    queries, enzymes = [torch.randn(3,4) for _ in range(3)], [torch.randn(7,4) for _ in range(3)]
    qa, ea = torch.randn(3,6), torch.randn(7,6)
    actual = ensemble_embeddings(queries, qa, .25) @ ensemble_embeddings(enzymes, ea, .25).T
    expected = .75 * sum(q@e.T for q,e in zip(queries,enzymes))/3 + .25*(qa@ea.T)
    torch.testing.assert_close(actual, expected)
