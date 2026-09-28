import numpy as np
import torch

from scripts.generalization_density_gate import nearest_support, support_gate, gate_candidates, balanced_value


def test_support_excludes_self_during_training_calibration():
    features = torch.tensor([[1., 0.], [0., 1.], [.6, .8]])
    expected = torch.tensor([.6, .8, .8])
    assert torch.allclose(nearest_support(features, features, torch.arange(3), batch_size=1), expected)
    assert torch.allclose(nearest_support(features, features, batch_size=2), torch.ones(3))


def test_gate_is_independent_of_opposing_candidates_and_conservative_outside_support():
    similarity = torch.tensor([.1, .5, .75, 1.])
    whole = support_gate(similarity, .5, 1., .5)
    assert torch.equal(whole, torch.tensor([0., 0., .25, .5]))
    assert torch.equal(whole, torch.cat([support_gate(part, .5, 1., .5) for part in similarity.split(2)]))
    assert torch.equal(support_gate(similarity, 1., 1., .5), torch.tensor([0., 0., 0., .5]))
    assert len(gate_candidates()) == 31


def test_selection_equally_weights_seen_and_unseen_direction_cells():
    summary = {direction: {stratum: dict(reactzyme_mrr=value) for stratum, value in
                          (("seen_reaction", .9), ("unseen_reaction", .1), ("all", .89))}
               for direction in ("reaction_to_enzyme", "enzyme_to_reaction")}
    assert np.isclose(balanced_value(summary), .5)
