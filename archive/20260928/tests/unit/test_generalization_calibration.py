import pytest
import torch
from horizyn.generalization_calibration import MeanAffinityCalibration
from horizyn.generalization_retrieval import canonical_dot


def test_zero_delegates_without_changing_dimension_or_values():
    r, e = torch.randn(3, 7), torch.randn(9, 7)
    model = MeanAffinityCalibration(r.double().mean(0), e.double().mean(0))
    assert model.encode_reactions(r) is r
    assert model.encode_enzymes(e) is e
    assert torch.equal(canonical_dot(r, e), canonical_dot(model.encode_reactions(r), model.encode_enzymes(e)))


def test_augmented_dot_realizes_stored_bias_coordinates_and_batch_independence():
    torch.manual_seed(42)
    r, e = torch.randn(3, 7), torch.randn(9, 7)
    mr, me = torch.randn(7, dtype=torch.float64), torch.randn(7, dtype=torch.float64)
    model = MeanAffinityCalibration(mr, me, .5, 2.)
    ra, ea = model.encode_reactions(r), model.encode_enzymes(e)
    rb = (-2 * (r.double() * me).sum(1)).float()
    eb = (-.5 * (e.double() * mr).sum(1)).float()
    expected = (r.double() @ e.double().T + rb.double()[:, None] + eb.double()[None, :]).float()
    assert torch.equal(canonical_dot(ra, ea), expected)
    assert torch.equal(ra[[2, 0]], model.encode_reactions(r[[2, 0]]))
    assert torch.equal(ea[[3]], model.encode_enzymes(e[[3]]))
    assert torch.equal(canonical_dot(ra, ea)[[2, 0]][:, [3]], canonical_dot(model.encode_reactions(r[[2, 0]]), model.encode_enzymes(e[[3]])))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):MeanAffinityCalibration(torch.ones(3), torch.ones(4))
    model = MeanAffinityCalibration(torch.ones(3), torch.ones(3))
    with pytest.raises(ValueError):model.encode_enzymes(torch.ones(2, 3, dtype=torch.float64))
    with pytest.raises(ValueError):model.encode_reactions(torch.full((1, 3), float('nan')))
