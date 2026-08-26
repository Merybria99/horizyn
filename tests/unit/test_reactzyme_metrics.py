import pytest
import torch

from horizyn.metrics import reactzyme_mean_reciprocal_rank, reciprocal_rank


def test_reactzyme_mrr_averages_all_positive_ranks():
    scores = torch.arange(100, 0, -1, dtype=torch.float32)
    targets = torch.tensor([0, 99], dtype=torch.long)

    assert reciprocal_rank(scores, targets).item() == pytest.approx(1.0)
    assert reactzyme_mean_reciprocal_rank(scores, targets).item() == pytest.approx(
        (1.0 + 1.0 / 100.0) / 2.0
    )


def test_reactzyme_mrr_matches_first_positive_rr_for_one_target():
    scores = torch.tensor([0.2, 0.7, 0.5, 0.9])
    targets = torch.tensor([2], dtype=torch.long)

    expected = reciprocal_rank(scores, targets)
    actual = reactzyme_mean_reciprocal_rank(scores, targets)

    assert actual.item() == pytest.approx(expected.item())


def test_reactzyme_mrr_ignores_padding():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    targets = torch.tensor([1, 3, -1, -1], dtype=torch.long)

    assert reactzyme_mean_reciprocal_rank(scores, targets).item() == pytest.approx(
        (1.0 / 2.0 + 1.0 / 4.0) / 2.0
    )


def test_reactzyme_mrr_deduplicates_positive_indices():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    targets = torch.tensor([1, 1, 3], dtype=torch.long)

    assert reactzyme_mean_reciprocal_rank(scores, targets).item() == pytest.approx(
        (1.0 / 2.0 + 1.0 / 4.0) / 2.0
    )


def test_reactzyme_mrr_returns_zero_without_targets():
    scores = torch.tensor([0.9, 0.8])
    targets = torch.tensor([-1, -1], dtype=torch.long)

    assert reactzyme_mean_reciprocal_rank(scores, targets).item() == 0.0


@pytest.mark.parametrize(
    "scores,targets,error",
    [
        (torch.ones(2, 2), torch.tensor([0]), "expects 1D tensors"),
        (torch.ones(2), torch.tensor([0], dtype=torch.int32), "dtype torch.long"),
        (torch.ones(2), torch.tensor([2]), "out-of-range"),
    ],
)
def test_reactzyme_mrr_validates_inputs(scores, targets, error):
    with pytest.raises(ValueError, match=error):
        reactzyme_mean_reciprocal_rank(scores, targets)
