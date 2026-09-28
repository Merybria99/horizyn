from types import SimpleNamespace

import pytest
import torch

from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule


def small_module(**kwargs):
    return ProteinPooledLitModule(
        query_encoder_dims=[4, 4], target_encoder_dims=[4, 4], embedding_dim=4,
        residue_dim=4, pooling="mean", loss_name="BidirectionalSampledMultiPositiveInfoNCELoss",
        sampled_require_both_directions=True, **kwargs,
    )


def test_guard_rejects_one_direction_without_support():
    module = small_module()
    dists = torch.rand(1, 2, requires_grad=True)
    with pytest.raises(RuntimeError, match="e2r"):
        module._loss_with_components(
            dists, torch.tensor([0]), torch.tensor([0]), torch.rand(1, 4), torch.rand(2, 4),
            biological_negative_mask=torch.tensor([[False, True]]),
            random_negative_mask=torch.zeros(1, 2, dtype=torch.bool),
        )


def test_supported_both_directions_have_finite_gradient():
    module = small_module()
    dists = torch.rand(2, 2, requires_grad=True)
    loss, components = module._loss_with_components(
        dists, torch.tensor([0, 1]), torch.tensor([0, 1]), torch.rand(2, 4), torch.rand(2, 4),
        biological_negative_mask=torch.tensor([[False, True], [False, False]]),
        random_negative_mask=torch.zeros(2, 2, dtype=torch.bool),
    )
    loss.backward()
    assert torch.isfinite(dists.grad).all()
    assert components["r2e_valid_anchors"] == 1
    assert components["e2r_valid_anchors"] == 1


def test_indexed_positive_lookup_does_not_iterate_global_adjacency():
    class IndexedLookup:
        def batch_positive_indices(self, queries, targets):
            assert queries == ["r"] and targets == ["e"]
            return [0], [0]

        def get(self, *args):
            raise AssertionError("Should use bounded indexed lookup")

    module = small_module(positive_pair_source="all_known_in_batch")
    module.trainer = SimpleNamespace(datamodule=SimpleNamespace(_train_query_to_targets=IndexedLookup()))
    q, p = module._positive_pair_indices(["r"], ["e"], ["r"], ["e"])
    assert q.tolist() == p.tolist() == [0]


@pytest.mark.parametrize("enabled", [False, True])
def test_checked_indexed_negative_sharing_is_opt_in(enabled):
    class IndexedLookup:
        def batch_negative_masks(self, queries, targets, pair_queries, pair_targets, types):
            assert enabled
            assert pair_queries == ["r0"] and pair_targets == ["e1"]
            assert types == ["random_negative"]
            return [[False, False], [False, False]], [[False, True], [True, False]]

    module = small_module(sampled_share_indexed_negatives=enabled)
    module.trainer = SimpleNamespace(datamodule=SimpleNamespace(indexed_training_pairs=IndexedLookup()))
    bio, random = module._typed_negative_masks(
        ["r0", "r1"], ["e0", "e1"], ["r0"], ["e1"], ["random_negative"],
        torch.tensor([0, 1]), torch.tensor([0, 1]),
    )
    assert not bio.any()
    assert random.tolist() == [[False, True], [enabled, False]]


def test_indexed_negative_sharing_does_not_override_validation_policy():
    class IndexedLookup:
        def batch_negative_masks(self, *args):
            raise AssertionError("Training-only policy must not touch validation")

    module = small_module(sampled_share_indexed_negatives=True)
    module.trainer = SimpleNamespace(datamodule=SimpleNamespace(indexed_training_pairs=IndexedLookup()))
    _, random = module._typed_negative_masks(
        ["r0", "r1"], ["e0", "e1"], [], [], [],
        torch.tensor([0, 1]), torch.tensor([0, 1]), use_unlabelled_as_random=True,
    )
    assert random.tolist() == [[False, True], [True, False]]
