import numpy as np
import pytest

torch = pytest.importorskip("torch")

from horizyn.capability.capability_losses import (
    multilabel_bce_or_zero,
    positive_masks_from_pair_maps,
)
from horizyn.capability.enzyme_capability_dataset import (
    CapabilityVectorDataset,
    TargetWithCapabilityDataset,
    TargetWithTextDataset,
    TextVectorDataset,
)
from horizyn.datasets.base import BaseDataset
from horizyn.model import GatedEnzymeFeatureFusion


def test_missing_capability_vector_zero_with_mask(tmp_path):
    path = tmp_path / "capability.npz"
    np.savez_compressed(path, ids=np.array(["e1"]), vectors=np.ones((1, 4), dtype=np.float32))
    capability = CapabilityVectorDataset(path)
    target = BaseDataset(
        keys=["e1", "e2"],
        array_data=[{"target_vec": torch.ones(3)}, {"target_vec": torch.zeros(3)}],
        use_key_to_idx=True,
    )
    merged = TargetWithCapabilityDataset(target, capability)
    assert len(merged) == 2
    assert bool(merged["e1"]["capability_mask"].item())
    assert not bool(merged["e2"]["capability_mask"].item())
    assert torch.allclose(merged["e2"]["capability_vec"], torch.zeros(4))


def test_missing_text_vector_zero_with_mask(tmp_path):
    path = tmp_path / "text.npz"
    np.savez_compressed(path, ids=np.array(["e1"]), vectors=np.ones((1, 6), dtype=np.float32))
    text = TextVectorDataset(path)
    target = BaseDataset(
        keys=["e1", "e2"],
        array_data=[{"target_vec": torch.ones(3)}, {"target_vec": torch.zeros(3)}],
        use_key_to_idx=True,
    )
    merged = TargetWithTextDataset(target, text)
    assert len(merged) == 2
    assert bool(merged["e1"]["text_mask"].item())
    assert not bool(merged["e2"]["text_mask"].item())
    assert torch.allclose(merged["e2"]["text_vec"], torch.zeros(6))


def test_gate_masks_missing_capability_branch():
    fusion = GatedEnzymeFeatureFusion(
        raw_dim=3,
        pooled_dim=3,
        hyperbolic_dim=2,
        capability_dim=4,
        output_dim=5,
    )
    _fused, gates = fusion(
        torch.randn(2, 3),
        torch.randn(2, 3),
        torch.randn(2, 2),
        capability_vector=torch.randn(2, 4),
        capability_mask=torch.tensor([True, False]),
        return_gates=True,
    )
    assert gates[1, 3].item() == pytest.approx(0.0, abs=1e-6)


def test_all_known_positive_masks_are_multi_positive():
    r2e, e2r = positive_masks_from_pair_maps(
        ["r1", "r2"],
        ["e1", "e2", "e3"],
        reaction_to_enzymes={"r1": {"e1", "e2"}, "r2": {"e3"}},
        enzyme_to_reactions={"e1": {"r1"}, "e2": {"r1"}, "e3": {"r2"}},
    )
    assert r2e.tolist() == [[True, True, False], [False, False, True]]
    assert e2r.tolist() == [[True, False], [True, False], [False, True]]


def test_masked_bce_ignores_unknown_label_rows():
    logits = torch.tensor([[10.0], [-10.0]])
    targets = torch.tensor([[1.0], [1.0]])
    sample_mask = torch.tensor([True, False])
    loss = multilabel_bce_or_zero(logits, targets, sample_mask)
    assert loss.item() < 1e-3
