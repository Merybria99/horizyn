"""Tests for enzyme BioFP target datasets."""

import numpy as np
import pytest
import torch

from horizyn.capability.enzyme_capability_dataset import (
    BioFPTargetDataset,
    CapabilityVectorDataset,
    TargetWithBioFPTargetDataset,
    TargetWithCapabilityDataset,
)
from horizyn.datasets.base import BaseDataset
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from scripts.pretrain_enzyme_biofp_split import active_biofp_keys


def _write_biofp_npz(path):
    np.savez(
        path,
        ids=np.asarray(["prot1", "prot2"]),
        center_targets=np.asarray([[1.0, 0.0], [0.25, 0.75]], dtype=np.float32),
        center_mask=np.asarray([True, True]),
        center_denominator=np.asarray([2.0, 4.0], dtype=np.float32),
        cofactor_targets=np.asarray([[0.0], [1.0]], dtype=np.float32),
        cofactor_mask=np.asarray([False, True]),
        cofactor_denominator=np.asarray([0.0, 1.0], dtype=np.float32),
        transition_targets=np.asarray([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=np.float32),
        transition_mask=np.asarray([True, True]),
        transition_denominator=np.asarray([2.0, 3.0], dtype=np.float32),
    )


def test_biofp_target_dataset_loads_soft_targets(tmp_path):
    path = tmp_path / "biofp.npz"
    _write_biofp_npz(path)

    dataset = BioFPTargetDataset(path)
    sample = dataset["prot2"]

    assert dataset.target_dims == {"center": 2, "cofactor": 1, "transition": 3}
    assert torch.allclose(sample["biofp_center_targets"], torch.tensor([0.25, 0.75]))
    assert sample["biofp_cofactor_mask"].item() is True
    assert sample["biofp_transition_denominator"].item() == pytest.approx(3.0)


def test_target_with_biofp_targets_preserves_missing_targets(tmp_path):
    path = tmp_path / "biofp.npz"
    _write_biofp_npz(path)
    biofp = BioFPTargetDataset(path)
    targets = BaseDataset(
        keys=["prot1", "missing"],
        array_data=[
            {"target_vec": torch.tensor([1.0, 2.0])},
            {"target_vec": torch.tensor([3.0, 4.0])},
        ],
        use_key_to_idx=True,
    )

    merged = TargetWithBioFPTargetDataset(targets, biofp)
    present = merged["prot1"]
    missing = merged["missing"]

    assert merged.missing_count == 1
    assert torch.equal(present["target_vec"], torch.tensor([1.0, 2.0]))
    assert present["biofp_center_mask"].item() is True
    assert missing["biofp_center_mask"].item() is False
    assert missing["biofp_cofactor_denominator"].item() == 0.0
    assert torch.equal(missing["biofp_transition_targets"], torch.zeros(3))


def test_biofp_target_dataset_discovers_minimal_schema_and_per_label_masks(tmp_path):
    path = tmp_path / "minimal.npz"
    np.savez(
        path,
        ids=np.asarray(["prot1", "prot2"]),
        mechanism_targets=np.asarray([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32),
        mechanism_mask=np.asarray([[True, True], [True, False]]),
        mechanism_denominator=np.asarray([[1.0, 0.2], [0.5, 0.0]], dtype=np.float32),
        mechanism_confidence=np.asarray([[1.0, 0.2], [0.5, 0.0]], dtype=np.float32),
        cofactor_targets=np.asarray([[0.0], [1.0]], dtype=np.float32),
        cofactor_mask=np.asarray([[False], [True]]),
        cofactor_denominator=np.asarray([[0.0], [1.0]], dtype=np.float32),
        cofactor_confidence=np.asarray([[0.0], [1.0]], dtype=np.float32),
    )

    dataset = BioFPTargetDataset(path)

    assert dataset.FAMILIES == ("cofactor", "mechanism")
    assert dataset.target_dims == {"cofactor": 1, "mechanism": 2}
    assert dataset["prot2"]["biofp_mechanism_mask"].tolist() == [True, False]
    assert dataset["prot1"]["biofp_mechanism_confidence"].tolist() == pytest.approx(
        [1.0, 0.2]
    )
    assert active_biofp_keys(dataset, {"mechanism": 1.0}) == {"prot1", "prot2"}
    assert active_biofp_keys(dataset, {"cofactor": 1.0}) == {"prot2"}
    assert active_biofp_keys(
        dataset,
        {"mechanism": 0.65, "cofactor": 0.35},
    ) == {"prot1", "prot2"}


def test_capability_wrapper_sets_mask_for_present_vectors(tmp_path):
    vector_path = tmp_path / "capability.npz"
    np.savez(
        vector_path,
        ids=np.asarray(["prot1"]),
        vectors=np.asarray([[0.2, 0.8]], dtype=np.float32),
    )
    capability = CapabilityVectorDataset(vector_path)
    targets = BaseDataset(
        keys=["prot1", "missing"],
        array_data=[
            {"target_vec": torch.tensor([1.0])},
            {"target_vec": torch.tensor([2.0])},
        ],
        use_key_to_idx=True,
    )

    merged = TargetWithCapabilityDataset(targets, capability)

    assert merged["prot1"]["capability_mask"].item() is True
    assert merged["missing"]["capability_mask"].item() is False


def test_biofp_auxiliary_loss_uses_soft_targets_and_masks():
    module = ProteinPooledLitModule(
        query_encoder_dims=[3, 6],
        target_encoder_dims=[6, 6],
        embedding_dim=6,
        residue_dim=4,
        pooling="mean",
        enzyme_input_mode="raw_mean_sleec_biofp_split",
        biofp_center_dim=2,
        biofp_cofactor_dim=1,
        biofp_transition_dim=3,
        biofp_seq_dim=4,
        biofp_dim=2,
        biofp_hidden_dim=8,
        biofp_aux_weight=0.03,
        beta=10.0,
        loss_name="FullBatchMLNCELoss",
    )

    loss, components = module._biofp_auxiliary_loss(
        pooling_details={
            "biofp_logits_center": torch.zeros(2, 2),
            "biofp_logits_cofactor": torch.zeros(2, 1),
            "biofp_logits_transition": torch.zeros(2, 3),
        },
        biofp_targets={
            "biofp_center_targets": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "biofp_center_mask": torch.tensor([True, True]),
            "biofp_center_denominator": torch.tensor([2.0, 4.0]),
            "biofp_cofactor_targets": torch.tensor([[0.0], [1.0]]),
            "biofp_cofactor_mask": torch.tensor([False, True]),
            "biofp_cofactor_denominator": torch.tensor([0.0, 1.0]),
            "biofp_transition_targets": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            "biofp_transition_mask": torch.tensor([True, True]),
            "biofp_transition_denominator": torch.tensor([1.0, 3.0]),
        },
    )

    assert loss.item() == pytest.approx(0.69314718)
    assert components["biofp_center_active"].item() == 2
    assert components["biofp_cofactor_active"].item() == 1
    assert components["biofp_transition_active"].item() == 2


def test_biofp_auxiliary_loss_supports_dynamic_per_label_confidence():
    module = ProteinPooledLitModule(
        query_encoder_dims=[3, 6],
        target_encoder_dims=[6, 6],
        embedding_dim=6,
        residue_dim=4,
        pooling="mean",
        enzyme_input_mode="raw_mean_sleec_biological_factorized",
        hyperbolic_hyp_dim=3,
        enzyme_block_dims={
            "core": 2,
            "site": 1,
            "mechanism": 1,
            "cofactor": 1,
            "ec": 1,
        },
        enzyme_block_weights={
            "core": 0.4,
            "site": 0.2,
            "mechanism": 0.15,
            "cofactor": 0.15,
            "ec": 0.1,
        },
        biofp_family_dims={"mechanism": 2, "cofactor": 1},
        biofp_family_weights={"mechanism": 0.65, "cofactor": 0.35},
        biofp_aux_weight=0.03,
        beta=10.0,
        loss_name="FullBatchMLNCELoss",
    )

    loss, components = module._biofp_auxiliary_loss(
        pooling_details={
            "biofp_logits_mechanism": torch.zeros(2, 2),
            "biofp_logits_cofactor": torch.zeros(2, 1),
        },
        biofp_targets={
            "biofp_mechanism_targets": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "biofp_mechanism_mask": torch.tensor([[True, False], [True, True]]),
            "biofp_mechanism_denominator": torch.tensor([[1.0, 0.0], [0.5, 0.2]]),
            "biofp_mechanism_confidence": torch.tensor([[1.0, 0.0], [0.5, 0.2]]),
            "biofp_cofactor_targets": torch.tensor([[0.0], [1.0]]),
            "biofp_cofactor_mask": torch.tensor([[False], [True]]),
            "biofp_cofactor_denominator": torch.tensor([[0.0], [1.0]]),
            "biofp_cofactor_confidence": torch.tensor([[0.0], [1.0]]),
        },
    )

    assert loss.item() == pytest.approx(0.69314718)
    assert components["biofp_mechanism_active"].item() == 2
    assert components["biofp_mechanism_active_labels"].item() == 3
    assert components["biofp_cofactor_active_labels"].item() == 1


def test_biofp_auxiliary_loss_has_rank_stable_schema_without_valid_labels():
    module = ProteinPooledLitModule(
        query_encoder_dims=[3, 6],
        target_encoder_dims=[6, 6],
        embedding_dim=6,
        residue_dim=4,
        pooling="mean",
        enzyme_input_mode="raw_mean_sleec_biological_factorized",
        hyperbolic_hyp_dim=3,
        enzyme_block_dims={
            "core": 2,
            "site": 1,
            "mechanism": 1,
            "cofactor": 1,
            "ec": 1,
        },
        enzyme_block_weights={
            "core": 0.4,
            "site": 0.2,
            "mechanism": 0.15,
            "cofactor": 0.15,
            "ec": 0.1,
        },
        biofp_family_dims={"mechanism": 2},
        biofp_family_weights={"mechanism": 1.0},
        biofp_aux_weight=0.03,
        beta=10.0,
        loss_name="FullBatchMLNCELoss",
    )
    pooling_details = {"biofp_logits_mechanism": torch.zeros(2, 2)}
    valid_targets = {
        "biofp_mechanism_targets": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "biofp_mechanism_mask": torch.ones(2, 2, dtype=torch.bool),
        "biofp_mechanism_denominator": torch.ones(2, 2),
    }
    empty_targets = {
        "biofp_mechanism_targets": torch.zeros(2, 2),
        "biofp_mechanism_mask": torch.zeros(2, 2, dtype=torch.bool),
        "biofp_mechanism_denominator": torch.zeros(2, 2),
    }

    valid_loss, valid_components = module._biofp_auxiliary_loss(
        pooling_details=pooling_details,
        biofp_targets=valid_targets,
    )
    empty_loss, empty_components = module._biofp_auxiliary_loss(
        pooling_details=pooling_details,
        biofp_targets=empty_targets,
    )

    assert valid_loss is not None
    assert empty_loss is not None
    assert empty_loss.item() == 0.0
    assert list(empty_components) == list(valid_components)
    assert empty_components["biofp_mechanism_active"].item() == 0.0
    assert empty_components["biofp_mechanism_active_labels"].item() == 0.0
    assert empty_components["weighted_biofp_mechanism"].item() == 0.0
