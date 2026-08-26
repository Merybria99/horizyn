import numpy as np
import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
h5py = pytest.importorskip("h5py")

from horizyn.capability.enzyme_capability_dataset import (
    CapabilityVectorDataset,
    EnzymeCapabilityPretrainDataset,
    TargetWithCapabilityDataset,
    enzyme_capability_collate,
)
from horizyn.capability.enzyme_capability_model import (
    EnzymeCapabilityEncoder,
    EnzymeCapabilityLitModule,
)
from horizyn.capability.capability_losses import (
    BiologicalFactorSupConLoss,
    CompositeBiologicalSupConLoss,
    multilabel_bce_or_zero,
    positive_masks_from_pair_maps,
)
from horizyn.datasets.base import BaseDataset
from horizyn.model import GatedEnzymeFeatureFusion
from scripts.export_enzyme_capability_vectors import _infer_biological_family_names
from scripts.pretrain_enzyme_capability import split_dataset_by_enzyme_id


def test_capability_encoder_output_is_normalized():
    model = EnzymeCapabilityEncoder(
        input_dims={"prot5_mean": 8, "prot5_sleec": 8, "lorentz_tangent": 4},
        capability_dim=6,
        hidden_dim=12,
        dropout=0.0,
    )
    out = model(
        {
            "prot5_mean": torch.randn(3, 8),
            "prot5_sleec": torch.randn(3, 8),
            "lorentz_tangent": torch.randn(3, 4),
        }
    )
    assert out.shape == (3, 6)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_factorized_capability_encoder_returns_normalized_family_vectors():
    model = EnzymeCapabilityEncoder(
        input_dims={"prot5_mean": 8, "prot5_sleec": 8, "lorentz_tangent": 4},
        capability_dim=10,
        hidden_dim=12,
        dropout=0.0,
        capability_mode="factorized_biological",
        family_dim=5,
    )
    out, families = model(
        {
            "prot5_mean": torch.randn(3, 8),
            "prot5_sleec": torch.randn(3, 8),
            "lorentz_tangent": torch.randn(3, 4),
        },
        return_family_vectors=True,
    )
    assert out.shape == (3, 10)
    assert set(families) == {
        "global",
        "cofactor",
        "reaction_center",
        "substrate",
        "product",
    }
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)
    for value in families.values():
        assert value.shape == (3, 5)
        assert torch.allclose(value.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_factorized_capability_encoder_uses_multiquery_residue_pooling():
    model = EnzymeCapabilityEncoder(
        input_dims={"prot5_mean": 8, "prot5_sleec": 8, "lorentz_tangent": 4},
        capability_dim=10,
        hidden_dim=12,
        dropout=0.0,
        capability_mode="factorized_biological",
        family_dim=5,
        residue_input_dim=8,
        use_residue_multiquery_pooling=True,
        residue_pool_scale=0.1,
        biological_family_names=(
            "global",
            "cofactor",
            "transition",
            "reaction_center",
            "substrate",
            "product",
        ),
    )
    out, families = model(
        {
            "prot5_mean": torch.randn(3, 8),
            "prot5_sleec": torch.randn(3, 8),
            "lorentz_tangent": torch.randn(3, 4),
        },
        residue_embeddings=torch.randn(3, 7, 8),
        residue_mask=torch.ones(3, 7, dtype=torch.bool),
        return_family_vectors=True,
    )
    assert out.shape == (3, 10)
    assert set(families) == {
        "global",
        "cofactor",
        "transition",
        "reaction_center",
        "substrate",
        "product",
    }
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(
        model.residue_pooler.last_attention_weights.sum(dim=-1),
        torch.ones(3, 6),
        atol=1e-5,
    )


def test_checkpoint_family_name_inference_supports_legacy_and_stage1():
    legacy_state = {
        "enzyme_encoder.family_heads.global.weight": torch.empty(1, 1),
        "enzyme_encoder.family_heads.cofactor.weight": torch.empty(1, 1),
        "enzyme_encoder.family_heads.reaction_center.weight": torch.empty(1, 1),
        "enzyme_encoder.family_heads.substrate.weight": torch.empty(1, 1),
        "enzyme_encoder.family_heads.product.weight": torch.empty(1, 1),
    }
    assert _infer_biological_family_names({}, legacy_state) == (
        "global",
        "cofactor",
        "reaction_center",
        "substrate",
        "product",
    )
    assert _infer_biological_family_names(
        {"biological_family_names": ["global", "transition"]},
        legacy_state,
    ) == ("global", "transition")


def test_enzyme_level_validation_split_has_no_overlap():
    class TinyDataset:
        def __init__(self):
            self.pairs = pd.DataFrame(
                {
                    "enzyme_id": ["e1", "e1", "e2", "e2", "e3", "e4"],
                    "reaction_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
                }
            )

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            return self.pairs.iloc[idx].to_dict()

    dataset = TinyDataset()
    train_subset, val_subset, stats = split_dataset_by_enzyme_id(dataset, 0.5, 13)
    train_enzymes = set(dataset.pairs.iloc[train_subset.indices]["enzyme_id"].astype(str))
    val_enzymes = set(dataset.pairs.iloc[val_subset.indices]["enzyme_id"].astype(str))
    assert not train_enzymes & val_enzymes
    assert stats["train_val_enzyme_overlap"] == 0


def test_exported_capability_vector_shape(tmp_path):
    path = tmp_path / "capability.npz"
    np.savez_compressed(path, ids=np.array(["e1", "e2"]), vectors=np.zeros((2, 256), dtype=np.float32))
    dataset = CapabilityVectorDataset(path)
    assert len(dataset) == 2
    assert dataset["e1"].shape == (256,)


def test_missing_capability_vector_preserves_target_and_masks(tmp_path):
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
    assert merged["e1"]["capability_mask"].item() is True
    assert merged["e2"]["capability_mask"].item() is False
    assert torch.allclose(merged["e2"]["capability_vec"], torch.zeros(4))


def test_missing_capability_vector_masks_gate_branch():
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
    assert gates.shape == (2, 4)
    assert gates[1, 3].item() == pytest.approx(0.0, abs=1e-6)


def test_positive_set_loss_multi_positive_from_train_maps():
    r2e, e2r = positive_masks_from_pair_maps(
        ["r1", "r2"],
        ["e1", "e2", "e3"],
        reaction_to_enzymes={"r1": {"e1", "e2"}, "r2": {"e3"}},
        enzyme_to_reactions={"e1": {"r1"}, "e2": {"r1"}, "e3": {"r2"}},
    )
    assert r2e.tolist() == [[True, True, False], [False, False, True]]
    assert e2r.tolist() == [[True, False], [True, False], [False, True]]


def test_attribute_loss_masks_missing_labels():
    logits = torch.tensor([[10.0], [-10.0]])
    targets = torch.tensor([[1.0], [1.0]])
    sample_mask = torch.tensor([True, False])
    loss = multilabel_bce_or_zero(logits, targets, sample_mask)
    assert loss.item() < 1e-3


def test_biological_factor_supcon_uses_shared_labels_as_positives():
    loss_fn = BiologicalFactorSupConLoss(temperature=0.1)
    embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ]
        ),
        dim=-1,
    )
    labels = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    result = loss_fn(embeddings, labels, torch.ones(3, dtype=torch.bool))
    assert result["metrics/valid_anchors"].item() == 2
    assert result["metrics/mean_positives"].item() == pytest.approx(1.0)
    assert result["loss"].item() < 0.1


def test_biological_factor_supcon_masks_missing_labels():
    loss_fn = BiologicalFactorSupConLoss(temperature=0.1)
    embeddings = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    labels = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )
    mask = torch.tensor([True, True, False])
    result = loss_fn(embeddings, labels, mask)
    assert result["metrics/valid_anchors"].item() == 2
    assert result["metrics/known_samples"].item() == 2
    assert result["metrics/masked_missing"].item() == 1


def test_composite_biological_supcon_requires_multi_family_context():
    loss_fn = CompositeBiologicalSupConLoss(
        temperature=0.1,
        positive_threshold=0.2,
        min_positive_families=2,
        min_known_families=2,
    )
    embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        dim=-1,
    )
    labels_by_family = {
        "reaction_center": torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        "cofactor": torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        "substrate": torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        "product": torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
    }
    masks_by_family = {
        family_name: torch.ones(4, dtype=torch.bool)
        for family_name in labels_by_family
    }
    result = loss_fn(embeddings, labels_by_family, masks_by_family)
    assert torch.isfinite(result["loss"])
    assert result["metrics/valid_anchors"].item() == 4
    assert result["metrics/mean_positives"].item() == pytest.approx(1.0)
    assert result["metrics/mean_hard_negatives"].item() > 0


def test_stage1_composite_requires_transition_or_core_cofactor():
    loss_fn = CompositeBiologicalSupConLoss(
        temperature=0.1,
        positive_threshold=0.35,
        min_positive_families=2,
        min_known_families=2,
        family_weights={
            "cofactor": 0.35,
            "transition": 0.40,
            "substrate": 0.15,
            "product": 0.10,
            "reaction_center": 0.0,
        },
        positive_family_names=("cofactor", "transition", "substrate", "product"),
        known_family_names=("cofactor", "transition", "substrate", "product"),
        required_positive_families=("cofactor", "transition"),
    )
    embeddings = torch.nn.functional.normalize(torch.randn(4, 5), dim=-1)
    labels_by_family = {
        "reaction_center": torch.tensor(
            [[1.0], [1.0], [1.0], [0.0]]
        ),
        "cofactor": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        "transition": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        "substrate": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        ),
        "product": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        ),
    }
    masks_by_family = {
        family_name: torch.ones(4, dtype=torch.bool)
        for family_name in labels_by_family
    }
    result = loss_fn(embeddings, labels_by_family, masks_by_family)
    assert torch.isfinite(result["loss"])
    assert result["metrics/valid_anchors"].item() == 4
    assert result["metrics/mean_positives"].item() == pytest.approx(1.0)


def test_reaction_demand_bio_composite_objective_backpropagates():
    model = EnzymeCapabilityLitModule(
        enzyme_input_dims={"prot5_mean": 8, "prot5_sleec": 8, "lorentz_tangent": 4},
        reaction_demand_dim=12,
        label_output_dims={
            "combined_core_cofactor_targets": 2,
            "reaction_center_targets": 2,
            "substrate_targets": 2,
            "product_targets": 2,
        },
        capability_dim=8,
        hidden_dim=16,
        family_dim=8,
        dropout=0.0,
        capability_mode="factorized_biological",
        pretraining_objective="reaction_demand_bio_composite",
        r2e_capability_weight=1.0,
        e2r_capability_weight=0.3,
        cofactor_subspace_weight=0.2,
        substrate_subspace_weight=0.1,
        product_subspace_weight=0.1,
        bio_global_weight=0.0,
        bio_cofactor_weight=0.1,
        bio_reaction_center_weight=0.0,
        bio_substrate_weight=0.05,
        bio_product_weight=0.05,
        bio_composite_weight=0.35,
        same_center_hard_negative_weight=0.2,
        bio_attribute_weight=0.05,
    )
    model.log = lambda *args, **kwargs: None
    model.reaction_to_enzymes = {
        "r1": {"e1", "e2"},
        "r2": {"e2"},
        "r3": {"e3"},
        "r4": {"e4"},
    }
    model.enzyme_to_reactions = {
        "e1": {"r1"},
        "e2": {"r1", "r2"},
        "e3": {"r3"},
        "e4": {"r4"},
    }
    model.reaction_family_masks = {
        rid: {
            "cofactor": True,
            "reaction_center": True,
            "substrate": True,
            "product": True,
        }
        for rid in model.reaction_to_enzymes
    }
    model.enzyme_family_masks = {
        eid: {
            "cofactor": True,
            "reaction_center": True,
            "substrate": True,
            "product": True,
        }
        for eid in model.enzyme_to_reactions
    }
    batch = {
        "enzyme_id": ["e1", "e2", "e3", "e4"],
        "reaction_id": ["r1", "r2", "r3", "r4"],
        "prot5_mean": torch.randn(4, 8),
        "prot5_sleec": torch.randn(4, 8),
        "lorentz_tangent": torch.randn(4, 4),
        "reaction_demand_vec": torch.randn(4, 12),
        "combined_core_cofactor_targets": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        "combined_core_cofactor_targets_mask": torch.ones(4, dtype=torch.bool),
        "reaction_center_targets": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        ),
        "reaction_center_targets_mask": torch.ones(4, dtype=torch.bool),
        "substrate_targets": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        "substrate_targets_mask": torch.ones(4, dtype=torch.bool),
        "product_targets": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        "product_targets_mask": torch.ones(4, dtype=torch.bool),
    }
    loss = model._shared_step(batch, "train")
    loss.backward()
    assert torch.isfinite(loss)


def test_pretrain_dataset_returns_positive_pair(tmp_path):
    pair_path = tmp_path / "pairs.parquet"
    pd.DataFrame(
        [
            {
                "enzyme_id": "e1",
                "reaction_id": "r1",
                "is_positive": 1,
                "reaction_demand_vector_id": "r1",
                "enzyme_capability_label_id": "e1",
                "source_split": "train",
            }
        ]
    ).to_parquet(pair_path, index=False)
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame(
        [
            {
                "enzyme_id": "e1",
                "cofactor_labels_train": ["NAD"],
                "reaction_center_labels_train": [],
                "substrate_class_labels_train": [],
                "product_class_labels_train": [],
                "reaction_type_labels_train": ["oxidoreduction"],
                "ec_numbers_train": ["1.1.1.1"],
            }
        ]
    ).to_parquet(labels_path, index=False)
    demand_path = tmp_path / "demand.npz"
    np.savez_compressed(demand_path, ids=np.array(["r1"]), vectors=np.ones((1, 12), dtype=np.float32))
    reaction_features_path = tmp_path / "reaction_features.parquet"
    pd.DataFrame(
        [
            {
                "reaction_id": "r1",
                "canonical_reaction_smiles": "CCO>>CC=O",
                "core_cofactor_labels": ["NAD"],
                "cofactor_labels": ["NAD"],
                "metal_ion_labels": [],
                "reaction_center_coarse_labels": ["redox_like"],
                "reaction_center_raw_labels": ["bond_order_change_C_O"],
                "substrate_class_labels": ["alcohol"],
                "product_class_labels": ["ketone"],
            }
        ]
    ).to_parquet(reaction_features_path, index=False)
    for name, dim in {"prot5_mean": 8, "prot5_sleec": 8, "lorentz_tangent": 4}.items():
        np.savez_compressed(
            tmp_path / f"{name}.npz",
            ids=np.array(["e1"]),
            vectors=np.ones((1, dim), dtype=np.float32),
        )
    dataset = EnzymeCapabilityPretrainDataset(
        pair_capability_training_path=pair_path,
        reaction_demand_vectors_path=demand_path,
        enzyme_capability_labels_path=labels_path,
        enzyme_feature_paths={
            "prot5_mean": tmp_path / "prot5_mean.npz",
            "prot5_sleec": tmp_path / "prot5_sleec.npz",
            "lorentz_tangent": tmp_path / "lorentz_tangent.npz",
        },
        enzyme_label_vocabs={
            "cofactor_labels": ["NAD"],
            "core_cofactor_labels": ["NAD"],
            "reaction_center_labels": [],
            "substrate_class_labels": [],
            "product_class_labels": [],
            "reaction_type_labels": ["oxidoreduction"],
            "ec_labels": ["1.1.1.1"],
        },
        reaction_features_path=reaction_features_path,
    )
    sample = dataset[0]
    assert sample["enzyme_id"] == "e1"
    assert sample["reaction_id"] == "r1"
    assert sample["reaction_demand_vec"].shape == (12,)
    assert bool(sample["reaction_cofactor_mask"])
    assert bool(sample["reaction_reaction_center_mask"])
    assert bool(sample["reaction_substrate_mask"])
    assert bool(sample["reaction_product_mask"])
    assert bool(sample["enzyme_cofactor_mask"])


def test_pretrain_dataset_collates_multiquery_residue_embeddings(tmp_path):
    pair_path = tmp_path / "pairs.parquet"
    pd.DataFrame(
        [
            {
                "enzyme_id": "e1",
                "reaction_id": "r1",
                "is_positive": 1,
                "reaction_demand_vector_id": "r1",
                "enzyme_capability_label_id": "e1",
                "source_split": "train",
            },
            {
                "enzyme_id": "e2",
                "reaction_id": "r2",
                "is_positive": 1,
                "reaction_demand_vector_id": "r2",
                "enzyme_capability_label_id": "e2",
                "source_split": "train",
            },
        ]
    ).to_parquet(pair_path, index=False)
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame(
        [
            {
                "enzyme_id": "e1",
                "cofactor_labels_train": ["NAD"],
                "reaction_center_labels_train": ["redox_like"],
                "substrate_class_labels_train": ["alcohol"],
                "product_class_labels_train": ["ketone"],
                "reaction_type_labels_train": ["oxidoreduction"],
                "ec_numbers_train": ["1.1.1.1"],
            },
            {
                "enzyme_id": "e2",
                "cofactor_labels_train": ["PLP"],
                "reaction_center_labels_train": ["c_n_change"],
                "substrate_class_labels_train": ["amino_acid_like"],
                "product_class_labels_train": ["keto_acid_like"],
                "reaction_type_labels_train": ["transamination"],
                "ec_numbers_train": ["2.6.1.1"],
            },
        ]
    ).to_parquet(labels_path, index=False)
    demand_path = tmp_path / "demand.npz"
    np.savez_compressed(
        demand_path,
        ids=np.array(["r1", "r2"]),
        vectors=np.ones((2, 12), dtype=np.float32),
    )
    for name, dim in {"prot5_mean": 8, "prot5_sleec": 8, "lorentz_tangent": 4}.items():
        np.savez_compressed(
            tmp_path / f"{name}.npz",
            ids=np.array(["e1", "e2"]),
            vectors=np.ones((2, dim), dtype=np.float32),
        )
    residue_path = tmp_path / "prot5_residue.h5"
    with h5py.File(residue_path, "w") as h5:
        h5.create_dataset("ids", data=np.array([b"e1", b"e2"]))
        h5.create_dataset("offsets", data=np.array([0, 3, 8], dtype=np.int64))
        h5.create_dataset("vectors", data=np.ones((8, 8), dtype=np.float32))
    dataset = EnzymeCapabilityPretrainDataset(
        pair_capability_training_path=pair_path,
        reaction_demand_vectors_path=demand_path,
        enzyme_capability_labels_path=labels_path,
        enzyme_feature_paths={
            "prot5_mean": tmp_path / "prot5_mean.npz",
            "prot5_sleec": tmp_path / "prot5_sleec.npz",
            "lorentz_tangent": tmp_path / "lorentz_tangent.npz",
        },
        enzyme_residue_path=residue_path,
        enzyme_label_vocabs={
            "cofactor_labels": ["NAD", "PLP"],
            "reaction_center_labels": ["redox_like", "c_n_change"],
            "substrate_class_labels": ["alcohol", "amino_acid_like"],
            "product_class_labels": ["ketone", "keto_acid_like"],
            "reaction_type_labels": ["oxidoreduction", "transamination"],
            "ec_labels": ["1.1.1.1", "2.6.1.1"],
        },
    )
    batch = enzyme_capability_collate([dataset[0], dataset[1]])
    assert batch["prot5_residue_embeddings"].shape == (2, 5, 8)
    assert batch["prot5_residue_mask"].tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
