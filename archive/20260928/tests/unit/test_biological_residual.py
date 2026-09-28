from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from horizyn.biological_residual import (
    FUNCTIONAL_TOKEN_SCHEMA_VERSION,
    FunctionalTokenH5Dataset,
    PromiscuityAwareBiologicalResidual,
    ProtectedScoreFusion,
    select_functional_token_indices,
)
from horizyn.config import load_config
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import TupleDataset
from horizyn.losses import DecoupledAllPositiveInfoNCELoss
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_conditioned_data_module import (
    HypergraphBatchSampler,
    ReplayConcatDataset,
    _construct_replay_data_module,
    _union_positive_lookups,
)
from scripts.build_biological_residual_pretrain_split import build_strict_graph
from scripts.materialize_biological_residual_split_configs import (
    materialize_split_configs,
)
from scripts.cache_sleec_functional_tokens import _cache_is_compatible
from scripts.evaluate_biological_residual import (
    evaluate_score_matrix,
    validate_exact_base_cache,
)
from scripts.train_protein_pooling import _load_partial_model_warm_start


def test_alpha_zero_exactly_recovers_base_scores() -> None:
    base = torch.randn(4, 5)
    local = torch.randn(4, 5)
    fusion = ProtectedScoreFusion(alpha=0.0)
    fused = fusion(base, local)
    assert fused.data_ptr() == base.data_ptr()
    assert torch.equal(fused, base)


def test_stage_b_validates_often_enough_to_write_its_monitored_checkpoint() -> None:
    config = load_config("configs/reactzyme_reaction_smi_biological_residual.yaml")
    assert config.training.validation_enabled is True
    assert config.training.check_val_every_n_epoch <= config.training.max_epochs
    assert config.logging.checkpoint_on_validation_end is True


@pytest.mark.parametrize("split", ["enzyme_smi", "time"])
def test_materialized_biological_residual_configs_use_split_specific_f3(
    tmp_path,
    split,
) -> None:
    shared_source = tmp_path / "source_tokens.h5"
    shared_reactzyme = tmp_path / "reactzyme_tokens.h5"
    paths = materialize_split_configs(
        split,
        output_dir=tmp_path / split,
        shared_source_tokens=shared_source,
        shared_reactzyme_tokens=shared_reactzyme,
    )

    source = load_config(paths["source_config"])
    finetune = load_config(paths["finetune_config"])
    expected_root = f"runs/biological_residual_{split}"
    expected_f3 = (
        f"runs/reactzyme_reaction_features_v1/checkpoints/{split}/"
        "F3_set_chemistry/protein-pooling-epoch=29.ckpt"
    )

    assert source.training.init_from_checkpoint == expected_f3
    assert source.data.train_pairs_path == f"{expected_root}/data/source_pretrain/train_pairs.csv"
    assert source.data.protein_functional_tokens_path == str(shared_source)
    assert finetune.data.train_pairs_path == (
        f"data/revised_protocols/reactzyme_paper/{split}/train_pairs.csv"
    )
    assert finetune.data.validation_pairs_path == (
        f"data/revised_protocols/reactzyme_paper/{split}/validation_pairs.csv"
    )
    assert f"features/{split}/train" in finetune.data.train_reaction_unimol2_embeds_path
    assert finetune.data.protein_functional_tokens_path == str(shared_reactzyme)
    assert finetune.data.source_replay.config_path == str(paths["source_config"])
    assert finetune.training.init_from_checkpoint == (
        f"{expected_root}/checkpoints/source_pretrain/last.ckpt"
    )
    assert finetune.model.biological_residual.fusion_alpha == pytest.approx(0.05)


def test_exact_evaluation_cache_requires_float32_manifest(tmp_path) -> None:
    cache = tmp_path / "base.h5"
    with h5py.File(cache, "w") as handle:
        handle.create_dataset("vectors", data=np.ones((1, 2), dtype=np.float32))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"signature": {"precision": "32", "output_dtype": "float32"}}),
        encoding="utf-8",
    )
    validate_exact_base_cache(cache)

    reduced_cache = tmp_path / "reduced.h5"
    with h5py.File(reduced_cache, "w") as handle:
        handle.create_dataset("vectors", data=np.ones((1, 2), dtype=np.float16))
    with pytest.raises(ValueError, match="float32 cache"):
        validate_exact_base_cache(reduced_cache)


def test_functional_cache_compatibility_decodes_byte_metadata(tmp_path) -> None:
    cache = tmp_path / "functional.h5"
    metadata = {"top_k": 2, "context_k": 1}
    with h5py.File(cache, "w") as handle:
        handle.attrs["schema_version"] = FUNCTIONAL_TOKEN_SCHEMA_VERSION
        handle.attrs["metadata_json"] = np.bytes_(json.dumps(metadata))
        handle.create_dataset("ids", data=np.asarray(["e1"], dtype="S2"))
        handle.create_dataset("token_vectors", data=np.zeros((1, 3, 4), dtype=np.float16))
    assert _cache_is_compatible(cache, ["e1"], metadata)


def test_warm_start_preserves_stage_configured_fusion_alpha(tmp_path) -> None:
    class DummyBiologicalModule(torch.nn.Module):
        def __init__(self, alpha: float) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.base = torch.nn.Linear(2, 2)
            self.model.biological_residual = torch.nn.Module()
            self.model.biological_residual.fusion = ProtectedScoreFusion(alpha=alpha)
            self.training_stage = "biological_residual"

    source = DummyBiologicalModule(alpha=0.0)
    with torch.no_grad():
        source.model.base.weight.fill_(3.0)
    checkpoint = tmp_path / "stage-a.ckpt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)

    destination = DummyBiologicalModule(alpha=0.05)
    _load_partial_model_warm_start(destination, checkpoint)
    assert destination.model.biological_residual.fusion.alpha.item() == pytest.approx(0.05)
    assert torch.equal(destination.model.base.weight, source.model.base.weight)


def test_functional_selection_combines_top_and_context_without_duplicates() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    indices, mask = select_functional_token_indices(scores, valid, top_k=2, context_k=3)
    selected = indices[0, mask[0]].tolist()
    assert selected[:2] == [1, 3]
    assert len(selected) == len(set(selected)) == 5


def test_functional_token_h5_round_trip(tmp_path) -> None:
    path = tmp_path / "tokens.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = FUNCTIONAL_TOKEN_SCHEMA_VERSION
        handle.create_dataset("ids", data=np.asarray(["p1"], dtype="S2"))
        handle.create_dataset("token_vectors", data=np.ones((1, 3, 4), dtype=np.float16))
        handle.create_dataset("token_mask", data=np.asarray([[1, 1, 0]], dtype=bool))
        handle.create_dataset("sleec_scores", data=np.asarray([[0.9, 0.5, 0]], dtype=np.float16))
        handle.create_dataset("token_positions", data=np.asarray([[2, 7, 0]], dtype=np.int32))
    dataset = FunctionalTokenH5Dataset(path)
    sample = dataset["p1"]
    assert sample["biological_residue_tokens"].shape == (3, 4)
    assert sample["biological_residue_mask"].tolist() == [True, True, False]
    assert sample["biological_residue_positions"].tolist() == [2, 7, 0]


def _residual_inputs() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    reaction = {
        "reactant_embeddings": torch.randn(3, 4, 8),
        "product_embeddings": torch.randn(3, 4, 8),
        "reactant_padding_mask": torch.tensor(
            [[False, False, True, True], [False, False, False, True], [False, True, True, True]]
        ),
        "product_padding_mask": torch.tensor(
            [[False, True, True, True], [False, False, True, True], [False, False, True, True]]
        ),
        "reactant_chirality_embeddings": torch.randn(3, 4, 5),
        "product_chirality_embeddings": torch.randn(3, 4, 5),
        "reactant_chirality_padding_mask": torch.tensor(
            [[False, False, True, True], [False, False, False, True], [False, True, True, True]]
        ),
        "product_chirality_padding_mask": torch.tensor(
            [[False, True, True, True], [False, False, True, True], [False, False, True, True]]
        ),
        "has_chirality": torch.tensor([True, True, False]),
    }
    residues = torch.randn(4, 6, 7)
    residue_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, True, True],
            [True, True, False, False, False, False],
        ]
    )
    sleec = torch.rand(4, 6) * residue_mask
    return reaction, residues, residue_mask, sleec


def test_biological_residual_shapes_gradients_and_bounded_fusion() -> None:
    reaction, residues, residue_mask, sleec = _residual_inputs()
    module = PromiscuityAwareBiologicalResidual(
        residue_dim=7,
        unimol_dim=8,
        chiro_dim=5,
        token_dim=12,
        heads=3,
        layers=1,
        dropout=0.0,
        max_molecules=8,
        target_chunk_size=2,
        fusion_alpha=0.1,
    )
    base = torch.randn(3, 4)
    output = module(reaction, residues, residue_mask, sleec, base_score=base)
    assert output["local_score"].shape == (3, 4)
    assert output["fused_score"].shape == (3, 4)
    assert torch.all((output["fused_score"] - base).abs() <= 0.100001)
    output["fused_score"].sum().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_missing_unimol2_rows_exactly_fall_back_to_base_score() -> None:
    reaction, residues, residue_mask, sleec = _residual_inputs()
    reaction["has_unimol2"] = torch.tensor([True, False, True])
    module = PromiscuityAwareBiologicalResidual(
        residue_dim=7,
        unimol_dim=8,
        chiro_dim=5,
        token_dim=12,
        heads=3,
        layers=1,
        dropout=0.0,
        fusion_alpha=0.1,
    ).eval()
    base = torch.randn(3, 4)
    output = module(reaction, residues, residue_mask, sleec, base_score=base)
    assert torch.equal(output["local_score"][1], torch.zeros_like(output["local_score"][1]))
    assert torch.equal(output["fused_score"][1], base[1])


def test_unimol2_and_chiro_sets_do_not_require_molecule_alignment() -> None:
    reaction, residues, residue_mask, sleec = _residual_inputs()
    reaction["reactant_chirality_embeddings"] = torch.randn(3, 2, 5)
    reaction["product_chirality_embeddings"] = torch.randn(3, 3, 5)
    reaction["reactant_chirality_padding_mask"] = torch.tensor(
        [[False, True], [False, False], [False, True]]
    )
    reaction["product_chirality_padding_mask"] = torch.tensor(
        [[False, True, True], [False, False, True], [False, False, False]]
    )
    module = PromiscuityAwareBiologicalResidual(
        residue_dim=7,
        unimol_dim=8,
        chiro_dim=5,
        token_dim=12,
        heads=3,
        layers=1,
        dropout=0.0,
        max_molecules=16,
    ).eval()
    output = module(reaction, residues, residue_mask, sleec)
    assert output["local_score"].shape == (3, 4)
    assert torch.isfinite(output["local_score"]).all()


def test_molecule_set_encoding_is_permutation_invariant() -> None:
    reaction, residues, residue_mask, sleec = _residual_inputs()
    module = PromiscuityAwareBiologicalResidual(
        residue_dim=7,
        unimol_dim=8,
        chiro_dim=5,
        token_dim=12,
        heads=3,
        layers=1,
        dropout=0.0,
        max_molecules=8,
    ).eval()
    first = module(reaction, residues, residue_mask, sleec)["local_score"]
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(reaction)
    for key in (
        "reactant_embeddings",
        "reactant_padding_mask",
        "reactant_chirality_embeddings",
        "reactant_chirality_padding_mask",
    ):
        permuted[key] = reaction[key][:, permutation]
    second = module(permuted, residues, residue_mask, sleec)["local_score"]
    assert torch.allclose(first, second, atol=2e-6, rtol=2e-6)


def test_capped_molecule_set_encoding_remains_permutation_invariant() -> None:
    reaction, residues, residue_mask, sleec = _residual_inputs()
    module = PromiscuityAwareBiologicalResidual(
        residue_dim=7,
        unimol_dim=8,
        chiro_dim=5,
        token_dim=12,
        heads=3,
        layers=1,
        dropout=0.0,
        max_molecules=3,
    ).eval()
    first = module(reaction, residues, residue_mask, sleec)["local_score"]
    permuted = dict(reaction)
    reactant_permutation = torch.tensor([2, 0, 3, 1])
    product_permutation = torch.tensor([1, 3, 0, 2])
    for prefix, permutation in (
        ("reactant", reactant_permutation),
        ("product", product_permutation),
    ):
        for suffix in (
            "embeddings",
            "padding_mask",
            "chirality_embeddings",
            "chirality_padding_mask",
        ):
            key = f"{prefix}_{suffix}"
            permuted[key] = reaction[key][:, permutation]
    second = module(permuted, residues, residue_mask, sleec)["local_score"]
    assert torch.allclose(first, second, atol=2e-6, rtol=2e-6)

    first.sum().backward()
    pool_gradient = module.reaction_encoder.pool_score.weight.grad
    assert pool_gradient is not None
    assert torch.isfinite(pool_gradient).all()


def test_unknown_negative_weight_reduces_unlabelled_negative_pressure() -> None:
    dists = torch.tensor([[0.1, 0.3, 0.5], [0.4, 0.1, 0.2]])
    query_idx = torch.tensor([0, 1])
    target_idx = torch.tensor([0, 1])
    full = DecoupledAllPositiveInfoNCELoss(beta=5.0, unknown_negative_weight=1.0)
    pu = DecoupledAllPositiveInfoNCELoss(beta=5.0, unknown_negative_weight=0.25)
    assert pu(dists, query_idx, target_idx) < full(dists, query_idx, target_idx)
    with pytest.raises(ValueError):
        DecoupledAllPositiveInfoNCELoss(unknown_negative_weight=0.0)


def test_hypergraph_sampler_keeps_grouped_positive_rows_in_local_batch() -> None:
    pair_rows = [
        {"query_id": "r1", "target_id": "e1"},
        {"query_id": "r1", "target_id": "e2"},
        {"query_id": "r2", "target_id": "e1"},
        {"query_id": "r3", "target_id": "e1"},
        {"query_id": "r4", "target_id": "e4"},
        {"query_id": "r5", "target_id": "e5"},
    ]
    pairs = BaseDataset(keys=[str(i) for i in range(len(pair_rows))], array_data=pair_rows)
    query_data = BaseDataset(keys=[f"r{i}" for i in range(1, 6)], array_data=list(range(5)))
    target_data = BaseDataset(keys=[f"e{i}" for i in range(1, 6)], array_data=list(range(5)))
    dataset = TupleDataset(
        tuple_dataset=pairs,
        key_name_to_dataset={"query_id": query_data, "target_id": target_data},
    )
    sampler = HypergraphBatchSampler(
        dataset,
        batch_size=6,
        anchors_per_batch=2,
        positives_per_anchor=2,
        reaction_anchor_fraction=0.5,
        seed=3,
    )
    batch = next(iter(sampler))
    selected = [pair_rows[index] for index in batch]
    reaction_counts = {value: 0 for value in ("r1", "r2", "r3", "r4", "r5")}
    enzyme_counts = {value: 0 for value in ("e1", "e2", "e3", "e4", "e5")}
    for row in selected:
        reaction_counts[row["query_id"]] += 1
        enzyme_counts[row["target_id"]] += 1
    assert max(reaction_counts.values()) >= 2
    assert max(enzyme_counts.values()) >= 2


def _pair_dataset(prefix: str, pair_rows: list[dict[str, str]]) -> TupleDataset:
    pairs = BaseDataset(
        keys=[f"{prefix}-{index}" for index in range(len(pair_rows))],
        array_data=pair_rows,
    )
    query_ids = sorted({row["query_id"] for row in pair_rows})
    target_ids = sorted({row["target_id"] for row in pair_rows})
    query_data = BaseDataset(
        keys=query_ids,
        array_data=[{"feature": torch.tensor([float(index)])} for index in range(len(query_ids))],
    )
    target_data = BaseDataset(
        keys=target_ids,
        array_data=[{"target": torch.tensor([float(index)])} for index in range(len(target_ids))],
    )
    return TupleDataset(
        tuple_dataset=pairs,
        key_name_to_dataset={"query_id": query_data, "target_id": target_data},
    )


def test_replay_concat_preserves_source_pair_metadata_and_fraction() -> None:
    primary = _pair_dataset(
        "primary",
        [
            {"query_id": "r1", "target_id": "e1"},
            {"query_id": "r1", "target_id": "e2"},
        ],
    )
    replay = _pair_dataset(
        "source",
        [
            {"query_id": "s1", "target_id": "p1"},
            {"query_id": "s1", "target_id": "p2"},
        ],
    )
    combined = ReplayConcatDataset(primary, replay, replay_fraction=0.5, seed=11)
    assert len(combined) == 4
    assert combined.realized_replay_fraction == 0.5
    assert combined["replay:source-0"]["query_id"] == "s1"
    sampler = HypergraphBatchSampler(
        combined,
        batch_size=4,
        anchors_per_batch=1,
        positives_per_anchor=2,
        reaction_anchor_fraction=1.0,
        seed=1,
    )
    assert "s1" not in sampler.reaction_hyperedges
    batch = next(iter(sampler))
    assert sum(index >= len(primary) for index in batch) == 2


def test_replay_positive_mask_retains_unsampled_source_annotations() -> None:
    merged = _union_positive_lookups(
        {"r1_f": ["e1"]},
        {"r1_f": ["e2"], "source_f": ["p1", "p2"]},
    )
    assert merged == {
        "r1_f": ["e1", "e2"],
        "source_f": ["p1", "p2"],
    }


def test_replay_data_module_construction_avoids_lightning_frame_error() -> None:
    module = _construct_replay_data_module(
        {
            "train_pairs_path": "train_pairs.csv",
            "test_pairs_path": "test_pairs.csv",
            "train_reactions_path": "train_reactions.csv",
            "test_reactions_path": "test_reactions.csv",
            "protein_residue_embeds_path": "proteins.h5",
            "validation_enabled": False,
        }
    )
    assert module.train_pairs_path.name == "train_pairs.csv"


def test_strict_source_graph_removes_heldout_chemistry_and_protein_identity() -> None:
    reactions = [
        {"reaction_id": "s1", "reaction_smiles": "CCO>>CC=O"},
        {"reaction_id": "s2", "reaction_smiles": "CCN>>CC=N"},
        {"reaction_id": "s3", "reaction_smiles": "CCC>>CC=C"},
        {"reaction_id": "s4", "reaction_smiles": "CO>>C=O"},
    ]
    pairs = [
        {"reaction_id": "s1", "protein_id": "uprot_safe"},
        {"reaction_id": "s2", "protein_id": "uprot_heldout"},
        {"reaction_id": "s3", "protein_id": "uprot_safe"},
        {"reaction_id": "s3", "protein_id": "uprot_safe"},
        {"reaction_id": "s4", "protein_id": "uprot_heldout"},
    ]
    heldout_pairs = [
        {
            "reaction_id": "h1",
            "reaction_smiles": "CCN.CC=N",
            "protein_id": "prot_heldout",
        }
    ]
    heldout_reactions = [
        {"reaction_id": "h2", "reaction_smiles": "CC=O.CCO"}
    ]
    kept_pairs, kept_reactions, counters = build_strict_graph(
        pairs, reactions, heldout_pairs, heldout_reactions
    )
    assert kept_pairs == [{"reaction_id": "s3", "protein_id": "uprot_safe"}]
    assert kept_reactions == [reactions[2]]
    assert counters["heldout_reaction"] == 2
    assert counters["heldout_protein"] == 1
    assert counters["duplicate_edge"] == 1


def test_cached_biological_training_path_has_finite_residual_gradients() -> None:
    torch.manual_seed(13)
    module = ProteinPooledLitModule(
        query_encoder_dims=[4, 4],
        target_encoder_dims=[7, 4],
        embedding_dim=4,
        residue_dim=7,
        pooling="mean",
        query_encoder_type="mlp",
        loss_name="DecoupledAllPositiveInfoNCELoss",
        positive_pair_source="observed_pairs",
        unknown_negative_weight=0.25,
        training_stage="biological_residual",
        biological_residual_enabled=True,
        biological_residual_token_dim=12,
        biological_residual_heads=3,
        biological_residual_layers=1,
        biological_residual_dropout=0.0,
        biological_residual_max_molecules=6,
        biological_residual_fusion_alpha=0.05,
        biological_residual_fused_loss_weight=1.0,
        biological_residual_local_loss_weight=0.2,
        biological_residual_guard_weight=0.1,
        reaction_unimol_dim=8,
        reaction_chienn_dim=5,
    )
    query_ids = ["q1", "q1", "q2", "q2"]
    target_ids = ["e1", "e2", "e1", "e3"]
    query_base_by_id = {name: torch.randn(4) for name in set(query_ids)}
    target_base_by_id = {name: torch.randn(4) for name in set(target_ids)}
    reaction_by_id = {
        name: (torch.randn(2, 8), torch.randn(1, 8)) for name in set(query_ids)
    }
    residue_by_id = {
        name: (torch.randn(5, 7), torch.rand(5)) for name in set(target_ids)
    }
    batch = {
        "query_id": query_ids,
        "target_id": target_ids,
        "cached_query_embedding": torch.stack([query_base_by_id[name] for name in query_ids]),
        "cached_target_embedding": torch.stack([target_base_by_id[name] for name in target_ids]),
        "reactant_embeddings": torch.stack([reaction_by_id[name][0] for name in query_ids]),
        "product_embeddings": torch.stack([reaction_by_id[name][1] for name in query_ids]),
        "reactant_padding_mask": torch.zeros(4, 2, dtype=torch.bool),
        "product_padding_mask": torch.zeros(4, 1, dtype=torch.bool),
        "biological_residue_tokens": torch.stack(
            [residue_by_id[name][0] for name in target_ids]
        ),
        "biological_residue_mask": torch.ones(4, 5, dtype=torch.bool),
        "biological_sleec_scores": torch.stack(
            [residue_by_id[name][1] for name in target_ids]
        ),
    }
    loss, count, _stats, components = module._compute_cached_full_batch_loss(
        batch,
        positive_pair_source="observed_pairs",
        structure_terms_enabled=False,
    )
    assert count == 4
    assert torch.isfinite(loss)
    assert torch.isfinite(components["ranking_guard"])
    loss.backward()
    residual_parameters = list(module.model.biological_residual.parameters())
    assert any(parameter.grad is not None for parameter in residual_parameters)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in residual_parameters
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in module.model.named_parameters()
        if not name.startswith("biological_residual.")
    )


def test_biological_evaluator_computes_both_graph_directions() -> None:
    scores = torch.tensor([[3.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    metrics = evaluate_score_matrix(
        scores,
        ["q1", "q2"],
        ["e1", "e2", "e3"],
        {"q1": ["e1", "e2"], "q2": ["e3"]},
    )
    assert metrics["reaction_to_enzyme/first_positive_mrr"] == 1.0
    assert metrics["enzyme_to_reaction/first_positive_mrr"] == 1.0
    assert metrics["balanced_first_positive_mrr"] == 1.0
