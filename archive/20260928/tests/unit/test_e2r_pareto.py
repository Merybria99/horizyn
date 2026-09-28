import json

import pytest
import torch

from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import TupleDataset
from horizyn.losses import MultiAlignmentRetrievalLoss
from horizyn.model import (
    E2RReactionAdapter,
    ProteinPooledDualModel,
    R2EEnzymeAdapter,
)
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_conditioned_data_module import (
    DirectionalHardNegativeBatchSampler,
)
from horizyn.utils.collate import residue_collate_fn


def test_e2r_adapter_changes_only_the_requested_retrieval_direction():
    model = ProteinPooledDualModel(
        query_encoder_kwargs={
            "input_dim": 4,
            "output_dim": 4,
            "num_layers": 0,
            "normalise_output": True,
        },
        target_encoder_kwargs={
            "input_dim": 4,
            "output_dim": 4,
            "num_layers": 0,
            "normalise_output": True,
        },
        residue_dim=4,
        pooling="mean",
        e2r_adapter_enabled=True,
        e2r_adapter_hidden_dim=8,
        e2r_adapter_dropout=0.0,
        e2r_adapter_gate_init=0.2,
    )
    model.eval()
    inputs = torch.randn(3, 4)

    baseline = model.query_encoder(inputs)
    reaction_to_enzyme = model.encode_queries(
        inputs,
        retrieval_direction="reaction_to_enzyme",
    )
    enzyme_to_reaction = model.encode_queries(
        inputs,
        retrieval_direction="enzyme_to_reaction",
    )

    assert torch.equal(reaction_to_enzyme, baseline)
    assert not torch.allclose(enzyme_to_reaction, baseline)
    assert torch.allclose(
        enzyme_to_reaction.norm(dim=-1),
        torch.ones(3),
        atol=1e-6,
    )


def test_factorized_directional_adapter_uses_all_enabled_inputs():
    adapter = E2RReactionAdapter(
        embedding_dim=16,
        hidden_dim=24,
        dropout=0.0,
        gate_init=0.1,
        use_factorized_inputs=True,
        use_directional_inputs=True,
        reaction_model_dim=7,
        unimol_dim=5,
        chiro_dim=3,
        chemistry_dim=11,
        directional_dim=13,
        directional_hidden_dim=6,
    )
    base = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
    query_inputs = {
        "reaction_embedding": torch.randn(2, 7),
        "reactant_embeddings": torch.randn(2, 2, 5),
        "product_embeddings": torch.randn(2, 3, 5),
        "reactant_padding_mask": torch.tensor([[False, True], [False, False]]),
        "product_padding_mask": torch.tensor([[False, False, True], [False, True, True]]),
        "reactant_chirality_embeddings": torch.randn(2, 2, 3),
        "product_chirality_embeddings": torch.randn(2, 3, 3),
        "reactant_chirality_padding_mask": torch.tensor([[False, True], [False, False]]),
        "product_chirality_padding_mask": torch.tensor([[False, False, True], [False, True, True]]),
        "reaction_chemistry_vector": torch.randn(2, 11),
        "reaction_directional_vector": torch.randn(2, 13),
        "has_reaction_directional": torch.tensor([True, False]),
    }

    output, details = adapter(base, query_inputs, return_details=True)
    output.sum().backward()

    assert output.shape == (2, 16)
    assert details["e2r_adapter_gate"].shape == (2,)
    assert adapter.raw_gate.grad is not None
    assert adapter.directional_projection[1].weight.grad is not None
    for projection in adapter.factorized_projections.values():
        assert projection[1].weight.grad is not None


def test_r2e_adapter_changes_only_reaction_to_enzyme_targets():
    model = ProteinPooledDualModel(
        query_encoder_kwargs={
            "input_dim": 4,
            "output_dim": 4,
            "num_layers": 0,
            "normalise_output": True,
        },
        target_encoder_kwargs={
            "input_dim": 4,
            "output_dim": 4,
            "num_layers": 0,
            "normalise_output": True,
        },
        residue_dim=4,
        pooling="mean",
        r2e_adapter_enabled=True,
        r2e_adapter_hidden_dim=8,
        r2e_adapter_dropout=0.0,
        r2e_adapter_gate_init=0.2,
    )
    model.eval()
    residues = torch.randn(3, 5, 4)
    mask = torch.zeros(3, 5, dtype=torch.bool)

    pooled = model.pool_residues(residues, residue_padding_mask=mask)
    baseline = model.target_encoder(pooled)
    reaction_to_enzyme = model.encode_targets(
        residues,
        residue_padding_mask=mask,
        retrieval_direction="reaction_to_enzyme",
    )
    enzyme_to_reaction = model.encode_targets(
        residues,
        residue_padding_mask=mask,
        retrieval_direction="enzyme_to_reaction",
    )

    assert torch.equal(enzyme_to_reaction, baseline)
    assert not torch.allclose(reaction_to_enzyme, baseline)
    assert torch.allclose(
        reaction_to_enzyme.norm(dim=-1),
        torch.ones(3),
        atol=1e-6,
    )


def test_factorized_r2e_adapter_uses_every_enzyme_block():
    adapter = R2EEnzymeAdapter(
        embedding_dim=16,
        hidden_dim=24,
        dropout=0.0,
        gate_init=0.1,
        use_factorized_inputs=True,
        block_dims={"core": 8, "site": 4, "mechanism": 2, "cofactor": 1, "ec": 1},
        block_weights={
            "core": 0.55,
            "site": 0.20,
            "mechanism": 0.12,
            "cofactor": 0.08,
            "ec": 0.05,
        },
    )
    base = torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)

    output, details = adapter(base, return_details=True)
    output.sum().backward()

    assert output.shape == (3, 16)
    assert details["r2e_adapter_gate"].shape == (3,)
    assert adapter.raw_gate.grad is not None
    for projection in adapter.block_projections.values():
        assert projection[1].weight.grad is not None


def test_bidirectional_adapters_are_isolated_by_retrieval_direction():
    model = ProteinPooledDualModel(
        query_encoder_kwargs={
            "input_dim": 4,
            "output_dim": 4,
            "num_layers": 0,
            "normalise_output": True,
        },
        target_encoder_kwargs={
            "input_dim": 4,
            "output_dim": 4,
            "num_layers": 0,
            "normalise_output": True,
        },
        residue_dim=4,
        pooling="mean",
        e2r_adapter_enabled=True,
        e2r_adapter_dropout=0.0,
        r2e_adapter_enabled=True,
        r2e_adapter_dropout=0.0,
    )
    model.eval()
    reactions = torch.randn(2, 4)
    residues = torch.randn(2, 3, 4)
    mask = torch.zeros(2, 3, dtype=torch.bool)

    base_reactions = model.query_encoder(reactions)
    base_enzymes = model.target_encoder(model.pool_residues(residues, residue_padding_mask=mask))
    r2e_reactions = model.encode_queries(
        reactions,
        retrieval_direction="reaction_to_enzyme",
    )
    e2r_reactions = model.encode_queries(
        reactions,
        retrieval_direction="enzyme_to_reaction",
    )
    r2e_enzymes = model.encode_targets(
        residues,
        residue_padding_mask=mask,
        retrieval_direction="reaction_to_enzyme",
    )
    e2r_enzymes = model.encode_targets(
        residues,
        residue_padding_mask=mask,
        retrieval_direction="enzyme_to_reaction",
    )

    assert torch.equal(r2e_reactions, base_reactions)
    assert not torch.allclose(e2r_reactions, base_reactions)
    assert not torch.allclose(r2e_enzymes, base_enzymes)
    assert torch.equal(e2r_enzymes, base_enzymes)


def test_r2e_training_stage_freezes_everything_except_r2e_adapter():
    module = ProteinPooledLitModule(
        query_encoder_dims=[4, 4],
        target_encoder_dims=[4, 4],
        embedding_dim=4,
        residue_dim=4,
        pooling="mean",
        e2r_adapter_enabled=True,
        r2e_adapter_enabled=True,
        training_stage="r2e_adapter",
    )

    trainable = {name for name, parameter in module.named_parameters() if parameter.requires_grad}

    assert trainable
    assert all(name.startswith("model.r2e_adapter.") for name in trainable)
    assert any(name.startswith("model.e2r_adapter.") for name, _ in module.named_parameters())


def test_bidirectional_training_stage_freezes_parent_and_trains_both_adapters():
    module = ProteinPooledLitModule(
        query_encoder_dims=[4, 4],
        target_encoder_dims=[4, 4],
        embedding_dim=4,
        residue_dim=4,
        pooling="mean",
        e2r_adapter_enabled=True,
        r2e_adapter_enabled=True,
        training_stage="bidirectional_adapters",
    )

    trainable = {name for name, parameter in module.named_parameters() if parameter.requires_grad}

    assert any(name.startswith("model.e2r_adapter.") for name in trainable)
    assert any(name.startswith("model.r2e_adapter.") for name in trainable)
    assert all(name.startswith(("model.e2r_adapter.", "model.r2e_adapter.")) for name in trainable)


def test_bidirectional_training_step_updates_both_direction_specific_adapters():
    module = ProteinPooledLitModule(
        query_encoder_dims=[4, 4],
        target_encoder_dims=[4, 4],
        embedding_dim=4,
        residue_dim=4,
        pooling="mean",
        loss_name="MultiAlignmentRetrievalLoss",
        lambda_r2e=1.0,
        lambda_e2r=1.0,
        e2r_adapter_enabled=True,
        e2r_adapter_dropout=0.0,
        r2e_adapter_enabled=True,
        r2e_adapter_dropout=0.0,
        e2r_identity_weight=0.05,
        r2e_identity_weight=0.05,
        training_stage="bidirectional_adapters",
        log_attention_stats=False,
    )
    batch = residue_collate_fn(
        [
            {
                "query_id": "r0",
                "target_id": "p0",
                "query_vec": torch.randn(4),
                "residue_embeddings": torch.randn(3, 4),
            },
            {
                "query_id": "r1",
                "target_id": "p1",
                "query_vec": torch.randn(4),
                "residue_embeddings": torch.randn(2, 4),
            },
        ]
    )

    loss, _, _, components = module._compute_full_batch_loss(
        batch,
        structure_terms_enabled=False,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["r2e"] > 0
    assert components["e2r"] > 0
    assert components["r2e_identity"] >= 0
    assert components["e2r_identity"] >= 0
    assert module.model.r2e_adapter.raw_gate.grad is not None
    assert module.model.e2r_adapter.raw_gate.grad is not None
    assert all(
        parameter.grad is None
        for name, parameter in module.named_parameters()
        if not name.startswith(("model.e2r_adapter.", "model.r2e_adapter."))
    )


def test_r2e_training_step_applies_identity_regularization():
    module = ProteinPooledLitModule(
        query_encoder_dims=[4, 4],
        target_encoder_dims=[4, 4],
        embedding_dim=4,
        residue_dim=4,
        pooling="mean",
        loss_name="MultiAlignmentRetrievalLoss",
        lambda_r2e=1.0,
        lambda_e2r=0.0,
        r2e_adapter_enabled=True,
        r2e_adapter_dropout=0.0,
        r2e_identity_weight=0.05,
        training_stage="r2e_adapter",
        log_attention_stats=False,
    )
    batch = residue_collate_fn(
        [
            {
                "query_id": "r0",
                "target_id": "p0",
                "query_vec": torch.randn(4),
                "residue_embeddings": torch.randn(3, 4),
            },
            {
                "query_id": "r1",
                "target_id": "p1",
                "query_vec": torch.randn(4),
                "residue_embeddings": torch.randn(2, 4),
            },
        ]
    )

    loss, _, _, components = module._compute_full_batch_loss(
        batch,
        structure_terms_enabled=False,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["r2e_identity"] >= 0
    assert components["weighted_r2e_identity"] >= 0
    assert module.model.r2e_adapter.raw_gate.grad is not None


def test_e2r_hard_negative_loss_operates_on_transposed_candidate_axis():
    dists = torch.tensor(
        [
            [0.10, 0.80],
            [0.11, 0.20],
            [0.70, 0.21],
        ]
    )
    query_idx = torch.tensor([0, 1], dtype=torch.long)
    target_idx = torch.tensor([0, 1], dtype=torch.long)
    loss_fn = MultiAlignmentRetrievalLoss(
        beta=4.0,
        lambda_r2e=0.0,
        lambda_e2r=1.0,
        lambda_e2r_hard_neg=0.5,
        e2r_hard_neg_top_k=1,
    )

    loss, components = loss_fn(
        dists,
        query_idx,
        target_idx,
        return_components=True,
    )

    assert torch.isfinite(loss)
    assert components["e2r_hard_neg"] > 0
    assert components["weighted_e2r_hard_neg"] > 0
    assert torch.equal(components["weighted_r2e_hard_neg"], torch.tensor(0.0))


def test_multi_alignment_accepts_distinct_directional_distance_matrices():
    r2e_dists = torch.tensor(
        [[0.1, 0.8], [0.7, 0.2]],
        requires_grad=True,
    )
    e2r_dists = torch.tensor(
        [[0.2, 0.6], [0.9, 0.1]],
        requires_grad=True,
    )
    indices = torch.tensor([0, 1], dtype=torch.long)
    loss_fn = MultiAlignmentRetrievalLoss(
        beta=4.0,
        lambda_r2e=1.0,
        lambda_e2r=1.0,
    )

    loss, components = loss_fn(
        r2e_dists,
        indices,
        indices,
        e2r_dists=e2r_dists,
        return_components=True,
    )
    loss.backward()

    assert components["r2e"] > 0
    assert components["e2r"] > 0
    assert r2e_dists.grad is not None
    assert e2r_dists.grad is not None


def test_e2r_hard_negative_sampler_includes_negative_reaction(tmp_path):
    pairs = BaseDataset(
        keys=["pair0", "pair1", "pair2"],
        array_data=[
            {"query_id": "r0", "target_id": "p0"},
            {"query_id": "r1", "target_id": "p1"},
            {"query_id": "r2", "target_id": "p2"},
        ],
    )
    queries = BaseDataset(
        keys=["r0", "r1", "r2"],
        array_data=torch.arange(3, dtype=torch.float32).unsqueeze(1),
    )
    targets = BaseDataset(
        keys=["p0", "p1", "p2"],
        array_data=torch.arange(3, dtype=torch.float32).unsqueeze(1),
    )
    dataset = TupleDataset(
        tuple_dataset=pairs,
        key_name_to_dataset={"query_id": queries, "target_id": targets},
    )
    pool_path = tmp_path / "e2r_hard_negatives.json"
    pool_path.write_text(
        json.dumps(
            {
                "schema_version": "directional_hard_negatives_v2",
                "direction": "enzyme_to_reaction",
                "anchor_to_negatives": {"p0": ["r1"]},
            }
        ),
        encoding="utf-8",
    )
    sampler = DirectionalHardNegativeBatchSampler(
        dataset,
        batch_size=3,
        hard_negative_pools_path=pool_path,
        direction="enzyme_to_reaction",
        anchor_queries_per_batch=1,
        positives_per_query=1,
        negatives_per_query=1,
        seed=42,
    )

    batch = next(iter(sampler))

    assert 0 in batch
    assert 1 in batch


def test_dense_reaction_features_are_train_fitted_and_have_masks(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from horizyn.capability.reaction_dense_features import (
        DENSE_REACTION_DIM,
        build_dense_reaction_feature_splits,
    )

    paths = {}
    contents = {
        "train": [
            ("r0", "CCO>>CC=O"),
            ("r1", "CC(=O)O>>CCO"),
        ],
        "validation": [("r2", "CCO>>CCN")],
        "test": [("r3", "not_a_reaction")],
    }
    for split, rows in contents.items():
        path = tmp_path / f"{split}.csv"
        lines = ["reaction_id,reaction_smiles"]
        lines.extend(f"{reaction_id},{smiles}" for reaction_id, smiles in rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[split] = path

    report = build_dense_reaction_feature_splits(
        train_reactions_path=paths["train"],
        validation_reactions_path=paths["validation"],
        test_reactions_path=paths["test"],
        out_dir=tmp_path / "dense",
    )
    train = dict(
        __import__("numpy").load(
            tmp_path / "dense/train_dense_reaction_features.npz",
            allow_pickle=True,
        )
    )
    test = dict(
        __import__("numpy").load(
            tmp_path / "dense/test_dense_reaction_features.npz",
            allow_pickle=True,
        )
    )

    assert train["vectors"].shape == (2, DENSE_REACTION_DIM)
    assert test["vectors"].shape == (1, DENSE_REACTION_DIM)
    assert test["mask"].tolist() == [False]
    assert report["splits"]["train"]["parse_coverage"] == 1.0
