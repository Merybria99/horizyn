import h5py
import numpy as np
import pytest
import torch

from horizyn.config import DotDict, validate_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset, truncate_residue_embeddings
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import TupleDataset
from horizyn.model import (
    ProteinAttentionPooling,
    ProteinMeanPooling,
    ProteinPooledDualModel,
    ReactionFingerprintAttentionPool,
    ReactionConditionedAttentionPooling,
    ReactionConditionedDualModel,
    SLEECFunctionalPool,
)
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_conditioned_data_module import (
    DirectionalHardNegativeBatchSampler,
    ReactionConditionedDataModule,
    ReactionDegreeBalancedBatchSampler,
)
from horizyn.reaction_conditioned_lightning_module import ReactionConditionedLitModule
from horizyn.utils import residue_collate_fn


def _write_stage1_checkpoint(path, input_dim=4, hidden_dim=5):
    torch.save(
        {
            "unwrapped_model_state_dict": {
                "net.0.weight": torch.randn(hidden_dim, input_dim),
                "net.0.bias": torch.randn(hidden_dim),
                "net.2.weight": torch.randn(1, hidden_dim),
                "net.2.bias": torch.randn(1),
            }
        },
        path,
    )


def _write_fixed_reaction_h5(path, ids, base_value, dim=3):
    vectors = np.full((len(ids), dim), base_value, dtype=np.float32)
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array([value.encode("utf-8") for value in ids]))
        h5_file.create_dataset("vectors", data=vectors)


def _write_ragged_reaction_h5(path, ids, base_value, dim=2):
    vectors = np.full((len(ids), dim), base_value, dtype=np.float32)
    offsets = np.arange(len(ids) + 1, dtype=np.int64)
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array([value.encode("utf-8") for value in ids]))
        h5_file.create_dataset("reactant_vectors", data=vectors)
        h5_file.create_dataset("reactant_offsets", data=offsets)
        h5_file.create_dataset("product_vectors", data=vectors)
        h5_file.create_dataset("product_offsets", data=offsets)


def test_residue_hdf5_loader_reads_ragged_vectors(tmp_path):
    h5_path = tmp_path / "residues.h5"
    vectors = np.arange(18, dtype=np.float32).reshape(6, 3)
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array([b"p1", b"p2"]))
        h5_file.create_dataset("vectors", data=vectors)
        h5_file.create_dataset("offsets", data=np.array([0, 2, 6], dtype=np.int64))

    dataset = ResidueEmbedDataset(str(h5_path), in_memory=False)

    assert len(dataset) == 2
    assert dataset.vec_dim == 3
    assert torch.equal(dataset["p1"]["residue_embeddings"], torch.from_numpy(vectors[:2]))
    assert torch.equal(dataset["p2"]["residue_embeddings"], torch.from_numpy(vectors[2:]))


def test_residue_hdf5_drop_empty_preserves_source_lengths(tmp_path):
    h5_path = tmp_path / "residues_with_empty.h5"
    vectors = np.arange(6, dtype=np.float32).reshape(3, 2)
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array([b"empty", b"p1", b"p2"]))
        h5_file.create_dataset("vectors", data=vectors)
        h5_file.create_dataset("offsets", data=np.array([0, 0, 1, 3], dtype=np.int64))

    dataset = ResidueEmbedDataset(str(h5_path), in_memory=False, drop_empty=True)

    assert dataset.keys == ["p1", "p2"]
    assert dataset.lengths.tolist() == [1, 2]
    assert dataset.length_by_key == {"empty": 0, "p1": 1, "p2": 2}


def test_truncate_residue_embeddings_ends_center():
    embeddings = torch.arange(8, dtype=torch.float32).unsqueeze(1)

    truncated = truncate_residue_embeddings(embeddings, max_tokens=4)

    assert truncated.squeeze(1).tolist() == [0.0, 3.0, 4.0, 7.0]


def test_residue_collate_padding_mask():
    batch = [
        {
            "query_id": "q1",
            "target_id": "p1",
            "query_vec": torch.tensor([1.0, 2.0]),
            "residue_embeddings": torch.ones(2, 3),
        },
        {
            "query_id": "q2",
            "target_id": "p2",
            "query_vec": torch.tensor([3.0, 4.0]),
            "residue_embeddings": torch.full((4, 3), 2.0),
        },
    ]

    collated = residue_collate_fn(batch)

    assert collated["query_vec"].shape == (2, 2)
    assert collated["residue_embeddings"].shape == (2, 4, 3)
    assert collated["residue_padding_mask"].tolist() == [
        [False, False, True, True],
        [False, False, False, False],
    ]
    assert collated["query_id"] == ["q1", "q2"]
    assert collated["target_id"] == ["p1", "p2"]


def _sampler_tuple_dataset():
    pairs = BaseDataset(
        keys=[f"pair{index}" for index in range(10)],
        array_data=[
            {"query_id": query_id, "target_id": f"p{index}"}
            for index, query_id in enumerate(
                ["r0", "r0", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]
            )
        ],
    )
    queries = BaseDataset(
        keys=[f"r{index}" for index in range(8)],
        array_data=torch.arange(8, dtype=torch.float32).unsqueeze(1),
    )
    targets = BaseDataset(
        keys=[f"p{index}" for index in range(10)],
        array_data=torch.arange(10, dtype=torch.float32).unsqueeze(1),
    )
    return TupleDataset(
        tuple_dataset=pairs,
        key_name_to_dataset={"query_id": queries, "target_id": targets},
    )


def test_hard_negative_sampler_rejects_impossible_unique_batch(tmp_path):
    dataset = _sampler_tuple_dataset()
    pool_path = tmp_path / "hard_negatives.json"
    pool_path.write_text('{"anchor_to_negatives": {"r0": ["p4"]}}', encoding="utf-8")

    with pytest.raises(ValueError, match="cannot exceed"):
        DirectionalHardNegativeBatchSampler(
            dataset,
            batch_size=len(dataset) + 1,
            hard_negative_pools_path=pool_path,
        )


def test_reaction_degree_balanced_sampler_shards_one_global_batch_without_overlap():
    dataset = _sampler_tuple_dataset()
    rank0 = ReactionDegreeBalancedBatchSampler(
        dataset,
        batch_size=3,
        degree_exponent=0.5,
        seed=17,
        rank=0,
        world_size=2,
    )
    rank1 = ReactionDegreeBalancedBatchSampler(
        dataset,
        batch_size=3,
        degree_exponent=0.5,
        seed=17,
        rank=1,
        world_size=2,
    )

    batch0 = next(iter(rank0))
    batch1 = next(iter(rank1))
    queries0 = {dataset.tuple_dataset[dataset.keys[index]]["query_id"] for index in batch0}
    queries1 = {dataset.tuple_dataset[dataset.keys[index]]["query_id"] for index in batch1}

    assert len(batch0) == len(batch1) == 3
    assert queries0.isdisjoint(queries1)
    assert len(queries0 | queries1) == 6


def test_reaction_degree_balanced_sampler_is_seed_and_epoch_deterministic():
    dataset = _sampler_tuple_dataset()
    first = ReactionDegreeBalancedBatchSampler(dataset, batch_size=4, seed=9)
    second = ReactionDegreeBalancedBatchSampler(dataset, batch_size=4, seed=9)

    first_epoch = list(iter(first))
    assert first_epoch == list(iter(second))
    assert first_epoch != list(iter(first))


def test_residue_collate_pads_residue_labels():
    batch = [
        {
            "query_id": "q1",
            "target_id": "p1",
            "query_vec": torch.tensor([1.0, 2.0]),
            "residue_embeddings": torch.ones(2, 3),
            "residue_labels": torch.tensor([1.0, 0.0]),
            "residue_label_mask": torch.tensor([True, False]),
        },
        {
            "query_id": "q2",
            "target_id": "p2",
            "query_vec": torch.tensor([3.0, 4.0]),
            "residue_embeddings": torch.full((4, 3), 2.0),
            "residue_labels": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "residue_label_mask": torch.tensor([True, True, False, True]),
        },
    ]

    collated = residue_collate_fn(batch)

    assert collated["residue_labels"].shape == (2, 4)
    assert collated["residue_label_mask"].shape == (2, 4)
    assert collated["residue_labels"].tolist() == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
    assert collated["residue_label_mask"].tolist() == [
        [True, False, False, False],
        [True, True, False, True],
    ]


def test_residue_collate_pads_score_residue_embeddings():
    batch = [
        {
            "query_id": "q1",
            "target_id": "p1",
            "query_vec": torch.tensor([1.0, 2.0]),
            "residue_embeddings": torch.ones(2, 3),
            "score_residue_embeddings": torch.ones(2, 4),
        },
        {
            "query_id": "q2",
            "target_id": "p2",
            "query_vec": torch.tensor([3.0, 4.0]),
            "residue_embeddings": torch.full((4, 3), 2.0),
            "score_residue_embeddings": torch.full((4, 4), 3.0),
        },
    ]

    collated = residue_collate_fn(batch)

    assert collated["residue_embeddings"].shape == (2, 4, 3)
    assert collated["score_residue_embeddings"].shape == (2, 4, 4)
    assert collated["score_residue_padding_mask"].tolist() == [
        [False, False, True, True],
        [False, False, False, False],
    ]
    assert collated["score_residue_embeddings"][0, 2:].sum().item() == 0.0


def test_protein_mean_pooling_masks_padding():
    pooling = ProteinMeanPooling()
    residues = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])

    pooled, weights = pooling(residues, mask, return_attention=True)

    assert torch.allclose(pooled, torch.tensor([[2.0, 1.0]]))
    assert torch.allclose(weights, torch.tensor([[0.5, 0.5, 0.0]]))


def test_protein_attention_pooling_matches_manual_softmax():
    pooling = ProteinAttentionPooling(hidden_dim=2, attention_bias=False)
    with torch.no_grad():
        pooling.attention.weight.copy_(torch.tensor([[1.0, 0.0]]))

    residues = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [4.0, 4.0]]])
    mask = torch.tensor([[True, True, False]])

    pooled, weights = pooling(residues, mask, return_attention=True)

    manual_weights = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    manual_pooled = manual_weights[0] * residues[0, 0] + manual_weights[1] * residues[0, 1]
    assert torch.allclose(weights[0, :2], manual_weights)
    assert weights[0, 2].item() == 0.0
    assert torch.allclose(pooled[0], manual_pooled)


def test_sleec_functional_pool_smoke_masks_padding():
    torch.manual_seed(0)
    pooler = SLEECFunctionalPool(hidden_dim=4, scorer_hidden_dim=6, mode="topk")
    residues = torch.randn(3, 7, 4)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False],
        ]
    )

    pooled, details = pooler(residues, attention_mask=mask, return_details=True)

    assert pooled.shape == (3, 4)
    for key in ("logits", "scores", "weights"):
        assert details[key].shape == (3, 7)
        assert torch.isfinite(details[key]).all()
    assert torch.isfinite(pooled).all()
    assert torch.allclose(details["weights"].sum(dim=1), torch.ones(3))
    assert details["weights"][~mask].sum().item() == 0.0


def test_sleec_functional_pool_threshold_fallback_to_masked_average():
    torch.manual_seed(1)
    pooler = SLEECFunctionalPool(
        hidden_dim=3,
        scorer_hidden_dim=5,
        mode="threshold",
        threshold=2.0,
    )
    residues = torch.randn(2, 5, 3)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )

    pooled, details = pooler(residues, attention_mask=mask, return_details=True)

    average_weights = mask.to(dtype=residues.dtype)
    average_weights = average_weights / average_weights.sum(dim=1, keepdim=True)
    expected = torch.einsum("bl,blh->bh", average_weights, residues)
    assert torch.allclose(pooled, expected)
    assert torch.allclose(details["weights"], average_weights)
    assert details["weights"][~mask].sum().item() == 0.0


def test_sleec_functional_pool_uses_external_score_embeddings(tmp_path):
    checkpoint = tmp_path / "stage1.ckpt"
    _write_stage1_checkpoint(checkpoint, input_dim=4, hidden_dim=5)
    pooler = SLEECFunctionalPool(
        hidden_dim=3,
        score_hidden_dim=4,
        scorer_hidden_dim=5,
        mode="threshold",
        threshold=0.0,
        checkpoint_path=str(checkpoint),
        freeze_scorer=True,
    )
    residues = torch.randn(2, 6, 3)
    score_residues = torch.randn(2, 6, 4)
    mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]
    )

    pooled, details = pooler(
        residues,
        attention_mask=mask,
        score_embeddings=score_residues,
        return_details=True,
    )

    assert pooled.shape == (2, 3)
    assert details["logits"].shape == (2, 6)
    assert details["scores"].shape == (2, 6)
    assert details["weights"].shape == (2, 6)
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(details["weights"]).all()
    assert torch.allclose(details["weights"].sum(dim=1), torch.ones(2))
    assert details["weights"][~mask].sum().item() == 0.0
    assert all(not parameter.requires_grad for parameter in pooler.scorer.parameters())


def test_sleec_functional_pool_rejects_misaligned_score_embeddings():
    pooler = SLEECFunctionalPool(hidden_dim=3, score_hidden_dim=4, scorer_hidden_dim=5)
    residues = torch.randn(2, 6, 3)
    score_residues = torch.randn(2, 5, 4)

    with pytest.raises(ValueError, match="align"):
        pooler(residues, score_embeddings=score_residues)


def test_protein_pooled_dual_model_score_shapes_for_pooling_modes():
    for pooling in ("mean", "avg", "attention", "sleec"):
        model = ProteinPooledDualModel(
            query_encoder_kwargs={
                "input_dim": 2,
                "output_dim": 3,
                "num_layers": 0,
                "widths": [],
                "normalise_output": True,
            },
            target_encoder_kwargs={
                "input_dim": 4,
                "output_dim": 3,
                "num_layers": 0,
                "widths": [],
                "normalise_output": True,
            },
            residue_dim=4,
            pooling=pooling,
            sleec_scorer_hidden_dim=8,
        )
        query_vecs = torch.randn(2, 2)
        residues = torch.randn(2, 5, 4)
        padding_mask = torch.zeros(2, 5, dtype=torch.bool)

        query_embeds, target_embeds, attention = model(
            query_vecs,
            residues,
            residue_padding_mask=padding_mask,
            return_attention=True,
        )

        assert query_embeds.shape == (2, 3)
        assert target_embeds.shape == (2, 3)
        assert attention.shape == (2, 5)
        assert torch.allclose(attention.sum(dim=1), torch.ones(2))


def test_reaction_fingerprint_attention_pool_shapes_and_gradients():
    pooler = ReactionFingerprintAttentionPool(
        input_dim=12,
        rdkit_dim=8,
        drfp_dim=4,
        token_dim=3,
        hidden_dim=5,
        dropout=0.0,
    )
    fingerprints = torch.randn(4, 12)

    pooled, details = pooler(fingerprints, return_details=True)
    assert pooled.shape == (4, 3)
    assert details["logits"].shape == (4, 3)
    assert details["weights"].shape == (4, 3)
    assert details["entropy"].shape == (4,)
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(details["weights"]).all()
    assert torch.allclose(details["weights"].sum(dim=1), torch.ones(4))

    pooled.pow(2).mean().backward()
    assert pooler.reactant_projection.weight.grad is not None
    assert pooler.product_projection.weight.grad is not None
    assert pooler.drfp_projection.weight.grad is not None
    assert pooler.attention[0].weight.grad is not None


def test_protein_pooled_dual_model_with_reaction_fingerprint_attention():
    model = ProteinPooledDualModel(
        query_encoder_kwargs={
            "input_dim": 3,
            "output_dim": 2,
            "num_layers": 0,
            "widths": [],
            "normalise_output": True,
        },
        target_encoder_kwargs={
            "input_dim": 4,
            "output_dim": 2,
            "num_layers": 0,
            "widths": [],
            "normalise_output": True,
        },
        residue_dim=4,
        pooling="mean",
        reaction_fingerprint_attention_enabled=True,
        reaction_fingerprint_attention_input_dim=12,
        reaction_fingerprint_attention_rdkit_dim=8,
        reaction_fingerprint_attention_drfp_dim=4,
        reaction_fingerprint_attention_token_dim=3,
        reaction_fingerprint_attention_hidden_dim=5,
    )
    query_vecs = torch.randn(3, 12)
    residues = torch.randn(3, 5, 4)
    padding_mask = torch.zeros(3, 5, dtype=torch.bool)

    query_embeds, query_attention = model.encode_queries(query_vecs, return_attention=True)
    assert query_embeds.shape == (3, 2)
    assert query_attention["reaction_fingerprint"].shape == (3, 3)
    assert torch.allclose(
        query_attention["reaction_fingerprint"].sum(dim=1),
        torch.ones(3),
    )

    query_embeds, target_embeds = model(
        query_vecs,
        residues,
        residue_padding_mask=padding_mask,
    )
    assert query_embeds.shape == (3, 2)
    assert target_embeds.shape == (3, 2)
    loss = (query_embeds * target_embeds).sum()
    loss.backward()
    assert model.reaction_fingerprint_attention is not None
    assert model.reaction_fingerprint_attention.reactant_projection.weight.grad is not None
    assert model.query_encoder.main_nn[0].weight.grad is not None


def test_reaction_conditioned_attention_pooling_matches_manual_softmax():
    pooling = ReactionConditionedAttentionPooling(residue_dim=2, embedding_dim=2)
    with torch.no_grad():
        pooling.residue_projection.weight.copy_(torch.eye(2))

    reaction = torch.tensor([[1.0, 0.0]])
    residues = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [2.0, 0.0]]])
    mask = torch.tensor([[False, False, True]])

    pooled, attention = pooling(reaction, residues, mask, return_attention=True)

    manual_scores = torch.tensor([1.0, 0.0])
    manual_attention = torch.softmax(manual_scores, dim=0)
    manual_pooled = manual_attention[0] * residues[0, 0] + manual_attention[1] * residues[0, 1]

    assert torch.allclose(attention[0, 0, :2], manual_attention)
    assert attention[0, 0, 2].item() == 0.0
    assert torch.allclose(pooled[0, 0], manual_pooled)


def test_reaction_conditioned_low_rank_attention_pooling_matches_manual_softmax():
    pooling = ReactionConditionedAttentionPooling(
        residue_dim=3,
        embedding_dim=4,
        attention_rank=2,
    )
    with torch.no_grad():
        pooling.reaction_projection.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        )
        pooling.residue_projection.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    reaction = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    residues = torch.tensor([[[1.0, 0.0, 7.0], [0.0, 2.0, 8.0]]])
    mask = torch.tensor([[False, False]])

    pooled, attention = pooling(reaction, residues, mask, return_attention=True)

    manual_attention = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    manual_pooled = manual_attention[0] * residues[0, 0] + manual_attention[1] * residues[0, 1]

    assert torch.allclose(attention[0, 0], manual_attention)
    assert torch.allclose(pooled[0, 0], manual_pooled)


def test_reaction_conditioned_dual_model_score_matrix_shape():
    model = ReactionConditionedDualModel(
        query_encoder_kwargs={
            "input_dim": 2,
            "output_dim": 3,
            "num_layers": 0,
            "widths": [],
            "normalise_output": True,
        },
        target_encoder_kwargs={
            "input_dim": 3,
            "output_dim": 3,
            "num_layers": 0,
            "widths": [],
            "normalise_output": True,
        },
        residue_dim=3,
        embedding_dim=3,
        attention_rank=2,
    )

    query_vecs = torch.randn(2, 2)
    residues = torch.randn(4, 5, 3)
    mask = torch.zeros(4, 5, dtype=torch.bool)

    scores = model.score_matrix(query_vecs, residues, mask)

    assert scores.shape == (2, 4)
    assert torch.isfinite(scores).all()


def test_reaction_conditioned_dual_model_projected_value_score_mode():
    model = ReactionConditionedDualModel(
        query_encoder_kwargs={
            "input_dim": 2,
            "output_dim": 3,
            "num_layers": 0,
            "widths": [],
            "normalise_output": True,
        },
        target_encoder_kwargs={
            "input_dim": 4,
            "output_dim": 3,
            "num_layers": 1,
            "widths": [8],
            "normalise_output": True,
        },
        residue_dim=4,
        embedding_dim=3,
        attention_rank=2,
        score_mode="projected_value",
    )

    query_vecs = torch.randn(2, 2)
    residues = torch.randn(5, 6, 4)
    mask = torch.zeros(5, 6, dtype=torch.bool)

    scores = model.score_matrix(query_vecs, residues, mask)

    assert model.target_encoder is None
    assert scores.shape == (2, 5)
    assert torch.isfinite(scores).all()


def test_reaction_conditioned_lightning_training_step():
    module = ReactionConditionedLitModule(
        query_encoder_dims=[2, 3],
        target_encoder_dims=[3, 3],
        embedding_dim=3,
        residue_dim=3,
    )
    batch = residue_collate_fn(
        [
            {
                "query_id": "q1",
                "target_id": "p1",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(3, 3),
            },
            {
                "query_id": "q1",
                "target_id": "p2",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(2, 3),
            },
            {
                "query_id": "q2",
                "target_id": "p1",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(3, 3),
            },
        ]
    )

    loss = module.training_step(batch, batch_idx=0)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_reaction_conditioned_lightning_training_step_with_hybrid_query_dict():
    module = ReactionConditionedLitModule(
        query_encoder_dims=[4, 3],
        target_encoder_dims=[3, 3],
        embedding_dim=3,
        residue_dim=3,
        query_encoder_type="hybrid_reaction",
        reaction_unimol_dim=2,
    )
    batch = residue_collate_fn(
        [
            {
                "query_id": "q1",
                "target_id": "p1",
                "query_vec": {
                    "fingerprint": torch.randn(4),
                    "has_unimol2": torch.tensor(True),
                    "reactant_embeddings": torch.randn(1, 2),
                    "product_embeddings": torch.randn(2, 2),
                },
                "residue_embeddings": torch.randn(3, 3),
            },
            {
                "query_id": "q1",
                "target_id": "p2",
                "query_vec": {
                    "fingerprint": torch.randn(4),
                    "has_unimol2": torch.tensor(True),
                    "reactant_embeddings": torch.randn(1, 2),
                    "product_embeddings": torch.randn(2, 2),
                },
                "residue_embeddings": torch.randn(2, 3),
            },
            {
                "query_id": "q2",
                "target_id": "p1",
                "query_vec": {
                    "fingerprint": torch.randn(4),
                    "has_unimol2": torch.tensor(False),
                    "reactant_embeddings": torch.zeros(1, 2),
                    "product_embeddings": torch.zeros(1, 2),
                },
                "residue_embeddings": torch.randn(3, 3),
            },
        ]
    )

    loss = module.training_step(batch, batch_idx=0)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_protein_pooled_lightning_training_step_logs_attention_stats():
    module = ProteinPooledLitModule(
        query_encoder_dims=[2, 3],
        target_encoder_dims=[3, 3],
        embedding_dim=3,
        residue_dim=3,
        pooling="attention",
        attention_logging_interval=1,
    )
    batch = residue_collate_fn(
        [
            {
                "query_id": "q1",
                "target_id": "p1",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(3, 3),
            },
            {
                "query_id": "q1",
                "target_id": "p2",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(2, 3),
            },
            {
                "query_id": "q2",
                "target_id": "p1",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(3, 3),
            },
        ]
    )

    loss = module.training_step(batch, batch_idx=0)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_protein_pooled_lightning_with_reaction_fingerprint_attention():
    module = ProteinPooledLitModule(
        query_encoder_dims=[3, 3],
        target_encoder_dims=[3, 3],
        embedding_dim=3,
        residue_dim=3,
        pooling="mean",
        reaction_fingerprint_attention_enabled=True,
        reaction_fingerprint_attention_input_dim=12,
        reaction_fingerprint_attention_rdkit_dim=8,
        reaction_fingerprint_attention_drfp_dim=4,
        reaction_fingerprint_attention_token_dim=3,
        reaction_fingerprint_attention_hidden_dim=5,
        attention_logging_interval=1,
    )
    batch = residue_collate_fn(
        [
            {
                "query_id": "q1",
                "target_id": "p1",
                "query_vec": torch.randn(12),
                "residue_embeddings": torch.randn(3, 3),
            },
            {
                "query_id": "q1",
                "target_id": "p2",
                "query_vec": torch.randn(12),
                "residue_embeddings": torch.randn(2, 3),
            },
            {
                "query_id": "q2",
                "target_id": "p1",
                "query_vec": torch.randn(12),
                "residue_embeddings": torch.randn(3, 3),
            },
        ]
    )

    loss = module.training_step(batch, batch_idx=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    pooler = module.model.reaction_fingerprint_attention
    assert pooler is not None
    assert pooler.reactant_projection.weight.grad is not None


def test_protein_pooled_validation_logs_bidirectional_retrieval_metrics():
    from unittest.mock import patch

    module = ProteinPooledLitModule(
        query_encoder_dims=[2, 2],
        target_encoder_dims=[2, 2],
        embedding_dim=2,
        residue_dim=2,
        pooling="mean",
        validation_retrieval_metrics=True,
        retrieval_metric_top_k=[1],
    )

    class MockDataModule:
        _val_retrieval_target_candidate_ids = ["prot1", "prot2"]
        _val_retrieval_query_candidate_ids = ["rxn1", "rxn2"]
        _query_to_targets = {"rxn1": ["prot1"], "rxn2": ["prot2"]}
        _target_to_queries = {"prot1": ["rxn1"], "prot2": ["rxn2"]}

    class MockTrainer:
        datamodule = MockDataModule()

    module.trainer = MockTrainer()

    def encode_targets(
        residue_embeddings,
        residue_padding_mask=None,
        score_residue_embeddings=None,
        score_residue_padding_mask=None,
        **kwargs,
    ):
        del kwargs
        del residue_padding_mask, score_residue_embeddings, score_residue_padding_mask
        return torch.nn.functional.normalize(residue_embeddings[:, 0, :], dim=-1)

    def encode_queries(query_vec):
        return torch.nn.functional.normalize(query_vec, dim=-1)

    module.model.encode_targets = encode_targets
    module.model.encode_queries = encode_queries
    module.on_validation_epoch_start()

    target_lookup_batch = residue_collate_fn(
        [
            {
                "target_id": "prot1",
                "target_lookup_row_idx": 0,
                "residue_embeddings": torch.tensor([[1.0, 0.0], [0.5, 0.0]]),
            },
            {
                "target_id": "prot2",
                "target_lookup_row_idx": 1,
                "residue_embeddings": torch.tensor([[0.0, 1.0], [0.0, 0.5]]),
            },
        ]
    )
    query_lookup_batch = {
        "query_id": ["rxn1", "rxn2"],
        "query_lookup_row_idx": torch.tensor([0, 1]),
        "query_vec": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    reaction_to_enzyme_batch = {
        "query_id": ["rxn1", "rxn2"],
        "query_vec": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    enzyme_to_reaction_batch = residue_collate_fn(
        [
            {
                "target_id": "prot1",
                "target_query_row_idx": 0,
                "residue_embeddings": torch.tensor([[1.0, 0.0], [0.5, 0.0]]),
            },
            {
                "target_id": "prot2",
                "target_query_row_idx": 1,
                "residue_embeddings": torch.tensor([[0.0, 1.0], [0.0, 0.5]]),
            },
        ]
    )

    with patch.object(module, "log") as mocked_log:
        module.validation_step(target_lookup_batch, batch_idx=0, dataloader_idx=1)
        module.validation_step(query_lookup_batch, batch_idx=0, dataloader_idx=2)
        module.validation_step(reaction_to_enzyme_batch, batch_idx=0, dataloader_idx=3)
        module.validation_step(enzyme_to_reaction_batch, batch_idx=0, dataloader_idx=4)

    assert torch.allclose(
        module.val_target_lookup_table,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert torch.allclose(
        module.val_query_lookup_table,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    logged_names = [call.args[0] for call in mocked_log.call_args_list]
    assert "val/reaction_to_enzyme/top_1" in logged_names
    assert "val/enzyme_to_reaction/top_1" in logged_names
    assert "val/reaction_to_enzyme/mrr" in logged_names
    assert "val/enzyme_to_reaction/mrr" in logged_names


def test_protein_pooled_lightning_sleec_training_step_with_residue_labels():
    module = ProteinPooledLitModule(
        query_encoder_dims=[2, 3],
        target_encoder_dims=[3, 3],
        embedding_dim=3,
        residue_dim=3,
        pooling="sleec",
        sleec_scorer_hidden_dim=4,
        lambda_residue=0.5,
        attention_logging_interval=1,
    )
    batch = residue_collate_fn(
        [
            {
                "query_id": "q1",
                "target_id": "p1",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(3, 3),
                "residue_labels": torch.tensor([1.0, 0.0, 1.0]),
                "residue_label_mask": torch.tensor([True, True, True]),
            },
            {
                "query_id": "q1",
                "target_id": "p2",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(2, 3),
                "residue_labels": torch.tensor([0.0, 1.0]),
                "residue_label_mask": torch.tensor([True, False]),
            },
            {
                "query_id": "q2",
                "target_id": "p1",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(3, 3),
                "residue_labels": torch.tensor([1.0, 0.0, 1.0]),
                "residue_label_mask": torch.tensor([True, True, True]),
            },
        ]
    )

    loss = module.training_step(batch, batch_idx=0)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_protein_pooled_lightning_sleec_with_external_score_embeddings(tmp_path):
    checkpoint = tmp_path / "stage1.ckpt"
    _write_stage1_checkpoint(checkpoint, input_dim=4, hidden_dim=5)
    module = ProteinPooledLitModule(
        query_encoder_dims=[2, 3],
        target_encoder_dims=[3, 3],
        embedding_dim=3,
        residue_dim=3,
        pooling="sleec",
        sleec_mode="threshold",
        sleec_threshold=0.34,
        sleec_scorer_hidden_dim=5,
        sleec_score_hidden_dim=4,
        sleec_checkpoint_path=str(checkpoint),
        sleec_freeze_scorer=True,
        attention_logging_interval=1,
    )
    batch = residue_collate_fn(
        [
            {
                "query_id": "q1",
                "target_id": "p1",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(3, 3),
                "score_residue_embeddings": torch.randn(3, 4),
            },
            {
                "query_id": "q1",
                "target_id": "p2",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(2, 3),
                "score_residue_embeddings": torch.randn(2, 4),
            },
            {
                "query_id": "q2",
                "target_id": "p1",
                "query_vec": torch.randn(2),
                "residue_embeddings": torch.randn(3, 3),
                "score_residue_embeddings": torch.randn(3, 4),
            },
        ]
    )

    loss = module.training_step(batch, batch_idx=0)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert all(
        not parameter.requires_grad for parameter in module.model.pooling.scorer.parameters()
    )


def test_reaction_conditioned_data_module_smoke(tmp_path):
    train_pairs = tmp_path / "train_pairs.csv"
    test_pairs = tmp_path / "test_pairs.csv"
    train_rxns = tmp_path / "train_rxns.csv"
    test_rxns = tmp_path / "test_rxns.csv"
    residues_h5 = tmp_path / "residues.h5"

    train_pairs.write_text("pr_id,reaction_id,protein_id\n" "1,rxn1,prot1\n" "2,rxn2,prot2\n")
    test_pairs.write_text("pr_id,reaction_id,protein_id\n1,rxn1,prot1\n")
    train_rxns.write_text("reaction_id,reaction_smiles\n" "rxn1,CCO>>CC=O\n" "rxn2,C>>CC\n")
    test_rxns.write_text("reaction_id,reaction_smiles\nrxn1,CCO>>CC=O\n")

    with h5py.File(residues_h5, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array([b"prot1", b"prot2"]))
        h5_file.create_dataset("vectors", data=np.random.randn(10, 8).astype(np.float32))
        h5_file.create_dataset("offsets", data=np.array([0, 4, 10], dtype=np.int64))

    data_module = ReactionConditionedDataModule(
        train_pairs_path=str(train_pairs),
        test_pairs_path=str(test_pairs),
        train_reactions_path=str(train_rxns),
        test_reactions_path=str(test_rxns),
        protein_residue_embeds_path=str(residues_h5),
        train_batch_size=2,
        retrieval_batch_size=1,
        num_workers=0,
        residue_dim=8,
        max_protein_tokens=4,
        standardize_reactions=False,
    )

    data_module.setup("fit")
    batch = next(iter(data_module.train_dataloader()))

    assert "query_vec" in batch
    assert "residue_embeddings" in batch
    assert "residue_padding_mask" in batch
    assert batch["query_vec"].shape[0] == 2
    assert batch["residue_embeddings"].shape[2] == 8
    assert batch["residue_embeddings"].shape[1] <= 4


def test_reaction_conditioned_multimodal_uses_split_specific_h5_paths(tmp_path):
    train_pairs = tmp_path / "train_pairs.csv"
    test_pairs = tmp_path / "test_pairs.csv"
    train_rxns = tmp_path / "train_rxns.csv"
    test_rxns = tmp_path / "test_rxns.csv"
    residues_h5 = tmp_path / "residues.h5"
    directional_ids = ["rxn1_f", "rxn1_r"]

    train_pairs.write_text("pr_id,reaction_id,protein_id\n1,rxn1,prot1\n")
    test_pairs.write_text("pr_id,reaction_id,protein_id\n1,rxn1,prot1\n")
    train_rxns.write_text("reaction_id,reaction_smiles\nrxn1,CCO>>CC=O\n")
    test_rxns.write_text("reaction_id,reaction_smiles\nrxn1,CCO>>CC=O\n")
    with h5py.File(residues_h5, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array([b"prot1"]))
        h5_file.create_dataset("vectors", data=np.random.randn(4, 8).astype(np.float32))
        h5_file.create_dataset("offsets", data=np.array([0, 4], dtype=np.int64))

    train_t5 = tmp_path / "train_t5.h5"
    val_t5 = tmp_path / "val_t5.h5"
    train_unimol = tmp_path / "train_unimol.h5"
    val_unimol = tmp_path / "val_unimol.h5"
    train_chiro = tmp_path / "train_chiro.h5"
    val_chiro = tmp_path / "val_chiro.h5"
    _write_fixed_reaction_h5(train_t5, directional_ids, base_value=1.0, dim=3)
    _write_fixed_reaction_h5(val_t5, directional_ids, base_value=9.0, dim=3)
    _write_ragged_reaction_h5(train_unimol, directional_ids, base_value=2.0, dim=2)
    _write_ragged_reaction_h5(val_unimol, directional_ids, base_value=8.0, dim=2)
    _write_ragged_reaction_h5(train_chiro, directional_ids, base_value=3.0, dim=2)
    _write_ragged_reaction_h5(val_chiro, directional_ids, base_value=7.0, dim=2)

    data_module = ReactionConditionedDataModule(
        train_pairs_path=str(train_pairs),
        test_pairs_path=str(test_pairs),
        train_reactions_path=str(train_rxns),
        test_reactions_path=str(test_rxns),
        protein_residue_embeds_path=str(residues_h5),
        reaction_representation="multimodal_reaction_attention",
        train_reaction_t5v2_embeds_path=str(train_t5),
        validation_reaction_t5v2_embeds_path=str(val_t5),
        train_reaction_unimol2_embeds_path=str(train_unimol),
        validation_reaction_unimol2_embeds_path=str(val_unimol),
        train_reaction_chiro_embeds_path=str(train_chiro),
        validation_reaction_chiro_embeds_path=str(val_chiro),
        reaction_model_dim=3,
        reaction_unimol_dim=2,
        reaction_chiro_dim=2,
        reaction_use_chiro=True,
        residue_dim=8,
        standardize_reactions=False,
    )

    train_queries = data_module._create_query_dataset(train_rxns, split_name="train")
    validation_queries = data_module._create_query_dataset(
        test_rxns,
        split_name="validation",
    )

    assert torch.allclose(train_queries["rxn1_f"]["reaction_embedding"], torch.full((3,), 1.0))
    assert torch.allclose(validation_queries["rxn1_f"]["reaction_embedding"], torch.full((3,), 9.0))
    assert torch.allclose(
        train_queries["rxn1_f"]["reactant_embeddings"],
        torch.full((1, 2), 2.0),
    )
    assert torch.allclose(
        validation_queries["rxn1_f"]["reactant_chirality_embeddings"],
        torch.full((1, 2), 7.0),
    )


def test_reaction_conditioned_data_module_dual_residue_sources(tmp_path):
    train_pairs = tmp_path / "train_pairs.csv"
    test_pairs = tmp_path / "test_pairs.csv"
    train_rxns = tmp_path / "train_rxns.csv"
    test_rxns = tmp_path / "test_rxns.csv"
    residues_h5 = tmp_path / "residues.h5"
    score_residues_h5 = tmp_path / "score_residues.h5"

    train_pairs.write_text("pr_id,reaction_id,protein_id\n" "1,rxn1,prot1\n" "2,rxn2,prot2\n")
    test_pairs.write_text("pr_id,reaction_id,protein_id\n1,rxn1,prot1\n")
    train_rxns.write_text("reaction_id,reaction_smiles\n" "rxn1,CCO>>CC=O\n" "rxn2,C>>CC\n")
    test_rxns.write_text("reaction_id,reaction_smiles\nrxn1,CCO>>CC=O\n")

    offsets = np.array([0, 4, 10], dtype=np.int64)
    with h5py.File(residues_h5, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array([b"prot1", b"prot2"]))
        h5_file.create_dataset("vectors", data=np.random.randn(10, 8).astype(np.float32))
        h5_file.create_dataset("offsets", data=offsets)
    with h5py.File(score_residues_h5, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array([b"prot1", b"prot2"]))
        h5_file.create_dataset("vectors", data=np.random.randn(10, 5).astype(np.float32))
        h5_file.create_dataset("offsets", data=offsets)

    data_module = ReactionConditionedDataModule(
        train_pairs_path=str(train_pairs),
        test_pairs_path=str(test_pairs),
        train_reactions_path=str(train_rxns),
        test_reactions_path=str(test_rxns),
        protein_residue_embeds_path=str(residues_h5),
        protein_score_residue_embeds_path=str(score_residues_h5),
        train_batch_size=2,
        retrieval_batch_size=1,
        num_workers=0,
        residue_dim=8,
        score_residue_dim=5,
        max_protein_tokens=4,
        standardize_reactions=False,
    )

    data_module.setup("fit")
    batch = next(iter(data_module.train_dataloader()))

    assert batch["residue_embeddings"].shape[2] == 8
    assert batch["score_residue_embeddings"].shape[2] == 5
    assert batch["residue_embeddings"].shape[:2] == batch["score_residue_embeddings"].shape[:2]
    assert torch.equal(batch["residue_padding_mask"], batch["score_residue_padding_mask"])


def test_reaction_conditioned_config_validation_accepts_residue_path():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "data/train_pairs.csv",
                "test_pairs_path": "data/test_pairs.csv",
                "train_reactions_path": "data/train_rxns.csv",
                "test_reactions_path": "data/test_rxns.csv",
                "protein_residue_embeds_path": "data/residues.h5",
                "residue_dim": 1024,
                "max_protein_tokens": 1024,
                "protein_truncation": "ends_center",
            },
            "model": {
                "name": "ReactionConditionedDualModel",
                "query_encoder_dims": [2048, 512],
                "target_encoder_dims": [1024, 512],
                "embedding_dim": 512,
            },
            "training": {"max_epochs": 1},
        }
    )

    validate_config(config)


def test_config_validation_accepts_split_specific_multimodal_reaction_paths():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "data/train_pairs.csv",
                "test_pairs_path": "data/val_pairs.csv",
                "train_reactions_path": "data/train_rxns.csv",
                "test_reactions_path": "data/val_rxns.csv",
                "protein_residue_embeds_path": "data/residues.h5",
                "reaction_representation": "multimodal_reaction_attention",
                "train_reaction_t5v2_embeds_path": "features/train/reactiont5v2.h5",
                "validation_reaction_t5v2_embeds_path": "features/val/reactiont5v2.h5",
                "train_reaction_unimol2_embeds_path": "features/train/unimol2.h5",
                "validation_reaction_unimol2_embeds_path": "features/val/unimol2.h5",
                "train_reaction_chiro_embeds_path": "features/train/chiro.h5",
                "validation_reaction_chiro_embeds_path": "features/val/chiro.h5",
                "reaction_model_dim": 3,
                "reaction_unimol_dim": 2,
                "reaction_chiro_dim": 2,
            },
            "model": {
                "name": "ReactionConditionedDualModel",
                "query_encoder_type": "multimodal_reaction_attention",
                "query_encoder_dims": [3, 3],
                "target_encoder_dims": [8, 3],
                "embedding_dim": 3,
            },
            "training": {"max_epochs": 1},
        }
    )

    validate_config(config)


def test_protein_pooled_config_validation_accepts_attention_pooling():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "data/train_pairs.csv",
                "test_pairs_path": "data/test_pairs.csv",
                "train_reactions_path": "data/train_rxns.csv",
                "test_reactions_path": "data/test_rxns.csv",
                "protein_residue_embeds_path": "data/residues.h5",
                "residue_dim": 1024,
                "max_protein_tokens": 1024,
                "protein_truncation": "ends_center",
            },
            "model": {
                "name": "ProteinPooledDualModel",
                "pooling": "attention",
                "query_encoder_dims": [2048, 512],
                "target_encoder_dims": [1024, 512],
                "embedding_dim": 512,
                "protein_attention_pooling": {
                    "attention_bias": True,
                    "return_attention": False,
                },
            },
            "training": {"max_epochs": 1},
        }
    )

    validate_config(config)


def test_protein_pooled_config_validation_accepts_reaction_fingerprint_attention():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "data/train_pairs.csv",
                "test_pairs_path": "data/test_pairs.csv",
                "train_reactions_path": "data/train_rxns.csv",
                "test_reactions_path": "data/test_rxns.csv",
                "protein_residue_embeds_path": "data/residues.h5",
                "residue_dim": 1024,
                "rdkit_fp_dim": 1024,
                "drfp_dim": 1024,
            },
            "model": {
                "name": "ProteinPooledDualModel",
                "pooling": "attention",
                "query_encoder_dims": [512, 512],
                "target_encoder_dims": [1024, 512],
                "embedding_dim": 512,
                "reaction_fingerprint_attention": {
                    "enabled": True,
                    "token_dim": 512,
                    "hidden_dim": 256,
                    "dropout": 0.0,
                    "attention_bias": True,
                },
                "reaction_hyperbolic_encoder": {
                    "checkpoint_path": None,
                },
            },
            "training": {"max_epochs": 1},
        }
    )

    validate_config(config)


def test_protein_pooled_config_validation_rejects_reaction_attention_with_hyperbolic():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "data/train_pairs.csv",
                "test_pairs_path": "data/test_pairs.csv",
                "train_reactions_path": "data/train_rxns.csv",
                "test_reactions_path": "data/test_rxns.csv",
                "protein_residue_embeds_path": "data/residues.h5",
                "residue_dim": 1024,
            },
            "model": {
                "name": "ProteinPooledDualModel",
                "pooling": "attention",
                "query_encoder_dims": [512, 512],
                "target_encoder_dims": [1024, 512],
                "embedding_dim": 512,
                "reaction_fingerprint_attention": {
                    "enabled": True,
                    "token_dim": 512,
                },
                "reaction_hyperbolic_encoder": {
                    "checkpoint_path": "checkpoints/reaction_stage1.ckpt",
                },
            },
            "training": {"max_epochs": 1},
        }
    )

    with pytest.raises(ValueError, match="reaction_fingerprint_attention"):
        validate_config(config)


def test_protein_pooled_config_validation_accepts_sleec_pooling():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "data/train_pairs.csv",
                "test_pairs_path": "data/test_pairs.csv",
                "train_reactions_path": "data/train_rxns.csv",
                "test_reactions_path": "data/test_rxns.csv",
                "protein_residue_embeds_path": "data/residues.h5",
                "residue_dim": 1024,
                "max_protein_tokens": 1024,
                "protein_truncation": "ends_center",
            },
            "model": {
                "name": "ProteinPooledDualModel",
                "pooling": "sleec",
                "query_encoder_dims": [2048, 512],
                "target_encoder_dims": [1024, 512],
                "embedding_dim": 512,
                "sleec_pooling": {
                    "mode": "topk",
                    "topk_fraction": 0.2,
                    "threshold": 0.5,
                    "scorer_hidden_dim": 128,
                },
            },
            "training": {"max_epochs": 1, "lambda_residue": 0.0},
        }
    )

    validate_config(config)


def test_protein_pooled_config_validation_accepts_external_sleec_scores():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "data/train_pairs.csv",
                "test_pairs_path": "data/test_pairs.csv",
                "train_reactions_path": "data/train_rxns.csv",
                "test_reactions_path": "data/test_rxns.csv",
                "protein_residue_embeds_path": "data/prott5_residues.h5",
                "protein_score_residue_embeds_path": "data/esm2_residues.h5",
                "residue_dim": 1024,
                "score_residue_dim": 1280,
                "max_protein_tokens": 1022,
                "protein_truncation": "ends_center",
            },
            "model": {
                "name": "ProteinPooledDualModel",
                "pooling": "sleec",
                "query_encoder_dims": [2048, 512],
                "target_encoder_dims": [1024, 512],
                "embedding_dim": 512,
                "sleec_pooling": {
                    "mode": "threshold",
                    "threshold": 0.34,
                    "topk_fraction": 0.2,
                    "scorer_hidden_dim": 256,
                    "score_hidden_dim": 1280,
                    "score_embedding_source": "external",
                    "checkpoint_path": "checkpoints/stage1.ckpt",
                    "freeze_scorer": True,
                },
            },
            "training": {"max_epochs": 1, "lambda_residue": 0.0},
        }
    )

    validate_config(config)
