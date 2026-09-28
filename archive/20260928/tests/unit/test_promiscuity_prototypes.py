from __future__ import annotations

from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

from horizyn.losses import BidirectionalSampledMultiPositiveInfoNCELoss
from horizyn.model import ResidualEnzymePrototypeHead
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_conditioned_data_module import (
    EnzymeGroupedBatchSampler,
    NamedEmbeddingDataset,
    ReactionConditionedDataModule,
)
from scripts.cache_frozen_prototype_embeddings import (
    file_signature,
    resolve_enzyme_ids,
    write_embedding_hdf5,
)


def test_single_prototype_score_is_exact_cosine() -> None:
    head = ResidualEnzymePrototypeHead(
        embedding_dim=3,
        prototype_count=1,
        bottleneck_dim=2,
        aggregation_temperature=0.1,
    )
    reactions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    enzymes = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

    actual = head.score(reactions, enzymes)
    expected = F.normalize(reactions, dim=-1) @ F.normalize(enzymes, dim=-1).t()

    assert not head.state_dict()
    assert torch.allclose(actual, expected, atol=1e-6)


def test_two_prototypes_are_normalized_and_trainable() -> None:
    torch.manual_seed(7)
    head = ResidualEnzymePrototypeHead(
        embedding_dim=8,
        prototype_count=2,
        bottleneck_dim=4,
        residual_gate_init=0.05,
        aggregation_temperature=0.1,
    )
    reactions = torch.randn(3, 8)
    enzymes = torch.randn(4, 8)

    scores, details = head.score(
        reactions,
        enzymes,
        return_details=True,
    )
    loss = -scores.mean()
    loss.backward()

    assert scores.shape == (3, 4)
    assert details["responsibilities"].shape == (3, 4, 2)
    assert torch.allclose(
        details["responsibilities"].sum(dim=-1),
        torch.ones(3, 4),
        atol=1e-6,
    )
    prototypes = head(enzymes)
    assert torch.allclose(prototypes.norm(dim=-1), torch.ones(4, 2), atol=1e-6)
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in head.parameters()
    )


def test_prototype_cosine_supports_signed_square_root_features() -> None:
    torch.manual_seed(9)
    head = ResidualEnzymePrototypeHead(
        embedding_dim=4,
        prototype_count=2,
        bottleneck_dim=3,
    )
    reactions = torch.tensor([[4.0, -1.0, 0.25, 0.0]])
    enzymes = torch.tensor([[1.0, -4.0, 0.0, 0.25]])

    actual = head.score(reactions, enzymes, feature_power=0.5)
    prototypes, details = head(enzymes, return_details=True)
    transformed_reactions = torch.sign(reactions) * torch.abs(reactions).sqrt()
    transformed_prototypes = torch.sign(prototypes) * torch.abs(prototypes).sqrt()
    component_scores = torch.einsum(
        "qd,tkd->qtk",
        F.normalize(transformed_reactions, dim=-1),
        F.normalize(transformed_prototypes, dim=-1),
    )
    expected = head.aggregation_temperature * torch.logsumexp(
        component_scores / head.aggregation_temperature
        + details["log_priors"].unsqueeze(0),
        dim=-1,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_bidirectional_sampled_loss_ignores_unlabelled_cells() -> None:
    loss_fn = BidirectionalSampledMultiPositiveInfoNCELoss(beta=2.0)
    query_idx = torch.tensor([0, 1], dtype=torch.long)
    target_idx = torch.tensor([0, 1], dtype=torch.long)
    biological = torch.tensor(
        [[False, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    random_negative = torch.zeros(2, 3, dtype=torch.bool)
    dists = torch.tensor(
        [[0.1, 0.8, -10.0], [0.9, 0.2, -10.0]],
        requires_grad=True,
    )

    baseline, components = loss_fn(
        dists,
        query_idx,
        target_idx,
        biological,
        random_negative,
        return_components=True,
    )
    changed = dists.detach().clone()
    changed[:, 2] = 100.0
    changed_loss = loss_fn(
        changed,
        query_idx,
        target_idx,
        biological,
        random_negative,
    )

    assert torch.allclose(baseline, changed_loss)
    assert components["r2e_valid_anchors"].item() == 2
    assert components["e2r_valid_anchors"].item() == 2
    baseline.backward()
    assert torch.isfinite(dists.grad).all()


class _FakeTupleDataset:
    def __init__(self) -> None:
        self.keys = [str(index) for index in range(6)]
        rows = [
            ("r0", "e0"),
            ("r1", "e0"),
            ("r2", "e1"),
            ("r3", "e1"),
            ("r4", "e2"),
            ("r5", "e3"),
        ]
        self.tuple_dataset = {
            str(index): {"query_id": query_id, "target_id": target_id}
            for index, (query_id, target_id) in enumerate(rows)
        }

    def __len__(self) -> int:
        return len(self.keys)


def test_enzyme_grouped_sampler_places_multiple_positive_rows_in_batch() -> None:
    dataset = _FakeTupleDataset()
    sampler = EnzymeGroupedBatchSampler(
        dataset,  # type: ignore[arg-type]
        batch_size=4,
        anchors_per_batch=1,
        positives_per_anchor=2,
        seed=3,
    )

    batch = next(iter(sampler))
    target_ids = [dataset.tuple_dataset[dataset.keys[index]]["target_id"] for index in batch]

    assert len(batch) == 4
    assert len(set(batch)) == 4
    assert max(target_ids.count(target_id) for target_id in set(target_ids)) >= 2


def test_cached_base_embeddings_train_only_the_prototype_head() -> None:
    torch.manual_seed(11)
    module = ProteinPooledLitModule(
        query_encoder_dims=[4, 4],
        target_encoder_dims=[4, 4],
        embedding_dim=4,
        residue_dim=4,
        pooling="mean",
        training_stage="prototype_only",
        enzyme_prototype_count=2,
        enzyme_prototype_bottleneck_dim=3,
    )
    batch = {
        "cached_query_embedding": torch.randn(4, 4),
        "cached_target_embedding": torch.randn(4, 4),
        "query_id": ["q0", "q1", "q0", "q2"],
        "target_id": ["e0", "e1", "e2", "e2"],
        "pair_type": ["positive"] * 4,
    }

    loss, batch_size, attention, components = module._compute_cached_full_batch_loss(
        batch,
        positive_pair_source="observed_pairs",
    )
    loss.backward()

    assert batch_size == 4
    assert attention == {}
    assert isinstance(components, dict)
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in module.model.enzyme_prototype_head.parameters()
    )
    assert all(
        parameter.grad is None
        for name, parameter in module.model.named_parameters()
        if not name.startswith("enzyme_prototype_head.")
    )


def test_frozen_embedding_cache_round_trip(tmp_path: Path) -> None:
    cache_path = tmp_path / "base.h5"
    expected = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )
    write_embedding_hdf5(
        cache_path,
        ["prot_a", "prot_b"],
        expected,
        checkpoint=tmp_path / "parent.ckpt",
        config=tmp_path / "config.yaml",
    )

    dataset = NamedEmbeddingDataset(cache_path, "cached_target_embedding")

    assert dataset.keys == ["prot_a", "prot_b"]
    assert dataset.vec_dim == 3
    assert torch.equal(dataset["prot_b"]["cached_target_embedding"], expected[1])


def test_frozen_embedding_cache_can_preserve_float32(tmp_path: Path) -> None:
    cache_path = tmp_path / "base.h5"
    expected = torch.tensor([[1.0001, -0.33333334]], dtype=torch.float32)
    write_embedding_hdf5(
        cache_path,
        ["prot_a"],
        expected,
        checkpoint=tmp_path / "parent.ckpt",
        config=tmp_path / "config.yaml",
        output_dtype="float32",
    )
    with h5py.File(cache_path, "r") as handle:
        assert str(handle["vectors"].dtype) == "float32"
        assert handle.attrs["dtype"] == "float32"
    dataset = NamedEmbeddingDataset(cache_path, "cached_target_embedding")
    assert torch.equal(dataset["prot_a"]["cached_target_embedding"], expected[0])


def test_cache_resolves_unique_protein_prefix_variants() -> None:
    resolved, missing = resolve_enzyme_ids(
        {"P1", "uprot_P2", "missing"},
        {"prot_P1", "prot_P2", "prot_P3"},
    )

    assert resolved == {"prot_P1", "prot_P2"}
    assert missing == ["missing"]


def test_content_signature_survives_idempotent_file_rewrite(tmp_path: Path) -> None:
    input_path = tmp_path / "pairs.csv"
    input_path.write_text("same content\n", encoding="utf-8")
    first = file_signature(input_path, content_hash=True)
    input_path.write_text("same content\n", encoding="utf-8")
    second = file_signature(input_path, content_hash=True)

    assert first == second
    assert "mtime_ns" not in first


def test_cached_data_module_bypasses_raw_tower_inputs(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.csv"
    pairs_path.write_text(
        "pr_id,reaction_id,protein_id\n0,r0,e0\n1,r1,e1\n",
        encoding="utf-8",
    )
    enzyme_cache = tmp_path / "enzymes.h5"
    reaction_cache = tmp_path / "reactions.h5"
    write_embedding_hdf5(
        enzyme_cache,
        ["e0", "e1"],
        torch.randn(2, 4),
        checkpoint=tmp_path / "parent.ckpt",
        config=tmp_path / "config.yaml",
    )
    write_embedding_hdf5(
        reaction_cache,
        ["r0_f", "r1_f"],
        torch.randn(2, 4),
        checkpoint=tmp_path / "parent.ckpt",
        config=tmp_path / "config.yaml",
    )
    data_module = ReactionConditionedDataModule(
        train_pairs_path=str(pairs_path),
        test_pairs_path=str(pairs_path),
        train_reactions_path=str(tmp_path / "unused_train_reactions.csv"),
        test_reactions_path=str(tmp_path / "unused_validation_reactions.csv"),
        protein_residue_embeds_path=str(tmp_path / "unused_residues.h5"),
        cached_enzyme_base_embeds_path=str(enzyme_cache),
        cached_train_reaction_base_embeds_path=str(reaction_cache),
        train_batch_size=2,
        num_workers=0,
        reaction_direction_mode="forward_only",
        validation_enabled=False,
    )

    data_module.setup("fit")
    batch = next(iter(data_module.train_dataloader()))

    assert batch["cached_query_embedding"].shape == (2, 4)
    assert batch["cached_target_embedding"].shape == (2, 4)
    assert sorted(batch["query_id"]) == ["r0_f", "r1_f"]
