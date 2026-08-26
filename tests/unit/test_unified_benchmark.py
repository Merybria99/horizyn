from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from horizyn.benchmarks.retrieval import (
    BenchmarkTask,
    evaluate_embedding_retrieval_direction,
    evaluate_retrieval_direction,
    group_pairs,
    l2_normalize_embeddings,
    load_candidate_keys_from_residue,
    load_benchmark_suite,
    needs_score_residue_embeddings,
    rank_metrics_for_query,
    screening_metrics_for_query,
    select_score_residue_h5,
    read_id_list,
    run_benchmark_task,
    validate_task_inputs,
)
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset


def test_rank_metrics_for_query_multi_positive():
    scores = torch.tensor([0.9, 0.2, 0.8, 0.1])
    metrics = rank_metrics_for_query(scores, [2, 3], top_k_values=[1, 2, 3])

    assert metrics["top_1"] == 0.0
    assert metrics["top_2"] == 1.0
    assert metrics["top_3"] == 1.0
    assert metrics["top_2_n"] == 0.5
    assert metrics["mean_rank"] == 3.0
    assert metrics["mrr"] == 0.5
    assert metrics["r_precision"] == 0.5
    assert round(metrics["avg_precision"], 6) == round((1 / 2 + 2 / 4) / 2, 6)


def test_screening_metrics_for_query_has_expected_keys():
    scores = torch.tensor([0.95, 0.1, 0.8, 0.2])
    metrics = screening_metrics_for_query(
        scores,
        [0, 2],
        bedroc_alphas=[20.0],
        ef_fractions=[0.5],
    )

    assert set(metrics) == {"bedroc_20", "ef_0_5"}
    assert metrics["ef_0_5"] == 2.0
    assert 0.0 <= metrics["bedroc_20"] <= 1.0


def test_chunked_and_dense_retrieval_metrics_are_identical():
    query_ids = ["q1", "q2", "q3"]
    candidate_ids = ["p1", "p2", "p3", "p4"]
    query_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    candidate_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    positives = {"q1": ["p1"], "q2": ["p2"], "q3": ["p1", "p3"]}
    dense_scores = (
        torch.nn.functional.normalize(query_embeds, dim=-1)
        @ torch.nn.functional.normalize(candidate_embeds, dim=-1).T
    )

    dense = evaluate_retrieval_direction(
        query_ids, candidate_ids, dense_scores, positives, [1, 2, 4]
    )
    chunked = evaluate_embedding_retrieval_direction(
        query_ids,
        candidate_ids,
        query_embeds,
        candidate_embeds,
        positives,
        [1, 2, 4],
        scoring_mode="cosine",
        query_batch_size=2,
    )

    assert chunked == dense


def test_retrieval_ties_use_stable_candidate_order():
    metrics = rank_metrics_for_query(torch.tensor([0.5, 0.5, 0.1]), [1], [1, 2])

    assert metrics["top_1"] == 0.0
    assert metrics["top_2"] == 1.0
    assert metrics["mrr"] == 0.5


def test_cosine_scoring_rejects_zero_norm_and_non_finite_embeddings():
    with pytest.raises(ValueError, match="zero-norm"):
        l2_normalize_embeddings(torch.tensor([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        l2_normalize_embeddings(torch.tensor([[float("nan"), 1.0]]))


def test_candidate_file_rejects_duplicate_ids(tmp_path):
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("p1\np1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate IDs"):
        read_id_list(candidate_path)


def test_residue_candidate_alignment_survives_dropped_empty_row(tmp_path):
    h5_path = tmp_path / "residue.h5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array([b"empty", b"p1", b"p2"]))
        h5_file.create_dataset("vectors", data=np.ones((3, 2), dtype=np.float32))
        h5_file.create_dataset("offsets", data=np.array([0, 0, 1, 3], dtype=np.int64))
    dataset = ResidueEmbedDataset(str(h5_path), drop_empty=True)
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("p1\np2\n", encoding="utf-8")

    keys, stats = load_candidate_keys_from_residue(dataset, candidate_path)

    assert keys == ["p1", "p2"]
    assert stats["zero_length_candidate_count"] == 0

    candidate_path.write_text("empty\np1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="zero_length"):
        load_candidate_keys_from_residue(dataset, candidate_path)


def test_validation_fails_when_positive_is_outside_candidate_manifest(tmp_path):
    task = BenchmarkTask(
        name="strict",
        task_type="retrieval",
        dataset="toy",
        task_label="toy",
        split="test",
        pairs=tmp_path / "pairs.csv",
        reactions=tmp_path / "reactions.csv",
    )
    reaction_inputs = BaseDataset(keys=["r1"], array_data=torch.ones(1, 2), use_key_to_idx=True)

    with pytest.raises(ValueError, match="outside the candidate manifest"):
        validate_task_inputs(
            task,
            reaction_inputs,
            ["p1"],
            ["p1"],
            [("r1", "missing")],
            {},
        )


def test_validation_rejects_exact_train_evaluation_pair_leakage(tmp_path):
    train_pairs = tmp_path / "train_pairs.csv"
    train_pairs.write_text("reaction_id,protein_id\nr1,p1\n", encoding="utf-8")
    task = BenchmarkTask(
        name="strict",
        task_type="retrieval",
        dataset="toy",
        task_label="toy",
        split="test",
        pairs=tmp_path / "pairs.csv",
        reactions=tmp_path / "reactions.csv",
        train_pairs=train_pairs,
    )
    reaction_inputs = BaseDataset(keys=["r1"], array_data=torch.ones(1, 2), use_key_to_idx=True)

    with pytest.raises(ValueError, match="leaks 1 exact positive pair"):
        validate_task_inputs(
            task,
            reaction_inputs,
            ["p1"],
            ["p1"],
            [("r1", "p1")],
            {},
        )


def test_validate_only_benchmark_emits_v2_provenance_end_to_end(tmp_path):
    pairs_path = tmp_path / "pairs.csv"
    reactions_path = tmp_path / "reactions.csv"
    embeddings_path = tmp_path / "proteins.h5"
    checkpoint_path = tmp_path / "model.ckpt"
    config_path = tmp_path / "config.yaml"
    pairs_path.write_text("reaction_id,protein_id\nr1,p1\n", encoding="utf-8")
    reactions_path.write_text("reaction_id,reaction_smiles\nr1,CCO>>CC=O\n", encoding="utf-8")
    with h5py.File(embeddings_path, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array([b"p1"]))
        h5_file.create_dataset("vectors", data=np.ones((1, 4), dtype=np.float32))
    checkpoint_path.write_bytes(b"validate-only checkpoint identity")
    config_path.write_text(
        f"""
data:
  train_pairs_path: {pairs_path}
  test_pairs_path: {pairs_path}
  train_reactions_path: {reactions_path}
  test_reactions_path: {reactions_path}
  protein_embeds_path: {embeddings_path}
model:
  name: DualContrastiveModel
  query_encoder_dims: [2048, 8, 4]
  target_encoder_dims: [4, 8, 4]
  embedding_dim: 4
training:
  max_epochs: 1
""",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        name="strict_e2e",
        task_type="retrieval",
        dataset="toy",
        task_label="toy",
        split="test",
        pairs=pairs_path,
        reactions=reactions_path,
        candidate_embedding_h5=embeddings_path,
        directions=("reaction_to_enzyme",),
    )

    result = run_benchmark_task(
        task,
        checkpoint_path,
        config_path,
        device="cpu",
        validate_only=True,
    )

    assert result["schema_version"] == 2
    assert result["metric_schema_version"] == 2
    assert result["validation"]["split_manifest"]["schema_version"] == 2
    assert result["artifact_manifest"]["artifact_type"] == "benchmark_validation"
    assert result["artifact_manifest"]["candidate_ids_sha256"]


def test_group_pairs_filters_allowed_sets():
    reaction_to_proteins, protein_to_reactions = group_pairs(
        [("r1", "p1"), ("r1", "p2"), ("r2", "p1")],
        allowed_reactions={"r1"},
        allowed_proteins={"p1"},
    )

    assert reaction_to_proteins == {"r1": ["p1"]}
    assert protein_to_reactions == {"p1": ["r1"]}


def test_load_benchmark_suite_resolves_paths(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
version: 1
tasks:
  - name: toy
    type: retrieval
    pairs: data/toy/pairs.csv
    reactions: data/toy/reactions.csv
    candidate_embedding_h5: data/toy/proteins.h5
    candidate_residue_h5:
      prott5: data/toy/proteins_prott5_residue.h5
    candidate_score_residue_h5:
      esm2: data/toy/proteins_esm2_residue.h5
    directions: [reaction_to_enzyme]
    top_k: [1, 5]
""",
        encoding="utf-8",
    )
    tasks = load_benchmark_suite(suite, project_root=Path("/project"))

    assert len(tasks) == 1
    assert tasks[0].name == "toy"
    assert tasks[0].pairs == Path("/project/data/toy/pairs.csv")
    assert tasks[0].candidate_embedding_h5 == Path("/project/data/toy/proteins.h5")
    assert tasks[0].candidate_score_residue_h5["esm2"] == Path(
        "/project/data/toy/proteins_esm2_residue.h5"
    )
    assert tasks[0].directions == ("reaction_to_enzyme",)
    assert tasks[0].top_k == (1, 5)


def test_external_score_residue_selection_uses_manifest_field(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
version: 1
tasks:
  - name: toy
    type: retrieval
    pairs: data/toy/pairs.csv
    reactions: data/toy/reactions.csv
    candidate_residue_h5:
      prott5: data/toy/proteins_prott5_residue.h5
      esm2: data/toy/proteins_esm2_residue.h5
    candidate_score_residue_h5:
      esm2: data/toy/proteins_esm2_score_residue.h5
""",
        encoding="utf-8",
    )
    task = load_benchmark_suite(suite, project_root=Path("/project"))[0]
    config = SimpleNamespace(
        model={"sleec_pooling": {"score_embedding_source": "external"}},
        data={},
    )

    assert needs_score_residue_embeddings(config)
    assert select_score_residue_h5(task, "esm2") == Path(
        "/project/data/toy/proteins_esm2_score_residue.h5"
    )
