from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from horizyn.benchmarks.retrieval import (
    BenchmarkTask,
    LEVEL1_BENCHMARK_PROTOCOLS,
    candidate_pool_policy,
    evaluate_embedding_retrieval_direction,
    evaluate_retrieval_direction,
    expand_bidirectional_pairs,
    group_pairs,
    l2_normalize_embeddings,
    load_candidate_keys_from_residue,
    load_benchmark_suite,
    needs_score_residue_embeddings,
    rank_metrics_for_query,
    read_pairs,
    restrict_candidates_to_test_positives,
    screening_metrics_for_query,
    select_score_residue_h5,
    read_id_list,
    run_benchmark_task,
    validate_task_inputs,
    validate_level1_benchmark_sources,
    validate_level1_candidate_pool,
)
from horizyn.artifacts import sha256_strings
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
    assert metrics["first_positive_mrr"] == 0.5
    assert metrics["reactzyme_mrr"] == pytest.approx((1 / 2 + 1 / 4) / 2)
    assert metrics["r_precision"] == 0.5
    assert round(metrics["avg_precision"], 6) == round((1 / 2 + 2 / 4) / 2, 6)


def test_reactzyme_metric_protocol_uses_published_all_positive_mrr():
    metrics = rank_metrics_for_query(
        torch.tensor([0.9, 0.2, 0.8, 0.1]),
        [2, 3],
        [1, 2, 5],
        metric_protocol="reactzyme",
    )

    assert metrics["mrr"] == pytest.approx((1 / 2 + 1 / 4) / 2)
    assert metrics["reactzyme_mrr"] == metrics["mrr"]
    assert metrics["first_positive_mrr"] == 0.5
    assert metrics["top_5_n"] == pytest.approx(2 / 5)


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


def test_candidate_manifest_can_be_the_release_fasta(tmp_path):
    candidate_path = tmp_path / "proteins.fasta"
    candidate_path.write_text(
        ">p2 description with spaces\nACDE\n>p1\nFGHI\n",
        encoding="utf-8",
    )

    assert read_id_list(candidate_path) == ["p2", "p1"]


def test_csv_readers_reject_short_rows_instead_of_creating_none_id(tmp_path):
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("protein_id,description\nmissing-cell\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing cell"):
        read_id_list(candidates, column="description")

    pairs = tmp_path / "pairs.csv"
    pairs.write_text("reaction_id,protein_id\nr1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty ID"):
        read_pairs(pairs, "reaction_id", "protein_id")


def test_test_positive_candidate_protocol_is_derived_and_enforced(tmp_path):
    task = BenchmarkTask(
        name="reactzyme_time",
        task_type="retrieval",
        dataset="ReactZyme",
        task_label="time",
        split="time",
        pairs=tmp_path / "pairs.csv",
        reactions=tmp_path / "reactions.csv",
        metric_protocol="reactzyme",
        candidates_from_test_positives=True,
    )

    keys, stats = restrict_candidates_to_test_positives(
        task,
        [("r1", "p2"), ("r2", "p1")],
        ["train-only", "p1", "p2"],
        {"candidate_manifest_count": 3},
    )

    assert keys == ["p1", "p2"]
    assert stats["candidate_manifest_count"] == 2
    assert stats["candidate_count"] == 2
    assert stats["requested_candidate_count"] == 2
    assert stats["candidate_source_store_count"] == 3
    with pytest.raises(ValueError, match="missing 1 test-positive"):
        restrict_candidates_to_test_positives(task, [("r1", "missing")], ["p1"], {})


def test_horizyn_protocol_expands_forward_and_reverse_queries(tmp_path):
    task = BenchmarkTask(
        name="horizyn_sota",
        task_type="retrieval",
        dataset="Horizyn",
        task_label="Horizyn",
        split="test",
        pairs=tmp_path / "pairs.csv",
        reactions=tmp_path / "reactions.csv",
        metric_protocol="horizyn",
        bidirectional_reactions=True,
    )

    assert expand_bidirectional_pairs(task, [("r1", "p1")]) == [
        ("r1_f", "p1"),
        ("r1_r", "p1"),
    ]


def test_custom_horizyn_task_is_not_mislabeled_as_the_published_pool(tmp_path):
    task = BenchmarkTask(
        name="horizyn_stress",
        task_type="retrieval",
        dataset="Horizyn",
        task_label="stress",
        split="deployment",
        pairs=tmp_path / "pairs.csv",
        reactions=tmp_path / "reactions.csv",
        metric_protocol="horizyn",
    )

    assert candidate_pool_policy(task, ["p1", "p2"], [("r1", "p1")]) == (
        "full_embedding_store"
    )


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
    assert result["metric_schema_version"] == 3
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


def test_level1_protocol_rejects_structural_deviations_at_load_time(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
tasks:
  - name: mislabeled_horizyn
    type: retrieval
    benchmark_protocol: horizyn_release_v1
    metric_protocol: horizyn
    bidirectional_reactions: true
    pairs: pairs.csv
    train_pairs: train.csv
    reactions: reactions.csv
    directions: [reaction_to_enzyme]
    top_k: [1, 10]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not implement horizyn_release_v1"):
        load_benchmark_suite(suite, project_root=tmp_path)


def test_level1_protocol_verifies_semantic_split_and_candidate_identity(tmp_path, monkeypatch):
    pairs_path = tmp_path / "test_pairs.csv"
    train_path = tmp_path / "train_pairs.csv"
    reactions_path = tmp_path / "reactions.csv"
    candidate_path = tmp_path / "proteins.fasta"
    pairs_path.write_text("reaction_id,protein_id\nr1,p1\nr1,p2\n", encoding="utf-8")
    train_path.write_text("reaction_id,protein_id\nr0,p0\n", encoding="utf-8")
    reactions_path.write_text(
        "reaction_id,reaction_smiles\nr1,CCO>>CC=O\n",
        encoding="utf-8",
    )
    candidate_path.write_text(">p1\nAAAA\n>p2\nCCCC\n>p3\nDDDD\n", encoding="utf-8")
    test_pairs = [("r1", "p1"), ("r1", "p2")]
    train_pairs = [("r0", "p0")]
    reaction_records = [("r1", "CCO>>CC=O")]
    candidate_ids = ["p1", "p2", "p3"]
    monkeypatch.setitem(
        LEVEL1_BENCHMARK_PROTOCOLS,
        "toy_release_v1",
        {
            "upstream_revision": "test-revision",
            "metric_protocol": "horizyn",
            "candidates_from_test_positives": False,
            "bidirectional_reactions": True,
            "directions": ("reaction_to_enzyme",),
            "top_k": (1,),
            "test_pair_count": len(test_pairs),
            "test_pair_digest": sha256_strings(
                sorted(f"{reaction}\t{protein}" for reaction, protein in test_pairs)
            ),
            "train_pair_count": len(train_pairs),
            "train_pair_digest": sha256_strings(
                sorted(f"{reaction}\t{protein}" for reaction, protein in train_pairs)
            ),
            "reaction_row_count": len(reaction_records),
            "reaction_digest": sha256_strings(
                sorted(f"{reaction}\t{smiles}" for reaction, smiles in reaction_records)
            ),
            "raw_query_count": 1,
            "evaluation_query_count": 2,
            "positive_protein_count": 2,
            "candidate_count": len(candidate_ids),
            "candidate_digest": sha256_strings(sorted(candidate_ids)),
        },
    )
    task = BenchmarkTask(
        name="toy",
        task_type="retrieval",
        dataset="toy",
        task_label="toy",
        split="test",
        pairs=pairs_path,
        train_pairs=train_path,
        reactions=reactions_path,
        candidate_ids=candidate_path,
        metric_protocol="horizyn",
        benchmark_protocol="toy_release_v1",
        bidirectional_reactions=True,
        directions=("reaction_to_enzyme",),
        top_k=(1,),
    )

    source_stats = validate_level1_benchmark_sources(task, test_pairs)
    candidate_stats = validate_level1_candidate_pool(task, candidate_ids)

    assert source_stats["release_compliance"] == "source_files_verified"
    assert source_stats["declared_candidate_count"] == 3
    assert candidate_stats["release_candidate_pool_compliance"] == "verified"


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("top_k", [1.5], "positive integers"),
        ("top_k", [1, 1], "duplicates"),
        ("directions", ["reaction_to_enyzme"], "unsupported directions"),
        ("bedroc_alphas", [float("inf")], "finite"),
        ("ef_fractions", [1.1], r"\[0, 1\]"),
    ],
)
def test_load_benchmark_suite_rejects_invalid_metric_schema(tmp_path, field, value, message):
    import yaml

    suite = tmp_path / "suite.yaml"
    payload = {
        "tasks": [
            {
                "name": "toy",
                "type": "retrieval",
                "pairs": "pairs.csv",
                "reactions": "reactions.csv",
                field: value,
            }
        ]
    }
    suite.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_benchmark_suite(suite, project_root=tmp_path)


def test_screening_metrics_remain_finite_for_extreme_alpha():
    metrics = screening_metrics_for_query(
        torch.tensor([0.9, 0.8, 0.1, 0.0]),
        [0],
        bedroc_alphas=[1e308],
        ef_fractions=[0.5],
    )

    assert metrics["bedroc_1e+308"] == pytest.approx(1.0)
