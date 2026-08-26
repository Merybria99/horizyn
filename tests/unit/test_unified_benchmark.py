from pathlib import Path
from types import SimpleNamespace

import torch

from horizyn.benchmarks.retrieval import (
    group_pairs,
    load_benchmark_suite,
    needs_score_residue_embeddings,
    rank_metrics_for_query,
    screening_metrics_for_query,
    select_score_residue_h5,
)


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
