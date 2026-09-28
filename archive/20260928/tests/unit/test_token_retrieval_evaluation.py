"""Regression checks for using the actual CIRCE validation/test protocols."""
from collections import defaultdict
import csv
from types import SimpleNamespace

import pytest
import torch

from horizyn.token_retrieval_evaluation import (
    CirceEvaluationPlan, circe_directional_metrics, evaluate_circe_score_matrix,
)
from scripts.evaluate_protein_pooling import append_retrieval_metrics, mean_metric_results


def test_validation_uses_stable_candidate_order_for_ties():
    scores = torch.tensor([[1., 1., 0., -1.]])
    result, per_query = circe_directional_metrics(
        scores, ["q"], ["p0", "p1", "p2", "p3"], {"q": ["p1", "p3"]}, "validation")
    assert result["first_positive_mrr"] == 0.5
    assert result["reactzyme_mrr"] == pytest.approx(0.375)
    assert per_query["mean_rank"].tolist() == [2.0]


def test_validation_macro_average_handles_partial_final_batch():
    anchors = [f"q{i}" for i in range(514)]
    scores = torch.tensor([[1., 0.]]).expand(514, -1)
    positives = {q: ["p0" if i < 512 else "p1"] for i, q in enumerate(anchors)}
    result, _ = circe_directional_metrics(scores, anchors, ["p0", "p1"], positives, "validation")
    assert result["num_queries"] == 514
    assert result["mrr"] == pytest.approx(513 / 514)


def test_test_metrics_call_original_standalone_with_ties_and_duplicate_positives():
    scores = torch.tensor([[1., 1., 0., -1.], [0., 3., 2., 1.]])
    positives = {"q0": ["p1", "p3", "p3"], "q1": ["p0", "p2"]}
    expected = defaultdict(list)
    for row, q in zip(scores, positives):
        append_retrieval_metrics(expected, row, torch.tensor([int(e[1:]) for e in positives[q]]))
    actual, _ = circe_directional_metrics(
        scores, list(positives), ["p0", "p1", "p2", "p3"], positives, "test")
    assert actual.pop("num_queries") == 2
    assert actual == mean_metric_results(expected)
    assert "avg_precision" in actual and "r_precision" in actual


@pytest.fixture
def protocol(tmp_path):
    rows = [("a", "r1", "p2"), ("b", "r1", "p2"), ("c", "r2", "p1")]
    for split in ["validation", "test"]:
        with (tmp_path / f"{split}_pairs.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pr_id", "reaction_id", "protein_id"])
            writer.writerows(rows)
        (tmp_path / f"{split}_candidate_ids.txt").write_text("p3\np1\np2\n")
    return SimpleNamespace(split_dir=tmp_path, sequences={"p1": "A", "p2": "B", "p3": "C"},
                           smiles={"r1": "C", "r2": "O"})


def test_candidate_order_forward_ids_and_positive_semantics_match_circe(protocol):
    validation = CirceEvaluationPlan.from_protocol(protocol, "validation")
    test = CirceEvaluationPlan.from_protocol(protocol, "test")
    assert validation.candidate_ids == ["p3", "p1", "p2"]
    assert test.candidate_ids == ["p2", "p1"]
    assert validation.query_ids == test.query_ids == ["r1_f", "r2_f"]
    assert validation.query_to_targets["r1_f"] == ["p2"]
    assert test.query_to_targets["r1_f"] == ["p2", "p2"]


def test_distractors_are_candidates_but_not_enzyme_query_anchors(protocol):
    plan = CirceEvaluationPlan.from_protocol(protocol, "validation")
    result = evaluate_circe_score_matrix(torch.tensor([[0., 0.2, 0.8], [0., 0.9, 0.1]]), plan)
    assert result["num_enzyme_candidates"] == 3
    assert result["enzyme_to_reaction/num_queries"] == 2
    assert result["reaction_to_enzyme/mrr"] == 1.
    assert result["enzyme_to_reaction/mrr"] == 1.


def test_missing_benchmark_protein_fails_without_changing_candidate_pool(protocol):
    del protocol.sequences["p2"]
    with pytest.raises(ValueError):
        CirceEvaluationPlan.from_protocol(protocol, "validation")
    with pytest.raises(ValueError):
        CirceEvaluationPlan.from_protocol(protocol, "test")


def test_harmonic_validation_and_arithmetic_test_summary_keep_circe_names(protocol):
    plan = CirceEvaluationPlan.from_protocol(protocol, "validation")
    result = evaluate_circe_score_matrix(torch.tensor([[0., 1., 0.5], [0., 0.2, 0.9]]), plan)
    r2e = result["reaction_to_enzyme/mrr"]
    e2r = result["enzyme_to_reaction/mrr"]
    assert result["mean_bidirectional_mrr"] == (r2e + e2r) / 2
    assert result["balanced_mrr"] == pytest.approx(2 * r2e * e2r / (r2e + e2r))
