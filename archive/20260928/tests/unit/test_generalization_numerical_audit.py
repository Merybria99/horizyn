import numpy as np
import pytest

from scripts.generalization_numerical_audit import independent_ranks


def test_direct_comparisons_resolve_positive_ties_by_candidate_index():
    scores = np.array([[.5, .5, .5], [.9, .1, .1]], np.float32)
    report, ranks, _ = independent_ranks(scores, np.array([0, 1]), np.array([2, 1]), batch_size=1)
    assert ranks["stable"].tolist() == [3, 2]
    assert ranks["optimistic"].tolist() == [1, 2]
    assert ranks["average"].tolist() == [2, 2.5]
    assert ranks["pessimistic"].tolist() == [3, 3]
    assert report["metrics_by_tie_policy"]["stable"]["reactzyme_mrr"] == pytest.approx(5 / 12)
    assert report["metrics_by_tie_policy"]["optimistic"]["reactzyme_mrr"] == .75
    assert report["incidence"]["queries_with_any_score_tie"] == 2
    assert report["incidence"]["queries_with_top_score_tie"] == 1
    assert report["incidence"]["positive_edges_with_tied_score"] == 2


def test_all_positive_macro_mrr_is_distinct_from_first_positive_mrr():
    scores = np.array([[.9, .8, .7], [.9, .8, .7]], np.float32)
    report, _, _ = independent_ranks(scores, np.array([0, 0, 1]), np.array([0, 2, 1]))
    metrics = report["metrics_by_tie_policy"]["stable"]
    assert metrics["reactzyme_mrr"] == pytest.approx(((1 + 1 / 3) / 2 + 1 / 2) / 2)
    assert metrics["first_positive_mrr"] == .75
    assert report["incidence"]["positive_edges_with_tied_score"] == 0
