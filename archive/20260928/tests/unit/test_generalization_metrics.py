import pytest
import torch

from scripts.generalization_metrics import evaluate_scores


def test_stable_ties_duplicate_edges_and_all_positive_macro():
    result = evaluate_scores(torch.tensor([[1., 1., 1.], [0., 2., 1.]]), {
        "reaction_index": [0, 0, 0, 1], "enzyme_index": [1, 2, 2, 0],
        "reaction_seen": [True, False], "enzyme_seen": [False, True, True],
    }, batch_size=1)
    r2e = result["summary"]["reaction_to_enzyme"]
    assert r2e["all"]["reactzyme_mrr"] == pytest.approx(((1/2 + 1/3)/2 + 1/3)/2)
    assert r2e["all"]["first_positive_mrr"] == pytest.approx((1/2 + 1/3)/2)
    assert r2e["all"]["num_positive_edges"] == 3
    assert result["per_positive"]["reaction_to_enzyme"]["rank"].tolist() == [2, 3, 3]
    e2r = result["summary"]["enzyme_to_reaction"]
    assert e2r["seen_reaction"]["num_queries"] == 2
    assert e2r["unseen_reaction"]["num_queries"] == 1
    assert e2r["unseen_reaction"]["reactzyme_mrr"] == pytest.approx(0.5)


def test_unlabelled_queries_omitted_and_empty_strata_explicit():
    result = evaluate_scores(torch.eye(2), {
        "reaction_index": [0], "enzyme_index": [0], "reaction_seen": [True, True],
    })
    summary = result["summary"]["reaction_to_enzyme"]
    assert summary["all"]["num_queries"] == 1
    assert summary["unseen_reaction"]["num_queries"] == 0
    assert summary["unseen_reaction"]["reactzyme_mrr"] is None


def test_out_of_range_truth_rejected():
    with pytest.raises(ValueError, match="outside"):
        evaluate_scores(torch.eye(2), {"reaction_index": [2], "enzyme_index": [0]})
