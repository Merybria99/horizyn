import csv

import pytest
import torch

from scripts.evaluate_horizyn1_circe_v2 import (
    query_truth, evaluate_direction, read_ids, implementation_signatures,
)


def test_cold_query_truth_includes_cross_quadrant_positives(tmp_path):
    path = tmp_path / "gold.csv"
    with path.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reaction_id", "protein_id"])
        writer.writerows([("r_test", "e_train"), ("r_test", "e_test"), ("r_train", "e_test"), ("r_train", "e_train")])
    r, e = query_truth(path, {"r_test"}, {"e_test"})
    assert dict(r) == {"r_test": {"e_train", "e_test"}}
    assert dict(e) == {"e_test": {"r_train", "r_test"}}
    with pytest.raises(ValueError, match="lack gold"):
        query_truth(path, {"r_missing"}, {"e_test"})


def test_full_catalog_metrics_and_missing_gold_guard():
    q, c = torch.tensor([[1., 0.]]), torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    kwargs = dict(batch_size=1, chunk_size=2, device="cpu", label="r2e")
    results = evaluate_direction(q, c, ["r"], {"r": {"a", "b"}}, {"a": 0, "b": 1, "c": 2}, **kwargs)
    assert results["mrr"] == 1
    assert results["recall_1"] == .5
    assert results["candidates"] == 3
    with pytest.raises(ValueError, match="gold positives missing"):
        evaluate_direction(q, c, ["r"], {"r": {"missing"}}, {"a": 0, "b": 1, "c": 2}, **kwargs)


def test_duplicate_candidate_ids_rejected(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("x\nx\n")
    with pytest.raises(ValueError, match="unique"):
        read_ids(path)


def test_embedding_cache_covers_encoder_implementation():
    signatures = implementation_signatures()
    assert {"horizyn/model.py", "horizyn/protein_pooling_lightning_module.py",
            "scripts/evaluate_protein_pooling.py", "horizyn/reaction_features.py"} <= signatures.keys()
    assert all(len(value) == 64 for value in signatures.values())
