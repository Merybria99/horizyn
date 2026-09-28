import hashlib
import json
import sys

import numpy as np
import pytest

from scripts.generalization_official_evaluate import bootstrap_difference, reaction_cluster_bootstrap, checked_official_truth, EXPECTED, SPLITS, main
from scripts.generalization_export import atomic_json, digest


def records(mrr):
    values = np.asarray(mrr, dtype=np.float64)
    return dict(query_index=np.arange(len(values)), reactzyme_mrr=values,
                first_rank=np.full(len(values), 2), top_1=values * 0,
                top_5=values * 0 + 1, top_10=values * 0 + 1)


def test_paired_query_bootstrap_constant_difference_and_identity_guard():
    base, method = records([.1, .2, .3]), records([.35, .45, .55])
    result = bootstrap_difference(method, base, 1000, 42)["reactzyme_mrr"]
    assert result["delta"] == pytest.approx(.25)
    assert result["lower_95"] == pytest.approx(.25)
    assert result["upper_95"] == pytest.approx(.25)
    method["query_index"] = np.array([1, 0, 2])
    with pytest.raises(ValueError, match="identical query IDs"):
        bootstrap_difference(method, base, 100, 42)


def test_reaction_group_bootstrap_retains_pooled_query_weighting():
    base, method = records([0, 0, 0]), records([1, 1, 0])
    result = reaction_cluster_bootstrap(method, base, np.array([10, 10, 20]), 1000, 42)["reactzyme_mrr"]
    # Giving each reaction group an equal point-estimate weight would yield
    # 1/2; the official enzyme-query estimand must remain 2/3.
    assert result["delta"] == pytest.approx(2 / 3)
    assert result["queries"] == 3 and result["clusters"] == 2
    assert result["lower_95"] == 0 and result["upper_95"] == 1


def test_official_truth_matches_hashed_source_not_only_matrix_shape(tmp_path, monkeypatch):
    monkeypatch.setitem(EXPECTED, "synthetic", (2, 2))
    source = tmp_path / "official_pairs.csv"
    source.write_text("reaction_id,protein_id\nR0,P0\nR1,P1\n")
    catalog = dict(reactions=["R0", "R1"], proteins=["P0", "P1"])
    atomic_json(tmp_path / "catalog.json", catalog)
    manifest = dict(split="synthetic", freeze=dict(freeze_sha256="frozen"),
                    inputs=dict(pairs=dict(path=str(source), sha256=digest(source))))
    atomic_json(tmp_path / "manifest.json", manifest)
    atomic_json(tmp_path / "complete.json", dict(split="synthetic", freeze_sha256="frozen",
        manifest_signature=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()))
    atomic_json(tmp_path / "feature_bundle_receipt.json", dict(inputs=dict(
        catalog=dict(sha256=digest(tmp_path / "catalog.json")))))
    np.savez(tmp_path / "pairs.npz", test=np.array([[0, 0], [1, 1]]))
    checked, edges = checked_official_truth(tmp_path, "synthetic", "frozen", {})
    assert checked == catalog and edges.tolist() == [[0, 0], [1, 1]]
    # Retains both query and candidate universes, but changes biological labels.
    np.savez(tmp_path / "pairs.npz", test=np.array([[0, 1], [1, 0]]))
    with pytest.raises(ValueError, match="hashed official associations"):
        checked_official_truth(tmp_path, "synthetic", "frozen", {})


def test_complete_six_cell_report_has_fixed_primary_and_known_metrics(tmp_path, monkeypatch):
    freeze_path = tmp_path / "freeze.json"
    atomic_json(freeze_path, dict(frozen_before_held_out_evaluation=True, primary_seed=42))
    freeze_sha = digest(freeze_path)
    methods = [dict(label="F3", primary=False, panels={}),
               dict(label="combined_seed42", primary=True, seed=42, panels={})]
    tiger = dict(source="https://example.test/reference", method="synthetic reference")
    for split in SPLITS:
        monkeypatch.setitem(EXPECTED, split, (2, 2))
        directory = tmp_path / f"features_test_{split}"
        directory.mkdir()
        pairs_path = directory / "source.csv"
        pairs_path.write_text("reaction_id,protein_id\nR0,P0\nR1,P1\n")
        atomic_json(directory / "catalog.json", dict(reactions=["R0", "R1"], proteins=["P0", "P1"]))
        catalog_sha = digest(directory / "catalog.json")
        manifest = dict(split=split, freeze=dict(freeze_sha256=freeze_sha),
                        inputs=dict(pairs=dict(path=str(pairs_path), sha256=digest(pairs_path))))
        atomic_json(directory / "manifest.json", manifest)
        atomic_json(directory / "complete.json", dict(split=split, freeze_sha256=freeze_sha,
            manifest_signature=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()))
        atomic_json(directory / "feature_bundle_receipt.json", dict(inputs=dict(catalog=dict(sha256=catalog_sha))))
        np.savez(directory / "pairs.npz", test=np.array([[0, 0], [1, 1]]))
        prediction = tmp_path / f"prediction_{split}"
        prediction.mkdir()
        # Baseline mistakes one query in each direction: (1 + 1/2) / 2.
        np.savez(prediction / "scores.npz", baseline=np.array([[.4, .8], [.1, .5]], np.float32),
                 selected=np.eye(2, dtype=np.float32))
        atomic_json(prediction / "complete.json", dict(labels_used=False,
            output_sha256=digest(prediction / "scores.npz"), inputs=dict(catalog=dict(sha256=catalog_sha))))
        for method, score_key in zip(methods, ("baseline", "selected")):
            method["panels"][split] = dict(path=str(prediction / "scores.npz"), score_key=score_key)
        tiger[split] = {direction: dict(mrr=.5, hit1=.5, hit10=1)
                        for direction in ("reaction_to_enzyme", "enzyme_to_reaction")}
    atomic_json(tmp_path / "tiger.json", tiger)
    atomic_json(tmp_path / "scores.json", dict(freeze=dict(path=str(freeze_path), sha256=freeze_sha),
        baseline_method="F3", methods=methods))
    output = tmp_path / "evaluated"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--manifest", str(tmp_path / "scores.json"),
        "--feature-root", str(tmp_path), "--output", str(output), "--tiger-reference", str(tmp_path / "tiger.json"),
        "--bootstrap-replicates", "100"])
    main()
    result = json.loads((output / "summary.json").read_text())
    assert result["primary_method"] == "combined_seed42"
    assert result["six_cell_macro"] == dict(F3=.75, combined_seed42=1.)
    assert len(json.loads((output / "paired_bootstrap.json").read_text())) == 30
    assert len(json.loads((output / "reaction_cluster_bootstrap.json").read_text())) == 15
