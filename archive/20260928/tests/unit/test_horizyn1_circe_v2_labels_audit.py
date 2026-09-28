"""End-to-end tiny-source checks for the independent CIRCE-v2 label audit."""

import csv
import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDITOR = load_script("audit_horizyn1_circe_v2_labels")
EXPORTER = load_script("build_horizyn1_circe_v2_labels")
REACTIONS = load_script("build_horizyn1_reaction_annotations")
NATIVE = load_script("extract_horizyn1_uniprot_annotations").implementation
GRAPH = load_script("audit_horizyn1_reconstruction")


def write_rows(path, columns, rows, delimiter="\t"):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(columns)
        writer.writerows(rows)


def read_rows(path, delimiter="\t"):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def change_rows(path, mutate, delimiter="\t"):
    rows = read_rows(path, delimiter)
    mutate(rows)
    write_rows(path, list(rows[0]), [[row[key] for key in rows[0]] for row in rows], delimiter)


@pytest.fixture
def completed(tmp_path):
    root = tmp_path / "run"
    for folder in ("raw", "clustered", "logs", "annotations", "downloads"):
        (root / folder).mkdir(parents=True)
    sequences = {"P1": "MAAA", "P2": "MCCC", "P3": "MEEE", "M1": "MAAT", "M2": "MCCD", "M3": "MEEA"}
    (root / "raw/raw_proteins.fasta").write_text("".join(f">{key}\n{seq}\n" for key, seq in sequences.items()))
    (root / "clustered/proteins.fasta").write_text("".join(f">{key}\n{sequences[key]}\n" for key in ("P1", "P2", "P3")))
    (root / "clustered/clusters.tsv").write_text("P1\tP1\nP1\tM1\nP2\tP2\nP2\tM2\nP3\tP3\nP3\tM3\n")
    chemistry = [(f"Rh_{i}", "C" * i + ">>" + "C" * i + "O") for i in range(1, 5)]
    write_rows(root / "raw/raw_reactions.tsv", ["reaction_id", "reaction_smiles"], chemistry)
    pairs = [("Rh_1", "P1", "fixture"), ("Rh_2", "M1", "fixture"), ("Rh_2", "P2", "fixture"), ("Rh_3", "M3", "fixture"), ("Rh_4", "P3", "fixture")]
    write_rows(root / "raw/raw_pairs.tsv", ["reaction_id", "protein_id", "sources"], pairs)
    write_rows(root / "clustered/pairs.tsv", ["reaction_id", "protein_id", "sources"], [(rid, {"M1": "P1", "M3": "P3"}.get(pid, pid), source) for rid, pid, source in pairs])
    (root / "raw/raw_manifest.json").write_text(json.dumps({"raw_proteins": 6, "raw_reactions": 4, "raw_pairs": 5}))
    (root / "clustered/clustered_manifest.json").write_text(json.dumps({"clustered_proteins": 3, "clustered_members": 6, "clustered_pairs": 5}))
    graph = GRAPH.Audit(root, "clustered").run()
    assert graph["status"] == "passed", graph
    assert graph["artifact_signatures"] == {str(Path(path).resolve()): AUDITOR.signature(Path(path)) for path in graph["paths"].values()}
    (root / "logs/clustered_integrity_audit.json").write_text(json.dumps(graph))
    annotation = root / "annotations"
    native_path = annotation / "native_uniprot.tsv.gz"
    native_manifest = annotation / "native_uniprot_manifest.json"
    native_source = root / "downloads/uniprot_trembl.dat"
    evidence = {
        "P1": ("1.1.1.1", "NAD+"), "P2": ("1.2.3.4", "[4Fe-4S] cluster"),
        "M1": ("2.1.1.1", "FAD"), "M2": (None, None), "M3": ("3.1.1.1", "heme"),
    }
    records = []
    for protein_id, (ec, cofactor) in evidence.items():
        sequence = sequences[protein_id]
        text = f"ID   ENTRY Unreviewed; 4 AA.\nAC   {protein_id};\n"
        if ec:
            text += f"DE   RecName: Full=Fixture; EC={ec};\n"
        if cofactor:
            text += f"CC   -!- COFACTOR: Name={cofactor};\n"
        records.append(text + f"SQ   SEQUENCE   4 AA;\n     {sequence}\n//\n")
    native_source.write_text("".join(records))
    NATIVE.extract_uniprot_annotations(input_path=native_source, target_fasta=root / "raw/raw_proteins.fasta", output_path=native_path, manifest_path=native_manifest, source_kind="trembl")
    bridge = annotation / "cached_reactions.csv"
    feature_path = annotation / "cached_features.parquet"
    write_rows(bridge, ["reaction_id", "reaction_smiles", "source_reaction_ids"], [(f"rxn_{i}", chemistry[i-1][1], f"Rh_{i}") for i in range(1, 4)], ",")
    pd.DataFrame([
        {"reaction_id": f"rxn_{i}", "raw_reaction_smiles": chemistry[i-1][1], "reaction_type_labels": [kind], "core_cofactor_labels": [cofactor]}
        for i, kind, cofactor in ((1, "oxidoreduction", "NAD"), (2, "hydrolysis", "FAD"), (3, "phosphorylation", "SAM"))
    ]).to_parquet(feature_path, index=False)
    lookup = annotation / "reaction_annotations.json"
    lookup_manifest = annotation / "reaction_annotations_manifest.json"
    REACTIONS.run_build(raw_reactions=root / "raw/raw_reactions.tsv", cached_features=feature_path, cached_reactions=bridge, output=lookup, manifest=lookup_manifest)
    output_dir = annotation / "circe_v2"
    EXPORTER.build_labels(representative_fasta=root / "clustered/proteins.fasta", cluster_map=root / "clustered/clusters.tsv", native_annotations=native_path, native_manifest=native_manifest, reaction_annotations=lookup, reaction_manifest=lookup_manifest, association_pairs=root / "raw/raw_pairs.tsv", output_dir=output_dir, pair_scope="unsplit_inventory")
    return root


def audit(root):
    return AUDITOR.Audit(root).run()


def failing_checks(report):
    return {row["check"] for row in report["errors"]}


def rewrite_npz(root, mutate):
    path = root / "annotations/circe_v2/enzyme_biofp_targets.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)


def test_complete_fixture_passes_real_independent_checks(completed):
    report = audit(completed)
    assert report["status"] == "passed", report
    assert report["observed_counts"]["native_rows"] == 6
    assert report["observed_counts"]["representatives"] == 3
    assert report["observed_counts"]["row_coverage"] == {"mechanism": 2, "native_cofactor": 2, "reaction_cofactor": 2, "cofactor": 2, "direct_ec": 2}
    assert report["observed_counts"]["pair_scope"] == "unsplit_inventory"


@pytest.mark.parametrize("case,expected_check", [
    ("shape", "array_shapes"), ("ids", "npz_ids"), ("dtype", "array_dtypes"),
    ("nan", "array_values"), ("range", "array_values"), ("negative_mask", "positive_only_masks"),
    ("member_mechanism", "source_profile_union"), ("member_cofactor", "source_profile_union"),
    ("confidence", "confidence"), ("cofactor_union", "source_profile_union"),
])
def test_npz_faults_are_caught_beyond_manifest_signatures(completed, case, expected_check):
    def mutate(arrays):
        if case == "shape": arrays["mechanism_targets"] = arrays["mechanism_targets"][:, :7]
        elif case == "ids": arrays["ids"] = arrays["ids"][[1, 0, 2]]
        elif case == "dtype": arrays["cofactor_targets"] = arrays["cofactor_targets"].astype(np.float64)
        elif case == "nan": arrays["mechanism_targets"][0, 0] = np.nan
        elif case == "range": arrays["mechanism_targets"][0, 0] = 2
        elif case == "negative_mask": arrays["mechanism_mask"][2, 1] = True
        elif case == "member_mechanism":
            arrays["mechanism_targets"][2, 1] = 1
            arrays["mechanism_mask"][2, 1] = True
        elif case == "member_cofactor":
            arrays["native_cofactor_targets"][2, 7] = 1
            arrays["native_cofactor_mask"][2, 7] = True
        elif case == "confidence": arrays["cofactor_confidence"][0, 0] = 0.4
        else: arrays["cofactor_targets"][0, 1] = 1
    rewrite_npz(completed, mutate)
    report = audit(completed)
    assert report["status"] == "failed"
    assert expected_check in failing_checks(report), report


def test_direct_ec_cannot_inherit_member_labels(completed):
    path = completed / "annotations/circe_v2/enzyme_ec_labels.csv"
    with path.open("a") as handle:
        handle.write("P3,3.1.1.1,4\n")
    report = audit(completed)
    assert "direct_ec" in failing_checks(report)


def test_direct_ec_missing_row_is_caught(completed):
    path = completed / "annotations/circe_v2/enzyme_ec_labels.csv"
    rows = read_rows(path, ",")
    write_rows(path, list(rows[0]), [[rows[0][key] for key in rows[0]]], ",")
    assert "direct_ec" in failing_checks(audit(completed))


@pytest.mark.parametrize("case", ["conflict_eligibility", "summary_union", "member_counts"])
def test_cluster_member_evidence_is_a_guard_not_gold(completed, case):
    directory = completed / "annotations/circe_v2"
    if case == "conflict_eligibility":
        path = directory / "candidate_eligibility.csv"
        change_rows(path, lambda rows: rows[0].update(ec_negative_candidate_eligible="1", biological_negative_candidate_eligible="1", reason="eligible"), ",")
        expected = "candidate_eligibility"
    else:
        path = directory / "cluster_annotation_summary.tsv.gz"
        change_rows(path, lambda rows: rows[0].update(**({"member_ec_evidence": '["1.1.1.1"]'} if case == "summary_union" else {"member_count": "99"})))
        expected = "cluster_ec_summary"
    assert expected in failing_checks(audit(completed))


def test_native_member_coverage_requires_explicit_unknown_rows(completed):
    path = completed / "annotations/native_uniprot.tsv.gz"
    rows = read_rows(path)
    rows = [row for row in rows if row["protein_id"] != "P3"]
    write_rows(path, list(rows[0]), [[row[key] for key in rows[0]] for row in rows])
    assert "native_coverage" in failing_checks(audit(completed))


def test_native_representative_hash_must_match_its_sequence(completed):
    path = completed / "annotations/native_uniprot.tsv.gz"
    change_rows(path, lambda rows: rows[0].update(sequence_sha256="0" * 64))
    assert "representative_native_sequence" in failing_checks(audit(completed))


def test_pair_scope_and_training_flag_cannot_disagree(completed):
    path = completed / "annotations/circe_v2/label_manifest.json"
    payload = json.loads(path.read_text())
    payload["training_split_declared"] = True
    path.write_text(json.dumps(payload))
    assert "pair_scope" in failing_checks(audit(completed))


@pytest.mark.parametrize("case", ["wrong_descriptor", "omitted_reaction", "wrong_source"])
def test_independent_cache_lookup_recompute_rejects_forged_rows(completed, case):
    path = completed / "annotations/reaction_annotations.json"
    lookup = json.loads(path.read_text())
    if case == "wrong_descriptor":
        lookup["reactions"]["Rh_1"]["mechanism"] = ["phosphate_transfer"]
    elif case == "omitted_reaction":
        del lookup["reactions"]["Rh_1"]
    else:
        lookup["reactions"]["Rh_1"]["source_reaction_id"] = "rxn_2"
    path.write_text(json.dumps(lookup))
    report = audit(completed)
    assert "reaction_lookup" in failing_checks(report) or "reaction_chemistry" in failing_checks(report)


def test_graph_gate_cannot_reuse_pass_after_same_count_rewrite(completed):
    path = completed / "clustered/clusters.tsv"
    path.write_text(path.read_text())
    report = audit(completed)
    assert report["status"] == "incomplete"
    assert "graph_gate" in failing_checks(report)


def test_corrupt_npz_yields_failure_report_instead_of_uncaught_exception(completed):
    path = completed / "annotations/circe_v2/enzyme_biofp_targets.npz"
    path.write_bytes(b"PK\x03\x04not a complete ZIP file")
    report = audit(completed)
    assert report["status"] == "failed"
    assert "artifact_reading" in failing_checks(report)


@pytest.mark.parametrize("relative", ["annotations/circe_v2/enzyme_biofp_targets.npz", "annotations/native_uniprot_manifest.json", "logs/clustered_integrity_audit.json"])
def test_missing_artifacts_are_incomplete(completed, relative):
    (completed / relative).unlink()
    assert audit(completed)["status"] == "incomplete"


def test_partial_artifact_is_incomplete(completed):
    (completed / "annotations/circe_v2/enzyme_biofp_targets.npz.partial").write_text("partial")
    assert audit(completed)["status"] == "incomplete"


def test_input_mutation_during_audit_is_incomplete(completed, monkeypatch):
    original = AUDITOR.Audit.verify

    def changing(self, *args):
        original(self, *args)
        with self.paths["associations"].open("a") as handle:
            handle.write("Rh_1\tP2\tfixture\n")
    monkeypatch.setattr(AUDITOR.Audit, "verify", changing)
    report = audit(completed)
    assert report["status"] == "incomplete"
    assert "input_stability" in failing_checks(report)


def test_partial_appearing_during_audit_is_incomplete(completed, monkeypatch):
    original = AUDITOR.Audit.verify

    def changing(self, *args):
        original(self, *args)
        self.paths["biofp"].with_name(self.paths["biofp"].name + ".partial").write_text("in progress")
    monkeypatch.setattr(AUDITOR.Audit, "verify", changing)
    assert audit(completed)["status"] == "incomplete"


@pytest.mark.parametrize("alias", ["direct", "hardlink", "symlink"])
def test_report_alias_safety_preserves_data(completed, alias):
    source = completed / "annotations/circe_v2/enzyme_biofp_targets.npz"
    before = source.read_bytes()
    target = completed / "audit.json"
    if alias == "direct": target = source
    elif alias == "hardlink": os.link(source, target)
    else: target.symlink_to(source)
    with pytest.raises(ValueError, match="aliases|symlink"):
        AUDITOR.write_report(target, {"status": "passed"}, [source])
    assert source.read_bytes() == before


def test_atomic_report_is_complete_and_has_no_partial_left(completed):
    instance = AUDITOR.Audit(completed)
    report = instance.run()
    target = completed / "logs/label_audit.json"
    AUDITOR.write_report(target, report, list(instance.paths.values()))
    assert json.loads(target.read_text())["status"] == "passed"
    assert not list(target.parent.glob(".label_audit.json.*.partial"))
