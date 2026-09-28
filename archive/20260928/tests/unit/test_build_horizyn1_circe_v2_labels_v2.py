"""Versioned representative cofactor targets, annotation status, and provenance."""

import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[2]


def import_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = import_script("build_horizyn1_circe_v2_labels_v2")

EXPECTED_MECHANISMS = [
    "redox_carbonyl_interconversion", "phosphate_transfer", "acyl_transfer",
    "glycosyl_transfer", "c_n_transfer_transamination", "sulfur_thiol_chemistry",
    "stereochemical_rearrangement", "hydrolysis_condensation",
]
EXPECTED_COFACTORS = [
    "NAD_NADP", "FAD_FMN", "PLP", "TPP", "CoA", "SAM", "FeS", "heme",
    "quinone", "thiol_lipoate", "metal_Mg", "metal_Mn", "metal_Zn",
    "metal_Fe", "metal_Cu", "metal_Co", "metal_Ni", "metal_Ca", "metal_K",
    "metal_Na", "metal_generic", "metal_divalent", "cobalamin", "biotin",
    "molybdopterin_tungsten", "F420", "F430", "folate", "pyruvoyl",
    "dipyrromethane", "phosphopantetheine", "unknown",
]
NATIVE_COLUMNS = ["protein_id", "annotation_status", "ec_numbers", "cofactor_names"]


def write_table(path, columns, rows, delimiter="\t"):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def native_row(protein_id, ec=(), cofactor=(), status="matched"):
    return (protein_id, status, json.dumps(list(ec)), json.dumps(list(cofactor)))


def read_table(path, delimiter=","):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


@pytest.fixture
def inputs(tmp_path):
    representative_fasta = tmp_path / "representatives.fasta"
    representative_fasta.write_text("".join(f">{value}\nMAAA\n" for value in "ABCDEFGH"))
    cluster_map = tmp_path / "clusters.tsv"
    cluster_map.write_text(
        "".join(f"{value}\t{value}\n" for value in "ABCDEFGH")
        + "A\ta_member\nB\tb_member\nC\tc_member\nE\te_member\n"
    )
    native_annotations = write_table(
        tmp_path / "native.tsv.gz",
        NATIVE_COLUMNS,
        [
            native_row("A", ["1.1.1.1"], ["NAD+"]),
            native_row("a_member", ["1.1.1.9"], ["FAD"]),
            native_row("B", ["1.1.1.2", "1.1.-.-"], ["pyridoxal phosphate"]),
            native_row("b_member", status="matched_unannotated"),
            native_row("C", ["1.1.-.-"]),
            native_row("c_member", ["1.1.1.8"]),
            native_row("D", ["1.1.1.3", "2.1.1.4"]),
            native_row("E", status="unresolved"),
            native_row("e_member", ["3.1.1.1"], ["heme"]),
            native_row("F", ["2.1.1.2"]),
            native_row("G", ["1.1.1.3"]),
            native_row("H", ["1.1.1.4"], ["zinc"]),
        ],
    )
    reaction_annotations = tmp_path / "reactions.json"
    reaction_annotations.write_text(json.dumps({
        "schema_version": "horizyn1_circe_v2_reaction_annotations_v2",
        "label_semantics": "positive_only_unknown_not_negative",
        "cofactor_vocabulary_version": "circe_cofactor_v2",
        "cofactor_unknown_label": "unknown",
        "families": {"mechanism": EXPECTED_MECHANISMS, "cofactor": EXPECTED_COFACTORS},
        "reactions": {
            "redox": {"chemistry_match": "exact", "evidence_type": "reaction_associated_descriptor", "mechanism": [EXPECTED_MECHANISMS[0]],
                      "cofactor": ["NAD_NADP"]},
            "member_only": {"chemistry_match": "exact", "evidence_type": "reaction_associated_descriptor", "mechanism": [EXPECTED_MECHANISMS[1]],
                            "cofactor": ["FAD_FMN"]},
            "empty": {"chemistry_match": "exact", "evidence_type": "reaction_associated_descriptor", "mechanism": [], "cofactor": []},
        },
    }))
    association_pairs = write_table(
        tmp_path / "own_source_pairs.tsv", ["reaction_id", "protein_id"],
        [("redox", "A"), ("member_only", "a_member"), ("redox", "B"),
         ("unknown", "C"), ("member_only", "c_member"), ("redox", "D"),
         ("member_only", "e_member"), ("redox", "G"), ("empty", "H")],
    )
    return {
        "representative_fasta": representative_fasta,
        "cluster_map": cluster_map,
        "native_annotations": native_annotations,
        "reaction_annotations": reaction_annotations,
        "association_pairs": association_pairs,
        "output_dir": tmp_path / "labels",
        "pair_scope": "unsplit_inventory",
    }


def test_native_labels_are_representative_only_and_preserve_incomplete_multi_ec(inputs):
    MODULE.build_labels(**inputs)
    rows = read_table(inputs["output_dir"] / "enzyme_ec_labels.csv")
    assert {(row["protein_id"], row["ec_number"], int(row["known_depth"])) for row in rows} == {
        ("A", "1.1.1.1", 4), ("B", "1.1.1.2", 4), ("B", "1.1.-.-", 2),
        ("C", "1.1.-.-", 2), ("D", "1.1.1.3", 4), ("D", "2.1.1.4", 4),
        ("F", "2.1.1.2", 4), ("G", "1.1.1.3", 4), ("H", "1.1.1.4", 4),
    }
    with np.load(inputs["output_dir"] / "enzyme_biofp_targets.npz", allow_pickle=False) as data:
        native = data["native_cofactor_targets"]
        assert np.argwhere(native[:, :31]).tolist() == [[0, 0], [1, 2], [7, 12]]
        assert native[4, 31] == 1  # Member-only heme must not become E's label.
        assert native[7, 12] == 1  # Zinc now has its specific metal class.
        assert native[7, 31] == 0


def test_weak_labels_use_own_raw_associations_and_positive_only_masks(inputs):
    MODULE.build_labels(**inputs)
    with np.load(inputs["output_dir"] / "enzyme_biofp_targets.npz", allow_pickle=False) as data:
        assert data["ids"].tolist() == list("ABCDEFGH")
        assert data["mechanism_targets"].shape == (8, 8)
        assert data["cofactor_targets"].shape == (8, 32)
        assert np.argwhere(data["mechanism_targets"]).tolist() == [[0, 0], [1, 0], [3, 0], [6, 0]]
        assert not data["mechanism_targets"][:, 1].any()  # Member-only phosphate transfer.
        assert not data["cofactor_targets"][:, 1].any()  # Member-only FAD.
        for family in ["mechanism", "cofactor", "native_cofactor", "reaction_cofactor"]:
            targets, mask = data[f"{family}_targets"], data[f"{family}_mask"]
            assert targets.dtype == np.float32
            assert mask.dtype == bool
            np.testing.assert_array_equal(mask, targets > 0)
        np.testing.assert_allclose(data["mechanism_confidence"], data["mechanism_targets"] * 0.4)
        assert data["cofactor_confidence"][0, 0] == 1.0  # Native wins over overlapping weak evidence.
        assert data["cofactor_confidence"][1, 2] == 1.0
        assert data["cofactor_confidence"][1, 0] == pytest.approx(0.4)
        assert not data["mechanism_mask"][[2, 4, 5, 7]].any()
        assert not data["cofactor_confidence"][~data["cofactor_mask"]].any()


def test_eligibility_uses_member_ec_only_as_a_conflict_guard(inputs):
    report = MODULE.build_labels(**inputs)
    rows = {row["protein_id"]: row for row in read_table(inputs["output_dir"] / "candidate_eligibility.csv")}
    assert {key: (int(row["ec_negative_candidate_eligible"]), int(row["biological_negative_candidate_eligible"]))
            for key, row in rows.items()} == {
        "A": (0, 0), "B": (1, 1), "C": (0, 0), "D": (0, 0),
        "E": (0, 0), "F": (1, 0), "G": (1, 1), "H": (1, 0),
    }
    assert rows["A"]["reason"] == "cluster_member_ec_conflict"
    assert rows["C"]["reason"] == "missing_or_ambiguous_direct_ec"
    assert rows["E"]["members_with_ec"] == "1"
    summaries = {row["protein_id"]: row for row in read_table(
        inputs["output_dir"] / "cluster_annotation_summary.tsv.gz", delimiter="\t"
    )}
    assert json.loads(summaries["A"]["member_ec_evidence"]) == ["1.1.1.1", "1.1.1.9"]
    assert json.loads(summaries["A"]["own_ec_labels"]) == ["1.1.1.1"]
    assert json.loads(summaries["E"]["own_ec_labels"]) == []
    assert report["negative_candidate_eligibility"]["ec_eligible"] == 4
    assert report["negative_candidate_eligibility"]["biological_eligible"] == 2


def test_manifest_records_exact_vocab_coverage_and_unsplit_provenance(inputs):
    report = MODULE.build_labels(**inputs)
    assert report["families"] == {"mechanism": EXPECTED_MECHANISMS, "cofactor": EXPECTED_COFACTORS}
    assert report["schema_version"] == MODULE.SCHEMA
    assert report["pair_scope"] == "unsplit_inventory"
    assert report["training_split_declared"] is False
    assert report["representative_count"] == 8
    assert report["raw_member_count"] == 12
    assert report["association_rows"] == 9
    assert report["representative_own_association_rows"] == 6
    assert report["representative_own_annotated_association_rows"] == 5
    assert report["row_coverage"] == {
        "mechanism": 4, "cofactor": 5, "native_cofactor": 3, "reaction_cofactor": 4, "direct_ec": 7,
    }
    assert "zero biological supervision confidence" in report["mask_semantics"]
    assert any("training-only source associations" in value for value in report["limitations"])
    saved = json.loads((inputs["output_dir"] / "label_manifest.json").read_text())
    assert saved == report
    vocab = json.loads((inputs["output_dir"] / "enzyme_biofp_vocab.json").read_text())
    assert vocab["families"] == report["families"]
    assert set(report["input_signatures"]) == {str(inputs[key].resolve()) for key in [
        "representative_fasta", "cluster_map", "native_annotations", "reaction_annotations", "association_pairs",
    ]} | {
        str((ROOT / "horizyn/capability/biological_targets.py").resolve()),
        str((ROOT / "horizyn/capability/cofactor_vocabulary_v2.py").resolve()),
        str((ROOT / "scripts/build_horizyn1_reaction_annotations_v2.py").resolve()),
    }
    for key, observed in report["output_signatures"].items():
        assert MODULE.signature(Path(report["outputs"][key])) == observed
    assert not list(inputs["output_dir"].glob("*.partial"))


def test_train_scope_uses_only_the_supplied_training_associations(inputs):
    inputs["pair_scope"] = "train"
    write_table(inputs["association_pairs"], ["reaction_id", "protein_id"], [("redox", "B")])
    report = MODULE.build_labels(**inputs)
    assert report["training_split_declared"] is True
    assert report["pair_scope"] == "train"
    assert report["association_rows"] == 1
    with np.load(inputs["output_dir"] / "enzyme_biofp_targets.npz") as data:
        assert np.argwhere(data["mechanism_mask"]).tolist() == [[1, 0]]


def test_gzipped_csv_training_associations_are_supported(inputs, tmp_path):
    inputs["association_pairs"] = write_table(
        tmp_path / "train.csv.gz", ["reaction_id", "protein_id"], [("redox", "B")], delimiter=",",
    )
    inputs["pair_scope"] = "train"
    report = MODULE.build_labels(**inputs)
    assert report["association_rows"] == 1
    assert report["row_coverage"]["mechanism"] == 1


def test_unchanged_resume_does_not_rewrite_outputs(inputs, monkeypatch):
    original = MODULE.build_labels(**inputs)
    snapshots = {key: MODULE.signature(Path(path)) for key, path in original["outputs"].items()}
    monkeypatch.setattr(MODULE, "fasta_ids", lambda _path: pytest.fail("Resume unexpectedly rebuilt labels"))
    assert MODULE.build_labels(**inputs) == original
    assert snapshots == {key: MODULE.signature(Path(path)) for key, path in original["outputs"].items()}


@pytest.mark.parametrize("change", ["input", "output", "partial", "scope"])
def test_resume_rejects_changed_inputs_outputs_or_scope(inputs, change):
    MODULE.build_labels(**inputs)
    if change == "input":
        with inputs["association_pairs"].open("a") as handle:
            handle.write("redox\tB\n")
    elif change == "output":
        with (inputs["output_dir"] / "enzyme_ec_labels.csv").open("a") as handle:
            handle.write("A,9.9.9.9,4\n")
    elif change == "partial":
        (inputs["output_dir"] / "enzyme_ec_labels.csv.partial").write_text("unfinished")
    else:
        inputs["pair_scope"] = "train"
    with pytest.raises(ValueError, match="Existing label manifest"):
        MODULE.build_labels(**inputs)


@pytest.mark.parametrize("case", [
    "missing", "duplicate", "unknown", "unmatched_labels", "bad_json", "schema",
    "unknown_status", "unannotated_has_labels",
])
def test_native_member_validation_fails_closed(inputs, case):
    rows = read_table(inputs["native_annotations"], delimiter="\t")
    if case == "missing":
        rows = [row for row in rows if row["protein_id"] != "a_member"]
    elif case == "duplicate":
        rows.append(rows[0].copy())
    elif case == "unknown":
        rows[0]["protein_id"] = "absent"
    elif case == "unmatched_labels":
        rows[0]["annotation_status"] = "sequence_mismatch"
    elif case == "bad_json":
        rows[0]["ec_numbers"] = '{"not": "a list"}'
    elif case == "unknown_status":
        rows[0]["annotation_status"] = "invented_status"
        rows[0]["ec_numbers"] = rows[0]["cofactor_names"] = "[]"
    elif case == "unannotated_has_labels":
        rows[0]["annotation_status"] = "matched_unannotated"
    columns = NATIVE_COLUMNS[:-1] if case == "schema" else NATIVE_COLUMNS
    write_table(inputs["native_annotations"], columns, [[row[column] for column in columns] for row in rows])
    with pytest.raises(ValueError):
        MODULE.build_labels(**inputs)
    assert not (inputs["output_dir"] / "label_manifest.json").exists()


@pytest.mark.parametrize("case", ["duplicate_member", "unknown_rep", "missing_self", "bad_row"])
def test_cluster_membership_validation(inputs, case):
    text = inputs["cluster_map"].read_text()
    if case == "duplicate_member":
        text += "A\ta_member\n"
    elif case == "unknown_rep":
        text += "absent\tmember\n"
    elif case == "missing_self":
        text = text.replace("A\tA\n", "")
    else:
        text += "A\tnew_member\textra\n"
    inputs["cluster_map"].write_text(text)
    with pytest.raises(ValueError):
        MODULE.build_labels(**inputs)


@pytest.mark.parametrize("text", ["", ">A\nMAA\n>A\nMBB\n"])
def test_empty_or_duplicate_representative_ids_are_rejected(inputs, text):
    inputs["representative_fasta"].write_text(text)
    with pytest.raises(ValueError):
        MODULE.build_labels(**inputs)


@pytest.mark.parametrize("case", ["vocab_size", "vocab_order", "nonexact_match", "unknown_feature"])
def test_reaction_annotation_schema_validation(inputs, case):
    lookup = json.loads(inputs["reaction_annotations"].read_text())
    if case == "vocab_size":
        lookup["families"]["mechanism"].append("invented")
    elif case == "vocab_order":
        lookup["families"]["cofactor"].reverse()
    elif case == "nonexact_match":
        lookup["reactions"]["redox"]["chemistry_match"] = "approximate"
    else:
        lookup["reactions"]["redox"]["mechanism"] = ["invented"]
    inputs["reaction_annotations"].write_text(json.dumps(lookup))
    with pytest.raises((ValueError, KeyError)):
        MODULE.build_labels(**inputs)


@pytest.mark.parametrize("case", ["unknown_protein", "bad_schema"])
def test_pair_validation(inputs, case):
    columns = ["reaction_id", "protein_id"] if case == "unknown_protein" else ["reaction_id", "wrong"]
    write_table(inputs["association_pairs"], columns, [("redox", "absent")])
    with pytest.raises(ValueError):
        MODULE.build_labels(**inputs)


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink", "direct"])
def test_output_aliasing_an_input_is_rejected_without_modification(inputs, alias_kind):
    output = inputs["output_dir"] / "enzyme_ec_labels.csv"
    output.parent.mkdir()
    source = inputs["association_pairs"]
    original = source.read_bytes()
    if alias_kind == "direct":
        output.write_bytes(original)
        inputs["association_pairs"] = output
    elif alias_kind == "symlink":
        output.symlink_to(source)
    else:
        os.link(source, output)
    with pytest.raises(ValueError, match="aliases"):
        MODULE.build_labels(**inputs)
    assert inputs["association_pairs"].read_bytes() == original


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink", "direct"])
def test_staging_output_aliasing_an_input_is_rejected_without_modification(inputs, alias_kind):
    staging = inputs["output_dir"] / "enzyme_ec_labels.csv.partial"
    staging.parent.mkdir()
    source = inputs["association_pairs"]
    original = source.read_bytes()
    if alias_kind == "direct":
        staging.write_bytes(original)
        inputs["association_pairs"] = staging
    elif alias_kind == "symlink":
        staging.symlink_to(source)
    else:
        os.link(source, staging)
    with pytest.raises(ValueError, match="aliases"):
        MODULE.build_labels(**inputs)
    assert inputs["association_pairs"].read_bytes() == original


def test_input_changed_during_export_cannot_produce_completion_manifest(inputs, monkeypatch):
    original_open = MODULE.open_text

    def change_after_snapshot(path):
        if path == inputs["association_pairs"]:
            with path.open("a") as handle:
                handle.write("redox\tB\n")
        return original_open(path)

    monkeypatch.setattr(MODULE, "open_text", change_after_snapshot)
    with pytest.raises(ValueError, match="input changed"):
        MODULE.build_labels(**inputs)
    assert not (inputs["output_dir"] / "label_manifest.json").exists()


@pytest.mark.parametrize("filename", ["enzyme_ec_labels.csv", "enzyme_biofp_targets.npz.partial"])
def test_orphan_outputs_require_explicit_force(inputs, filename):
    output = inputs["output_dir"] / filename
    output.parent.mkdir()
    output.write_text("existing unfinished work")
    with pytest.raises(ValueError, match="Incomplete label outputs"):
        MODULE.build_labels(**inputs)
    assert output.read_text() == "existing unfinished work"


def test_force_failure_removes_stale_completion_manifest(inputs):
    MODULE.build_labels(**inputs)
    inputs["cluster_map"].write_text("invalid\n")
    with pytest.raises(ValueError):
        MODULE.build_labels(**inputs, force=True)
    assert not (inputs["output_dir"] / "label_manifest.json").exists()


def test_force_rebuild_replaces_stale_labels_and_provenance(inputs):
    MODULE.build_labels(**inputs)
    inputs["pair_scope"] = "train"
    write_table(inputs["association_pairs"], ["reaction_id", "protein_id"], [("redox", "B")])
    report = MODULE.build_labels(**inputs, force=True)
    assert report["pair_scope"] == "train"
    assert report["association_rows"] == 1
    assert MODULE.build_labels(**inputs) == report


@pytest.mark.parametrize("name,expected", [
    ("vanadium", [31]), ("coenzyme B12", [22]),
    ("[4Fe-4S] cluster", [6]), ("[2Fe-2S] cluster", [6]),
    ("coenzyme B", [9]), ("NADP+", [0]),
])
def test_native_cofactor_mapping_uses_biological_names_not_substring_accidents(inputs, name, expected):
    rows = read_table(inputs["native_annotations"], delimiter="\t")
    for row in rows:
        if row["protein_id"] == "H":
            row["cofactor_names"] = json.dumps([name])
    write_table(inputs["native_annotations"], NATIVE_COLUMNS,
                [[row[column] for column in NATIVE_COLUMNS] for row in rows])
    MODULE.build_labels(**inputs)
    with np.load(inputs["output_dir"] / "enzyme_biofp_targets.npz") as data:
        assert np.flatnonzero(data["native_cofactor_targets"][7]).tolist() == expected


@pytest.mark.parametrize("ec,depth", [("1.2.3.4", 4), ("1.2.-.-", 2), ("1.-.-.-", 1), ("1.2.3", 0)])
def test_known_ec_depth(ec, depth):
    assert MODULE.known_depth(ec) == depth


@pytest.mark.parametrize("labels,expected", [
    ({"1.2.3.4", "1.2.-.-"}, "1.2.3.4"),
    ({"1.2.-.-"}, None),
    ({"1.2.3.4", "1.2.3.5"}, None),
    ({"1.2.3.4", "2.-.-.-"}, None),
    (set(), None),
])
def test_consistent_complete_ec_does_not_invent_or_hide_labels(labels, expected):
    assert MODULE.consistent_complete_ec(labels) == expected


def test_unknown_is_separate_per_source_and_never_combined_with_known(inputs):
    report = MODULE.build_labels(**inputs)
    assert report["unknown_counts"] == {
        "native_cofactor": 5, "reaction_cofactor": 4, "cofactor": 3,
    }
    with np.load(inputs["output_dir"] / "enzyme_biofp_targets.npz", allow_pickle=False) as data:
        for family in ("native_cofactor", "reaction_cofactor", "cofactor"):
            targets, mask = data[f"{family}_targets"], data[f"{family}_mask"]
            np.testing.assert_array_equal(targets[:, 31], ~targets[:, :31].any(axis=1))
            np.testing.assert_array_equal(mask, targets > 0)
            denominator = data[f"{family}_denominator"]
            assert denominator.dtype == np.float32
            assert denominator.shape == (8,)
            np.testing.assert_array_equal(denominator, targets[:, :31].any(axis=1))
            assert report["row_coverage"][family] + report["unknown_counts"][family] == 8
        # D has no native cofactor annotation, but its own reaction gives NAD.
        assert data["native_cofactor_targets"][3, 31] == 1
        assert data["reaction_cofactor_targets"][3, 0] == 1
        assert data["cofactor_targets"][3, 31] == 0
        # H has zinc natively, and no known associated reaction cofactor.
        assert data["native_cofactor_targets"][7, 12] == 1
        assert data["reaction_cofactor_targets"][7, 31] == 1
        assert data["cofactor_targets"][7, 31] == 0
        assert not data["cofactor_confidence"][:, 31].any()
        for family, weight in (("native_cofactor", 1.0), ("reaction_cofactor", 0.4)):
            confidence = data[f"{family}_confidence"]
            assert confidence.dtype == np.float32
            assert confidence.shape == (8, 32)
            assert not confidence[:, 31].any()
            np.testing.assert_allclose(confidence[:, :31], data[f"{family}_targets"][:, :31] * weight)
        np.testing.assert_array_equal(
            data["cofactor_targets"][:, :31],
            data["native_cofactor_mask"][:, :31] | data["reaction_cofactor_mask"][:, :31],
        )


def test_npz_contract_is_self_describing_and_annotation_flags_are_boolean(inputs):
    report = MODULE.build_labels(**inputs)
    with np.load(inputs["output_dir"] / "enzyme_biofp_targets.npz", allow_pickle=False) as data:
        assert data["cofactor_labels"].dtype.kind == "U"
        assert data["cofactor_labels"].shape == (32,)
        assert data["cofactor_labels"].tolist() == EXPECTED_COFACTORS
        assert data["cofactor_unknown_index"].dtype == np.int64
        assert data["cofactor_unknown_index"].shape == ()
        assert data["cofactor_unknown_index"].item() == 31
        for key, expected in (
            ("cofactor_vocabulary_version", "circe_cofactor_v2"),
            ("annotation_semantics", "positive_only_role_aware_v2"),
            ("pair_scope", "unsplit_inventory"),
        ):
            assert data[key].dtype.kind == "U"
            assert data[key].shape == ()
            assert data[key].item() == expected
            assert report[key] == expected
        for key in ("native_cofactor_has_annotation", "native_cofactor_has_unmapped"):
            assert data[key].dtype == bool
            assert data[key].shape == (8,)
        assert data["native_cofactor_has_annotation"].tolist() == [True, True, False, False, False, False, False, True]
        assert not data["native_cofactor_has_unmapped"].any()
    assert report["positive_only"] is True
    assert report["non_biological_classes"] == {"cofactor": ["unknown"]}
    assert report["unmapped_native_cofactor_name_counts"] == {"all_members": {}, "representatives": {}}
    assert set(report["source_roles"]) == {"native_cofactor", "reaction_cofactor", "cofactor", "unknown"}


def test_unmapped_names_remain_in_source_and_are_counted_without_member_transfer(inputs):
    rows = read_table(inputs["native_annotations"], delimiter="\t")
    for row in rows:
        if row["protein_id"] == "H":
            row["cofactor_names"] = json.dumps(["vanadium"])
        elif row["protein_id"] == "B":
            row["cofactor_names"] = json.dumps(["pyridoxal phosphate", "vanadium"])
        elif row["protein_id"] == "e_member":
            row["cofactor_names"] = json.dumps(["heme", "novel prosthetic moiety"])
    write_table(inputs["native_annotations"], NATIVE_COLUMNS,
                [[row[column] for column in NATIVE_COLUMNS] for row in rows])
    original = inputs["native_annotations"].read_bytes()
    report = MODULE.build_labels(**inputs)
    assert inputs["native_annotations"].read_bytes() == original
    assert report["unmapped_native_cofactor_name_counts"] == {
        "all_members": {"novel prosthetic moiety": 1, "vanadium": 2},
        "representatives": {"vanadium": 2},
    }
    assert report["native_cofactor_annotation_counts"] == {
        "representatives_with_annotation": 3, "representatives_with_unmapped": 2,
        "all_members_with_annotation": 5, "all_members_with_unmapped": 3,
    }
    with np.load(inputs["output_dir"] / "enzyme_biofp_targets.npz", allow_pickle=False) as data:
        assert data["native_cofactor_has_unmapped"].tolist() == [False, True, False, False, False, False, False, True]
        # Known + unsupported is known; unsupported-only remains unknown.
        assert data["native_cofactor_targets"][1, 2] == 1
        assert data["native_cofactor_targets"][1, 31] == 0
        assert data["native_cofactor_targets"][7, 31] == 1
        assert not data["native_cofactor_targets"][7, :31].any()
        assert data["native_cofactor_has_annotation"][7]
        # Member-only heme and unsupported names do not become E annotations.
        assert not data["native_cofactor_has_annotation"][4]
        assert not data["native_cofactor_has_unmapped"][4]


@pytest.mark.parametrize("case", [
    "old_schema", "missing_schema", "wrong_identity", "wrong_unknown",
    "wrong_semantics", "unknown_status_label", "wrong_role",
    "nonlist_labels", "duplicate_labels", "wrong_lookup_type",
])
def test_version_and_role_contract_rejects_invalid_reaction_sources(inputs, case):
    lookup = json.loads(inputs["reaction_annotations"].read_text())
    if case == "old_schema":
        lookup["schema_version"] = "horizyn1_circe_v2_reaction_annotations_v1"
    elif case == "missing_schema":
        del lookup["schema_version"]
    elif case == "wrong_identity":
        lookup["cofactor_vocabulary_version"] = "circe_cofactor_v1"
    elif case == "wrong_unknown":
        lookup["cofactor_unknown_label"] = "none"
    elif case == "wrong_semantics":
        lookup["label_semantics"] = "dense_negative"
    elif case == "unknown_status_label":
        lookup["reactions"]["redox"]["cofactor"] = ["unknown"]
    elif case == "wrong_role":
        lookup["reactions"]["redox"]["evidence_type"] = "direct_native"
    elif case == "nonlist_labels":
        lookup["reactions"]["redox"]["cofactor"] = "NAD_NADP"
    elif case == "duplicate_labels":
        lookup["reactions"]["redox"]["cofactor"] = ["NAD_NADP", "NAD_NADP"]
    else:
        lookup["reactions"] = []
    inputs["reaction_annotations"].write_text(json.dumps(lookup))
    with pytest.raises(ValueError):
        MODULE.build_labels(**inputs)
    assert not (inputs["output_dir"] / "label_manifest.json").exists()


@pytest.mark.parametrize("force", [False, True])
def test_frozen_legacy_export_cannot_be_overwritten_even_with_force(inputs, force):
    inputs["output_dir"].mkdir()
    manifest = inputs["output_dir"] / "label_manifest.json"
    original = json.dumps({"schema_version": "horizyn1_circe_v2_labels_v1"})
    manifest.write_text(original)
    with pytest.raises(ValueError, match="another schema"):
        MODULE.build_labels(**inputs, force=force)
    assert manifest.read_text() == original


def test_optional_source_manifests_and_implementation_are_bound(inputs, tmp_path):
    import hashlib

    for key in ("native_manifest", "reaction_manifest"):
        path = tmp_path / f"{key}.json"
        path.write_text("{}")
        inputs[key] = path
    report = MODULE.build_labels(**inputs)
    assert report["implementation_sha256"] == hashlib.sha256(Path(MODULE.__file__).read_bytes()).hexdigest()
    for key in ("native_manifest", "reaction_manifest"):
        assert report["sources"][key] == str(inputs[key])
        assert report["input_signatures"][str(inputs[key].resolve())] == MODULE.signature(inputs[key])


def test_exporter_implementation_cannot_alias_an_output(inputs):
    output = inputs["output_dir"] / "enzyme_ec_labels.csv"
    output.parent.mkdir()
    output.symlink_to(Path(MODULE.__file__).resolve())
    with pytest.raises(ValueError, match="aliases"):
        MODULE.build_labels(**inputs, force=True)
