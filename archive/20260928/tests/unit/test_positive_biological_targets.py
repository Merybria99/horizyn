"""Observed-positive export stays sparse and training-edge restricted."""

import json
import subprocess
import sys

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from horizyn.capability.positive_biological_targets import build_positive_targets


@pytest.fixture
def sources(tmp_path):
    paths = {name: tmp_path / filename for name, filename in {
        "train_pairs_path": "train_pairs.csv", "matched_members_path": "matched_members.csv",
        "directional_features_path": "directional_features.parquet", "ec_labels_path": "enzyme_ec.csv",
        "enzyme_cofactor_labels_path": "enzyme_cofactors.csv",
    }.items()}
    pd.DataFrame([
        {"protein_id": "prot_a", "reaction_id": "r1"},
        {"protein_id": "prot_b", "reaction_id": "r2"},
        {"protein_id": "prot_c", "reaction_id": "r3"},
    ]).to_csv(paths["train_pairs_path"], index=False)
    pd.DataFrame([
        {"source_protein_id": "prot_a", "source_reaction_id": "r1", "reaction_id": "mapped1", "match_confidence": 0.8},
        {"source_protein_id": "prot_a", "source_reaction_id": "r_heldout", "reaction_id": "mapped2", "match_confidence": 1.0},
        {"source_protein_id": "prot_outside", "source_reaction_id": "r1", "reaction_id": "mapped2", "match_confidence": 1.0},
    ]).to_csv(paths["matched_members_path"], index=False)
    pd.DataFrame([
        {"reaction_id": "mapped1", "reaction_type_labels": ["oxidation", "phosphorylation"], "core_cofactor_labels": ["NAD"], "reaction_center_mapping_confidence": 0.5},
        {"reaction_id": "mapped2", "reaction_type_labels": ["hydrolysis"], "core_cofactor_labels": ["PLP"], "reaction_center_mapping_confidence": 1.0},
    ]).to_parquet(paths["directional_features_path"], index=False)
    pd.DataFrame([
        {"protein_id": "prot_a", "ec_number": "1.2.3.4;1.2.3.5;2.7.-.-"},
        {"protein_id": "prot_b", "ec_number": "unknown"},
        {"protein_id": "prot_outside", "ec_number": "3.1.1.1"},
    ]).to_csv(paths["ec_labels_path"], index=False)
    pd.DataFrame([
        {"enzyme_id": "uprot_a", "enzyme_derived_core_cofactor_labels_train": "['FAD', 'NAD']"},
        {"enzyme_id": "uprot_b", "enzyme_derived_core_cofactor_labels_train": "['NAD independent', 'unknown', 'no PLP']"},
        {"enzyme_id": "uprot_outside", "enzyme_derived_core_cofactor_labels_train": "['PLP']"},
    ]).to_csv(paths["enzyme_cofactor_labels_path"], index=False)
    return paths


def labels_for(ids, arrays, metadata, protein, family):
    position = ids.index(protein)
    indices = arrays[f"{family}_positive_indices"][position]
    confidence = arrays[f"{family}_confidence"][position]
    return {metadata["families"][family][index]: float(weight) for index, weight in zip(indices, confidence) if index >= 0}


def test_positive_export_retains_multilabels_without_heldout_or_missing_negatives(sources):
    ids, arrays, metadata = build_positive_targets(**sources)
    assert ids == ["prot_a", "prot_b", "prot_c"]
    assert metadata["families"]["ec"] == ["1.2.3.4", "1.2.3.5", "2.7"]
    assert set(labels_for(ids, arrays, metadata, "prot_a", "ec")) == {"1.2.3.4", "1.2.3.5", "2.7"}
    mechanism = labels_for(ids, arrays, metadata, "prot_a", "mechanism")
    assert set(mechanism) == {"redox_carbonyl_interconversion", "phosphate_transfer"}
    assert all(value == pytest.approx(0.4) for value in mechanism.values())
    cofactors = labels_for(ids, arrays, metadata, "prot_a", "cofactor")
    assert cofactors == {"NAD_NADP": pytest.approx(0.4), "FAD_FMN": pytest.approx(0.4)}
    for family in metadata["families"]:
        for protein in ("prot_b", "prot_c"):
            assert labels_for(ids, arrays, metadata, protein, family) == {}
    assert metadata["filter_counts"]["members_outside_retained_train_edges"] == 2
    assert metadata["row_coverage"] == {"ec": 1, "cofactor": 1, "mechanism": 1}
    assert all(len(value) == 64 for value in metadata["source_sha256"].values())
    json.dumps(metadata, allow_nan=False)


def test_sparse_shapes_types_and_pickle_free_export(sources, tmp_path):
    ids, arrays, metadata = build_positive_targets(**sources)
    for family, vocabulary in metadata["families"].items():
        indices = arrays[f"{family}_positive_indices"]
        weights = arrays[f"{family}_confidence"]
        assert indices.dtype == np.int32
        assert weights.dtype == np.float32
        assert indices.shape == weights.shape
        assert indices.shape[1] <= 3
        assert int(arrays[f"{family}_vocab_size"]) == len(vocabulary)
        assert np.all(weights[indices < 0] == 0)
        assert np.all(weights[indices >= 0] > 0)
    output = tmp_path / "targets.npz"
    np.savez_compressed(output, ids=np.asarray(ids, dtype=str), **arrays)
    with np.load(output, allow_pickle=False) as loaded:
        assert loaded["ids"].tolist() == ids


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, "nonsense"])
def test_nonpositive_and_invalid_match_confidence_never_promoted(sources, bad):
    members = pd.read_csv(sources["matched_members_path"])
    members["match_confidence"] = bad
    members.to_csv(sources["matched_members_path"], index=False)
    sources["enzyme_cofactor_labels_path"] = None
    ids, arrays, metadata = build_positive_targets(**sources)
    assert labels_for(ids, arrays, metadata, "prot_a", "mechanism") == {}
    assert labels_for(ids, arrays, metadata, "prot_a", "cofactor") == {}


def test_zero_mapping_confidence_masks_mechanism_not_participant_presence(sources):
    features = pd.read_parquet(sources["directional_features_path"])
    features["reaction_center_mapping_confidence"] = 0.0
    features.to_parquet(sources["directional_features_path"], index=False)
    ids, arrays, metadata = build_positive_targets(**sources)
    assert labels_for(ids, arrays, metadata, "prot_a", "mechanism") == {}
    assert labels_for(ids, arrays, metadata, "prot_a", "cofactor")


def test_zero_annotation_confidence_masks_ec_and_enzyme_cofactor_rows(sources):
    for name in ("ec_labels_path", "enzyme_cofactor_labels_path"):
        frame = pd.read_csv(sources[name])
        frame["confidence"] = 0.0
        frame.to_csv(sources[name], index=False)
    ids, arrays, metadata = build_positive_targets(**sources)
    assert labels_for(ids, arrays, metadata, "prot_a", "ec") == {}
    # The retained reaction is still independent positive NAD evidence.
    assert labels_for(ids, arrays, metadata, "prot_a", "cofactor") == {"NAD_NADP": pytest.approx(0.32)}


def test_duplicates_use_max_confidence_not_observation_frequency(sources):
    members = pd.read_csv(sources["matched_members_path"])
    pd.concat([members] * 10).to_csv(sources["matched_members_path"], index=False)
    ids, arrays, metadata = build_positive_targets(**sources)
    assert labels_for(ids, arrays, metadata, "prot_a", "mechanism")["phosphate_transfer"] == pytest.approx(0.4)


@pytest.mark.parametrize("source,id_column", [("train_pairs_path", "protein_id"), ("ec_labels_path", "protein_id"), ("enzyme_cofactor_labels_path", "enzyme_id")])
def test_canonical_id_collision_fails_closed(sources, source, id_column):
    frame = pd.read_csv(sources[source])
    duplicate = frame.iloc[[0]].copy()
    duplicate[id_column] = "enzyme_a"
    pd.concat([frame, duplicate]).to_csv(sources[source], index=False)
    with pytest.raises(ValueError, match="collision"):
        build_positive_targets(**sources)


def test_duplicate_directional_id_is_ambiguous(sources):
    features = pd.read_parquet(sources["directional_features_path"])
    pd.concat([features, features.iloc[[0]]]).to_parquet(sources["directional_features_path"], index=False)
    with pytest.raises(ValueError, match="Duplicate reaction"):
        build_positive_targets(**sources)


def test_explicit_negative_pairs_are_not_retained(sources):
    pairs = pd.read_csv(sources["train_pairs_path"])
    pairs["Label"] = [0, 1, 1]
    pairs.to_csv(sources["train_pairs_path"], index=False)
    ids, arrays, metadata = build_positive_targets(**sources)
    assert ids == ["prot_b", "prot_c"]
    assert metadata["row_coverage"] == {"ec": 0, "cofactor": 0, "mechanism": 0}


def test_heldout_source_path_is_rejected(sources):
    sources["ec_labels_path"] = sources["ec_labels_path"].with_name("test_ec.csv")
    with pytest.raises(ValueError, match="held-out"):
        build_positive_targets(**sources)


def test_import_has_no_torch_dependency():
    result = subprocess.run([sys.executable, "-c", "import sys; import horizyn.capability.positive_biological_targets; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_writes_unicode_targets_and_refuses_overwrite(sources, tmp_path):
    output = tmp_path / "export"
    command = [sys.executable, "-m", "horizyn.capability.positive_biological_targets", "--output-dir", str(output)]
    for key, flag in {
        "train_pairs_path": "--train-pairs", "matched_members_path": "--matched-members",
        "directional_features_path": "--directional-features", "ec_labels_path": "--ec-labels",
        "enzyme_cofactor_labels_path": "--enzyme-cofactors",
    }.items():
        command.extend([flag, str(sources[key])])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["row_coverage"]["ec"] == 1
    with np.load(output / "positive_targets.npz", allow_pickle=False) as archive:
        assert archive["ids"].dtype.kind == "U"
        assert archive["ids"].tolist() == ["prot_a", "prot_b", "prot_c"]
    assert json.loads((output / "positive_vocab.json").read_text())["schema_version"] == "positive_bio_v1"
    repeat = subprocess.run(command, capture_output=True, text=True)
    assert repeat.returncode != 0
    assert "Refusing to overwrite" in repeat.stderr
    assert not list(output.glob(".positive_*"))
