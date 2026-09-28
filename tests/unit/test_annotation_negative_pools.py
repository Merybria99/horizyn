import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "build_annotation_negative_pools.py"
SPEC = importlib.util.spec_from_file_location("build_annotation_negative_pools", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_annotation_pools_are_typed_incompatible_and_deterministic(tmp_path):
    pairs_path = tmp_path / "pairs.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pr_id", "reaction_id", "protein_id", "reaction_smiles"],
        )
        writer.writeheader()
        for idx, (query_id, protein_id, smiles) in enumerate(
            [
                ("q1", "prot_a", "A>>B"),
                ("q2", "prot_b", "C>>D"),
                ("q3", "prot_c", "E>>F"),
                ("q4", "prot_d", "G>>H"),
                ("q5", "prot_e", "I>>J"),
                ("q6", "prot_f", "K>>L"),
            ]
        ):
            writer.writerow(
                {
                    "pr_id": idx,
                    "reaction_id": query_id,
                    "protein_id": protein_id,
                    "reaction_smiles": smiles,
                }
            )

    ec_path = tmp_path / "ec.csv"
    with ec_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "ec_number", "known_depth"])
        writer.writeheader()
        writer.writerows(
            [
                {"protein_id": "nr90_a", "ec_number": "1.1.1.1", "known_depth": 4},
                {"protein_id": "nr90_b", "ec_number": "1.1.1.2", "known_depth": 4},
                {"protein_id": "nr90_c", "ec_number": "1.1.1.1", "known_depth": 4},
                {"protein_id": "nr90_d", "ec_number": "2.1.1.1", "known_depth": 4},
                {"protein_id": "nr90_f", "ec_number": "1.1.1.3", "known_depth": 4},
                {"protein_id": "nr90_f", "ec_number": "3.1.-.-", "known_depth": 2},
            ]
        )

    biofp_path = tmp_path / "biofp.npz"
    mechanism_targets = np.zeros((6, 2), dtype=np.float32)
    mechanism_targets[:3, 0] = 1.0
    mechanism_targets[3:, 1] = 1.0
    mechanism_targets[5, :] = [1.0, 0.0]
    cofactor_targets = np.zeros((6, 2), dtype=np.float32)
    cofactor_targets[:3, 0] = 1.0
    cofactor_targets[5, 0] = 1.0
    np.savez(
        biofp_path,
        ids=np.array(["uprot_a", "uprot_b", "uprot_c", "uprot_d", "uprot_e", "uprot_f"]),
        mechanism_targets=mechanism_targets,
        mechanism_mask=np.ones_like(mechanism_targets, dtype=bool),
        cofactor_targets=cofactor_targets,
        cofactor_mask=np.ones_like(cofactor_targets, dtype=bool),
    )

    kwargs = dict(
        max_biological=4,
        max_random=2,
        ec_prefix_depth=2,
        biofp_threshold=0.5,
        min_biofp_similarity=0.5,
        seed=11,
    )
    first, report = MODULE.build_pools(pairs_path, ec_path, biofp_path, **kwargs)
    second, _ = MODULE.build_pools(pairs_path, ec_path, biofp_path, **kwargs)
    assert first == second
    q1 = first["reaction_to_negatives"]["q1"]
    assert q1["biological"] == ["prot_b"]
    assert "prot_a" not in q1["random"]
    assert "prot_b" not in q1["random"]
    assert "prot_c" not in q1["random"]
    assert "prot_f" not in q1["biological"]
    assert all(
        "prot_e" not in typed_pool["random"]
        for typed_pool in first["reaction_to_negatives"].values()
    )
    assert report["unique_reactions"] == 6
    assert report["eligible_single_ec_random_proteins"] == 4
    assert report["proteins_rejected_for_conflicting_ec"] >= 1


def test_biofp_similarity_treats_missing_cofactors_as_unknown():
    left = MODULE.BioFPProfile(
        mechanisms=frozenset({0, 1}),
        mechanism_observed=frozenset({0, 1}),
        cofactors=frozenset(),
        cofactor_observed=frozenset(),
    )
    right = MODULE.BioFPProfile(
        mechanisms=frozenset({0}),
        mechanism_observed=frozenset({0, 1}),
        cofactors=frozenset(),
        cofactor_observed=frozenset(),
    )

    assert MODULE.biofp_similarity(left, right) == 0.5


def test_stable_offset_rejects_an_empty_population():
    with pytest.raises(ValueError, match="modulus must be positive"):
        MODULE.stable_offset("q1", seed=42, modulus=0)


ELIGIBILITY_COLUMNS = [
    "protein_id",
    "ec_negative_candidate_eligible",
    "biological_negative_candidate_eligible",
]


def write_eligibility(path, rows, columns=ELIGIBILITY_COLUMNS):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


@pytest.fixture
def eligibility_inputs(tmp_path):
    protein_ids = list("abcdefgx")
    pairs_path = tmp_path / "pairs.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reaction_id", "protein_id", "reaction_smiles"])
        writer.writerows((f"q_{value}", f"prot_{value}", f"{value}>>product") for value in protein_ids)
    ec_path = tmp_path / "ec.csv"
    with ec_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "ec_number", "known_depth"])
        writer.writerows(
            (f"uprot_{value}", ec, depth)
            for value, ec, depth in [
                ("a", "1.1.1.1", 4),
                ("b", "1.1.1.2", 4),
                ("c", "1.1.1.3", 4),
                ("d", "1.1.1.4", 4),
                ("e", "1.1.1.5", 4),
                ("f", "1.1.1.1", 4),
                ("g", "1.1.1.6", 4),
                ("g", "3.1.-.-", 2),
                ("x", "1.1.1.1", 4),
                ("x", "1.1.1.2", 4),
            ]
        )
    biofp_path = tmp_path / "biofp.npz"
    targets = np.ones((len(protein_ids), 1), dtype=np.float32)
    np.savez(
        biofp_path,
        ids=np.array([f"nr90_{value}" for value in protein_ids]),
        mechanism_targets=targets,
        mechanism_mask=np.ones_like(targets, dtype=bool),
        cofactor_targets=targets,
        cofactor_mask=np.ones_like(targets, dtype=bool),
    )
    kwargs = dict(
        max_biological=32,
        max_random=32,
        ec_prefix_depth=2,
        biofp_threshold=0.5,
        min_biofp_similarity=0.5,
        seed=11,
    )
    return (pairs_path, ec_path, biofp_path), kwargs


def test_candidate_eligibility_filters_only_candidates_and_keeps_all_ecs(
    tmp_path, eligibility_inputs
):
    paths, kwargs = eligibility_inputs
    eligibility_path = write_eligibility(
        tmp_path / "eligibility.csv",
        [
            ("a", 0, 0),
            ("uprot_b", 1, 1),
            ("nr90_c", 1, 0),
            ("prot_d", 0, 1),  # Biological flag cannot override EC ineligibility.
            # e is missing: both pools must fail closed.
            ("f", 1, 1),
            ("g", 1, 1),  # Flags cannot override a conflicting partial EC annotation.
            ("x", 0, 0),  # Both positive ECs must still exclude matching candidates.
        ],
    )
    payload, report = MODULE.build_pools(
        *paths, candidate_eligibility_path=eligibility_path, **kwargs
    )
    assert payload["reaction_to_negatives"]["q_a"] == {
        "biological": ["prot_b"],
        "random": ["prot_c"],
    }
    assert payload["reaction_to_negatives"]["q_x"] == {
        "biological": [],
        "random": ["prot_c"],
    }
    for pools in payload["reaction_to_negatives"].values():
        assert not {"prot_a", "prot_d", "prot_e", "prot_g", "prot_x"} & (
            set(pools["biological"]) | set(pools["random"])
        )
    assert report["proteins_rejected_for_conflicting_ec"] == 2
    assert report["proteins_with_complete_ec"] == 8
    assert report["eligible_single_ec_random_proteins"] == 3
    assert report["eligible_single_ec_biofp_proteins"] == 2
    metadata = report["candidate_eligibility"]
    assert metadata == {
        "path": str(eligibility_path),
        "sha256": hashlib.sha256(eligibility_path.read_bytes()).hexdigest(),
        "rows": 7,
        "unique_canonical_proteins": 7,
        "duplicate_consistent_rows": 0,
        "ec_flag_eligible_proteins": 4,
        "biological_and_ec_flag_eligible_proteins": 3,
        "training_proteins_with_flags": 7,
        "training_proteins_missing_flags": 1,
        "training_proteins_passing_ec_flag": 4,
        "training_proteins_passing_biological_and_ec_flags": 3,
    }
    assert payload["criteria"]["candidate_eligibility"]["sha256"] == metadata["sha256"]


def test_candidate_eligibility_preserves_default_behavior(tmp_path, eligibility_inputs):
    paths, kwargs = eligibility_inputs
    default, default_report = MODULE.build_pools(*paths, **kwargs)
    explicit_none, explicit_none_report = MODULE.build_pools(
        *paths, candidate_eligibility_path=None, **kwargs
    )
    assert (default, default_report) == (explicit_none, explicit_none_report)
    assert "candidate_eligibility" not in default["criteria"]
    assert "candidate_eligibility" not in default_report
    eligibility_path = write_eligibility(
        tmp_path / "all_eligible.csv", [(value, 1, 1) for value in "abcdefgx"]
    )
    all_eligible, report = MODULE.build_pools(
        *paths, candidate_eligibility_path=eligibility_path, **kwargs
    )
    all_eligible["criteria"].pop("candidate_eligibility")
    report.pop("candidate_eligibility")
    assert (all_eligible, report) == (default, default_report)


def test_candidate_eligibility_rejects_empty_candidate_population(tmp_path, eligibility_inputs):
    paths, kwargs = eligibility_inputs
    path = write_eligibility(tmp_path / "empty.csv", [])
    with pytest.raises(ValueError, match="after candidate-eligibility filtering"):
        MODULE.build_pools(*paths, candidate_eligibility_path=path, **kwargs)


@pytest.mark.parametrize("value", ["true", "False", "", "2", "-1", "0.0", " 1", "1 "])
@pytest.mark.parametrize("column", [1, 2])
def test_candidate_eligibility_requires_strict_boolean_flags(tmp_path, value, column):
    row = ["protein", "1", "1"]
    row[column] = value
    path = write_eligibility(tmp_path / "invalid.csv", [row])
    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        MODULE.load_candidate_eligibility(path)


def test_candidate_eligibility_checks_canonical_duplicate_flags(tmp_path):
    path = write_eligibility(
        tmp_path / "duplicates.csv", [("prot_a", 1, 0), ("uprot_a", 1, 0)]
    )
    flags, report = MODULE.load_candidate_eligibility(path)
    assert flags == {"a": (True, False)}
    assert report["duplicate_consistent_rows"] == 1
    path = write_eligibility(path, [("prot_a", 1, 0), ("nr90_a", 1, 1)])
    with pytest.raises(ValueError, match="conflicting eligibility flags.*'a'"):
        MODULE.load_candidate_eligibility(path)


@pytest.mark.parametrize(
    "columns,rows,expected",
    [
        (["protein_id"], [("a",)], "must contain unique columns"),
        (ELIGIBILITY_COLUMNS + ["protein_id"], [("a", 1, 1, "a")], "unique columns"),
        (ELIGIBILITY_COLUMNS, [("a", 1)], "malformed eligibility CSV row"),
        (ELIGIBILITY_COLUMNS, [("a", 1, 1, "extra")], "malformed eligibility CSV row"),
        (ELIGIBILITY_COLUMNS, [("", 1, 1)], "protein_id must not be empty"),
        (ELIGIBILITY_COLUMNS, [("prot_", 1, 1)], "protein_id must not be empty"),
    ],
)
def test_candidate_eligibility_validates_csv_schema(tmp_path, columns, rows, expected):
    path = write_eligibility(tmp_path / "schema.csv", rows, columns)
    with pytest.raises(ValueError, match=expected):
        MODULE.load_candidate_eligibility(path)


def role_aware_arrays(ids=("a", "b"), native=({0}, {0}), reaction=(set(), set())):
    count = len(ids)
    arrays = {
        "ids": np.asarray(ids, dtype=str),
        "annotation_semantics": np.asarray(MODULE.ROLE_AWARE_SEMANTICS),
        "cofactor_vocabulary_version": np.asarray(MODULE.COFACTOR_VOCABULARY_VERSION),
        "cofactor_labels": np.asarray(MODULE.COFACTOR_LABELS),
        "cofactor_unknown_index": np.asarray(MODULE.UNKNOWN_INDEX, dtype=np.int64),
        "pair_scope": np.asarray("train"),
    }
    mechanism = np.zeros((count, 8), dtype=np.float32)
    mechanism[:, 0] = 1
    arrays.update(mechanism_targets=mechanism, mechanism_mask=mechanism.astype(bool),
                  mechanism_confidence=mechanism * np.float32(0.4))
    for family, rows in (("native_cofactor", native), ("reaction_cofactor", reaction)):
        targets = np.zeros((count, 32), dtype=np.float32)
        for index, values in enumerate(rows):
            targets[index, list(values)] = 1
            targets[index, MODULE.UNKNOWN_INDEX] = not bool(values)
        arrays[f"{family}_targets"] = targets
        arrays[f"{family}_mask"] = targets.astype(bool)
        confidence = targets * np.float32(1.0 if family == "native_cofactor" else 0.4)
        confidence[:, MODULE.UNKNOWN_INDEX] = 0
        arrays[f"{family}_confidence"] = confidence
    union = np.logical_or(arrays["native_cofactor_targets"], arrays["reaction_cofactor_targets"])
    union[:, MODULE.UNKNOWN_INDEX] = ~union[:, :MODULE.UNKNOWN_INDEX].any(axis=1)
    arrays.update(cofactor_targets=union.astype(np.float32), cofactor_mask=union)
    confidence = np.where(arrays["native_cofactor_targets"], 1.0,
                          np.where(arrays["reaction_cofactor_targets"], 0.4, 0.0)).astype(np.float32)
    confidence[:, MODULE.UNKNOWN_INDEX] = 0
    arrays["cofactor_confidence"] = confidence
    for family in ("cofactor", "native_cofactor", "reaction_cofactor"):
        arrays[f"{family}_denominator"] = arrays[f"{family}_targets"][:, :MODULE.UNKNOWN_INDEX].any(axis=1).astype(np.float32)
    return arrays


@pytest.mark.parametrize(
    "native,reaction,expected",
    [
        (({0}, {0}), (set(), set()), 0.90),
        (({0}, {1}), (set(), set()), 0.75),
        ((set(), set()), ({0}, {0}), 0.85),
        (({0}, {0}), ({1}, {1}), 1.00),
        (({0}, set()), (set(), {0}), 0.75),
        ((set(), set()), (set(), set()), 0.75),
        (({0}, set()), (set(), set()), 0.75),
        (({0, 1}, {1, 2}), (set(), set()), 0.80),
    ],
)
def test_role_aware_similarity_uses_distinct_positive_evidence(tmp_path, native, reaction, expected):
    path = tmp_path / "v2.npz"
    np.savez(path, **role_aware_arrays(native=native, reaction=reaction))
    profiles = MODULE.load_biofp(path, 0.5)
    assert MODULE.biofp_similarity(profiles["a"], profiles["b"]) == pytest.approx(expected)
    assert MODULE.UNKNOWN_INDEX not in profiles["a"].cofactors
    assert MODULE.UNKNOWN_INDEX not in profiles["a"].cofactor_observed


def test_role_aware_mechanism_jaccard_uses_union_without_joint_mask_restriction(tmp_path):
    arrays = role_aware_arrays(native=(set(), set()))
    arrays["mechanism_targets"][1, 1] = 1
    arrays["mechanism_mask"][1, 1] = True
    arrays["mechanism_confidence"][1, 1] = np.float32(0.4)
    path = tmp_path / "v2.npz"
    np.savez(path, **arrays)
    profiles = MODULE.load_biofp(path, 0.5)
    assert MODULE.biofp_similarity(profiles["a"], profiles["b"]) == 0.375


def test_role_aware_unknown_without_any_known_evidence_scores_zero(tmp_path):
    arrays = role_aware_arrays(native=(set(), set()))
    arrays["mechanism_targets"][:] = 0
    arrays["mechanism_mask"][:] = False
    arrays["mechanism_confidence"][:] = 0
    path = tmp_path / "v2.npz"
    np.savez(path, **arrays)
    profiles = MODULE.load_biofp(path, 0.5)
    assert set(profiles) == {"a", "b"}  # Retain unknown profiles for inspection.
    assert MODULE.biofp_similarity(profiles["a"], profiles["b"]) == 0
    # Even manually constructed profiles must not treat unknown as a shared cofactor.
    marked = profiles["a"]._replace(native_cofactors=frozenset({MODULE.UNKNOWN_INDEX}))
    assert MODULE.biofp_similarity(marked, marked) == 0


def test_role_aware_pool_grouping_retains_cofactor_roles(tmp_path):
    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reaction_id", "protein_id", "reaction_smiles"])
        writer.writerows((f"q_{value}", value, f"{value}>>product") for value in "abc")
    ec = tmp_path / "ec.csv"
    with ec.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "ec_number", "known_depth"])
        writer.writerows([("a", "1.1.1.1", 4), ("b", "1.1.1.2", 4), ("c", "1.1.1.2", 4)])
    path = tmp_path / "v2.npz"
    arrays = role_aware_arrays(ids=tuple("abc"), native=({0}, {0}, set()), reaction=(set(), set(), {0}))
    np.savez(path, **arrays)
    profiles = MODULE.load_biofp(path, 0.5)
    assert profiles["b"].cofactors == profiles["c"].cofactors
    assert MODULE.biofp_group_fields(profiles["b"]) != MODULE.biofp_group_fields(profiles["c"])
    kwargs = dict(max_biological=3, max_random=3, ec_prefix_depth=2,
                  biofp_threshold=0.5, min_biofp_similarity=0.875, seed=11)
    payload, report = MODULE.build_pools(pairs, ec, path, **kwargs)
    assert payload["reaction_to_negatives"]["q_a"]["biological"] == ["b"]
    assert payload["criteria"]["similarity_weights"] == {
        "mechanism": 0.75, "native_cofactor": 0.15, "reaction_cofactor": 0.10,
    }
    assert report["pair_scope"] == "train"
    arrays["pair_scope"] = np.asarray("unsplit_inventory")
    np.savez(path, **arrays)
    assert set(MODULE.load_biofp(path, 0.5)) == set("abc")
    with pytest.raises(ValueError, match="unsplit_inventory.*training"):
        MODULE.build_pools(pairs, ec, path, **kwargs)


@pytest.mark.parametrize(
    "field,replacement,expected",
    [
        ("annotation_semantics", np.asarray("unknown_future_semantics"), "annotation_semantics"),
        ("annotation_semantics", np.asarray([MODULE.ROLE_AWARE_SEMANTICS]), "annotation_semantics"),
        ("annotation_semantics", np.asarray(MODULE.ROLE_AWARE_SEMANTICS.encode()), "annotation_semantics"),
        ("cofactor_vocabulary_version", np.asarray("circe_cofactor_v1"), "cofactor_vocabulary_version"),
        ("cofactor_labels", np.asarray(MODULE.COFACTOR_LABELS[::-1]), "cofactor_labels"),
        ("cofactor_labels", np.asarray(MODULE.COFACTOR_LABELS, dtype="S"), "cofactor_labels"),
        ("cofactor_unknown_index", np.asarray(30, dtype=np.int64), "cofactor_unknown_index"),
        ("cofactor_unknown_index", np.asarray(31, dtype=np.int32), "cofactor_unknown_index"),
        ("pair_scope", np.asarray("test"), "pair_scope"),
        ("ids", np.asarray([["a", "b"]]), "ids"),
        ("mechanism_targets", np.ones((2, 7)), "shape"),
        ("native_cofactor_mask", np.zeros((2, 31)), "shape"),
        ("reaction_cofactor_targets", np.full((2, 32), 0.5), "binary"),
        ("cofactor_mask", np.full((2, 32), 2), "binary"),
        ("mechanism_mask", np.zeros((2, 8), dtype=bool), "positive-only"),
        ("cofactor_confidence", np.ones((2, 32)), "unknown must be zero"),
        ("native_cofactor_confidence", np.ones((2, 32)), "unknown must be zero"),
        ("reaction_cofactor_confidence", np.ones((2, 32)), "unknown must be zero"),
        ("cofactor_confidence", np.full((2, 32), np.nan), "finite"),
    ],
)
def test_role_aware_loader_rejects_malformed_explicit_contract(tmp_path, field, replacement, expected):
    arrays = role_aware_arrays()
    arrays[field] = replacement
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=expected):
        MODULE.load_biofp(path, 0.5)


@pytest.mark.parametrize("field", sorted(MODULE._V2_METADATA) + [
    "native_cofactor_targets", "cofactor_confidence", "mechanism_confidence",
    "native_cofactor_confidence", "reaction_cofactor_confidence",
])
def test_role_aware_loader_never_falls_back_when_metadata_or_roles_are_missing(tmp_path, field):
    arrays = role_aware_arrays()
    del arrays[field]
    path = tmp_path / "missing.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="missing"):
        MODULE.load_biofp(path, 0.5)


@pytest.mark.parametrize("family", ["cofactor", "native_cofactor", "reaction_cofactor"])
def test_role_aware_loader_validates_unknown_per_source_and_union(tmp_path, family):
    arrays = role_aware_arrays()
    for suffix in ("targets", "mask"):
        array = arrays[f"{family}_{suffix}"]
        array[0, MODULE.UNKNOWN_INDEX] = not bool(array[0, MODULE.UNKNOWN_INDEX])
    path = tmp_path / "bad_unknown.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="unknown must mark exactly"):
        MODULE.load_biofp(path, 0.5)


def test_role_aware_loader_rejects_union_that_discards_source_positives(tmp_path):
    arrays = role_aware_arrays()
    for suffix in ("targets", "mask"):
        array = arrays[f"cofactor_{suffix}"]
        array[0, 0] = 0
        array[0, 1] = 1
    path = tmp_path / "bad_union.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="union must equal"):
        MODULE.load_biofp(path, 0.5)


def test_legacy_profiles_and_positive_only_npz_keep_original_behavior(tmp_path):
    left = MODULE.BioFPProfile(frozenset({0, 1}), frozenset({0, 1}), frozenset(), frozenset())
    right = MODULE.BioFPProfile(frozenset({0}), frozenset({0, 1}), frozenset(), frozenset())
    assert MODULE.biofp_similarity(left, right) == 0.5
    assert MODULE.biofp_group_fields(left) == ((0, 1), (0, 1), (), ())
    path = tmp_path / "legacy.npz"
    mechanism = np.array([[1, 1], [1, 0]], dtype=np.float32)
    cofactor = np.array([[1, 0], [0, 1]], dtype=np.float32)
    np.savez(path, ids=np.array(["a", "b"], dtype=object),
             mechanism_targets=mechanism, mechanism_mask=mechanism.astype(bool),
             cofactor_targets=cofactor, cofactor_mask=cofactor.astype(bool))
    profiles = MODULE.load_biofp(path, 0.5, require_train_scope=True)
    assert MODULE.biofp_similarity(profiles["a"], profiles["b"]) == 1.0
    assert profiles["a"].annotation_semantics == ""


def test_profiles_cannot_silently_mix_legacy_and_role_aware_semantics():
    legacy = MODULE.BioFPProfile(frozenset({0}), frozenset({0}), frozenset(), frozenset())
    role_aware = legacy._replace(annotation_semantics=MODULE.ROLE_AWARE_SEMANTICS)
    with pytest.raises(ValueError, match="legacy and role-aware"):
        MODULE.biofp_similarity(legacy, role_aware)


@pytest.mark.parametrize("family", ["mechanism", "cofactor", "native_cofactor", "reaction_cofactor"])
@pytest.mark.parametrize("corruption", ["zero", "wrong_weight", "extra_positive"])
def test_role_aware_confidence_requires_exact_known_provenance(tmp_path, family, corruption):
    arrays = role_aware_arrays(reaction=({1}, {1}))
    confidence = arrays[f"{family}_confidence"]
    if corruption == "zero":
        confidence[:] = 0
    elif corruption == "wrong_weight":
        confidence[confidence > 0] = np.float32(0.2)
    else:
        confidence[0, 3] = np.float32(0.4)
    path = tmp_path / "bad_confidence.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=f"{family}_confidence.*exact known-positive provenance"):
        MODULE.load_biofp(path, 0.5)


def test_role_aware_loader_rejects_all_zero_confidence_before_scoring(tmp_path):
    arrays = role_aware_arrays()
    for family in ("mechanism", "cofactor", "native_cofactor", "reaction_cofactor"):
        arrays[f"{family}_confidence"][:] = 0
    path = tmp_path / "zero_confidence.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="exact known-positive provenance"):
        MODULE.load_biofp(path, 0.5, require_train_scope=True)


@pytest.mark.parametrize("require_train_scope", [False, True])
@pytest.mark.parametrize("signature", ["source_targets", "source_confidence", "source_denominator"])
def test_role_aware_loader_rejects_all_metadata_removed(tmp_path, signature, require_train_scope):
    arrays = role_aware_arrays()
    for name in MODULE._V2_METADATA:
        del arrays[name]
    keep = {
        "source_targets": {"native_cofactor_targets", "reaction_cofactor_targets"},
        "source_confidence": {"native_cofactor_confidence"},
        "source_denominator": {"reaction_cofactor_denominator"},
    }[signature]
    for name in list(arrays):
        if name.startswith(("native_cofactor_", "reaction_cofactor_")) and name not in keep:
            del arrays[name]
    path = tmp_path / "stripped_metadata.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="role-aware.*missing required v2 metadata"):
        MODULE.load_biofp(path, 0.5, require_train_scope=require_train_scope)


@pytest.mark.parametrize("width,with_sources", [(10, True), (17, True), (32, False)])
def test_legacy_array_widths_without_v2_signature_remain_supported(tmp_path, width, with_sources):
    mechanism = np.ones((2, 2), dtype=np.float32)
    cofactor = np.zeros((2, width), dtype=np.float32)
    cofactor[:, 0] = 1
    arrays = dict(ids=np.array(["a", "b"]), mechanism_targets=mechanism,
                  mechanism_mask=mechanism.astype(bool), cofactor_targets=cofactor,
                  cofactor_mask=cofactor.astype(bool), cofactor_denominator=np.ones(2))
    if with_sources:
        arrays.update(native_cofactor_targets=cofactor, reaction_cofactor_targets=cofactor,
                      native_cofactor_mask=cofactor.astype(bool), reaction_cofactor_mask=cofactor.astype(bool))
    path = tmp_path / "legacy_width.npz"
    np.savez(path, **arrays)
    profiles = MODULE.load_biofp(path, 0.5, require_train_scope=True)
    assert profiles["a"].annotation_semantics == ""
    assert MODULE.biofp_similarity(profiles["a"], profiles["b"]) == 1.0


@pytest.mark.parametrize("family", ["cofactor", "native_cofactor", "reaction_cofactor"])
@pytest.mark.parametrize("corruption", ["value", "shape", "dtype"])
def test_role_aware_loader_validates_supplied_known_evidence_denominators(tmp_path, family, corruption):
    arrays = role_aware_arrays()
    key = f"{family}_denominator"
    if corruption == "value":
        arrays[key][0] = 1 - arrays[key][0]
    elif corruption == "shape":
        arrays[key] = arrays[key][:1]
    else:
        arrays[key] = arrays[key].astype(np.float64)
    path = tmp_path / "bad_denominator.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=f"{key}.*exactly rows with known positives"):
        MODULE.load_biofp(path, 0.5)


def test_role_aware_scoring_does_not_require_unused_denominator_arrays(tmp_path):
    arrays = role_aware_arrays()
    for key in MODULE._V2_DENOMINATORS:
        del arrays[key]
    path = tmp_path / "no_denominators.npz"
    np.savez(path, **arrays)
    profiles = MODULE.load_biofp(path, 0.5)
    assert MODULE.biofp_similarity(profiles["a"], profiles["b"]) == pytest.approx(0.9)
