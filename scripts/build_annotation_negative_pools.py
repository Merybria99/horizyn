#!/usr/bin/env python3
"""Build CIRCE-V2 negative pools from training-set annotations only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.capability.cofactor_vocabulary_v2 import (
    COFACTOR_LABELS,
    COFACTOR_VOCABULARY_VERSION,
    UNKNOWN_INDEX,
)

ROLE_AWARE_SEMANTICS = "positive_only_role_aware_v2"
_V2_METADATA = frozenset({
    "annotation_semantics", "cofactor_vocabulary_version", "cofactor_labels",
    "cofactor_unknown_index", "pair_scope",
})
_V2_DENOMINATORS = frozenset({
    "cofactor_denominator", "native_cofactor_denominator", "reaction_cofactor_denominator",
})


class BioFPProfile(NamedTuple):
    mechanisms: frozenset[int]
    mechanism_observed: frozenset[int]
    cofactors: frozenset[int]
    cofactor_observed: frozenset[int]
    annotation_semantics: str = ""
    native_cofactors: frozenset[int] = frozenset()
    reaction_cofactors: frozenset[int] = frozenset()
    pair_scope: str = ""


BioFPGroupKey = tuple[Any, ...]


def canonical_protein_id(protein_id: str) -> str:
    value = str(protein_id).strip()
    for prefix in ("prot_", "uprot_", "nr90_"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def reaction_signature(reaction_smiles: str, reaction_id: str) -> str:
    value = str(reaction_smiles or "").strip()
    if ">>" not in value:
        return ".".join(sorted(filter(None, value.split(".")))) if value else str(reaction_id)
    left, right = value.split(">>", 1)
    return (
        ".".join(sorted(filter(None, left.split("."))))
        + ">>"
        + ".".join(sorted(filter(None, right.split("."))))
    )


def load_training_graph(
    path: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str], list[str]]:
    reaction_to_positives: dict[str, set[str]] = defaultdict(set)
    protein_to_reactions: dict[str, set[str]] = defaultdict(set)
    reaction_to_signature: dict[str, str] = {}
    proteins: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"reaction_id", "protein_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            query_id = row["reaction_id"].strip()
            protein_id = row["protein_id"].strip()
            reaction_to_positives[query_id].add(protein_id)
            protein_to_reactions[protein_id].add(query_id)
            proteins.add(protein_id)
            signature = reaction_signature(row.get("reaction_smiles", ""), query_id)
            previous = reaction_to_signature.setdefault(query_id, signature)
            if previous != signature:
                raise ValueError(f"Reaction {query_id!r} has inconsistent reaction SMILES")
    return reaction_to_positives, protein_to_reactions, reaction_to_signature, sorted(proteins)


def load_ec_labels(
    path: Path,
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    complete_labels: dict[str, set[str]] = defaultdict(set)
    all_labels: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"protein_id", "ec_number", "known_depth"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            try:
                depth = int(row["known_depth"])
            except (TypeError, ValueError):
                continue
            ec_number = row["ec_number"].strip()
            protein_id = canonical_protein_id(row["protein_id"])
            if ec_number:
                all_labels[protein_id].add(ec_number)
            if depth == 4 and len(ec_number.split(".")) == 4 and "-" not in ec_number:
                complete_labels[protein_id].add(ec_number)
    return (
        {protein_id: frozenset(values) for protein_id, values in complete_labels.items()},
        {protein_id: frozenset(values) for protein_id, values in all_labels.items()},
    )


def load_complete_ec_labels(path: Path) -> dict[str, frozenset[str]]:
    complete_labels, _all_labels = load_ec_labels(path)
    return complete_labels


def load_candidate_eligibility(
    path: Path,
) -> tuple[dict[str, tuple[bool, bool]], dict[str, Any]]:
    """Load candidate-only restrictions; never remove annotations from positives."""
    required = {
        "protein_id",
        "ec_negative_candidate_eligible",
        "biological_negative_candidate_eligible",
    }
    eligibility: dict[str, tuple[bool, bool]] = {}
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not required.issubset(fieldnames) or len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{path} must contain unique columns including {sorted(required)}")
        for row in reader:
            rows += 1
            if None in row or any(row.get(column) is None for column in required):
                raise ValueError(f"{path}:{reader.line_num}: malformed eligibility CSV row")
            protein_id = canonical_protein_id(row["protein_id"])
            if not protein_id:
                raise ValueError(f"{path}:{reader.line_num}: protein_id must not be empty")
            flags = []
            for column in (
                "ec_negative_candidate_eligible",
                "biological_negative_candidate_eligible",
            ):
                value = row[column]
                if value not in {"0", "1"}:
                    raise ValueError(
                        f"{path}:{reader.line_num}: {column} must be exactly 0 or 1"
                    )
                flags.append(value == "1")
            values = (flags[0], flags[1])
            previous = eligibility.setdefault(protein_id, values)
            if previous != values:
                raise ValueError(
                    f"{path}:{reader.line_num}: conflicting eligibility flags for "
                    f"canonical protein ID {protein_id!r}"
                )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return eligibility, {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "rows": rows,
        "unique_canonical_proteins": len(eligibility),
        "duplicate_consistent_rows": rows - len(eligibility),
        "ec_flag_eligible_proteins": sum(flags[0] for flags in eligibility.values()),
        "biological_and_ec_flag_eligible_proteins": sum(
            flags[0] and flags[1] for flags in eligibility.values()
        ),
    }


def ec_annotation_matches_complete(annotation: str, complete_ec: str) -> bool:
    annotation_parts = annotation.split(".")
    complete_parts = complete_ec.split(".")
    return len(annotation_parts) == len(complete_parts) and all(
        annotation_part == "-" or annotation_part == complete_part
        for annotation_part, complete_part in zip(annotation_parts, complete_parts)
    )


def _v2_metadata(payload: Any, path: Path) -> str | None:
    """An explicit version contract must validate completely or fail closed."""
    present = _V2_METADATA & set(payload.files)
    if not present:
        return None
    missing = _V2_METADATA - set(payload.files)
    if missing:
        raise ValueError(f"{path} is missing role-aware metadata: {sorted(missing)}")
    for name, expected in (
        ("annotation_semantics", ROLE_AWARE_SEMANTICS),
        ("cofactor_vocabulary_version", COFACTOR_VOCABULARY_VERSION),
    ):
        value = payload[name]
        if value.shape != () or value.dtype.kind != "U" or value.item() != expected:
            raise ValueError(f"{path}: invalid {name}; expected Unicode scalar {expected!r}")
    labels = payload["cofactor_labels"]
    if (labels.dtype.kind != "U" or labels.shape != (len(COFACTOR_LABELS),)
            or tuple(labels.tolist()) != COFACTOR_LABELS):
        raise ValueError(f"{path}: cofactor_labels must match the ordered v2 vocabulary")
    unknown = payload["cofactor_unknown_index"]
    if unknown.shape != () or unknown.dtype != np.dtype("int64") or unknown.item() != UNKNOWN_INDEX:
        raise ValueError(f"{path}: cofactor_unknown_index must be int64 scalar {UNKNOWN_INDEX}")
    scope = payload["pair_scope"]
    if (scope.shape != () or scope.dtype.kind != "U"
            or scope.item() not in {"train", "unsplit_inventory"}):
        raise ValueError(f"{path}: pair_scope must be train or unsplit_inventory")
    return str(scope.item())


def _validate_v2_arrays(arrays: dict[str, np.ndarray], path: Path) -> None:
    ids = arrays["ids"]
    if ids.ndim != 1 or ids.dtype.kind != "U":
        raise ValueError(f"{path}: ids must be a one-dimensional Unicode array")
    for family in ("mechanism", "cofactor", "native_cofactor", "reaction_cofactor"):
        width = 8 if family == "mechanism" else len(COFACTOR_LABELS)
        target, mask = arrays[f"{family}_targets"], arrays[f"{family}_mask"]
        shape = (len(ids), width)
        if target.shape != shape or mask.shape != shape:
            raise ValueError(f"{path}: {family} targets and mask must have shape {shape}")
        if target.dtype.kind not in "biuf" or mask.dtype.kind not in "biuf":
            raise ValueError(f"{path}: {family} targets and mask must be numeric binary arrays")
        if not np.all((target == 0) | (target == 1)) or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{path}: {family} targets and mask must be binary")
        if not np.array_equal(target, mask):
            raise ValueError(f"{path}: {family} mask must equal positive-only targets")
        if family != "mechanism":
            expected_unknown = ~target[:, :UNKNOWN_INDEX].any(axis=1)
            if not np.array_equal(target[:, UNKNOWN_INDEX], expected_unknown):
                raise ValueError(f"{path}: {family} unknown must mark exactly rows without known positives")
    union = arrays["cofactor_targets"][:, :UNKNOWN_INDEX]
    native = arrays["native_cofactor_targets"][:, :UNKNOWN_INDEX]
    reaction = arrays["reaction_cofactor_targets"][:, :UNKNOWN_INDEX]
    if not np.array_equal(union, np.logical_or(native, reaction)):
        raise ValueError(f"{path}: cofactor union must equal native OR reaction known positives")
    for family in ("mechanism", "cofactor", "native_cofactor", "reaction_cofactor"):
        confidence = arrays[f"{family}_confidence"]
        if confidence.shape != arrays[f"{family}_targets"].shape or confidence.dtype.kind not in "biuf":
            raise ValueError(f"{path}: {family}_confidence must match target shape")
        if not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError(f"{path}: {family}_confidence must be finite and in [0, 1]")
        if family != "mechanism" and np.any(confidence[:, UNKNOWN_INDEX] != 0):
            raise ValueError(f"{path}: {family}_confidence unknown must be zero")
        if family == "cofactor":
            expected_confidence = np.where(
                native, np.float32(1.0), np.where(reaction, np.float32(0.4), np.float32(0.0)),
            )
        else:
            targets = arrays[f"{family}_targets"]
            if family != "mechanism":
                targets = targets[:, :UNKNOWN_INDEX]
            weight = np.float32(1.0 if family == "native_cofactor" else 0.4)
            expected_confidence = targets.astype(np.float32) * weight
        observed_confidence = confidence if family == "mechanism" else confidence[:, :UNKNOWN_INDEX]
        if not np.array_equal(observed_confidence, expected_confidence):
            raise ValueError(f"{path}: {family}_confidence must match exact known-positive provenance weights")
        denominator_key = f"{family}_denominator"
        if denominator_key in arrays:
            denominator = arrays[denominator_key]
            expected_denominator = arrays[f"{family}_targets"][:, :UNKNOWN_INDEX].any(axis=1)
            if (denominator.shape != (len(ids),) or denominator.dtype != np.dtype("float32")
                    or not np.array_equal(denominator, expected_denominator)):
                raise ValueError(f"{path}: {denominator_key} must mark exactly rows with known positives")


def load_biofp(
    path: Path, threshold: float, *, require_train_scope: bool = False,
) -> dict[str, BioFPProfile]:
    with np.load(path, allow_pickle=True) as payload:
        required = {
            "ids",
            "mechanism_targets",
            "mechanism_mask",
            "cofactor_targets",
            "cofactor_mask",
        }
        pair_scope = _v2_metadata(payload, path)
        if pair_scope is not None:
            if require_train_scope and pair_scope != "train":
                raise ValueError(f"{path}: unsplit_inventory annotations cannot construct training negative pools")
            required.update({
                "native_cofactor_targets", "native_cofactor_mask",
                "reaction_cofactor_targets", "reaction_cofactor_mask",
                "mechanism_confidence", "cofactor_confidence",
                "native_cofactor_confidence", "reaction_cofactor_confidence",
            })
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} is missing BioFP arrays: {sorted(missing)}")
        arrays = {name: payload[name] for name in required}
        if pair_scope is not None:
            arrays.update({name: payload[name] for name in _V2_DENOMINATORS & set(payload.files)})
            _validate_v2_arrays(arrays, path)
        else:
            cofactor_shape = arrays["cofactor_targets"].shape
            source_targets = {"native_cofactor_targets", "reaction_cofactor_targets"}
            source_markers = {
                "native_cofactor_confidence", "reaction_cofactor_confidence",
                "native_cofactor_denominator", "reaction_cofactor_denominator",
            }
            if (len(cofactor_shape) == 2 and cofactor_shape[1] == len(COFACTOR_LABELS)
                    and (source_targets.issubset(payload.files) or source_markers.intersection(payload.files))):
                raise ValueError(f"{path}: role-aware cofactor arrays are missing required v2 metadata")
        ids = arrays["ids"].astype(str)
        mechanism_targets = arrays["mechanism_targets"]
        mechanism_mask = arrays["mechanism_mask"].astype(bool)
        cofactor_targets = arrays["cofactor_targets"]
        cofactor_mask = arrays["cofactor_mask"].astype(bool)
    profiles = {}
    for idx, protein_id in enumerate(ids):
        mechanism_observed = frozenset(np.flatnonzero(mechanism_mask[idx]).tolist())
        cofactor_observed = frozenset(np.flatnonzero(cofactor_mask[idx]).tolist())
        mechanisms = frozenset(
            np.flatnonzero(mechanism_mask[idx] & (mechanism_targets[idx] >= threshold)).tolist()
        )
        cofactors = frozenset(
            np.flatnonzero(cofactor_mask[idx] & (cofactor_targets[idx] >= threshold)).tolist()
        )
        if pair_scope is not None:
            known_native = frozenset(np.flatnonzero(
                arrays["native_cofactor_targets"][idx, :UNKNOWN_INDEX] == 1
            ).tolist())
            known_reaction = frozenset(np.flatnonzero(
                arrays["reaction_cofactor_targets"][idx, :UNKNOWN_INDEX] == 1
            ).tolist())
            profiles[canonical_protein_id(protein_id)] = BioFPProfile(
                mechanisms, mechanism_observed, cofactors - {UNKNOWN_INDEX},
                cofactor_observed - {UNKNOWN_INDEX}, ROLE_AWARE_SEMANTICS,
                known_native, known_reaction, pair_scope,
            )
        elif mechanisms:
            profiles[canonical_protein_id(protein_id)] = BioFPProfile(
                mechanisms=mechanisms,
                mechanism_observed=mechanism_observed,
                cofactors=cofactors,
                cofactor_observed=cofactor_observed,
            )
    return profiles


def masked_jaccard(
    left: frozenset[int],
    right: frozenset[int],
    left_observed: frozenset[int],
    right_observed: frozenset[int],
) -> float | None:
    jointly_observed = left_observed & right_observed
    if not jointly_observed:
        return None
    left = left & jointly_observed
    right = right & jointly_observed
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def biofp_similarity(
    left: BioFPProfile,
    right: BioFPProfile,
) -> float:
    if left.annotation_semantics or right.annotation_semantics:
        if left.annotation_semantics != ROLE_AWARE_SEMANTICS or right.annotation_semantics != ROLE_AWARE_SEMANTICS:
            raise ValueError("Cannot compare legacy and role-aware BioFP semantics")
        return (
            0.75 * positive_jaccard(left.mechanisms, right.mechanisms)
            + 0.15 * positive_jaccard(left.native_cofactors - {UNKNOWN_INDEX}, right.native_cofactors - {UNKNOWN_INDEX})
            + 0.10 * positive_jaccard(left.reaction_cofactors - {UNKNOWN_INDEX}, right.reaction_cofactors - {UNKNOWN_INDEX})
        )
    mechanism_score = masked_jaccard(
        left.mechanisms,
        right.mechanisms,
        left.mechanism_observed,
        right.mechanism_observed,
    )
    if mechanism_score is None:
        return 0.0
    cofactor_score = masked_jaccard(
        left.cofactors,
        right.cofactors,
        left.cofactor_observed,
        right.cofactor_observed,
    )
    if cofactor_score is None:
        return mechanism_score
    return 0.75 * mechanism_score + 0.25 * cofactor_score


def positive_jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    """Compare supported positives; an empty union contributes no evidence."""
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def biofp_group_fields(profile: BioFPProfile) -> tuple[Any, ...]:
    # Preserve legacy key strings, sorting, and seeded tie-breaking exactly.
    fields = tuple(tuple(sorted(values)) for values in profile[:4])
    if profile.annotation_semantics:
        fields += (
            profile.annotation_semantics, tuple(sorted(profile.native_cofactors)),
            tuple(sorted(profile.reaction_cofactors)), profile.pair_scope,
        )
    return fields


def stable_offset(value: str, seed: int, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus


def stable_order_key(value: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def build_pools(
    train_pairs_path: Path,
    ec_labels_path: Path,
    biofp_targets_path: Path,
    *,
    max_biological: int,
    max_random: int,
    ec_prefix_depth: int,
    biofp_threshold: float,
    min_biofp_similarity: float,
    seed: int,
    candidate_eligibility_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if max_biological < 0:
        raise ValueError("max_biological must be non-negative")
    if max_random <= 0:
        raise ValueError("max_random must be positive")
    if ec_prefix_depth not in {1, 2, 3}:
        raise ValueError("ec_prefix_depth must be 1, 2, or 3")
    if not 0.0 <= biofp_threshold <= 1.0:
        raise ValueError("biofp_threshold must be in [0, 1]")
    if not 0.0 <= min_biofp_similarity <= 1.0:
        raise ValueError("min_biofp_similarity must be in [0, 1]")

    reaction_to_positives, protein_to_reactions, reaction_signatures, proteins = (
        load_training_graph(train_pairs_path)
    )
    if not reaction_to_positives:
        raise ValueError("Training graph contains no reaction-protein pairs")
    complete_ec, all_ec = load_ec_labels(ec_labels_path)
    biofp = load_biofp(biofp_targets_path, biofp_threshold, require_train_scope=True)
    candidate_flags = None
    eligibility_report = None
    if candidate_eligibility_path is not None:
        candidate_flags, eligibility_report = load_candidate_eligibility(candidate_eligibility_path)

    def candidate_allowed(protein_id: str, *, biological: bool = False) -> bool:
        if candidate_flags is None:
            return True
        ec_allowed, bio_allowed = candidate_flags.get(
            canonical_protein_id(protein_id), (False, False)
        )
        return ec_allowed and (not biological or bio_allowed)

    # Keep ALL annotations, including those of ineligible candidates, for
    # positive-EC exclusion and internally conflicting-EC rejection below.
    protein_ec = {
        protein_id: complete_ec.get(canonical_protein_id(protein_id), frozenset())
        for protein_id in proteins
    }
    protein_biofp = {
        protein_id: biofp.get(canonical_protein_id(protein_id)) for protein_id in proteins
    }
    protein_has_conflicting_ec = {}
    for protein_id in proteins:
        canonical_id = canonical_protein_id(protein_id)
        complete_values = complete_ec.get(canonical_id, frozenset())
        annotations = all_ec.get(canonical_id, frozenset())
        protein_has_conflicting_ec[protein_id] = len(complete_values) > 1 or (
            len(complete_values) == 1
            and any(
                not ec_annotation_matches_complete(annotation, next(iter(complete_values)))
                for annotation in annotations
            )
        )
    eligible_biological = [
        protein_id
        for protein_id in proteins
        if len(protein_ec[protein_id]) == 1
        and not protein_has_conflicting_ec[protein_id]
        and protein_biofp[protein_id] is not None
        and protein_biofp[protein_id].mechanisms
        and candidate_allowed(protein_id, biological=True)
    ]
    eligible_random = [
        protein_id
        for protein_id in proteins
        if len(protein_ec[protein_id]) == 1 and not protein_has_conflicting_ec[protein_id]
        and candidate_allowed(protein_id)
    ]
    if not eligible_random:
        raise ValueError(
            "No training proteins have one internally consistent complete EC annotation; "
            "cannot construct safe random negatives"
            + (" after candidate-eligibility filtering" if candidate_flags is not None else "")
        )
    candidate_groups: dict[BioFPGroupKey, list[str]] = defaultdict(list)
    candidate_profiles: dict[BioFPGroupKey, BioFPProfile] = {}
    candidate_index: dict[tuple[str, int], set[BioFPGroupKey]] = defaultdict(set)
    for protein_id in eligible_biological:
        ec_number = next(iter(protein_ec[protein_id]))
        prefix = ".".join(ec_number.split(".")[:ec_prefix_depth])
        profile = protein_biofp[protein_id]
        group_key = (
            prefix,
            ec_number,
            *biofp_group_fields(profile),
        )
        candidate_groups[group_key].append(protein_id)
        candidate_profiles[group_key] = profile
        for mechanism in profile.mechanisms:
            candidate_index[(prefix, mechanism)].add(group_key)
    for members in candidate_groups.values():
        members.sort()

    protein_signatures = {
        protein_id: frozenset(
            reaction_signatures[query_id] for query_id in protein_to_reactions[protein_id]
        )
        for protein_id in proteins
    }

    random_order = list(eligible_random)
    rng = np.random.default_rng(seed)
    rng.shuffle(random_order)
    output: dict[str, dict[str, list[str]]] = {}
    biological_counts = []
    random_counts = []

    for query_id in sorted(reaction_to_positives):
        positives = reaction_to_positives[query_id]
        positive_ecs = frozenset(ec for protein_id in positives for ec in protein_ec[protein_id])
        positive_profiles = sorted(
            {
                protein_biofp[protein_id]
                for protein_id in positives
                if protein_biofp[protein_id] is not None
            },
            key=biofp_group_fields,
        )
        prefixes = {".".join(ec_number.split(".")[:ec_prefix_depth]) for ec_number in positive_ecs}
        candidate_group_keys: set[BioFPGroupKey] = set()
        for prefix in prefixes:
            for profile in positive_profiles:
                for mechanism in profile.mechanisms:
                    candidate_group_keys.update(candidate_index.get((prefix, mechanism), ()))

        scored_groups = []
        query_signature = reaction_signatures[query_id]
        for group_key in candidate_group_keys:
            candidate_ec = group_key[1]
            if candidate_ec in positive_ecs:
                continue
            candidate_profile = candidate_profiles[group_key]
            similarity = max(
                biofp_similarity(positive_profile, candidate_profile)
                for positive_profile in positive_profiles
            )
            if similarity >= min_biofp_similarity:
                scored_groups.append((similarity, group_key))
        scored_groups.sort(
            key=lambda item: (
                -item[0],
                stable_order_key(f"{query_id}:{item[1]}", seed),
            )
        )
        biological = []
        if max_biological == 0:
            scored_groups = []
        for _score, group_key in scored_groups:
            group_members = sorted(
                candidate_groups[group_key],
                key=lambda candidate_id: stable_order_key(f"{query_id}:{candidate_id}", seed),
            )
            for candidate_id in group_members:
                if candidate_id in positives or query_signature in protein_signatures[candidate_id]:
                    continue
                biological.append(candidate_id)
                if len(biological) == max_biological:
                    break
            if len(biological) == max_biological:
                break

        excluded = positives | set(biological)
        random_negatives = []
        start = stable_offset(query_id, seed, len(random_order))
        for step in range(len(random_order)):
            candidate_id = random_order[(start + step) % len(random_order)]
            if candidate_id in excluded:
                continue
            candidate_ecs = protein_ec[candidate_id]
            if positive_ecs & candidate_ecs:
                continue
            if query_signature in protein_signatures[candidate_id]:
                continue
            random_negatives.append(candidate_id)
            if len(random_negatives) == max_random:
                break

        output[query_id] = {"biological": biological, "random": random_negatives}
        biological_counts.append(len(biological))
        random_counts.append(len(random_negatives))

    payload = {
        "schema_version": 1,
        "source": "training_annotations_only",
        "criteria": {
            "biological": (
                "same EC prefix, different complete leaf EC, shared BioFP mechanism, "
                "mask-aware BioFP similarity threshold, and different observed "
                "reaction signature"
            ),
            "random": (
                "seeded single, internally consistent complete-EC training protein "
                "excluding positives, biological negatives, matching complete ECs, "
                "and matching observed reaction signatures"
            ),
            "ec_prefix_depth": ec_prefix_depth,
            "biofp_threshold": biofp_threshold,
            "min_biofp_similarity": min_biofp_similarity,
            "seed": seed,
        },
        "reaction_to_negatives": output,
    }
    report = {
        "training_pairs_path": str(train_pairs_path),
        "ec_labels_path": str(ec_labels_path),
        "biofp_targets_path": str(biofp_targets_path),
        "unique_reactions": len(reaction_to_positives),
        "unique_train_proteins": len(proteins),
        "proteins_with_complete_ec": sum(bool(protein_ec[value]) for value in proteins),
        "eligible_single_ec_biofp_proteins": len(eligible_biological),
        "eligible_single_ec_random_proteins": len(eligible_random),
        "proteins_rejected_for_conflicting_ec": sum(protein_has_conflicting_ec.values()),
        "reactions_with_biological_negatives": sum(value > 0 for value in biological_counts),
        "mean_biological_negatives": float(np.mean(biological_counts)),
        "min_biological_negatives": min(biological_counts),
        "max_biological_negatives": max(biological_counts),
        "reactions_with_random_negatives": sum(value > 0 for value in random_counts),
        "mean_random_negatives": float(np.mean(random_counts)),
        "min_random_negatives": min(random_counts),
        "max_random_negatives": max(random_counts),
    }
    if any(profile.annotation_semantics == ROLE_AWARE_SEMANTICS for profile in biofp.values()):
        payload["criteria"].update({
            "biological": (
                "same EC prefix, different complete leaf EC, shared BioFP mechanism, "
                "role-aware positive-set Jaccard threshold, and different observed "
                "reaction signature"
            ),
            "annotation_semantics": ROLE_AWARE_SEMANTICS,
            "cofactor_vocabulary_version": COFACTOR_VOCABULARY_VERSION,
            "similarity_weights": {"mechanism": 0.75, "native_cofactor": 0.15, "reaction_cofactor": 0.10},
            "unknown_cofactors": "excluded; missing evidence contributes zero without reweighting",
        })
        report.update({"annotation_semantics": ROLE_AWARE_SEMANTICS, "pair_scope": "train"})
    if candidate_flags is not None:
        eligibility_report.update(
            {
                "training_proteins_with_flags": sum(
                    canonical_protein_id(protein_id) in candidate_flags for protein_id in proteins
                ),
                "training_proteins_missing_flags": sum(
                    canonical_protein_id(protein_id) not in candidate_flags
                    for protein_id in proteins
                ),
                "training_proteins_passing_ec_flag": sum(
                    candidate_allowed(protein_id) for protein_id in proteins
                ),
                "training_proteins_passing_biological_and_ec_flags": sum(
                    candidate_allowed(protein_id, biological=True) for protein_id in proteins
                ),
            }
        )
        report["candidate_eligibility"] = eligibility_report
        payload["criteria"]["candidate_eligibility"] = {
            "path": str(candidate_eligibility_path),
            "sha256": eligibility_report["sha256"],
            "random": "ec_negative_candidate_eligible=1",
            "biological": (
                "ec_negative_candidate_eligible=1 AND biological_negative_candidate_eligible=1"
            ),
            "missing_proteins": "ineligible",
            "annotations": "retained for positive exclusion and EC conflict checks",
        }
    return payload, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--ec-labels", type=Path, required=True)
    parser.add_argument("--biofp-targets", type=Path, required=True)
    parser.add_argument(
        "--candidate-eligibility",
        type=Path,
        help="Optional candidate-only eligibility CSV; proteins absent from it are ineligible",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-biological", type=int, default=32)
    parser.add_argument("--max-random", type=int, default=32)
    parser.add_argument("--ec-prefix-depth", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--biofp-threshold", type=float, default=0.5)
    parser.add_argument("--min-biofp-similarity", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, report = build_pools(
        args.train_pairs,
        args.ec_labels,
        args.biofp_targets,
        max_biological=args.max_biological,
        max_random=args.max_random,
        ec_prefix_depth=args.ec_prefix_depth,
        biofp_threshold=args.biofp_threshold,
        min_biofp_similarity=args.min_biofp_similarity,
        seed=args.seed,
        candidate_eligibility_path=args.candidate_eligibility,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
