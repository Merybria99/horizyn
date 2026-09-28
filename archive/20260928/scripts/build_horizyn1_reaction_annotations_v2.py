#!/usr/bin/env python3
"""Reuse CIRCE-v2 reaction descriptors only for exact, unambiguous chemistry.

This reads small reaction tables, never sequence FASTA or pair tables. Exported
labels are positive reaction-associated descriptors, not protein annotations or
negative labels. Unmapped labels and omitted reactions remain unknown.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from horizyn.capability.biological_targets import (
    MECHANISM_LABELS,
    _labels,
    mechanism_groups,
)
from horizyn.capability.cofactor_vocabulary_v2 import (
    COFACTOR_LABELS,
    COFACTOR_VOCABULARY_VERSION,
    KNOWN_COFACTOR_LABELS,
    UNKNOWN_LABEL,
    cofactor_name_groups,
)

SCHEMA_VERSION = "horizyn1_circe_v2_reaction_annotations_v2"
FAMILIES = {"mechanism": list(MECHANISM_LABELS), "cofactor": list(COFACTOR_LABELS)}
LIMITATIONS = [
    "Positive reaction-associated descriptors are not direct sequence annotations.",
    "The cofactor vocabulary includes unknown for downstream enzyme exports; reaction descriptors contain only known positives.",
    "Metal groups preserve element identity, not oxidation state; original input names remain in the bound source artifacts.",
    "Missing reactions, empty families, and absent labels mean unknown, not negative.",
    "No EC labels are inferred or transferred by this export.",
    "Cached annotation provenance is retained; its release need not be UniProt 2023_05.",
    "Downstream sequence targets must use only the permitted training associations.",
    "Exact strings are required; no chemical canonicalization or direction reversal is attempted.",
]


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _descriptor(row: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mechanism = mechanism_groups(row)
    cofactors = cofactor_name_groups(
        _labels(row.get("core_cofactor_labels")) | _labels(row.get("cofactor_labels"))
    )
    return (
        tuple(label for label in MECHANISM_LABELS if label in mechanism),
        tuple(label for label in KNOWN_COFACTOR_LABELS if label in cofactors),
    )


def _feature_chemistry(row: dict[str, Any]) -> str:
    # The raw field preserves the input used to generate the cached features.
    # Do not let an incompatible raw field pass via a coincidentally matching
    # canonical field. Missing raw chemistry may use an explicit alternative.
    for column in ("raw_reaction_smiles", "reaction_smiles", "canonical_reaction_smiles"):
        value = _text(row.get(column))
        if value:
            return value
    return ""


def build_lookup(
    raw_reactions: dict[str, str],
    cached_reactions: Iterable[dict[str, Any]],
    feature_rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return sparse positive descriptors and a disjoint coverage accounting."""
    bridge: dict[str, str] = {}
    rhea_candidates: dict[str, set[str]] = defaultdict(set)
    for row in cached_reactions:
        cached_id, chemistry = _text(row.get("reaction_id")), _text(row.get("reaction_smiles"))
        if not cached_id or not chemistry:
            raise ValueError("Cached reaction bridge contains an empty ID or chemistry")
        if cached_id in bridge and bridge[cached_id] != chemistry:
            raise ValueError(f"Conflicting bridge chemistry for {cached_id}")
        bridge[cached_id] = chemistry
        for source in _text(row.get("source_reaction_ids")).split("|"):
            match = re.fullmatch(r"(?:Rh_|RHEA:)(\d+)", source.strip())
            if match:
                rhea_candidates[f"Rh_{match.group(1)}"].add(cached_id)

    features: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        cached_id = _text(row.get("reaction_id"))
        if not cached_id:
            raise ValueError("Cached features contain an empty reaction_id")
        features[cached_id].append(row)

    exported: dict[str, dict[str, Any]] = {}
    classifications: dict[str, list[str]] = {
        "matched": [], "unknown": [], "chemistry_mismatch": [], "ambiguous": []
    }
    unknown_reasons: dict[str, list[str]] = defaultdict(list)
    family_counts: Counter[str] = Counter()
    support = {family: Counter() for family in FAMILIES}
    for raw_id, chemistry in sorted(raw_reactions.items()):
        if raw_id.startswith("Rh_"):
            candidates = rhea_candidates.get(raw_id, set())
        elif raw_id.startswith("Em_"):
            candidates = {"rxn_" + raw_id[3:]} & bridge.keys()
        else:
            candidates = set()
        if not candidates:
            classifications["unknown"].append(raw_id)
            unknown_reasons["no_source_id_match"].append(raw_id)
            continue
        available = sorted(cached_id for cached_id in candidates if cached_id in features)
        if not available:
            classifications["unknown"].append(raw_id)
            unknown_reasons["no_cached_feature_row"].append(raw_id)
            continue
        exact = [
            (cached_id, row)
            for cached_id in available
            if bridge[cached_id] == chemistry
            for row in features[cached_id]
            if _feature_chemistry(row) == chemistry
        ]
        if not exact:
            classifications["chemistry_mismatch"].append(raw_id)
            continue
        descriptors = {_descriptor(row) for _, row in exact}
        if len(descriptors) != 1:
            classifications["ambiguous"].append(raw_id)
            continue
        mechanism, cofactors = next(iter(descriptors))
        if not mechanism and not cofactors:
            classifications["unknown"].append(raw_id)
            unknown_reasons["no_supported_positive_descriptor"].append(raw_id)
            continue
        matching_ids = sorted({cached_id for cached_id, _ in exact})
        exported[raw_id] = {
            "mechanism": list(mechanism),
            "cofactor": list(cofactors),
            "source_reaction_id": matching_ids[0],
            "source_reaction_ids": matching_ids,
            "chemistry_match": "exact",
            "evidence_type": "reaction_associated_descriptor",
        }
        classifications["matched"].append(raw_id)
        for family in FAMILIES:
            labels = exported[raw_id][family]
            family_counts[family] += bool(labels)
            support[family].update(labels)
    return exported, {
        "raw_reactions": len(raw_reactions),
        "cached_reaction_ids": len(bridge),
        "cached_feature_ids": len(features),
        "counts": {key: len(ids) for key, ids in classifications.items()},
        "reaction_ids_by_status": classifications,
        "unknown_reasons": dict(unknown_reasons),
        "family_positive_coverage": {family: family_counts[family] for family in FAMILIES},
        "label_support": {
            family: {label: support[family][label] for label in vocabulary}
            for family, vocabulary in FAMILIES.items()
        },
    }


def _read_csv(path: Path, delimiter: str, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain {sorted(required)}")
        return list(reader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _reject_aliases(inputs: list[Path], output: Path, manifest: Path) -> None:
    paths = [*inputs, output, manifest]
    for index in (len(inputs), len(inputs) + 1):
        target = paths[index]
        for other in paths[:index]:
            if target.resolve() == other.resolve() or (
                target.exists() and other.exists() and target.samefile(other)
            ):
                raise ValueError(f"Output aliases another artifact: {target} -> {other}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_build(
    *, raw_reactions: Path, cached_features: Path, cached_reactions: Path,
    output: Path, manifest: Path, force: bool = False,
) -> dict[str, Any]:
    sources = {"raw_reactions": raw_reactions, "cached_features": cached_features,
               "cached_reactions": cached_reactions}
    implementation = {
        "builder": Path(__file__),
        "biological_targets": PROJECT_ROOT / "horizyn/capability/biological_targets.py",
        "cofactor_vocabulary": PROJECT_ROOT / "horizyn/capability/cofactor_vocabulary_v2.py",
    }
    _reject_aliases([*sources.values(), *implementation.values()], output, manifest)
    all_inputs = {**sources, **implementation}
    before = {key: _signature(path) for key, path in all_inputs.items()}
    metadata = {
        key: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for key, path in all_inputs.items()
    }

    def verify_stability() -> None:
        if any(_signature(path) != before[key] for key, path in all_inputs.items()):
            raise RuntimeError("An input changed while constructing reaction annotations")

    verify_stability()
    if output.exists() or manifest.exists():
        previous = None
        completed_signatures = None
        if output.is_file() and manifest.is_file():
            completed_signatures = (_signature(output), _signature(manifest))
            try:
                previous = json.loads(manifest.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        if isinstance(previous, dict) and (
            previous.get("schema_version") == SCHEMA_VERSION
            and previous.get("status") == "complete"
            and previous.get("inputs") == metadata
            and isinstance(previous.get("output"), dict)
            and previous.get("output", {}).get("path") == str(output.resolve())
            and previous.get("output", {}).get("sha256") == _sha256(output)
        ):
            verify_stability()
            if completed_signatures != (_signature(output), _signature(manifest)):
                raise RuntimeError("Completed annotations changed during the resume check")
            return {**previous, "skipped": True}
        if not force:
            raise FileExistsError("Existing annotations do not match a completed build; use --force to rebuild")

    raw_rows = _read_csv(raw_reactions, "\t", {"reaction_id", "reaction_smiles"})
    raw: dict[str, str] = {}
    for row in raw_rows:
        key, chemistry = _text(row["reaction_id"]), _text(row["reaction_smiles"])
        if not key or not chemistry or key in raw:
            raise ValueError(f"Empty or duplicate raw reaction row: {key!r}")
        raw[key] = chemistry
    bridge = _read_csv(cached_reactions, ",", {"reaction_id", "reaction_smiles", "source_reaction_ids"})
    features = pd.read_parquet(cached_features)
    if "reaction_id" not in features.columns:
        raise ValueError("Cached features must contain reaction_id")
    if not {"raw_reaction_smiles", "reaction_smiles", "canonical_reaction_smiles"}.intersection(features.columns):
        raise ValueError("Cached features require explicit chemistry for verification")
    reactions, coverage = build_lookup(raw, bridge, features.to_dict("records"))
    verify_stability()
    payload = {
        "schema_version": SCHEMA_VERSION, "families": FAMILIES,
        "cofactor_vocabulary_version": COFACTOR_VOCABULARY_VERSION,
        "cofactor_unknown_label": UNKNOWN_LABEL,
        "reactions": reactions, "sources": metadata, "limitations": LIMITATIONS,
        "label_semantics": "positive_only_unknown_not_negative",
    }
    _atomic_json(output, payload)
    verify_stability()
    report = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "inputs": metadata,
        "cofactor_vocabulary_version": COFACTOR_VOCABULARY_VERSION,
        "cofactor_unknown_label": UNKNOWN_LABEL,
        "created_utc": datetime.now(timezone.utc).isoformat(), "coverage": coverage,
        "output": {"path": str(output.resolve()), "bytes": output.stat().st_size, "sha256": _sha256(output)},
        "limitations": LIMITATIONS,
    }
    verify_stability()
    _atomic_json(manifest, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("raw-reactions", "cached-features", "cached-reactions", "output", "manifest"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Explicitly rebuild mismatching or incomplete prior outputs")
    args = parser.parse_args()
    report = run_build(**vars(args))
    print(json.dumps({"skipped": report.get("skipped", False), "coverage": report["coverage"]["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
