"""Sparse observed-positive targets; absent annotations are strictly neutral.

Reaction-derived evidence is limited to retained training associations. Enzyme
annotation files are restricted to training protein IDs, which alone does not
prove their independence from historical or held-out reaction knowledge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horizyn.capability.biological_targets import (
    COFACTOR_LABELS, MECHANISM_LABELS, _labels, _reject_held_out_source,
    _sha256, canonical_protein_key, cofactor_groups, mechanism_groups,
)


SCHEMA_VERSION = "positive_bio_v1"
_CONFIDENCE_COLUMNS = ("confidence", "annotation_confidence", "label_confidence")
_ENZYME_COFACTOR_COLUMNS = (
    "enzyme_derived_core_cofactor_labels_train", "uniprot_core_cofactor_labels_train",
    "enzyme_uniprotkb_cofactor_labels_train",
)


def _positive_labels(value: Any) -> set[str]:
    # These source columns assert labels, not signed statements. Never turn
    # explicit absence text such as 'NAD independent' into an NAD observation.
    return {
        label for label in _labels(value)
        if re.search(r"(^|[\s_-])(no|not|non|without|none|unknown|negative|absent|independent)([\s_-]|$)", label, re.I) is None
    }


def _canonical_ec(value: str) -> str | None:
    """Canonical specified prefix, never a completed or imputed EC number."""
    parts = value.split(".")
    if not 1 <= len(parts) <= 4:
        return None
    known = []
    unknown = False
    for part in parts:
        if part == "-":
            unknown = True
        elif unknown or re.fullmatch(r"[1-9][0-9]*", part) is None:
            return None
        else:
            known.append(part)
    return ".".join(known) if known and int(known[0]) in range(1, 8) else None


def _confidence(row: dict[str, Any], columns: tuple[str, ...], default: float = 1.0) -> float:
    for column in columns:
        if column in row:
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                return 0.0
            return min(value, 1.0) if math.isfinite(value) and value > 0 else 0.0
    return default


def _id_key(value: Any, observed: dict[str, str], source: str) -> str:
    if pd.isna(value) or not str(value).strip():
        raise ValueError(f"Empty protein ID in {source}")
    original = str(value).strip()
    key = canonical_protein_key(original)
    previous = observed.setdefault(key, original)
    if not key or previous != original:
        raise ValueError(f"Canonical protein ID collision in {source}: {previous!r}, {original!r}")
    return key


def _csv(path: Path, required: set[str], optional: set[str] | None = None) -> pd.DataFrame:
    wanted = required | (optional or set())
    frame = pd.read_csv(path, usecols=lambda name: name in wanted, dtype=str)
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")
    return frame


def build_positive_targets(
    *,
    train_pairs_path: Path,
    matched_members_path: Path,
    directional_features_path: Path,
    ec_labels_path: Path,
    enzyme_cofactor_labels_path: Path | None = None,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    """Return sparse target rows in sorted training-protein order.

    Each family uses [N, maximum observed labels per protein] indices (padding
    -1) plus confidence (padding 0), never an N-by-vocabulary dense matrix.
    Repeated observations use their maximum confidence rather than multiplying
    supervision for well-studied proteins or duplicated source records.
    """
    sources = {
        "train_pairs": Path(train_pairs_path), "matched_members": Path(matched_members_path),
        "directional_features": Path(directional_features_path), "ec_labels": Path(ec_labels_path),
        "enzyme_cofactors": None if enzyme_cofactor_labels_path is None else Path(enzyme_cofactor_labels_path),
    }
    for role, path in sources.items():
        if path is not None:
            _reject_held_out_source(path, role)

    pairs = _csv(sources["train_pairs"], {"protein_id", "reaction_id"}, {"label", "Label"})
    for column in ("label", "Label"):
        if column in pairs:
            pairs = pairs.loc[pd.to_numeric(pairs[column], errors="coerce") == 1]
    if pairs.empty or pairs[["protein_id", "reaction_id"]].isna().any().any():
        raise ValueError("Training pairs must contain nonempty positive protein/reaction IDs")
    train_by_key: dict[str, str] = {}
    retained_edges = set()
    for protein, reaction in pairs[["protein_id", "reaction_id"]].itertuples(index=False, name=None):
        key = _id_key(protein, train_by_key, "training pairs")
        if not str(reaction).strip():
            raise ValueError("Empty training reaction ID")
        retained_edges.add((key, str(reaction)))
    ids = sorted(train_by_key.values())
    evidence: dict[str, dict[str, dict[str, float]]] = {family: {} for family in ("ec", "cofactor", "mechanism")}
    counts: Counter[str] = Counter()
    cofactor_source_flags: Counter[str] = Counter()

    def add(family: str, protein: str, labels: set[str], confidence: float) -> None:
        if not labels or not math.isfinite(confidence) or confidence <= 0:
            return
        row = evidence[family].setdefault(protein, {})
        for label in labels:
            row[label] = max(row.get(label, 0.0), min(confidence, 1.0))

    ec_rows = _csv(sources["ec_labels"], {"protein_id", "ec_number"}, set(_CONFIDENCE_COLUMNS))
    ec_ids: dict[str, str] = {}
    for row in ec_rows.to_dict("records"):
        key = _id_key(row["protein_id"], ec_ids, "EC labels")
        if key not in train_by_key:
            counts["ec_rows_outside_train"] += 1
            continue
        confidence = _confidence(row, _CONFIDENCE_COLUMNS)
        if confidence == 0:
            counts["ec_rows_without_positive_confidence"] += 1
            continue
        labels = set()
        for label in _labels(row["ec_number"]):
            canonical = _canonical_ec(label)
            if canonical is None:
                counts["unrecognized_ec_labels"] += 1
            else:
                labels.add(canonical)
        add("ec", train_by_key[key], labels, confidence)

    members = _csv(sources["matched_members"], {"source_protein_id", "source_reaction_id", "reaction_id"}, {"match_confidence"})
    features = pd.read_parquet(sources["directional_features"])
    if "reaction_id" not in features or features["reaction_id"].isna().any():
        raise ValueError("Directional features must contain nonempty reaction_id")
    if features["reaction_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate reaction IDs in directional features")
    feature_rows = {str(row["reaction_id"]): row for row in features.to_dict("records")}
    member_ids: dict[str, str] = {}
    for member in members.to_dict("records"):
        key = _id_key(member["source_protein_id"], member_ids, "matched members")
        if (key, str(member["source_reaction_id"])) not in retained_edges:
            counts["members_outside_retained_train_edges"] += 1
            continue
        row = feature_rows.get(str(member["reaction_id"]))
        if row is None:
            counts["members_without_features"] += 1
            continue
        match_confidence = _confidence(member, ("match_confidence",))
        if match_confidence == 0:
            counts["members_without_positive_confidence"] += 1
            continue
        protein = train_by_key[key]
        mapping_confidence = _confidence(row, ("mapping_confidence", "reaction_center_mapping_confidence"))
        add("mechanism", protein, mechanism_groups(row), match_confidence * mapping_confidence)
        reaction_cofactors = cofactor_groups(_positive_labels(row.get("core_cofactor_labels")) | _positive_labels(row.get("cofactor_labels")))
        add("cofactor", protein, reaction_cofactors, 0.4 * match_confidence)
        if reaction_cofactors:
            cofactor_source_flags["retained_reaction_participant_not_proven_requirement"] += 1
        if mapping_confidence == 0:
            counts["members_without_positive_mapping_confidence"] += 1

    if sources["enzyme_cofactors"] is not None:
        cofactor_rows = _csv(sources["enzyme_cofactors"], {"enzyme_id"}, set(_ENZYME_COFACTOR_COLUMNS) | set(_CONFIDENCE_COLUMNS) | {"quality_flags", "enzyme_cofactor_label_source"})
        available = [name for name in _ENZYME_COFACTOR_COLUMNS if name in cofactor_rows]
        if not available:
            raise ValueError("Enzyme cofactor file has no supported enzyme-derived/UniProt columns")
        cofactor_ids: dict[str, str] = {}
        for row in cofactor_rows.to_dict("records"):
            key = _id_key(row["enzyme_id"], cofactor_ids, "enzyme cofactor labels")
            if key not in train_by_key:
                counts["cofactor_rows_outside_train"] += 1
                continue
            confidence = _confidence(row, _CONFIDENCE_COLUMNS)
            if confidence == 0:
                counts["cofactor_rows_without_positive_confidence"] += 1
                continue
            flags = _labels(row.get("quality_flags"))
            for name in available:
                groups = cofactor_groups(_positive_labels(row.get(name)))
                if not groups:
                    continue
                # Most existing rows are molecule/structure-derived evidence,
                # not experimental statements of cofactor requirement.
                weight = 0.4
                if name == "enzyme_uniprotkb_cofactor_labels_train":
                    weight = 1.0 if "enzyme_cofactor_from_uniprotkb_experimental" in flags else 0.7
                add("cofactor", train_by_key[key], groups, confidence * weight)
                cofactor_source_flags[name] += 1
            cofactor_source_flags.update(flags)

    vocabularies = {
        "ec": sorted({label for row in evidence["ec"].values() for label in row}),
        "cofactor": list(COFACTOR_LABELS), "mechanism": list(MECHANISM_LABELS),
    }
    arrays: dict[str, np.ndarray] = {}
    coverage = {}
    support = {}
    for family, vocabulary in vocabularies.items():
        label_index = {label: index for index, label in enumerate(vocabulary)}
        width = max((len(row) for row in evidence[family].values()), default=0)
        # Keep one padding column for families with zero coverage.
        indices = np.full((len(ids), max(width, 1)), -1, dtype=np.int32)
        confidence = np.zeros(indices.shape, dtype=np.float32)
        family_support: Counter[str] = Counter()
        for row_index, protein in enumerate(ids):
            row = evidence[family].get(protein, {})
            for position, label in enumerate(sorted(row, key=label_index.__getitem__)):
                indices[row_index, position] = label_index[label]
                confidence[row_index, position] = row[label]
                family_support[label] += 1
        arrays[f"{family}_positive_indices"] = indices
        arrays[f"{family}_confidence"] = confidence
        arrays[f"{family}_vocab_size"] = np.asarray(len(vocabulary), dtype=np.int32)
        coverage[family] = int((indices >= 0).any(axis=1).sum())
        support[family] = {label: family_support[label] for label in vocabulary}

    metadata = {
        "schema_version": SCHEMA_VERSION, "families": vocabularies,
        "num_train_enzymes": len(ids), "num_retained_train_edges": len(retained_edges),
        "num_matched_members": len(members), "row_coverage": coverage,
        "coverage_fraction": {family: count / len(ids) for family, count in coverage.items()},
        "support": support, "filter_counts": dict(counts),
        "cofactor_source_flags": dict(cofactor_source_flags),
        "sources": {name: None if path is None else str(path) for name, path in sources.items()},
        "source_sha256": {name: None if path is None else _sha256(path) for name, path in sources.items()},
        "confidence_policy": {
            "aggregation": "maximum per protein/label; duplicate observations do not add weight",
            "ec": "source confidence or 1 when no confidence field exists",
            "mechanism": "positive match confidence times positive chemistry mapping confidence",
            "cofactor_reaction": 0.4, "cofactor_molecule_derived": 0.4,
            "cofactor_uniprotkb_experimental_flag": 1.0, "cofactor_uniprotkb_other": 0.7,
            "invalid_or_zero_confidence": "excluded, not promoted to positive confidence",
        },
        "limitations": [
            "EC and independent enzyme annotations are training-protein restricted, not proven temporally independent of held-out reaction knowledge.",
            "Mechanism vocabulary denotes coarse inferred transformations, not verified catalytic mechanisms.",
            "Molecule-derived and reaction-participant cofactor evidence is not proof of enzyme-required cofactors.",
            "Unlisted, unknown, unrecognized and negative labels contribute no biological supervision.",
        ],
    }
    return ids, arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", required=True, type=Path)
    parser.add_argument("--matched-members", required=True, type=Path)
    parser.add_argument("--directional-features", required=True, type=Path)
    parser.add_argument("--ec-labels", required=True, type=Path)
    parser.add_argument("--enzyme-cofactors", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output_npz = args.output_dir / "positive_targets.npz"
    output_vocab = args.output_dir / "positive_vocab.json"
    if output_npz.exists() or output_vocab.exists():
        raise FileExistsError(f"Refusing to overwrite existing positive targets in {args.output_dir}")
    ids, arrays, metadata = build_positive_targets(
        train_pairs_path=args.train_pairs, matched_members_path=args.matched_members,
        directional_features_path=args.directional_features, ec_labels_path=args.ec_labels,
        enzyme_cofactor_labels_path=args.enzyme_cofactors,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=args.output_dir, prefix=".positive_targets.", delete=False) as handle:
            temporary.append(Path(handle.name))
            np.savez_compressed(handle, ids=np.asarray(ids, dtype=str), **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output_dir, prefix=".positive_vocab.", delete=False) as handle:
            temporary.append(Path(handle.name))
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Exclusive hard links atomically publish complete files without ever
        # overwriting another invocation's output. Metadata is published last.
        os.link(temporary[0], output_npz)
        os.link(temporary[1], output_vocab)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    print(json.dumps({"positive_targets": str(output_npz), "train_enzymes": len(ids), "row_coverage": metadata["row_coverage"]}), flush=True)


if __name__ == "__main__":
    main()
