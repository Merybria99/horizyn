#!/usr/bin/env python3
"""Add train-pair-derived EC labels to capability reaction features.

The source-collapse pair files preserve source provenance in ``source_entries``.
This script uses that provenance to recover EC labels already present in the
source datasets:

- Horizyn train proteins via ``data/sota/uniprot_all_ec_labels.csv``.
- CLIPZyme train pairs via their direct ``ec`` column.
- SABIO-RK train pairs via ``ec_numbers_json``.

ReactZyme official split rows do not contain EC labels, so they remain unlabeled
unless another curated mapping is provided separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.capability.reaction_types import reaction_type_labels  # noqa: E402

EC_TOKEN_PATTERN = re.compile(r"[1-7](?:\.(?:\d+|-)){0,3}")


def parse_ec_numbers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    output: list[str] = []
    seen: set[str] = set()
    text = str(value).replace(",", ";").replace("EC-", "").replace("EC:", "")
    for raw_ec in text.split(";"):
        raw_ec = raw_ec.strip()
        if not raw_ec:
            continue
        match = EC_TOKEN_PATTERN.search(raw_ec)
        if match is None:
            continue
        parts = match.group(0).split(".")
        while len(parts) < 4:
            parts.append("-")
        normalized = ".".join(parts[:4])
        if normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        return []
    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return []


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def vocab(values: list[Any]) -> list[str]:
    return sorted({item for row in values for item in as_list(row)})


def count_with(values: Any) -> int:
    return int(sum(bool(as_list(row)) for row in values))


def build_label_vocabs(features: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "ec_labels": vocab(features["ec_numbers"].tolist()),
        "cofactor_labels": vocab(features["cofactor_labels"].tolist()),
        "product_class_labels": vocab(features["product_class_labels"].tolist()),
        "reaction_center_labels": vocab(features["reaction_center_coarse_labels"].tolist()),
        "reaction_center_raw_labels": vocab(features["reaction_center_raw_labels"].tolist()),
        "reaction_type_labels": vocab(features["reaction_type_labels"].tolist()),
        "substrate_class_labels": vocab(features["substrate_class_labels"].tolist()),
    }


def quality_report(features: pd.DataFrame) -> dict[str, Any]:
    flag_counts = Counter(flag for flags in features["quality_flags"] for flag in as_list(flags))
    return {
        "known_failure_modes": [],
        "missingness_by_ec_level1": {},
        "num_enzymes_total_train": 0,
        "num_enzymes_with_capability_labels": 0,
        "num_reactions_total": int(len(features)),
        "num_reactions_with_atom_mapping": int(
            sum(
                isinstance(status, dict) and status.get("reaction_center") == "ok"
                for status in features["extraction_status"]
            )
        ),
        "num_reactions_with_cofactor_labels": count_with(features["cofactor_labels"]),
        "num_reactions_with_drfp": count_with(features["drfp_active_bits"]),
        "num_reactions_with_ec": count_with(features["ec_numbers"]),
        "num_reactions_with_reaction_center_labels": count_with(
            features["reaction_center_coarse_labels"]
        ),
        "num_reactions_with_reaction_type_labels": count_with(features["reaction_type_labels"]),
        "num_reactions_with_substrate_product_labels": int(
            sum(
                bool(as_list(row["substrate_class_labels"]) or as_list(row["product_class_labels"]))
                for row in features.to_dict("records")
            )
        ),
        "num_reactions_with_valid_smiles": int(features["canonical_reaction_smiles"].notna().sum()),
        "quality_flag_counts": dict(flag_counts),
        "top_cofactor_labels": dict(
            Counter(label for labels in features["cofactor_labels"] for label in as_list(labels)).most_common(50)
        ),
        "top_product_class_labels": dict(
            Counter(
                label for labels in features["product_class_labels"] for label in as_list(labels)
            ).most_common(50)
        ),
        "top_reaction_center_labels": dict(
            Counter(
                label
                for labels in features["reaction_center_coarse_labels"]
                for label in as_list(labels)
            ).most_common(50)
        ),
        "top_reaction_type_labels": dict(
            Counter(
                label for labels in features["reaction_type_labels"] for label in as_list(labels)
            ).most_common(50)
        ),
        "top_substrate_class_labels": dict(
            Counter(
                label for labels in features["substrate_class_labels"] for label in as_list(labels)
            ).most_common(50)
        ),
    }


def load_horizyn_ec(path: Path) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ecs = parse_ec_numbers(row.get("ec_number"))
            if not ecs:
                continue
            for key in (row.get("protein_id"), row.get("uniprot_accession")):
                if key:
                    labels[str(key).strip()].update(ecs)
    return labels


def load_source_index_ec(path: Path, ec_column: str) -> dict[int, set[str]]:
    labels: dict[int, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            ecs = parse_ec_numbers(row.get(ec_column))
            if ecs:
                labels[idx].update(ecs)
    return labels


def source_entry_ec_labels(
    source_entry: str,
    *,
    horizyn_ec: dict[str, set[str]],
    clipzyme_index_ec: dict[int, set[str]],
    sabio_index_ec: dict[int, set[str]],
) -> tuple[set[str], str | None]:
    parts = source_entry.split(":")
    if len(parts) < 4:
        return set(), None
    source_name = parts[0]
    source_protein_id = parts[-2]
    try:
        source_idx = int(parts[-1])
    except ValueError:
        source_idx = -1

    if source_name == "horizyn_train":
        return set(horizyn_ec.get(source_protein_id, set())), "horizyn_uniprot"
    if source_name == "clipzyme_train" and source_idx >= 0:
        return set(clipzyme_index_ec.get(source_idx, set())), "clipzyme_direct"
    if source_name == "sabio_rk_novelty90_train" and source_idx >= 0:
        return set(sabio_index_ec.get(source_idx, set())), "sabio_direct"
    if source_name.startswith("reactzyme_"):
        return set(), "reactzyme_no_ec_column"
    return set(), None


def update_reaction_types(features: pd.DataFrame) -> pd.DataFrame:
    reaction_types: list[list[str]] = []
    quality_flags: list[list[str]] = []
    statuses: list[dict[str, str]] = []
    for row in features.to_dict("records"):
        old_flags = [flag for flag in as_list(row.get("quality_flags")) if flag != "reaction_type_from_ec_only"]
        status = row.get("extraction_status")
        if not isinstance(status, dict):
            status = {}
        labels, flags, type_status = reaction_type_labels(
            as_list(row.get("reaction_center_coarse_labels")),
            as_list(row.get("cofactor_labels")),
            as_list(row.get("substrate_class_labels")),
            as_list(row.get("product_class_labels")),
            as_list(row.get("ec_numbers")),
        )
        next_status = dict(status)
        next_status["reaction_type"] = type_status
        reaction_types.append(labels)
        quality_flags.append(sorted(set(old_flags) | set(flags)))
        statuses.append(next_status)

    out = features.copy()
    out["reaction_type_labels"] = reaction_types
    out["quality_flags"] = quality_flags
    out["extraction_status"] = statuses
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-pairs",
        default="data/standardized/retrieval_training_source_collapse/train_exact/train_pairs_valid_rxn_pseudo.csv",
    )
    parser.add_argument(
        "--reaction-features",
        default="data/processed/capability_features/train_exact/reaction_features.parquet",
    )
    parser.add_argument("--output-reaction-features", default=None)
    parser.add_argument("--horizyn-ec", default="data/sota/uniprot_all_ec_labels.csv")
    parser.add_argument("--clipzyme-pairs", default="data/paper/clipzyme/train/enzymemap/pairs.csv")
    parser.add_argument("--sabio-pairs", default="data/paper/sabio_rk/eval/novelty90/pairs.csv")
    parser.add_argument("--out-dir", default="data/processed/capability_features/train_exact")
    args = parser.parse_args()

    train_pairs_path = PROJECT_ROOT / args.train_pairs
    reaction_features_path = PROJECT_ROOT / args.reaction_features
    output_features_path = (
        PROJECT_ROOT / args.output_reaction_features if args.output_reaction_features else reaction_features_path
    )
    out_dir = PROJECT_ROOT / args.out_dir

    horizyn_ec = load_horizyn_ec(PROJECT_ROOT / args.horizyn_ec)
    clipzyme_index_ec = load_source_index_ec(PROJECT_ROOT / args.clipzyme_pairs, "ec")
    sabio_index_ec = load_source_index_ec(PROJECT_ROOT / args.sabio_pairs, "ec_numbers_json")

    pairs = pd.read_csv(train_pairs_path)
    reaction_to_ec: dict[str, set[str]] = defaultdict(set)
    enzyme_to_ec: dict[str, set[str]] = defaultdict(set)
    source_entries = Counter()
    source_entries_with_ec = Counter()
    pair_rows: list[dict[str, Any]] = []

    for row in pairs.to_dict("records"):
        reaction_id = str(row["reaction_id"])
        enzyme_id = str(row.get("protein_id") or row.get("enzyme_id"))
        pair_ecs: set[str] = set()
        pair_sources: set[str] = set()
        for source_entry in str(row.get("source_entries", "")).split("|"):
            if not source_entry:
                continue
            source_name = source_entry.split(":", 1)[0]
            source_entries[source_name] += 1
            ecs, source = source_entry_ec_labels(
                source_entry,
                horizyn_ec=horizyn_ec,
                clipzyme_index_ec=clipzyme_index_ec,
                sabio_index_ec=sabio_index_ec,
            )
            if ecs:
                source_entries_with_ec[source_name] += 1
                pair_ecs.update(ecs)
            if source:
                pair_sources.add(source)
        if pair_ecs:
            reaction_to_ec[reaction_id].update(pair_ecs)
            enzyme_to_ec[enzyme_id].update(pair_ecs)
        pair_rows.append(
            {
                "enzyme_id": enzyme_id,
                "reaction_id": reaction_id,
                "ec_numbers": ";".join(sorted(pair_ecs)),
                "ec_sources": ";".join(sorted(pair_sources)),
                "source_split": "train",
            }
        )

    features = pd.read_parquet(reaction_features_path)
    features["ec_numbers"] = features["reaction_id"].astype(str).map(
        lambda reaction_id: sorted(reaction_to_ec.get(reaction_id, set()))
    )
    features = update_reaction_types(features)
    output_features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_features_path, index=False)

    pd.DataFrame(pair_rows).to_csv(out_dir / "pair_ec_annotations.csv", index=False)
    pd.DataFrame(
        [
            {
                "reaction_id": reaction_id,
                "ec_number": ";".join(sorted(ecs)),
                "num_ec_numbers": len(ecs),
                "ec_label_source": "train_pair_union",
            }
            for reaction_id, ecs in sorted(reaction_to_ec.items())
        ]
    ).to_csv(out_dir / "reaction_ec_from_train_pairs.csv", index=False)
    pd.DataFrame(
        [
            {
                "enzyme_id": enzyme_id,
                "ec_number": ";".join(sorted(ecs)),
                "num_ec_numbers": len(ecs),
                "ec_label_source": "train_pair_union",
            }
            for enzyme_id, ecs in sorted(enzyme_to_ec.items())
        ]
    ).to_csv(out_dir / "enzyme_ec_from_train_pairs.csv", index=False)

    vocabs = build_label_vocabs(features)
    write_json(out_dir / "label_vocabs.json", vocabs)
    report = quality_report(features)
    report["top_ec_labels"] = dict(Counter(ec for ecs in features["ec_numbers"] for ec in ecs).most_common(50))
    write_json(out_dir / "annotation_quality_report.json", report)

    enzyme_to_reaction_union_ec: dict[str, set[str]] = defaultdict(set)
    for row in pairs.to_dict("records"):
        enzyme_id = str(row.get("protein_id") or row.get("enzyme_id"))
        reaction_id = str(row["reaction_id"])
        enzyme_to_reaction_union_ec[enzyme_id].update(reaction_to_ec.get(reaction_id, set()))

    ec_report = {
        "source": "train_pair_union_existing_ec_annotations",
        "train_pairs": int(len(pairs)),
        "train_pairs_with_ec": int(sum(bool(row["ec_numbers"]) for row in pair_rows)),
        "train_reactions_with_ec": int(len(reaction_to_ec)),
        "train_enzymes_with_direct_pair_ec": int(len(enzyme_to_ec)),
        "train_enzymes_with_reaction_union_ec": int(
            sum(bool(ec_numbers) for ec_numbers in enzyme_to_reaction_union_ec.values())
        ),
        "unique_ec_labels": int(len(vocabs["ec_labels"])),
        "unique_complete_ec4_labels": int(sum("-" not in ec.split(".") for ec in vocabs["ec_labels"])),
        "source_entry_counts": dict(source_entries),
        "source_entry_counts_with_ec": dict(source_entries_with_ec),
        "ec_sources": {
            "horizyn_uniprot": args.horizyn_ec,
            "clipzyme_direct": f"{args.clipzyme_pairs}:ec",
            "sabio_direct": f"{args.sabio_pairs}:ec_numbers_json",
            "reactzyme": "official train pair rows contain no EC column",
        },
    }
    write_json(out_dir / "ec_annotation_report.json", ec_report)
    print(json.dumps(ec_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
