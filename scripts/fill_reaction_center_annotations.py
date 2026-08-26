#!/usr/bin/env python3
"""Fill missing reaction-center labels with RXNMapper for an existing feature table."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from horizyn.capability.io import write_json
from horizyn.capability.reaction_center import (
    coarse_reaction_center_labels,
    extract_reaction_center_raw_labels,
)
from horizyn.capability.reaction_features import _collect_label_vocabs, _quality_report
from horizyn.capability.reaction_types import reaction_type_labels


def _bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def _status_value(status: Any, key: str) -> str | None:
    return status.get(key) if isinstance(status, dict) else None


def _list_values(value: Any) -> list[str]:
    if value is None or isinstance(value, str):
        return []
    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return []


def _needs_mapping(row: dict[str, Any], remap_ok_empty: bool) -> bool:
    canonical = str(row.get("canonical_reaction_smiles") or "")
    if ">>" not in canonical:
        return False
    center_status = _status_value(row.get("extraction_status"), "reaction_center")
    if center_status != "ok":
        return True
    return remap_ok_empty and not _list_values(row.get("reaction_center_raw_labels"))


def _map_batch(reactions: list[str], mapper) -> list[tuple[str | None, float | None]]:
    if not reactions:
        return []
    try:
        results = mapper.get_attention_guided_atom_maps(reactions)
        return [
            (
                str(result.get("mapped_rxn"))
                if isinstance(result, dict) and result.get("mapped_rxn")
                else None,
                float(result.get("confidence"))
                if isinstance(result, dict) and result.get("confidence") is not None
                else None,
            )
            for result in results
        ]
    except Exception:
        mapped: list[tuple[str | None, float | None]] = []
        for reaction in reactions:
            try:
                result = mapper.get_attention_guided_atom_maps([reaction])[0]
                mapped.append(
                    (
                        str(result.get("mapped_rxn")) if result.get("mapped_rxn") else None,
                        float(result.get("confidence"))
                        if result.get("confidence") is not None
                        else None,
                    )
                )
            except Exception:
                mapped.append((None, None))
        return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capability-dir",
        default="data/processed/capability_features/train_exact_rhea_reconstructed",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--remap-ok-empty",
        type=_bool,
        default=False,
        help="Also remap rows with reaction_center status ok but no raw labels.",
    )
    args = parser.parse_args()

    from rxnmapper import RXNMapper

    capability_dir = Path(args.capability_dir)
    path = capability_dir / "reaction_features.parquet"
    features = pd.read_parquet(path)
    rows = features.to_dict("records")
    target_indices = [
        idx for idx, row in enumerate(rows) if _needs_mapping(row, args.remap_ok_empty)
    ]
    mapper = RXNMapper()

    updated = 0
    mapped_without_labels = 0
    failed = 0
    status_before = Counter(
        _status_value(row.get("extraction_status"), "reaction_center") for row in rows
    )
    for start in range(0, len(target_indices), max(1, args.batch_size)):
        batch_indices = target_indices[start : start + max(1, args.batch_size)]
        reactions = [str(rows[idx]["canonical_reaction_smiles"]) for idx in batch_indices]
        mapped_batch = _map_batch(reactions, mapper)
        for idx, (mapped, confidence) in zip(batch_indices, mapped_batch, strict=True):
            row = rows[idx]
            status = dict(row.get("extraction_status") or {})
            flags = set(_list_values(row.get("quality_flags")))
            flags.discard("no_reaction_center")
            if not mapped:
                status["reaction_center"] = "failed"
                flags.add("no_reaction_center")
                failed += 1
            else:
                try:
                    raw = extract_reaction_center_raw_labels(mapped)
                    coarse = coarse_reaction_center_labels(raw)
                    row["reaction_center_raw_labels"] = raw
                    row["reaction_center_coarse_labels"] = coarse
                    row["reaction_center_mapping_confidence"] = confidence
                    status["reaction_center"] = "ok"
                    flags.add("reaction_center_from_rxnmapper")
                    types, type_flags, type_status = reaction_type_labels(
                        coarse,
                        _list_values(row.get("cofactor_labels")),
                        _list_values(row.get("substrate_class_labels")),
                        _list_values(row.get("product_class_labels")),
                        _list_values(row.get("ec_numbers")),
                    )
                    row["reaction_type_labels"] = types
                    status["reaction_type"] = type_status
                    flags.update(type_flags)
                    if raw:
                        updated += 1
                    else:
                        mapped_without_labels += 1
                except Exception:
                    status["reaction_center"] = "failed"
                    flags.add("no_reaction_center")
                    failed += 1
            row["extraction_status"] = status
            row["quality_flags"] = sorted(flags)
        done = min(start + max(1, args.batch_size), len(target_indices))
        print(f"Mapped reaction centers for {done}/{len(target_indices)} rows", flush=True)

    output_columns = list(features.columns)
    if "reaction_center_mapping_confidence" not in output_columns:
        output_columns.append("reaction_center_mapping_confidence")
    updated_features = pd.DataFrame(rows, columns=output_columns)
    updated_features.to_parquet(path, index=False)
    write_json(capability_dir / "label_vocabs.json", _collect_label_vocabs(rows))
    write_json(capability_dir / "annotation_quality_report.json", _quality_report(rows))
    sample = (
        updated_features.sample(n=min(50, len(updated_features)), random_state=13)
        if len(updated_features)
        else updated_features
    )
    sample[
        [
            "reaction_id",
            "canonical_reaction_smiles",
            "ec_numbers",
            "cofactor_labels",
            "reaction_center_coarse_labels",
            "substrate_class_labels",
            "product_class_labels",
            "reaction_type_labels",
            "quality_flags",
        ]
    ].to_csv(capability_dir / "reaction_feature_examples.csv", index=False)
    status_after = Counter(
        _status_value(row.get("extraction_status"), "reaction_center") for row in rows
    )
    report = {
        "capability_dir": str(capability_dir),
        "input_rows": int(len(features)),
        "target_rows": int(len(target_indices)),
        "updated_rows_with_raw_center_labels": int(updated),
        "mapped_rows_without_center_labels": int(mapped_without_labels),
        "failed_rows": int(failed),
        "reaction_center_status_before": dict(status_before),
        "reaction_center_status_after": dict(status_after),
    }
    write_json(capability_dir / "reaction_center_fill_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
