#!/usr/bin/env python3
"""Refresh biology-aware capability annotation columns without rerunning RXNMapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from horizyn.capability.annotation_dictionaries import annotation_dictionaries_payload
from horizyn.capability.cofactors import (
    cofactor_labels_for_substrate_filter,
    load_cofactor_aliases,
    split_cofactor_label_tiers,
)
from horizyn.capability.enzyme_labels import build_enzyme_capability_labels
from horizyn.capability.io import write_json
from horizyn.capability.reaction_demand import build_reaction_demand_vectors
from horizyn.capability.reaction_features import _collect_label_vocabs, _quality_report
from horizyn.capability.reaction_types import reaction_type_labels
from horizyn.capability.substrate_classes import (
    classify_reaction_substrates_products,
    substrate_product_transition_labels,
)


def _list_values(value: Any) -> list[str]:
    if value is None or isinstance(value, str):
        return []
    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return []


def refresh_revised_annotations(
    *,
    capability_dir: Path,
    cofactor_dictionary_path: Path | None,
    train_pairs_path: Path | None,
    rebuild_demand: bool,
    rebuild_enzyme_labels: bool,
) -> dict[str, int | str]:
    features_path = capability_dir / "reaction_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing reaction features: {features_path}")

    cofactor_aliases = load_cofactor_aliases(cofactor_dictionary_path)
    features = pd.read_parquet(features_path)
    rows = features.to_dict("records")

    filtered_changed = 0
    rows_with_cofactor_molecules_removed = 0
    rows_with_core_cofactors = 0
    rows_with_transitions = 0
    for row in rows:
        cofactors = _list_values(row.get("cofactor_labels"))
        tiers = split_cofactor_label_tiers(cofactors)
        row.update(tiers)
        if tiers["core_cofactor_labels"]:
            rows_with_core_cofactors += 1

        canonical = str(row.get("canonical_reaction_smiles") or "")
        substrate_unfiltered: list[str] = []
        product_unfiltered: list[str] = []
        substrate_labels: list[str] = []
        product_labels: list[str] = []
        transition_labels: list[str] = []
        flags = set(_list_values(row.get("quality_flags")))
        flags.discard("substrate_class_from_smarts")
        flags.discard("substrate_class_invalid_molecule")
        flags.discard("substrate_class_removed_cofactor_molecule")
        status = dict(row.get("extraction_status") or {})

        if ">>" in canonical:
            substrate_unfiltered, product_unfiltered, _, _ = (
                classify_reaction_substrates_products(canonical)
            )
            skip_labels = cofactor_labels_for_substrate_filter(cofactors)
            substrate_labels, product_labels, class_flags, class_status = (
                classify_reaction_substrates_products(
                    canonical,
                    cofactor_aliases=cofactor_aliases,
                    skip_cofactor_labels=skip_labels,
                )
            )
            transition_labels = substrate_product_transition_labels(
                substrate_labels,
                product_labels,
            )
            if "substrate_class_removed_cofactor_molecule" in class_flags:
                rows_with_cofactor_molecules_removed += 1
            flags.update(class_flags)
            status["substrate_product_class"] = class_status

        if (
            substrate_labels != _list_values(row.get("substrate_class_labels"))
            or product_labels != _list_values(row.get("product_class_labels"))
        ):
            filtered_changed += 1
        if transition_labels:
            rows_with_transitions += 1

        row["substrate_class_labels_unfiltered"] = substrate_unfiltered
        row["product_class_labels_unfiltered"] = product_unfiltered
        row["substrate_class_labels"] = substrate_labels
        row["product_class_labels"] = product_labels
        row["substrate_product_transition_labels"] = transition_labels
        row.setdefault("reaction_center_mapping_confidence", None)

        types, type_flags, type_status = reaction_type_labels(
            _list_values(row.get("reaction_center_coarse_labels")),
            cofactors,
            substrate_labels,
            product_labels,
            _list_values(row.get("ec_numbers")),
        )
        row["reaction_type_labels"] = types
        status["reaction_type"] = type_status
        flags.update(type_flags)
        row["extraction_status"] = status
        row["quality_flags"] = sorted(flags)

    updated = pd.DataFrame(rows)
    updated.to_parquet(features_path, index=False)
    write_json(capability_dir / "label_vocabs.json", _collect_label_vocabs(rows))
    write_json(capability_dir / "annotation_dictionaries.json", annotation_dictionaries_payload())
    write_json(capability_dir / "annotation_quality_report.json", _quality_report(rows))
    sample = updated.sample(n=min(50, len(updated)), random_state=13) if len(updated) else updated
    example_columns = [
        column
        for column in [
            "reaction_id",
            "canonical_reaction_smiles",
            "ec_numbers",
            "cofactor_labels",
            "core_cofactor_labels",
            "metal_ion_labels",
            "auxiliary_participant_labels",
            "reaction_center_coarse_labels",
            "substrate_class_labels",
            "product_class_labels",
            "substrate_product_transition_labels",
            "reaction_type_labels",
            "quality_flags",
        ]
        if column in sample.columns
    ]
    sample[example_columns].to_csv(capability_dir / "reaction_feature_examples.csv", index=False)

    if rebuild_demand:
        build_reaction_demand_vectors(
            reaction_features_path=features_path,
            reaction_drfp_path=capability_dir / "reaction_drfp.npz",
            out_dir=capability_dir,
        )
    if rebuild_enzyme_labels:
        if train_pairs_path is None:
            default_pairs = capability_dir / "train_pairs_directional_rhea_reconstructed.csv"
            train_pairs_path = default_pairs if default_pairs.exists() else None
        if train_pairs_path is None:
            raise ValueError("--rebuild-enzyme-labels requires --train-pairs")
        build_enzyme_capability_labels(
            train_pairs_path=train_pairs_path,
            reaction_features_path=features_path,
            out_dir=capability_dir,
        )

    report = {
        "capability_dir": str(capability_dir),
        "reaction_rows": int(len(updated)),
        "rows_with_core_cofactors": int(rows_with_core_cofactors),
        "rows_with_substrate_product_transitions": int(rows_with_transitions),
        "rows_with_cofactor_molecules_removed": int(rows_with_cofactor_molecules_removed),
        "rows_changed_by_cofactor_filtering": int(filtered_changed),
        "rebuilt_reaction_demand_vectors": int(bool(rebuild_demand)),
        "rebuilt_enzyme_capability_labels": int(bool(rebuild_enzyme_labels)),
    }
    write_json(capability_dir / "revised_focus_annotation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capability-dir",
        default="data/processed/capability_features/train_exact_rhea_reconstructed",
    )
    parser.add_argument(
        "--cofactor-dictionary",
        default="data/processed/capability_features/chebi_cofactor_dictionary.tsv",
    )
    parser.add_argument("--train-pairs", default=None)
    parser.add_argument("--rebuild-demand", action="store_true")
    parser.add_argument("--rebuild-enzyme-labels", action="store_true")
    args = parser.parse_args()

    dictionary = Path(args.cofactor_dictionary)
    report = refresh_revised_annotations(
        capability_dir=Path(args.capability_dir),
        cofactor_dictionary_path=dictionary if dictionary.exists() else None,
        train_pairs_path=Path(args.train_pairs) if args.train_pairs else None,
        rebuild_demand=bool(args.rebuild_demand),
        rebuild_enzyme_labels=bool(args.rebuild_enzyme_labels),
    )
    import json

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
