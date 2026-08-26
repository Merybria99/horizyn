"""Train-only enzyme capability label construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from horizyn.capability.io import ensure_dir, write_json


def read_pairs(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    enzyme_col = "enzyme_id" if "enzyme_id" in df.columns else "protein_id"
    if enzyme_col not in df.columns or "reaction_id" not in df.columns:
        raise ValueError(f"{path} must contain reaction_id and enzyme_id/protein_id")
    out = df.rename(columns={enzyme_col: "enzyme_id"})[["enzyme_id", "reaction_id"]].copy()
    out["enzyme_id"] = out["enzyme_id"].astype(str)
    out["reaction_id"] = out["reaction_id"].astype(str)
    return out


def union_list_column(values: list[Any]) -> list[str]:
    out: set[str] = set()
    for xs in values:
        if xs is None:
            continue
        if isinstance(xs, str):
            continue
        if isinstance(xs, Iterable):
            out.update(str(x) for x in xs if str(x))
    return sorted(out)


def ec_overlap_level(reaction_ecs: list[str], enzyme_ecs: list[str]) -> int | None:
    best = 0
    for reaction_ec in reaction_ecs:
        r_parts = str(reaction_ec).split(".")
        for enzyme_ec in enzyme_ecs:
            e_parts = str(enzyme_ec).split(".")
            depth = 0
            for left, right in zip(r_parts, e_parts):
                if left != right:
                    break
                depth += 1
            best = max(best, min(depth, 4))
    return best or None


def intersection_size(a: list[str], b: list[str]) -> int:
    return len(set(a) & set(b))


def union_group_column(group: pd.DataFrame, column: str) -> list[str]:
    if column not in group.columns:
        return []
    return union_list_column(group[column].tolist())


def _leakage_check(
    train_pairs: pd.DataFrame,
    validation_pairs_path: str | Path | None,
    test_pairs_path: str | Path | None,
) -> None:
    train_set = set(map(tuple, train_pairs[["enzyme_id", "reaction_id"]].values.tolist()))
    for label, path in (("validation", validation_pairs_path), ("test", test_pairs_path)):
        if path is None:
            continue
        other = read_pairs(path)
        other_set = set(map(tuple, other[["enzyme_id", "reaction_id"]].values.tolist()))
        overlap = train_set & other_set
        if overlap:
            raise AssertionError(
                f"{len(overlap)} {label} pairs also appear in train pairs; "
                "refusing to build capability labels with ambiguous split provenance"
            )


def build_enzyme_capability_labels(
    *,
    train_pairs_path: str | Path,
    reaction_features_path: str | Path,
    out_dir: str | Path,
    validation_pairs_path: str | Path | None = None,
    test_pairs_path: str | Path | None = None,
    sequence_cluster_map_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = ensure_dir(out_dir)
    pairs = read_pairs(train_pairs_path)
    _leakage_check(pairs, validation_pairs_path, test_pairs_path)
    features = pd.read_parquet(reaction_features_path)
    merged = pairs.merge(features, on="reaction_id", how="left", validate="many_to_one")

    enzyme_rows: list[dict[str, Any]] = []
    for enzyme_id, group in merged.groupby("enzyme_id", sort=True):
        ec_numbers = union_list_column(group["ec_numbers"].tolist())
        cofactors = union_list_column(group["cofactor_labels"].tolist())
        core_cofactors = union_group_column(group, "core_cofactor_labels")
        metal_ions = union_group_column(group, "metal_ion_labels")
        auxiliary_participants = union_group_column(group, "auxiliary_participant_labels")
        centers = union_list_column(group["reaction_center_coarse_labels"].tolist())
        substrates = union_list_column(group["substrate_class_labels"].tolist())
        products = union_list_column(group["product_class_labels"].tolist())
        transitions = union_group_column(group, "substrate_product_transition_labels")
        types = union_list_column(group["reaction_type_labels"].tolist())
        flags = {"train_only_labels"}
        missing_reaction_features = group["canonical_reaction_smiles"].isna()
        if missing_reaction_features.all():
            flags.add("enzyme_has_no_reaction_features")
        elif missing_reaction_features.any():
            flags.add("enzyme_has_some_missing_reaction_features")
        if not cofactors:
            flags.add("enzyme_has_no_cofactor_labels")
        if not core_cofactors:
            flags.add("enzyme_has_no_core_cofactor_labels")
        if not centers:
            flags.add("enzyme_has_no_reaction_center_labels")
        if not ec_numbers:
            flags.add("enzyme_has_no_ec_labels")
        if group["reaction_id"].nunique() > 1 or len(types) > 1:
            flags.add("enzyme_multifunctional")
        enzyme_rows.append(
            {
                "enzyme_id": enzyme_id,
                "train_reaction_ids": sorted(group["reaction_id"].dropna().astype(str).unique()),
                "ec_numbers_train": ec_numbers,
                "cofactor_labels_train": cofactors,
                "core_cofactor_labels_train": core_cofactors,
                "metal_ion_labels_train": metal_ions,
                "auxiliary_participant_labels_train": auxiliary_participants,
                "reaction_center_labels_train": centers,
                "substrate_class_labels_train": substrates,
                "product_class_labels_train": products,
                "substrate_product_transition_labels_train": transitions,
                "reaction_type_labels_train": types,
                "num_train_reactions": int(group["reaction_id"].nunique()),
                "num_distinct_reaction_types": len(types),
                "num_distinct_cofactors": len(cofactors),
                "num_distinct_core_cofactors": len(core_cofactors),
                "label_source": "train_only",
                "quality_flags": sorted(flags),
            }
        )
    enzyme_labels = pd.DataFrame(enzyme_rows)

    cluster_map: dict[str, str] = {}
    if sequence_cluster_map_path is not None:
        clusters = pd.read_csv(sequence_cluster_map_path)
        enzyme_col = "enzyme_id" if "enzyme_id" in clusters.columns else "protein_id"
        cluster_col = "sequence_cluster_id" if "sequence_cluster_id" in clusters.columns else "cluster_id"
        if enzyme_col in clusters.columns and cluster_col in clusters.columns:
            cluster_map = {
                str(row[enzyme_col]): str(row[cluster_col])
                for row in clusters.to_dict("records")
            }

    enzyme_lookup = {row["enzyme_id"]: row for row in enzyme_rows}
    pair_rows: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        enzyme_id = str(row["enzyme_id"])
        labels = enzyme_lookup[enzyme_id]
        reaction_ecs = union_list_column([row["ec_numbers"]])
        reaction_cofactors = union_list_column([row["cofactor_labels"]])
        reaction_core_cofactors = union_list_column([row.get("core_cofactor_labels")])
        reaction_metal_ions = union_list_column([row.get("metal_ion_labels")])
        reaction_auxiliary_participants = union_list_column([row.get("auxiliary_participant_labels")])
        reaction_types = union_list_column([row["reaction_type_labels"]])
        reaction_substrates = union_list_column([row["substrate_class_labels"]])
        reaction_transitions = union_list_column([row.get("substrate_product_transition_labels")])
        reaction_centers = (
            union_list_column([row["reaction_center_coarse_labels"]])
        )
        pair_rows.append(
            {
                "enzyme_id": enzyme_id,
                "reaction_id": str(row["reaction_id"]),
                "is_positive": 1,
                "reaction_demand_vector_id": str(row["reaction_id"]),
                "enzyme_capability_label_id": enzyme_id,
                "ec_overlap_level": ec_overlap_level(reaction_ecs, labels["ec_numbers_train"]),
                "cofactor_overlap": intersection_size(reaction_cofactors, labels["cofactor_labels_train"]),
                "core_cofactor_overlap": intersection_size(
                    reaction_core_cofactors,
                    labels["core_cofactor_labels_train"],
                ),
                "metal_ion_overlap": intersection_size(
                    reaction_metal_ions,
                    labels["metal_ion_labels_train"],
                ),
                "auxiliary_participant_overlap": intersection_size(
                    reaction_auxiliary_participants,
                    labels["auxiliary_participant_labels_train"],
                ),
                "reaction_type_overlap": intersection_size(reaction_types, labels["reaction_type_labels_train"]),
                "substrate_class_overlap": intersection_size(reaction_substrates, labels["substrate_class_labels_train"]),
                "substrate_product_transition_overlap": intersection_size(
                    reaction_transitions,
                    labels["substrate_product_transition_labels_train"],
                ),
                "center_label_overlap": intersection_size(reaction_centers, labels["reaction_center_labels_train"]),
                "sequence_cluster_id": cluster_map.get(enzyme_id),
                "source_split": "train",
            }
        )
    pair_training = pd.DataFrame(pair_rows)
    assert set(pair_training["source_split"].unique()) == {"train"}

    vocabs = {
        "ec_labels": sorted({x for row in enzyme_rows for x in row["ec_numbers_train"]}),
        "cofactor_labels": sorted({x for row in enzyme_rows for x in row["cofactor_labels_train"]}),
        "core_cofactor_labels": sorted(
            {x for row in enzyme_rows for x in row["core_cofactor_labels_train"]}
        ),
        "metal_ion_labels": sorted(
            {x for row in enzyme_rows for x in row["metal_ion_labels_train"]}
        ),
        "auxiliary_participant_labels": sorted(
            {x for row in enzyme_rows for x in row["auxiliary_participant_labels_train"]}
        ),
        "reaction_center_labels": sorted({x for row in enzyme_rows for x in row["reaction_center_labels_train"]}),
        "substrate_class_labels": sorted({x for row in enzyme_rows for x in row["substrate_class_labels_train"]}),
        "product_class_labels": sorted({x for row in enzyme_rows for x in row["product_class_labels_train"]}),
        "substrate_product_transition_labels": sorted(
            {x for row in enzyme_rows for x in row["substrate_product_transition_labels_train"]}
        ),
        "reaction_type_labels": sorted({x for row in enzyme_rows for x in row["reaction_type_labels_train"]}),
    }
    report = {
        "num_train_pairs": int(len(pairs)),
        "num_enzymes_total_train": int(len(enzyme_labels)),
        "num_enzymes_with_capability_labels": int(
            sum(
                bool(
                    row["cofactor_labels_train"]
                    or row["core_cofactor_labels_train"]
                    or row["ec_numbers_train"]
                    or row["reaction_center_labels_train"]
                    or row["substrate_class_labels_train"]
                    or row["product_class_labels_train"]
                    or row["reaction_type_labels_train"]
                )
                for row in enzyme_rows
            )
        ),
        "num_enzymes_with_ec": int(sum(bool(row["ec_numbers_train"]) for row in enzyme_rows)),
        "quality_flag_counts": dict(Counter(flag for row in enzyme_rows for flag in row["quality_flags"])),
    }

    enzyme_labels.to_parquet(out / "enzyme_capability_labels.parquet", index=False)
    pair_training.to_parquet(out / "pair_capability_training.parquet", index=False)
    write_json(out / "enzyme_label_vocabs.json", vocabs)
    write_json(out / "enzyme_capability_quality_report.json", report)
    return enzyme_labels, pair_training
