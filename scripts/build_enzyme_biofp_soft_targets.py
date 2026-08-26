#!/usr/bin/env python3
"""Build train-only soft enzyme BioFP targets for inline retrieval supervision."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


CENTER_LABELS = [
    "chirality_change",
    "phosphate_transfer_like",
    "p_o_change",
    "redox_like",
    "c_c_change",
    "c_n_change",
    "c_s_change",
    "bond_order_change",
]

COFACTOR_LABELS = [
    "NAD_NADP_redox",
    "flavin_redox",
    "PLP_like",
    "NTP_phosphoryl",
    "CoA_acyl",
    "SAM_methyl_or_radical",
    "sugar_nucleotide_glycosyl",
    "thiol_conjugation_or_redox",
    "heme_tetrapyrrole_redox",
    "FeS_electron_transfer",
]

EXPANDED_COFACTOR_LABELS = [
    "NAD_NADP_redox",
    "flavin_redox",
    "quinone_PQQ_redox",
    "PLP_like",
    "TPP_dependent",
    "NTP_phosphoryl",
    "CoA_acyl",
    "SAM_methyl_or_radical",
    "sugar_nucleotide_glycosyl",
    "thiol_conjugation_or_redox",
    "heme_tetrapyrrole_redox",
    "FeS_electron_transfer",
    "biotin_carboxylation",
    "molybdopterin_redox",
    "ascorbate_pterin_redox",
    "carotenoid_redox",
]

TRANSITION_LABELS = [
    "alcohol_to_carbonyl",
    "carbonyl_to_alcohol",
    "gain_phosphate",
    "loss_phosphate",
    "carboxylation",
    "decarboxylation",
    "methylation",
    "demethylation",
    "acyl_transfer",
    "glycosyl_transfer",
    "thioester_formation_or_cleavage",
    "amine_to_imine_or_carbonyl",
    "thiol_disulfide_or_conjugation",
    "stereochemical_inversion",
    "hydration_dehydration",
    "ring_closure_or_opening",
]

FAMILIES = {
    "center": CENTER_LABELS,
    "cofactor": COFACTOR_LABELS,
    "transition": TRANSITION_LABELS,
}

COFACTOR_LABEL_SETS = {
    "compact": COFACTOR_LABELS,
    "expanded": EXPANDED_COFACTOR_LABELS,
}

COMPACT_FINE_COFACTOR_TO_GROUP = {
    "NAD": "NAD_NADP_redox",
    "NADP": "NAD_NADP_redox",
    "FAD": "flavin_redox",
    "FMN": "flavin_redox",
    "flavin": "flavin_redox",
    "PQQ": "flavin_redox",
    "quinone": "flavin_redox",
    "PLP": "PLP_like",
    "pyridoxal": "PLP_like",
    "pyridoxine": "PLP_like",
    "ATP": "NTP_phosphoryl",
    "GTP": "NTP_phosphoryl",
    "CTP": "NTP_phosphoryl",
    "UTP": "NTP_phosphoryl",
    "CoA": "CoA_acyl",
    "SAM": "SAM_methyl_or_radical",
    "glutathione": "thiol_conjugation_or_redox",
    "mycothiol": "thiol_conjugation_or_redox",
    "bacillithiol": "thiol_conjugation_or_redox",
    "CoB": "thiol_conjugation_or_redox",
    "CoM": "thiol_conjugation_or_redox",
    "lipoate": "thiol_conjugation_or_redox",
    "heme": "heme_tetrapyrrole_redox",
    "F430": "heme_tetrapyrrole_redox",
    "siroheme": "heme_tetrapyrrole_redox",
    "cobalamin": "heme_tetrapyrrole_redox",
    "cobamamide": "heme_tetrapyrrole_redox",
    "FeS_cluster": "FeS_electron_transfer",
}

EXPANDED_FINE_COFACTOR_TO_GROUP = {
    **COMPACT_FINE_COFACTOR_TO_GROUP,
    "quinone": "quinone_PQQ_redox",
    "PQQ": "quinone_PQQ_redox",
    "TPP": "TPP_dependent",
    "biotin": "biotin_carboxylation",
    "molybdopterin": "molybdopterin_redox",
    "ascorbate": "ascorbate_pterin_redox",
    "pterin": "ascorbate_pterin_redox",
    "carotenoid": "carotenoid_redox",
}

COMPACT_COFACTOR_ARCHITECTURE_TO_GROUP = {
    "rossmann_NAD_NADP": "NAD_NADP_redox",
    "flavin_FAD_FMN": "flavin_redox",
    "PLP_dependent": "PLP_like",
    "P_loop_NTP_kinase": "NTP_phosphoryl",
    "CoA_acyl_carrier": "CoA_acyl",
    "SAM_methyl_or_radical": "SAM_methyl_or_radical",
    "thiol_or_protein_redox": "thiol_conjugation_or_redox",
    "heme_tetrapyrrole": "heme_tetrapyrrole_redox",
    "FeS_metal_cluster": "FeS_electron_transfer",
}

EXPANDED_COFACTOR_ARCHITECTURE_TO_GROUP = {
    **COMPACT_COFACTOR_ARCHITECTURE_TO_GROUP,
    "quinone_PQQ_redox": "quinone_PQQ_redox",
    "TPP_dependent": "TPP_dependent",
    "biotin_carboxylation": "biotin_carboxylation",
    "molybdopterin_redox": "molybdopterin_redox",
    "ascorbate_pterin_redox": "ascorbate_pterin_redox",
    "carotenoid_redox": "carotenoid_redox",
}

COFACTOR_FINE_MAPS = {
    "compact": COMPACT_FINE_COFACTOR_TO_GROUP,
    "expanded": EXPANDED_FINE_COFACTOR_TO_GROUP,
}
COFACTOR_ARCHITECTURE_MAPS = {
    "compact": COMPACT_COFACTOR_ARCHITECTURE_TO_GROUP,
    "expanded": EXPANDED_COFACTOR_ARCHITECTURE_TO_GROUP,
}

SUGAR_NUCLEOTIDE_LABELS = {"UDP", "GDP", "CDP", "TDP"}
GLYCOSYL_TRANSFER_TRANSITIONS = {
    "gain_glycoside",
    "loss_glycoside",
    "gain_sugar_like",
    "loss_sugar_like",
}
GLYCOSYL_TRANSFER_COFACTOR_RULES = {"none", "type", "type_or_transition"}


def _families_for(cofactor_label_set: str) -> dict[str, list[str]]:
    if cofactor_label_set not in COFACTOR_LABEL_SETS:
        raise ValueError(
            "cofactor_label_set must be one of "
            f"{sorted(COFACTOR_LABEL_SETS)}"
        )
    return {
        "center": CENTER_LABELS,
        "cofactor": COFACTOR_LABEL_SETS[cofactor_label_set],
        "transition": TRANSITION_LABELS,
    }


def _as_label_set(value: Any) -> set[str]:
    if value is None:
        return set()
    try:
        if bool(pd.isna(value)):
            return set()
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() in {"nan", "none", "null", "[]"}:
            return set()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        return {
            item.strip().strip("'\"")
            for chunk in raw.split(";")
            for item in chunk.split(",")
            if item.strip().strip("'\"")
        }
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {str(item) for item in value if str(item)}
    return set()


def _strip_direction_suffix(reaction_id: str) -> str:
    reaction_id = str(reaction_id)
    if reaction_id.endswith(("_f", "_r")):
        return reaction_id[:-2]
    return reaction_id


def _glycosyl_transfer_cofactor_signal(
    reaction_types: set[str],
    transitions: set[str],
    rule: str,
) -> bool:
    if rule not in GLYCOSYL_TRANSFER_COFACTOR_RULES:
        raise ValueError(
            "glycosyl_transfer_cofactor_rule must be one of "
            f"{sorted(GLYCOSYL_TRANSFER_COFACTOR_RULES)}"
        )
    if rule == "none":
        return False
    if "glycosyl_transfer" in reaction_types:
        return True
    return rule == "type_or_transition" and bool(transitions & GLYCOSYL_TRANSFER_TRANSITIONS)


def _compact_cofactor_groups(
    core_labels: set[str],
    *,
    all_labels: set[str] | None = None,
    architecture_bins: set[str] | None = None,
    reaction_types: set[str] | None = None,
    transitions: set[str] | None = None,
    glycosyl_transfer_cofactor_rule: str = "none",
    cofactor_label_set: str = "compact",
) -> set[str]:
    fine_map = COFACTOR_FINE_MAPS[cofactor_label_set]
    architecture_map = COFACTOR_ARCHITECTURE_MAPS[cofactor_label_set]
    all_labels = set() if all_labels is None else all_labels
    architecture_bins = set() if architecture_bins is None else architecture_bins
    reaction_types = set() if reaction_types is None else reaction_types
    transitions = set() if transitions is None else transitions
    out: set[str] = set()

    for label in core_labels:
        group = fine_map.get(label)
        if group is not None:
            out.add(group)
    for architecture_bin in architecture_bins:
        group = architecture_map.get(architecture_bin)
        if group is not None:
            out.add(group)

    sugar_sources = core_labels | all_labels
    if any("sugar" in label.lower() or label in SUGAR_NUCLEOTIDE_LABELS for label in sugar_sources):
        out.add("sugar_nucleotide_glycosyl")
    if _glycosyl_transfer_cofactor_signal(
        reaction_types,
        transitions,
        glycosyl_transfer_cofactor_rule,
    ):
        out.add("sugar_nucleotide_glycosyl")
    return out


def _cofactor_groups(
    row: dict[str, Any],
    *,
    glycosyl_transfer_cofactor_rule: str = "none",
    cofactor_label_set: str = "compact",
) -> set[str]:
    core = _as_label_set(row.get("core_cofactor_labels"))
    all_cofactors = _as_label_set(row.get("cofactor_labels"))
    reaction_types = _as_label_set(row.get("reaction_type_labels"))
    transitions = _as_label_set(row.get("substrate_product_transition_labels"))
    return _compact_cofactor_groups(
        core,
        all_labels=all_cofactors,
        reaction_types=reaction_types,
        transitions=transitions,
        glycosyl_transfer_cofactor_rule=glycosyl_transfer_cofactor_rule,
        cofactor_label_set=cofactor_label_set,
    )


def _transition_groups(row: dict[str, Any]) -> set[str]:
    transitions = _as_label_set(row.get("substrate_product_transition_labels"))
    reaction_types = _as_label_set(row.get("reaction_type_labels"))
    centers = _as_label_set(row.get("reaction_center_coarse_labels"))
    out: set[str] = set()
    if transitions & {
        "alcohol_to_aldehyde",
        "alcohol_to_ketone",
        "hydroxy_acid_to_keto_acid",
    } or reaction_types & {"oxidation", "oxidoreduction"}:
        out.add("alcohol_to_carbonyl")
    if transitions & {
        "aldehyde_to_alcohol",
        "ketone_to_alcohol",
        "keto_acid_to_hydroxy_acid",
    } or reaction_types & {"reduction"}:
        out.add("carbonyl_to_alcohol")
    if transitions & {
        "phosphorylation_like",
        "alcohol_to_phosphate_ester",
        "gain_phosphate_containing",
        "gain_phosphate_ester",
    } or reaction_types & {"phosphorylation"}:
        out.add("gain_phosphate")
    if transitions & {
        "dephosphorylation_like",
        "phosphate_ester_to_alcohol",
        "loss_phosphate_containing",
        "loss_phosphate_ester",
    } or reaction_types & {"dephosphorylation"}:
        out.add("loss_phosphate")
    if reaction_types & {"carboxylation"}:
        out.add("carboxylation")
    if transitions & {"loss_carboxylate"}:
        out.add("decarboxylation")
    if reaction_types & {"methyl_transfer"}:
        out.add("methylation")
    if transitions & {"loss_thioether"}:
        out.add("demethylation")
    if reaction_types & {"acyl_transfer"} or transitions & {
        "carboxylate_to_coa_thioester",
        "coa_thioester_to_carboxylate",
        "carboxylate_to_ester",
    }:
        out.add("acyl_transfer")
    if reaction_types & {"glycosyl_transfer"} or transitions & {
        "gain_glycoside",
        "loss_glycoside",
        "gain_sugar_like",
        "loss_sugar_like",
    }:
        out.add("glycosyl_transfer")
    if transitions & {"carboxylate_to_coa_thioester", "coa_thioester_to_carboxylate"}:
        out.add("thioester_formation_or_cleavage")
    if transitions & {
        "amine_to_imine",
        "ketone_to_imine",
        "amino_acid_to_keto_acid",
        "keto_acid_to_amino_acid",
    } or reaction_types & {"transamination"}:
        out.add("amine_to_imine_or_carbonyl")
    if transitions & {"thiol_to_disulfide", "disulfide_to_thiol"}:
        out.add("thiol_disulfide_or_conjugation")
    if reaction_types & {"stereochemical_inversion"} or "chirality_change" in centers:
        out.add("stereochemical_inversion")
    if reaction_types & {"hydrolysis"} and transitions & {"gain_alcohol", "loss_alcohol"}:
        out.add("hydration_dehydration")
    if transitions & {"gain_lactone", "loss_lactone", "gain_lactam", "loss_lactam"}:
        out.add("ring_closure_or_opening")
    return out


def _reaction_family_labels(
    row: dict[str, Any],
    *,
    glycosyl_transfer_cofactor_rule: str = "none",
    cofactor_label_set: str = "compact",
) -> dict[str, tuple[set[str], bool]]:
    center_source = _as_label_set(row.get("reaction_center_coarse_labels"))
    center = center_source & set(CENTER_LABELS)
    cofactor_source = _as_label_set(row.get("core_cofactor_labels"))
    reaction_types = _as_label_set(row.get("reaction_type_labels"))
    transitions = _as_label_set(row.get("substrate_product_transition_labels"))
    cofactor = _cofactor_groups(
        row,
        glycosyl_transfer_cofactor_rule=glycosyl_transfer_cofactor_rule,
        cofactor_label_set=cofactor_label_set,
    )
    glycosyl_rescue_known = _glycosyl_transfer_cofactor_signal(
        reaction_types,
        transitions,
        glycosyl_transfer_cofactor_rule,
    )
    transition_source = _as_label_set(row.get("substrate_product_transition_labels")) | _as_label_set(
        row.get("reaction_type_labels")
    )
    transition = _transition_groups(row)
    return {
        "center": (center, bool(center_source)),
        "cofactor": (cofactor, bool(cofactor_source) or glycosyl_rescue_known),
        "transition": (transition, bool(transition_source)),
    }


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if {"reaction_id", "protein_id"}.issubset(fieldnames):
            reaction_col, protein_col = "reaction_id", "protein_id"
        elif {"query_id", "target_id"}.issubset(fieldnames):
            reaction_col, protein_col = "query_id", "target_id"
        else:
            raise ValueError(
                f"{path} must contain reaction_id/protein_id or query_id/target_id columns"
            )
        pairs = {
            (str(row[protein_col]), _strip_direction_suffix(str(row[reaction_col])))
            for row in reader
            if str(row.get(protein_col, "")).strip()
            and str(row.get(reaction_col, "")).strip()
        }
    return sorted(pairs)


def _decode_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_allowed_reactions(path: Path) -> set[str]:
    """Read base reaction IDs from CSV, text, or HDF5 ids datasets."""
    if path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            if "ids" not in handle:
                raise KeyError(f"{path} must contain an 'ids' dataset")
            return {
                _strip_direction_suffix(_decode_id(value))
                for value in handle["ids"][:]
                if _decode_id(value).strip()
            }

    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if "," not in sample and "\t" not in sample:
            return {
                _strip_direction_suffix(line.strip())
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            }

        dialect = csv.Sniffer().sniff(sample)
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header")
        preferred_columns = ("reaction_id", "query_id", "id")
        id_column = next(
            (column for column in preferred_columns if column in reader.fieldnames),
            reader.fieldnames[0],
        )
        return {
            _strip_direction_suffix(str(row[id_column]))
            for row in reader
            if str(row.get(id_column, "")).strip()
        }


def _intersect_allowed_reactions(paths: list[Path] | None) -> set[str] | None:
    if not paths:
        return None
    allowed: set[str] | None = None
    for path in paths:
        reaction_ids = _read_allowed_reactions(path)
        allowed = reaction_ids if allowed is None else allowed & reaction_ids
    return allowed


def _read_enzyme_cofactor_labels(
    path: Path,
    *,
    enzyme_id_column: str = "enzyme_id",
    core_column: str = "combined_core_cofactor_labels_train",
    architecture_column: str | None = "cofactor_architecture_bins_train",
    cofactor_label_set: str = "compact",
) -> dict[str, tuple[set[str], bool]]:
    """Read enzyme-level cofactor labels and map them to compact BioFP groups."""

    columns = {enzyme_id_column, core_column}
    if architecture_column is not None:
        columns.add(architecture_column)
    df = pd.read_csv(path, usecols=lambda column: column in columns)
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required cofactor columns: {missing}")

    out: dict[str, tuple[set[str], bool]] = {}
    for row in df.to_dict("records"):
        enzyme_id = str(row[enzyme_id_column]).strip()
        if not enzyme_id or enzyme_id.lower() in {"nan", "none", "null"}:
            continue
        core_labels = _as_label_set(row.get(core_column))
        architecture_bins = (
            set()
            if architecture_column is None
            else _as_label_set(row.get(architecture_column))
        )
        groups = _compact_cofactor_groups(
            core_labels,
            architecture_bins=architecture_bins,
            cofactor_label_set=cofactor_label_set,
        )
        known = bool(core_labels) or bool(groups)
        previous_groups, previous_known = out.get(enzyme_id, (set(), False))
        out[enzyme_id] = (previous_groups | groups, previous_known or known)
    return out


def _add_enzyme_cofactor_observations(
    counts: dict[str, dict[str, np.ndarray]],
    denominators: dict[str, dict[str, float]],
    enzyme_cofactor_labels: dict[str, tuple[set[str], bool]],
    *,
    eligible_enzymes: set[str],
    index_by_label: dict[str, int],
    weight: float,
) -> dict[str, int]:
    if weight <= 0:
        raise ValueError("enzyme cofactor label weight must be positive")

    stats = {
        "known": 0,
        "positive": 0,
        "zero_positive_known": 0,
        "skipped_not_in_train_pairs": 0,
    }
    for enzyme_id, (groups, known) in enzyme_cofactor_labels.items():
        if enzyme_id not in eligible_enzymes:
            if known:
                stats["skipped_not_in_train_pairs"] += 1
            continue
        if not known:
            continue
        stats["known"] += 1
        if groups:
            stats["positive"] += 1
        else:
            stats["zero_positive_known"] += 1
        counts[enzyme_id]
        denominators[enzyme_id]["cofactor"] += weight
        for group in groups:
            idx = index_by_label.get(group)
            if idx is not None:
                counts[enzyme_id]["cofactor"][idx] += weight
    return stats


def build_soft_targets(
    train_pairs: Path,
    reaction_features: Path,
    allowed_reactions: set[str] | None = None,
    enzyme_cofactor_labels: Path | None = None,
    enzyme_cofactor_core_column: str = "combined_core_cofactor_labels_train",
    enzyme_cofactor_architecture_column: str | None = "cofactor_architecture_bins_train",
    enzyme_cofactor_weight: float = 1.0,
    glycosyl_transfer_cofactor_rule: str = "none",
    cofactor_label_set: str = "compact",
) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    families = _families_for(cofactor_label_set)
    if glycosyl_transfer_cofactor_rule not in GLYCOSYL_TRANSFER_COFACTOR_RULES:
        raise ValueError(
            "glycosyl_transfer_cofactor_rule must be one of "
            f"{sorted(GLYCOSYL_TRANSFER_COFACTOR_RULES)}"
        )
    features = pd.read_parquet(reaction_features)
    if "reaction_id" not in features.columns:
        raise ValueError("reaction_features must contain reaction_id")
    reaction_rows = {
        str(row["reaction_id"]): row
        for row in features.to_dict("records")
    }
    reaction_labels = {
        reaction_id: _reaction_family_labels(
            row,
            glycosyl_transfer_cofactor_rule=glycosyl_transfer_cofactor_rule,
            cofactor_label_set=cofactor_label_set,
        )
        for reaction_id, row in reaction_rows.items()
    }

    raw_pairs = _read_pairs(train_pairs)
    if allowed_reactions is None:
        pairs = raw_pairs
    else:
        pairs = [
            (enzyme_id, reaction_id)
            for enzyme_id, reaction_id in raw_pairs
            if reaction_id in allowed_reactions
        ]
    counts: dict[str, dict[str, np.ndarray]] = defaultdict(
        lambda: {
            family: np.zeros(len(labels), dtype=np.float32)
            for family, labels in families.items()
        }
    )
    denominators: dict[str, dict[str, float]] = defaultdict(
        lambda: {family: 0.0 for family in families}
    )
    index_by_family = {
        family: {label: idx for idx, label in enumerate(labels)}
        for family, labels in families.items()
    }
    skipped_pairs = 0
    for enzyme_id, reaction_id in pairs:
        labels_for_reaction = reaction_labels.get(reaction_id)
        if labels_for_reaction is None:
            skipped_pairs += 1
            continue
        for family, (labels, known) in labels_for_reaction.items():
            if not known:
                continue
            counts[enzyme_id]
            denominators[enzyme_id][family] += 1.0
            for label in labels:
                idx = index_by_family[family].get(label)
                if idx is not None:
                    counts[enzyme_id][family][idx] += 1.0

    enzyme_cofactor_stats: dict[str, int] | None = None
    if enzyme_cofactor_labels is not None:
        enzyme_cofactor_map = _read_enzyme_cofactor_labels(
            enzyme_cofactor_labels,
            core_column=enzyme_cofactor_core_column,
            architecture_column=enzyme_cofactor_architecture_column,
            cofactor_label_set=cofactor_label_set,
        )
        enzyme_cofactor_stats = _add_enzyme_cofactor_observations(
            counts,
            denominators,
            enzyme_cofactor_map,
            eligible_enzymes={enzyme_id for enzyme_id, _reaction_id in pairs},
            index_by_label=index_by_family["cofactor"],
            weight=enzyme_cofactor_weight,
        )

    ids = sorted(
        enzyme_id
        for enzyme_id, family_denominators in denominators.items()
        if any(value > 0 for value in family_denominators.values())
    )
    arrays: dict[str, np.ndarray] = {}
    support: dict[str, dict[str, int]] = {}
    zero_positive_known: dict[str, int] = {}
    for family, labels in families.items():
        targets = np.zeros((len(ids), len(labels)), dtype=np.float32)
        mask = np.zeros(len(ids), dtype=bool)
        denom = np.zeros(len(ids), dtype=np.float32)
        for row_idx, enzyme_id in enumerate(ids):
            family_denom = denominators[enzyme_id][family]
            denom[row_idx] = family_denom
            if family_denom > 0:
                mask[row_idx] = True
                targets[row_idx] = counts[enzyme_id][family] / family_denom
        arrays[f"{family}_targets"] = targets
        arrays[f"{family}_mask"] = mask
        arrays[f"{family}_denominator"] = denom
        support[family] = {
            label: int((targets[:, idx] > 0).sum())
            for idx, label in enumerate(labels)
        }
        zero_positive_known[family] = int((mask & (targets.sum(axis=1) == 0.0)).sum())

    metadata = {
        "families": families,
        "support": support,
        "zero_positive_known_targets": zero_positive_known,
        "num_train_pairs_unique_raw": len(raw_pairs),
        "num_train_pairs_unique": len(pairs),
        "num_train_pairs_filtered_by_allowed_reactions": len(raw_pairs) - len(pairs),
        "num_enzymes": len(ids),
        "skipped_pairs_missing_reaction_features": skipped_pairs,
        "num_allowed_reactions": None if allowed_reactions is None else len(allowed_reactions),
        "source_train_pairs": str(train_pairs),
        "source_reaction_features": str(reaction_features),
        "glycosyl_transfer_cofactor_rule": glycosyl_transfer_cofactor_rule,
        "cofactor_label_set": cofactor_label_set,
    }
    if enzyme_cofactor_labels is not None:
        metadata.update(
            {
                "source_enzyme_cofactor_labels": str(enzyme_cofactor_labels),
                "enzyme_cofactor_core_column": enzyme_cofactor_core_column,
                "enzyme_cofactor_architecture_column": enzyme_cofactor_architecture_column,
                "enzyme_cofactor_weight": enzyme_cofactor_weight,
                "enzyme_cofactor_label_stats": enzyme_cofactor_stats,
            }
        )
    return ids, arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", required=True, type=Path)
    parser.add_argument("--reaction-features", required=True, type=Path)
    parser.add_argument(
        "--allowed-reactions",
        action="append",
        type=Path,
        default=None,
        help=(
            "Optional CSV, text, or HDF5 file with reaction IDs to keep. "
            "May be supplied multiple times; IDs are intersected after stripping _f/_r."
        ),
    )
    parser.add_argument(
        "--enzyme-cofactor-labels",
        type=Path,
        default=None,
        help=(
            "Optional enhanced enzyme-level cofactor CSV. Compact cofactor labels "
            "are added as enzyme-level pseudo-observations."
        ),
    )
    parser.add_argument(
        "--enzyme-cofactor-core-column",
        default="combined_core_cofactor_labels_train",
        help="Column in --enzyme-cofactor-labels with fine core cofactor labels.",
    )
    parser.add_argument(
        "--enzyme-cofactor-architecture-column",
        default="cofactor_architecture_bins_train",
        help=(
            "Column in --enzyme-cofactor-labels with broad architecture bins. "
            "Use an empty string to ignore architecture bins."
        ),
    )
    parser.add_argument(
        "--enzyme-cofactor-weight",
        type=float,
        default=1.0,
        help="Pseudo-observation weight for enzyme-level cofactor labels.",
    )
    parser.add_argument(
        "--glycosyl-transfer-cofactor-rule",
        choices=sorted(GLYCOSYL_TRANSFER_COFACTOR_RULES),
        default="none",
        help=(
            "Optional rescue rule for sugar_nucleotide_glycosyl. 'type' uses "
            "glycosyl_transfer reaction-type evidence; 'type_or_transition' also "
            "uses glycoside/sugar transition evidence."
        ),
    )
    parser.add_argument(
        "--cofactor-label-set",
        choices=sorted(COFACTOR_LABEL_SETS),
        default="compact",
        help=(
            "Cofactor target vocabulary. 'compact' preserves the original "
            "10-label BioFP head; 'expanded' promotes clean core cofactor "
            "families such as TPP, biotin, quinone/PQQ, and molybdopterin."
        ),
    )
    parser.add_argument("--out-npz", required=True, type=Path)
    parser.add_argument("--out-vocab", required=True, type=Path)
    args = parser.parse_args()

    allowed_reactions = _intersect_allowed_reactions(args.allowed_reactions)
    enzyme_cofactor_architecture_column = (
        None
        if args.enzyme_cofactor_architecture_column == ""
        else args.enzyme_cofactor_architecture_column
    )
    ids, arrays, metadata = build_soft_targets(
        args.train_pairs,
        args.reaction_features,
        allowed_reactions=allowed_reactions,
        enzyme_cofactor_labels=args.enzyme_cofactor_labels,
        enzyme_cofactor_core_column=args.enzyme_cofactor_core_column,
        enzyme_cofactor_architecture_column=enzyme_cofactor_architecture_column,
        enzyme_cofactor_weight=args.enzyme_cofactor_weight,
        glycosyl_transfer_cofactor_rule=args.glycosyl_transfer_cofactor_rule,
        cofactor_label_set=args.cofactor_label_set,
    )
    if args.allowed_reactions:
        metadata["source_allowed_reactions"] = [str(path) for path in args.allowed_reactions]
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    args.out_vocab.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        ids=np.asarray(ids, dtype=object),
        **arrays,
    )
    args.out_vocab.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        "Wrote BioFP soft targets: "
        f"{args.out_npz} ({len(ids)} enzymes), vocab={args.out_vocab}"
    )


if __name__ == "__main__":
    main()
