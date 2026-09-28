"""Leakage-safe biological supervision targets for ReactZyme enzyme towers."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "biofp_minimal_v1"

MECHANISM_LABELS = (
    "redox_carbonyl_interconversion",
    "phosphate_transfer",
    "acyl_transfer",
    "glycosyl_transfer",
    "c_n_transfer_transamination",
    "sulfur_thiol_chemistry",
    "stereochemical_rearrangement",
    "hydrolysis_condensation",
)

COFACTOR_LABELS = (
    "NAD_NADP",
    "FAD_FMN",
    "PLP",
    "TPP",
    "CoA",
    "SAM",
    "FeS",
    "heme",
    "quinone",
    "thiol_lipoate",
)


def _labels(value: Any) -> set[str]:
    if value is None:
        return set()
    try:
        if bool(pd.isna(value)):
            return set()
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return set()
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return {str(item).strip() for item in parsed if str(item).strip()}
            except (SyntaxError, ValueError):
                pass
        return {
            token.strip().strip("'\"")
            for chunk in text.split(";")
            for token in chunk.split(",")
            if token.strip().strip("'\"")
        }
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _normalized_tokens(values: set[str]) -> set[str]:
    return {
        value.lower().replace("-", "_").replace(" ", "_")
        for value in values
    }


def mechanism_groups(row: dict[str, Any]) -> set[str]:
    centers = _normalized_tokens(_labels(row.get("reaction_center_coarse_labels")))
    transitions = _normalized_tokens(
        _labels(row.get("substrate_product_transition_labels"))
    )
    reaction_types = _normalized_tokens(_labels(row.get("reaction_type_labels")))
    all_tokens = centers | transitions | reaction_types
    out: set[str] = set()

    if all_tokens & {
        "redox_like", "oxidation", "reduction", "oxidoreduction",
        "alcohol_to_aldehyde", "alcohol_to_ketone", "aldehyde_to_alcohol",
        "ketone_to_alcohol", "hydroxy_acid_to_keto_acid",
        "keto_acid_to_hydroxy_acid",
    }:
        out.add("redox_carbonyl_interconversion")
    if all_tokens & {
        "phosphate_transfer_like", "p_o_change", "phosphorylation",
        "dephosphorylation", "phosphorylation_like", "dephosphorylation_like",
        "alcohol_to_phosphate_ester", "phosphate_ester_to_alcohol",
        "gain_phosphate_containing", "loss_phosphate_containing",
    }:
        out.add("phosphate_transfer")
    if all_tokens & {
        "acyl_transfer", "carboxylate_to_coa_thioester",
        "coa_thioester_to_carboxylate", "carboxylate_to_ester",
        "thioester_formation_or_cleavage",
    }:
        out.add("acyl_transfer")
    if all_tokens & {
        "glycosyl_transfer", "gain_glycoside", "loss_glycoside",
        "gain_sugar_like", "loss_sugar_like",
    }:
        out.add("glycosyl_transfer")
    if all_tokens & {
        "transamination", "amine_to_imine", "ketone_to_imine",
        "amino_acid_to_keto_acid", "keto_acid_to_amino_acid",
    }:
        out.add("c_n_transfer_transamination")
    if all_tokens & {
        "c_s_change", "s_s_change", "thiol_to_disulfide", "disulfide_to_thiol",
        "thiol_disulfide_or_conjugation", "sulfur_transfer",
    }:
        out.add("sulfur_thiol_chemistry")
    if all_tokens & {
        "chirality_change", "stereochemical_inversion", "isomerization",
        "racemization", "epimerization",
    }:
        out.add("stereochemical_rearrangement")
    if all_tokens & {
        "hydrolysis", "condensation", "hydration", "dehydration",
        "hydration_dehydration", "gain_lactone", "loss_lactone",
        "gain_lactam", "loss_lactam", "ring_closure_or_opening",
    }:
        out.add("hydrolysis_condensation")
    return out


def cofactor_groups(values: set[str]) -> set[str]:
    tokens = _normalized_tokens(values)
    joined = " ".join(sorted(tokens))
    out: set[str] = set()
    if any(token in joined for token in ("nad", "nadp", "nicotinamide")):
        out.add("NAD_NADP")
    if any(token in joined for token in ("fad", "fmn", "flavin")):
        out.add("FAD_FMN")
    if any(token in joined for token in ("plp", "pyridoxal", "pyridoxamine")):
        out.add("PLP")
    if any(token in joined for token in ("tpp", "thiamine_pyrophosphate", "thiamine_diphosphate")):
        out.add("TPP")
    if "coa" in joined or "coenzyme_a" in joined:
        out.add("CoA")
    if "sam" in joined or "s_adenosyl" in joined:
        out.add("SAM")
    if any(token in joined for token in ("fes", "fe_s", "iron_sulfur")):
        out.add("FeS")
    if any(token in joined for token in ("heme", "haem", "siroheme")):
        out.add("heme")
    if any(token in joined for token in ("quinone", "pqq", "ubiquinone", "menaquinone")):
        out.add("quinone")
    if any(token in joined for token in (
        "lipoate", "lipoamide", "glutathione", "mycothiol", "bacillithiol",
        "thioredoxin", "coenzyme_m", "coenzyme_b",
    )):
        out.add("thiol_lipoate")
    return out


def canonical_protein_key(protein_id: str) -> str:
    value = str(protein_id).strip()
    for prefix in ("protein_", "enzyme_", "uprot_", "prot_"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_held_out_source(path: Path, role: str) -> None:
    lowered = {part.lower() for part in path.parts}
    stem = path.stem.lower()
    held_out_name = stem in {"validation", "val", "test"} or stem.startswith(
        ("validation_", "val_", "test_")
    )
    if "validation" in lowered or "val" in lowered or "test" in lowered or held_out_name:
        raise ValueError(f"{role} must be train-only, got held-out path: {path}")


def _load_curated_cofactors(path: Path) -> dict[str, tuple[set[str], float]]:
    frame = pd.read_csv(path)
    if "enzyme_id" not in frame.columns:
        raise ValueError(f"{path} must contain enzyme_id")
    preferred_columns = (
        "enzyme_derived_core_cofactor_labels_train",
        "uniprot_core_cofactor_labels_train",
        "enzyme_uniprotkb_cofactor_labels_train",
    )
    available = [column for column in preferred_columns if column in frame.columns]
    if not available:
        raise ValueError(f"{path} has no curated UniProt cofactor columns")
    output: dict[str, tuple[set[str], float]] = {}
    for row in frame.to_dict("records"):
        enzyme_id = canonical_protein_key(row["enzyme_id"])
        values: set[str] = set()
        for column in available:
            values.update(_labels(row.get(column)))
        groups = cofactor_groups(values)
        if values or groups:
            previous, _confidence = output.get(enzyme_id, (set(), 0.0))
            output[enzyme_id] = (previous | groups, 1.0)
    return output


def _add_observation(
    positive: np.ndarray,
    denominator: np.ndarray,
    labels: set[str],
    vocabulary: tuple[str, ...],
    positive_weight: float,
    negative_weight: float,
) -> None:
    for index, label in enumerate(vocabulary):
        weight = positive_weight if label in labels else negative_weight
        denominator[index] += weight
        if label in labels:
            positive[index] += weight


def build_minimal_targets(
    *,
    train_pairs_path: Path,
    matched_members_path: Path,
    directional_features_path: Path,
    enzyme_cofactor_labels_path: Path | None = None,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    """Build mechanism/cofactor targets using training associations only."""

    for path, role in (
        (train_pairs_path, "train pairs"),
        (matched_members_path, "Rhea matches"),
        (directional_features_path, "directional features"),
    ):
        _reject_held_out_source(path, role)
    if enzyme_cofactor_labels_path is not None:
        _reject_held_out_source(enzyme_cofactor_labels_path, "enzyme cofactor labels")

    train_pairs = pd.read_csv(train_pairs_path)
    required_pairs = {"protein_id", "reaction_id"}
    if not required_pairs.issubset(train_pairs.columns):
        raise ValueError(f"{train_pairs_path} must contain {sorted(required_pairs)}")
    train_ids = sorted(set(train_pairs["protein_id"].astype(str)))
    train_id_by_key = {canonical_protein_key(value): value for value in train_ids}
    train_pair_keys = {
        (canonical_protein_key(row["protein_id"]), str(row["reaction_id"]))
        for row in train_pairs.to_dict("records")
    }

    members = pd.read_csv(matched_members_path)
    required_members = {"source_protein_id", "source_reaction_id", "reaction_id"}
    if not required_members.issubset(members.columns):
        raise ValueError(f"{matched_members_path} must contain {sorted(required_members)}")
    features = pd.read_parquet(directional_features_path)
    if "reaction_id" not in features.columns:
        raise ValueError(f"{directional_features_path} must contain reaction_id")
    feature_rows = {str(row["reaction_id"]): row for row in features.to_dict("records")}

    mechanism_positive: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(MECHANISM_LABELS), dtype=np.float32)
    )
    mechanism_denominator: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(MECHANISM_LABELS), dtype=np.float32)
    )
    cofactor_positive: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(COFACTOR_LABELS), dtype=np.float32)
    )
    cofactor_denominator: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(COFACTOR_LABELS), dtype=np.float32)
    )
    skipped_members = 0
    for member in members.to_dict("records"):
        protein_key = canonical_protein_key(member["source_protein_id"])
        if (protein_key, str(member["source_reaction_id"])) not in train_pair_keys:
            skipped_members += 1
            continue
        protein_id = train_id_by_key.get(protein_key)
        if protein_id is None:
            skipped_members += 1
            continue
        row = feature_rows.get(str(member["reaction_id"]))
        if row is None:
            skipped_members += 1
            continue
        labels = mechanism_groups(row)
        source_known = bool(
            _labels(row.get("reaction_center_coarse_labels"))
            | _labels(row.get("substrate_product_transition_labels"))
            | _labels(row.get("reaction_type_labels"))
        )
        match_confidence = float(member.get("match_confidence", 1.0) or 1.0)
        mapping_confidence = row.get(
            "mapping_confidence",
            row.get("reaction_center_mapping_confidence", 1.0),
        )
        try:
            mapping_confidence = float(mapping_confidence)
            if not np.isfinite(mapping_confidence):
                mapping_confidence = 1.0
        except (TypeError, ValueError):
            mapping_confidence = 1.0
        evidence = max(0.05, min(1.0, match_confidence * mapping_confidence))
        if source_known:
            _add_observation(
                mechanism_positive[protein_id],
                mechanism_denominator[protein_id],
                labels,
                MECHANISM_LABELS,
                evidence,
                evidence,
            )

        reaction_cofactors = cofactor_groups(
            _labels(row.get("core_cofactor_labels"))
            | _labels(row.get("cofactor_labels"))
        )
        if reaction_cofactors:
            _add_observation(
                cofactor_positive[protein_id],
                cofactor_denominator[protein_id],
                reaction_cofactors,
                COFACTOR_LABELS,
                0.4 * match_confidence,
                0.15 * match_confidence,
            )

    curated_count = 0
    if enzyme_cofactor_labels_path is not None:
        curated = _load_curated_cofactors(enzyme_cofactor_labels_path)
        for key, (groups, confidence) in curated.items():
            protein_id = train_id_by_key.get(key)
            if protein_id is None:
                continue
            curated_count += 1
            _add_observation(
                cofactor_positive[protein_id],
                cofactor_denominator[protein_id],
                groups,
                COFACTOR_LABELS,
                confidence,
                0.15,
            )

    arrays: dict[str, np.ndarray] = {}
    support: dict[str, dict[str, int]] = {}
    for family, vocabulary, positives_by_id, denominators_by_id in (
        ("mechanism", MECHANISM_LABELS, mechanism_positive, mechanism_denominator),
        ("cofactor", COFACTOR_LABELS, cofactor_positive, cofactor_denominator),
    ):
        positives = np.stack(
            [positives_by_id[protein_id] for protein_id in train_ids], axis=0
        )
        denominator = np.stack(
            [denominators_by_id[protein_id] for protein_id in train_ids], axis=0
        )
        mask = denominator > 0
        targets = np.divide(
            positives,
            denominator,
            out=np.zeros_like(positives),
            where=mask,
        )
        confidence = np.clip(denominator, 0.0, 1.0)
        arrays[f"{family}_targets"] = targets.astype(np.float32)
        arrays[f"{family}_mask"] = mask
        arrays[f"{family}_denominator"] = denominator.astype(np.float32)
        arrays[f"{family}_confidence"] = confidence.astype(np.float32)
        support[family] = {
            label: int(((targets[:, index] > 0) & mask[:, index]).sum())
            for index, label in enumerate(vocabulary)
        }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "families": {
            "mechanism": list(MECHANISM_LABELS),
            "cofactor": list(COFACTOR_LABELS),
        },
        "num_train_enzymes": len(train_ids),
        "num_matched_members": int(len(members)),
        "num_skipped_members": skipped_members,
        "num_curated_cofactor_enzymes": curated_count,
        "row_coverage": {
            family: int(arrays[f"{family}_mask"].any(axis=1).sum())
            for family in ("mechanism", "cofactor")
        },
        "support": support,
        "sources": {
            "train_pairs": str(train_pairs_path),
            "matched_members": str(matched_members_path),
            "directional_features": str(directional_features_path),
            "enzyme_cofactors": (
                None if enzyme_cofactor_labels_path is None else str(enzyme_cofactor_labels_path)
            ),
        },
        "source_sha256": {
            "train_pairs": _sha256(train_pairs_path),
            "matched_members": _sha256(matched_members_path),
            "directional_features": _sha256(directional_features_path),
            "enzyme_cofactors": (
                None
                if enzyme_cofactor_labels_path is None
                else _sha256(enzyme_cofactor_labels_path)
            ),
        },
    }
    return train_ids, arrays, metadata


def write_minimal_targets(
    output_npz: Path,
    output_vocab: Path,
    ids: list[str],
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_vocab.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, ids=np.asarray(ids, dtype=object), **arrays)
    output_vocab.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "COFACTOR_LABELS",
    "MECHANISM_LABELS",
    "SCHEMA_VERSION",
    "build_minimal_targets",
    "canonical_protein_key",
    "cofactor_groups",
    "mechanism_groups",
    "write_minimal_targets",
]
