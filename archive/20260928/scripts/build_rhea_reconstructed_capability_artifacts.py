#!/usr/bin/env python3
"""Build capability artifacts for the ReactZyme Rhea-reconstructed train variant."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from horizyn.capability.enzyme_labels import build_enzyme_capability_labels
from horizyn.capability.io import ensure_dir, write_json
from horizyn.capability.reaction_demand import build_reaction_demand_vectors
from horizyn.capability.reaction_features import (
    _collect_label_vocabs,
    _quality_report,
    build_reaction_features,
)


def _bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def _decode_ids(values: Any) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _load_drfp_by_id(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    ids = _decode_ids(payload["ids"])
    vectors = np.asarray(payload["vectors"])
    return {reaction_id: vectors[idx] for idx, reaction_id in enumerate(ids)}


def _write_feature_bundle(
    *,
    features: pd.DataFrame,
    drfp_by_id: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reaction_ids = features["reaction_id"].astype(str).tolist()
    missing = [reaction_id for reaction_id in reaction_ids if reaction_id not in drfp_by_id]
    if missing:
        raise ValueError(f"Missing DRFP vectors for {len(missing)} reactions, e.g. {missing[:5]}")
    features.to_parquet(out_dir / "reaction_features.parquet", index=False)
    np.savez_compressed(
        out_dir / "reaction_drfp.npz",
        vectors=np.stack([drfp_by_id[reaction_id] for reaction_id in reaction_ids]),
        ids=np.asarray(reaction_ids),
    )
    write_json(
        out_dir / "reaction_id_to_index.json",
        {reaction_id: idx for idx, reaction_id in enumerate(reaction_ids)},
    )
    rows = features.to_dict("records")
    write_json(out_dir / "label_vocabs.json", _collect_label_vocabs(rows))
    write_json(out_dir / "annotation_quality_report.json", _quality_report(rows))
    sample = features.sample(n=min(50, len(features)), random_state=13) if len(features) else features
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
    ].to_csv(out_dir / "reaction_feature_examples.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-pairs",
        default=(
            "data/standardized/retrieval_training_source_collapse/train_exact/"
            "train_pairs_valid_rxn_rhea_reconstructed.csv"
        ),
    )
    parser.add_argument(
        "--train-rxns",
        default=(
            "data/standardized/retrieval_training_source_collapse/train_exact/"
            "train_rxns_valid_rxn_rhea_reconstructed.csv"
        ),
    )
    parser.add_argument(
        "--base-capability-dir",
        default="data/processed/capability_features/train_exact",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/capability_features/train_exact_rhea_reconstructed",
    )
    parser.add_argument(
        "--rhea2ec",
        default="data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv",
    )
    parser.add_argument(
        "--rhea-chebi-smiles",
        default="data/paper/reactzyme/raw/rhea_molecules.tsv",
    )
    parser.add_argument(
        "--cofactor-dictionary",
        default="data/processed/capability_features/chebi_cofactor_dictionary.tsv",
    )
    parser.add_argument("--drfp-bits", type=int, default=2048)
    parser.add_argument("--use-rxnmapper", type=_bool, default=True)
    parser.add_argument("--rxnmapper-batch-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    base_dir = Path(args.base_capability_dir)
    train_rxns = pd.read_csv(args.train_rxns)
    train_pairs = pd.read_csv(args.train_pairs)
    train_rxn_ids = train_rxns["reaction_id"].astype(str).tolist()

    base_features = pd.read_parquet(base_dir / "reaction_features.parquet")
    base_features = base_features[base_features["reaction_id"].astype(str).isin(train_rxn_ids)].copy()
    base_drfp = _load_drfp_by_id(base_dir / "reaction_drfp.npz")
    existing_ids = set(base_features["reaction_id"].astype(str))
    missing_rxns = train_rxns[~train_rxns["reaction_id"].astype(str).isin(existing_ids)].copy()

    missing_dir = out_dir / "_missing_reaction_feature_build"
    if missing_dir.exists():
        shutil.rmtree(missing_dir)
    if len(missing_rxns):
        missing_dir.mkdir(parents=True)
        missing_csv = missing_dir / "missing_reactions.csv"
        missing_rxns.to_csv(missing_csv, index=False)
        missing_features = build_reaction_features(
            reaction_smiles_path=missing_csv,
            out_dir=missing_dir,
            train_pairs_path=args.train_pairs,
            rhea2ec_path=args.rhea2ec,
            rhea_chebi_smiles_path=args.rhea_chebi_smiles,
            cofactor_dictionary_path=args.cofactor_dictionary,
            drfp_bits=args.drfp_bits,
            use_rxnmapper=args.use_rxnmapper,
            rxnmapper_batch_size=args.rxnmapper_batch_size,
        )
        missing_drfp = _load_drfp_by_id(missing_dir / "reaction_drfp.npz")
    else:
        missing_features = pd.DataFrame(columns=base_features.columns)
        missing_drfp = {}

    features_by_id = {
        str(row["reaction_id"]): row
        for row in pd.concat([base_features, missing_features], ignore_index=True).to_dict("records")
    }
    missing_after_merge = [reaction_id for reaction_id in train_rxn_ids if reaction_id not in features_by_id]
    if missing_after_merge:
        raise ValueError(
            f"{len(missing_after_merge)} train reactions have no feature row after merge; "
            f"examples={missing_after_merge[:5]}"
        )
    merged_features = pd.DataFrame([features_by_id[reaction_id] for reaction_id in train_rxn_ids])
    drfp_by_id = {**base_drfp, **missing_drfp}
    _write_feature_bundle(features=merged_features, drfp_by_id=drfp_by_id, out_dir=out_dir)

    train_pairs_out = out_dir / "train_pairs_directional_rhea_reconstructed.csv"
    train_pairs.to_csv(train_pairs_out, index=False)
    build_reaction_demand_vectors(
        reaction_features_path=out_dir / "reaction_features.parquet",
        reaction_drfp_path=out_dir / "reaction_drfp.npz",
        out_dir=out_dir,
    )
    enzyme_labels, pair_training = build_enzyme_capability_labels(
        train_pairs_path=train_pairs_out,
        reaction_features_path=out_dir / "reaction_features.parquet",
        out_dir=out_dir,
    )
    report = {
        "train_pairs": int(len(train_pairs)),
        "train_reactions": int(len(train_rxns)),
        "base_feature_rows_reused": int(len(base_features)),
        "missing_reactions_annotated": int(len(missing_rxns)),
        "output_reaction_features": int(len(merged_features)),
        "output_enzyme_labels": int(len(enzyme_labels)),
        "output_pair_capability_training": int(len(pair_training)),
        "all_output_reactions_directional": bool(
            train_rxns["reaction_smiles"].fillna("").astype(str).str.contains(">>", regex=False).all()
        ),
        "rxnmapper_used_for_missing_reactions": bool(args.use_rxnmapper),
        "new_missing_reaction_center_policy": (
            "rxnmapper" if args.use_rxnmapper else "no_atom_map"
        ),
    }
    write_json(out_dir / "rhea_reconstructed_capability_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
