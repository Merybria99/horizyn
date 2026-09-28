#!/usr/bin/env python3
"""Build clean directional-only capability training files."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.capability.io import ensure_dir, write_json

try:
    import pandas as pd
except Exception:  # pragma: no cover - depends on runtime environment
    pd = None


def _read_pairs(path: str | Path) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to build directional capability splits")
    pairs = pd.read_csv(path)
    enzyme_col = "enzyme_id" if "enzyme_id" in pairs.columns else "protein_id"
    if enzyme_col not in pairs.columns or "reaction_id" not in pairs.columns:
        raise ValueError(f"{path} must contain reaction_id and enzyme_id/protein_id")
    out = pairs.copy()
    if enzyme_col != "enzyme_id":
        out = out.rename(columns={enzyme_col: "enzyme_id"})
    if "protein_id" not in out.columns:
        out["protein_id"] = out["enzyme_id"]
    out["reaction_id"] = out["reaction_id"].astype(str)
    out["enzyme_id"] = out["enzyme_id"].astype(str)
    out["protein_id"] = out["protein_id"].astype(str)
    if "pr_id" not in out.columns:
        out.insert(0, "pr_id", np.arange(len(out), dtype=np.int64))
    return out


def _valid_directional_reaction_ids(
    reaction_features_path: Path,
    reaction_demand_vectors_path: Path | None,
) -> set[str]:
    if pd is None:
        raise ImportError("pandas is required to build directional capability splits")
    features = pd.read_parquet(reaction_features_path)
    if "reaction_id" not in features.columns or "canonical_reaction_smiles" not in features.columns:
        raise ValueError(
            "reaction_features must contain reaction_id and canonical_reaction_smiles"
        )
    smiles = features["canonical_reaction_smiles"].fillna("").astype(str)
    valid = set(features.loc[smiles.str.contains(">>", regex=False), "reaction_id"].astype(str))
    if reaction_demand_vectors_path is not None:
        payload = np.load(reaction_demand_vectors_path, allow_pickle=True)
        ids = {
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in payload["ids"]
        }
        valid &= ids
    return valid


def _move_required(tmp_dir: Path, out_dir: Path, source_name: str, target_name: str) -> None:
    source = tmp_dir / source_name
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.move(str(source), str(out_dir / target_name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-pairs",
        default=(
            "data/standardized/retrieval_training_source_collapse/train_exact/"
            "train_pairs_valid_rxn_pseudo.csv"
        ),
    )
    parser.add_argument(
        "--reaction-features",
        default="data/processed/capability_features/train_exact/reaction_features.parquet",
    )
    parser.add_argument(
        "--reaction-demand-vectors",
        default="data/processed/capability_features/train_exact/reaction_demand_vectors.npz",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/capability_features/train_exact",
    )
    args = parser.parse_args()
    from horizyn.capability.enzyme_labels import build_enzyme_capability_labels

    out_dir = ensure_dir(args.out_dir)
    valid_reactions = _valid_directional_reaction_ids(
        Path(args.reaction_features),
        Path(args.reaction_demand_vectors) if args.reaction_demand_vectors else None,
    )
    pairs = _read_pairs(args.train_pairs)
    filtered = pairs[pairs["reaction_id"].isin(valid_reactions)].copy()
    if filtered.empty:
        raise ValueError("No train pairs remain after directional reaction filtering")

    directional_pairs_path = out_dir / "train_pairs_directional.csv"
    filtered.to_csv(directional_pairs_path, index=False)

    tmp_dir = out_dir / "_directional_label_build"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    build_enzyme_capability_labels(
        train_pairs_path=directional_pairs_path,
        reaction_features_path=args.reaction_features,
        out_dir=tmp_dir,
    )
    _move_required(
        tmp_dir,
        out_dir,
        "enzyme_capability_labels.parquet",
        "enzyme_capability_labels_directional.parquet",
    )
    _move_required(
        tmp_dir,
        out_dir,
        "pair_capability_training.parquet",
        "pair_capability_training_directional.parquet",
    )
    _move_required(
        tmp_dir,
        out_dir,
        "enzyme_label_vocabs.json",
        "enzyme_label_vocabs_directional.json",
    )
    _move_required(
        tmp_dir,
        out_dir,
        "enzyme_capability_quality_report.json",
        "enzyme_capability_quality_report_directional.json",
    )
    shutil.rmtree(tmp_dir)

    report = {
        "input_train_pairs": int(len(pairs)),
        "directional_train_pairs": int(len(filtered)),
        "removed_non_directional_or_missing_demand": int(len(pairs) - len(filtered)),
        "valid_directional_reactions": int(len(valid_reactions)),
        "output_train_pairs": str(directional_pairs_path),
        "output_pair_capability_training": str(
            out_dir / "pair_capability_training_directional.parquet"
        ),
        "output_enzyme_capability_labels": str(
            out_dir / "enzyme_capability_labels_directional.parquet"
        ),
    }
    write_json(out_dir / "directional_filter_report.json", report)


if __name__ == "__main__":
    main()
