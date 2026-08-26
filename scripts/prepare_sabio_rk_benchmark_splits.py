#!/usr/bin/env python3
"""Prepare leakage-filtered SABIO-RK retrieval benchmark splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENZYME_DISCOVERY_ROOT = PROJECT_ROOT.parent
DEFAULT_PAIRS = (
    ENZYME_DISCOVERY_ROOT
    / "sabio_rk_download/parquet_cache/enzyme_reaction_uniprot_pairs.parquet"
)
DEFAULT_SEQUENCES = ENZYME_DISCOVERY_ROOT / "sabio_rk_download/seqsim_fetch/new_sequences.tsv"
DEFAULT_SIMILARITY = (
    ENZYME_DISCOVERY_ROOT
    / "sabio_rk_download/seqsim_fetch/new_vs_old_max_similarity_by_source_dataset.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/paper/sabio_rk/eval"


def reaction_id_from_smiles(smiles: str) -> str:
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16]
    return f"sabio_rxn_{digest}"


def similarity_value_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if column.startswith("max_sim_")
        and not column.startswith("max_sim_seq_")
        and not column.startswith("max_sim_uniprotid_")
    ]


def load_valid_sabio_rows(
    pairs_path: Path,
    sequences_path: Path,
    similarity_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    pairs = pd.read_parquet(pairs_path)
    sequences = pd.read_csv(sequences_path, sep="\t", dtype=str)
    similarity = pd.read_csv(similarity_path, dtype=str)

    max_columns = similarity_value_columns(list(similarity.columns))
    if not max_columns:
        raise ValueError(f"No max_sim_* value columns found in {similarity_path}")

    similarity_values = similarity[max_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    similarity = similarity[["new_uniprot"]].copy()
    similarity["max_similarity_to_previous_dataset"] = similarity_values.max(axis=1)

    sequence_ids = set(sequences["accession"].dropna().astype(str))
    valid = pairs[
        pairs["reaction_smiles_status"].eq("complete")
        & pairs["reaction_smiles"].notna()
        & pairs["uniprot_accession"].notna()
    ].copy()
    valid["protein_id"] = valid["uniprot_accession"].astype(str)
    valid["reaction_smiles"] = valid["reaction_smiles"].astype(str)
    valid = valid[valid["protein_id"].isin(sequence_ids)]
    valid = valid.merge(
        similarity,
        left_on="protein_id",
        right_on="new_uniprot",
        how="inner",
    )
    valid["reaction_id"] = valid["reaction_smiles"].map(reaction_id_from_smiles)
    return valid, sequences, max_columns


def write_threshold_split(
    valid: pd.DataFrame,
    sequences: pd.DataFrame,
    threshold: float,
    output_root: Path,
) -> dict[str, object]:
    split_name = f"novelty{int(threshold)}"
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    subset = valid[valid["max_similarity_to_previous_dataset"] < threshold].copy()
    if subset.empty:
        raise ValueError(f"No SABIO-RK rows remain for threshold < {threshold}")

    pairs = subset[
        [
            "reaction_id",
            "protein_id",
            "reaction_smiles",
            "primary_sabio_reaction_id",
            "kinetic_law_id",
            "ec_numbers_json",
            "max_similarity_to_previous_dataset",
        ]
    ].drop_duplicates(subset=["reaction_id", "protein_id"])
    pairs = pairs.sort_values(["reaction_id", "protein_id"]).reset_index(drop=True)
    pairs.insert(0, "pr_id", range(len(pairs)))

    reactions = pairs[["reaction_id", "reaction_smiles"]].drop_duplicates("reaction_id")
    reactions = reactions.sort_values("reaction_id").reset_index(drop=True)

    candidate_ids = sorted(pairs["protein_id"].unique())
    filtered_sequences = sequences[sequences["accession"].isin(candidate_ids)].copy()
    filtered_sequences = filtered_sequences.sort_values("accession").reset_index(drop=True)

    pairs_path = split_dir / "pairs.csv"
    reactions_path = split_dir / "reactions.csv"
    candidates_path = split_dir / "candidate_ids.txt"
    sequences_path = split_dir / "sequences.tsv"
    metadata_path = split_dir / "metadata.json"

    pairs.to_csv(pairs_path, index=False)
    reactions.to_csv(reactions_path, index=False)
    filtered_sequences.to_csv(sequences_path, sep="\t", index=False)
    candidates_path.write_text("\n".join(candidate_ids) + "\n", encoding="utf-8")

    stats: dict[str, object] = {
        "split": split_name,
        "threshold": threshold,
        "similarity_rule": f"max similarity to any previous dataset < {threshold:g}",
        "pairs": str(pairs_path),
        "reactions": str(reactions_path),
        "candidate_ids": str(candidates_path),
        "sequences": str(sequences_path),
        "num_pairs": int(len(pairs)),
        "num_reactions": int(len(reactions)),
        "num_candidate_proteins": int(len(candidate_ids)),
        "num_sequences": int(len(filtered_sequences)),
        "min_similarity": float(pairs["max_similarity_to_previous_dataset"].min()),
        "max_similarity": float(pairs["max_similarity_to_previous_dataset"].max()),
        "mean_similarity": float(pairs["max_similarity_to_previous_dataset"].mean()),
    }
    metadata_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SABIO-RK complete-SMILES retrieval splits filtered by sequence novelty."
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--sequences", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--similarity", type=Path, default=DEFAULT_SIMILARITY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[50.0, 90.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid, sequences, max_columns = load_valid_sabio_rows(
        args.pairs,
        args.sequences,
        args.similarity,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for threshold in args.thresholds:
        summary.append(write_threshold_split(valid, sequences, threshold, args.output_root))

    summary_path = args.output_root / "summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    manifest = {
        "source_pairs": str(args.pairs),
        "source_sequences": str(args.sequences),
        "source_similarity": str(args.similarity),
        "similarity_columns": max_columns,
        "valid_complete_smiles_rows_with_sequence_and_similarity": int(len(valid)),
        "splits": summary,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
