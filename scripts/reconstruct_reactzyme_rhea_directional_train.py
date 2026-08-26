#!/usr/bin/env python3
"""Reconstruct directional Rhea reactions for ReactZyme train rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from horizyn.capability.reactzyme_rhea import (
    combine_source_collapse_variant,
    reconstruct_reactzyme_train_rhea,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reactzyme-eval-root",
        default="data/paper/reactzyme/eval",
    )
    parser.add_argument(
        "--cleaned-uniprot-rhea",
        default="data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv",
    )
    parser.add_argument(
        "--rhea-molecules",
        default="data/paper/reactzyme/raw/rhea_molecules.tsv",
    )
    parser.add_argument(
        "--base-train-pairs",
        default=(
            "data/standardized/retrieval_training_source_collapse/train_exact/"
            "train_pairs_valid_rxn_pseudo.csv"
        ),
    )
    parser.add_argument(
        "--base-train-rxns",
        default=(
            "data/standardized/retrieval_training_source_collapse/train_exact/"
            "train_rxns_valid_rxn_pseudo.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="data/standardized/retrieval_training_source_collapse/train_exact",
    )
    parser.add_argument("--suffix", default="_valid_rxn_rhea_reconstructed")
    parser.add_argument(
        "--audit-dir",
        default="data/paper/reactzyme/reconstructed_rhea_train",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    audit_dir = Path(args.audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    reconstructed_pairs, reconstructed_rxns, members, report = reconstruct_reactzyme_train_rhea(
        reactzyme_eval_root=args.reactzyme_eval_root,
        cleaned_uniprot_rhea_path=args.cleaned_uniprot_rhea,
        rhea_molecules_path=args.rhea_molecules,
    )
    reconstructed_pairs.to_csv(audit_dir / "reactzyme_train_directional_rhea_pairs.csv", index=False)
    reconstructed_rxns.to_csv(audit_dir / "reactzyme_train_directional_rhea_reactions.csv", index=False)
    members.to_csv(audit_dir / "reactzyme_train_directional_rhea_members.csv", index=False)

    base_pairs = pd.read_csv(args.base_train_pairs)
    base_rxns = pd.read_csv(args.base_train_rxns)
    combined_pairs, combined_rxns = combine_source_collapse_variant(
        base_pairs=base_pairs,
        base_reactions=base_rxns,
        reconstructed_pairs=reconstructed_pairs,
        reconstructed_reactions=reconstructed_rxns,
    )

    pair_path = out_dir / f"train_pairs{args.suffix}.csv"
    rxn_path = out_dir / f"train_rxns{args.suffix}.csv"
    combined_pairs.to_csv(pair_path, index=False)
    combined_rxns.to_csv(rxn_path, index=False)

    reactzyme_base_pairs = int(
        base_pairs["source_datasets"].fillna("").astype(str).str.contains("reactzyme").sum()
    )
    metadata = {
        "suffix": args.suffix,
        "base_train_pairs": str(args.base_train_pairs),
        "base_train_rxns": str(args.base_train_rxns),
        "output_train_pairs": str(pair_path),
        "output_train_rxns": str(rxn_path),
        "audit_dir": str(audit_dir),
        "policy": (
            "ReactZyme molecule-set rows are removed; train-only exact-sequence "
            "Rhea substrate/product reactions are added as directional positives."
        ),
        "base_pairs": int(len(base_pairs)),
        "base_reactions": int(len(base_rxns)),
        "base_reactzyme_molecule_set_pairs_removed": reactzyme_base_pairs,
        "base_non_reactzyme_pairs_kept": int(len(base_pairs) - reactzyme_base_pairs),
        "reconstructed_reactzyme_pairs_added": int(len(reconstructed_pairs)),
        "reconstructed_reactzyme_reactions_added": int(len(reconstructed_rxns)),
        "output_pairs": int(len(combined_pairs)),
        "output_reactions": int(len(combined_rxns)),
        "output_reactzyme_pairs": int(
            combined_pairs["source_datasets"].fillna("").astype(str).str.contains("reactzyme").sum()
        ),
        "output_directional_reactions_with_separator": int(
            combined_rxns["reaction_smiles"].fillna("").astype(str).str.contains(">>", regex=False).sum()
        ),
        "reconstruction_report": report,
    }
    metadata_path = out_dir / f"metadata{args.suffix}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (audit_dir / "reconstruction_report.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
