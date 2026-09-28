#!/usr/bin/env python3
"""Enhance enzyme capability cofactor labels with enzyme-side UniProt evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.capability.enzyme_cofactor_annotations import enhance_enzyme_cofactor_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capability-dir",
        default="data/processed/capability_features/train_exact_rhea_reconstructed",
        help="Capability artifact directory containing enzyme labels and pair table.",
    )
    parser.add_argument(
        "--train-pairs",
        default="data/standardized/retrieval_training_source_collapse/train_exact/train_pairs_valid_rxn_rhea_reconstructed.csv",
        help="Train-only pair table with source_entries provenance.",
    )
    parser.add_argument(
        "--uniprot-molecules",
        default="data/paper/reactzyme/raw/uniprot_molecules.tsv",
        help="UniProt accession to molecule-set table.",
    )
    parser.add_argument(
        "--cleaned-uniprot-rhea",
        default="data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv",
        help="Optional UniProt/Rhea table used to map sequence hashes back to accessions.",
    )
    parser.add_argument(
        "--reaction-features",
        default=None,
        help="Optional reaction_features.parquet path; defaults to capability-dir/reaction_features.parquet.",
    )
    parser.add_argument(
        "--cofactor-dictionary",
        default=None,
        help="Optional ChEBI cofactor dictionary TSV/CSV/JSON.",
    )
    parser.add_argument(
        "--uniprotkb-cofactors",
        default=None,
        help=(
            "Optional TSV cache with UniProtKB Entry/Cofactor columns. If omitted "
            "and --fetch-uniprotkb-cofactors is set, capability-dir/uniprotkb_cofactor_comments.tsv is used."
        ),
    )
    parser.add_argument(
        "--fetch-uniprotkb-cofactors",
        action="store_true",
        help="Fetch missing UniProtKB cofactor comments into the cache before annotating.",
    )
    parser.add_argument(
        "--uniprotkb-fetch-scope",
        choices=("missing_cofactor", "all_accessions"),
        default="missing_cofactor",
        help="Which accessions to fetch when --fetch-uniprotkb-cofactors is set.",
    )
    parser.add_argument("--uniprotkb-batch-size", type=int, default=250)
    parser.add_argument("--no-csv", action="store_true", help="Do not write enzyme_cofactor_labels_enhanced.csv")
    args = parser.parse_args()

    labels, pairs, report = enhance_enzyme_cofactor_annotations(
        capability_dir=args.capability_dir,
        train_pairs_path=args.train_pairs,
        uniprot_molecules_path=args.uniprot_molecules,
        cleaned_uniprot_rhea_path=args.cleaned_uniprot_rhea,
        reaction_features_path=args.reaction_features,
        cofactor_dictionary_path=args.cofactor_dictionary,
        uniprotkb_cofactor_path=args.uniprotkb_cofactors,
        fetch_uniprotkb_cofactors=args.fetch_uniprotkb_cofactors,
        uniprotkb_fetch_scope=args.uniprotkb_fetch_scope,
        uniprotkb_batch_size=args.uniprotkb_batch_size,
        write_csv=not args.no_csv,
    )
    print(f"enhanced enzyme labels: {len(labels):,}")
    print(f"enhanced pair rows: {len(pairs):,}")
    print(
        "core cofactor enzymes: "
        f"reaction={report['reaction_derived_core_cofactor_enzymes']:,}, "
        f"enzyme_side={report['enzyme_derived_core_cofactor_enzymes']:,}, "
        f"combined={report['combined_core_cofactor_enzymes']:,}, "
        f"new={report['new_core_cofactor_enzymes_from_enzyme_side']:,}"
    )
    if "uniprotkb_cofactor_rows_with_labels" in report:
        print(
            "UniProtKB cofactor comments: "
            f"rows_with_labels={report['uniprotkb_cofactor_rows_with_labels']:,}, "
            f"enzyme_labels={report['num_enzymes_with_uniprotkb_cofactor_labels']:,}"
        )


if __name__ == "__main__":
    main()
