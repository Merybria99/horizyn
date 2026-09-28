#!/usr/bin/env python3
"""Build a Horizyn-compatible EnzymeMap train split from the upstream repo."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENZYMEMAP = PROJECT_ROOT.parent / "sources/enzymemap/data/processed_reactions.csv.gz"
DEFAULT_PROTEINS = PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/proteins.fasta"
DEFAULT_OUT = PROJECT_ROOT / "data/paper/clipzyme/train/enzymemap"


def read_fasta_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                ids.add(line[1:].strip().split()[0])
    return ids


def reaction_id(reaction_smiles: str) -> str:
    return f"rxn_{hashlib.sha1(reaction_smiles.encode('utf-8')).hexdigest()[:16]}"


def parse_refs(value: str) -> list[str]:
    value = (value or "").strip()
    if not value or value == "[]":
        return []
    try:
        refs = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        refs = [item.strip() for item in value.split(",")]
    if isinstance(refs, str):
        refs = [refs]
    return [str(ref).strip() for ref in refs if str(ref).strip() and str(ref).strip() != "-"]


def upstream_dataset_labels(n_rows: int, val_frac: float, test_frac: float, seed: int) -> list[str]:
    # Mirrors templatecorr.helpers.split_data_df, used by EnzymeMap's
    # scripts/analysis_preprocess.py: numpy seed, shuffled row indices,
    # train first, then val, then test.
    indices = list(range(n_rows))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_end = int((1.0 - test_frac - val_frac) * n_rows)
    val_end = int((1.0 - test_frac) * n_rows)
    labels = [""] * n_rows
    for idx in indices[:train_end]:
        labels[idx] = "train"
    for idx in indices[train_end:val_end]:
        labels[idx] = "val"
    for idx in indices[val_end:]:
        labels[idx] = "test"
    return labels


def count_rows(path: Path) -> int:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _row in reader)


def write_outputs(args: argparse.Namespace) -> dict[str, object]:
    protein_ids = read_fasta_ids(args.proteins_fasta)
    n_rows = count_rows(args.enzymemap_processed)
    labels = upstream_dataset_labels(n_rows, args.val_frac, args.test_frac, args.seed)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.csv"
    reactions_path = out_dir / "reactions.csv"
    candidates_path = out_dir / "candidate_ids.txt"
    metadata_path = out_dir / "metadata.json"

    pair_fields = [
        "reaction_id",
        "protein_id",
        "reaction_smiles",
        "rxnid",
        "ec",
        "enzymemap_row",
        "upstream_dataset",
        "source",
        "protein_db",
    ]
    reaction_fields = ["reaction_id", "reaction_smiles", "rxnid", "ec"]

    seen_pairs: set[tuple[str, str]] = set()
    reactions: dict[str, dict[str, str]] = {}
    candidate_ids: set[str] = set()
    stats = {
        "source": "https://github.com/hesther/enzymemap",
        "source_file": str(args.enzymemap_processed),
        "proteins_fasta": str(args.proteins_fasta),
        "output_dir": str(out_dir),
        "split_seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "raw_rows": n_rows,
        "upstream_train_rows": 0,
        "rows_with_uniprot_like_refs": 0,
        "rows_without_refs": 0,
        "protein_refs_seen": 0,
        "protein_refs_not_in_fasta": 0,
        "duplicate_pairs": 0,
        "pairs_written": 0,
        "reactions_written": 0,
        "candidate_proteins": 0,
    }

    with gzip.open(args.enzymemap_processed, "rt", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        with pairs_path.open("w", newline="", encoding="utf-8") as pair_handle:
            pair_writer = csv.DictWriter(pair_handle, fieldnames=pair_fields)
            pair_writer.writeheader()
            for idx, row in enumerate(reader):
                if labels[idx] != "train":
                    continue
                stats["upstream_train_rows"] += 1
                refs = parse_refs(row.get("protein_refs", ""))
                if not refs:
                    stats["rows_without_refs"] += 1
                    continue
                if (row.get("protein_db") or "").strip().lower() not in {"uniprot", "swissprot"}:
                    continue
                stats["rows_with_uniprot_like_refs"] += 1
                rxn_smiles = (row.get("unmapped") or "").strip()
                if not rxn_smiles:
                    continue
                rid = reaction_id(rxn_smiles)
                rxnid = (row.get("rxn_idx") or "").strip()
                ec = (row.get("ec_num") or "").strip()
                for protein_id in refs:
                    stats["protein_refs_seen"] += 1
                    if protein_id not in protein_ids:
                        stats["protein_refs_not_in_fasta"] += 1
                        continue
                    key = (rid, protein_id)
                    if key in seen_pairs:
                        stats["duplicate_pairs"] += 1
                        continue
                    seen_pairs.add(key)
                    candidate_ids.add(protein_id)
                    reactions.setdefault(
                        rid,
                        {
                            "reaction_id": rid,
                            "reaction_smiles": rxn_smiles,
                            "rxnid": rxnid,
                            "ec": ec,
                        },
                    )
                    pair_writer.writerow(
                        {
                            "reaction_id": rid,
                            "protein_id": protein_id,
                            "reaction_smiles": rxn_smiles,
                            "rxnid": rxnid,
                            "ec": ec,
                            "enzymemap_row": str(idx),
                            "upstream_dataset": "train",
                            "source": row.get("source", ""),
                            "protein_db": row.get("protein_db", ""),
                        }
                    )

    with reactions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reaction_fields)
        writer.writeheader()
        for row in sorted(reactions.values(), key=lambda item: item["reaction_id"]):
            writer.writerow(row)

    candidates_path.write_text("\n".join(sorted(candidate_ids)) + ("\n" if candidate_ids else ""), encoding="utf-8")
    stats["pairs_written"] = len(seen_pairs)
    stats["reactions_written"] = len(reactions)
    stats["candidate_proteins"] = len(candidate_ids)
    metadata_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enzymemap-processed", type=Path, default=DEFAULT_ENZYMEMAP)
    parser.add_argument("--proteins-fasta", type=Path, default=DEFAULT_PROTEINS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    stats = write_outputs(parse_args())
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
