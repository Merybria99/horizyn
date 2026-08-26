#!/usr/bin/env python3
"""Prepare downloaded paper benchmark assets for Horizyn evaluation scripts.

The downloader preserves the upstream files under data/paper/*/raw and extracts
archives under data/paper/*/processed. This script adds lightweight CSV/FASTA
views with deterministic IDs so scripts/evaluate_paper_setting.py can consume
the benchmarks without changing the model.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import pickle
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def stable_id(prefix: str, value: str, length: int = 16) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_fasta(path: Path, records: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record_id in sorted(records):
            sequence = records[record_id]
            handle.write(f">{record_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def write_id_list(path: Path, ids: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_ids = sorted(set(ids))
    with path.open("w", encoding="utf-8") as handle:
        for item in unique_ids:
            handle.write(item + "\n")
    return len(unique_ids)


def load_torch_object(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("PyTorch is required to read ReactZyme .pt split files") from exc
    return torch.load(path, map_location="cpu")


def iter_reactzyme_pairs(path: Path) -> Iterable[tuple[str, str]]:
    payload = load_torch_object(path)
    if isinstance(payload, dict):
        values = payload.values()
    elif isinstance(payload, (list, tuple)):
        values = payload
    else:
        raise ValueError(f"Unsupported ReactZyme split object in {path}: {type(payload)}")

    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Expected (reaction_smiles, protein_sequence) pairs in {path}")
        reaction_smiles, protein_sequence = item
        if reaction_smiles and protein_sequence:
            yield str(reaction_smiles), str(protein_sequence)


def materialize_reactzyme_split(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
) -> dict[str, int | str]:
    split_records: dict[str, list[dict[str, str]]] = {"train": [], "test": []}
    reaction_rows: dict[str, dict[str, str]] = {}
    protein_sequences: dict[str, str] = {}

    for split_name, split_path in (("train", train_path), ("test", test_path)):
        seen_pairs: set[tuple[str, str]] = set()
        for reaction_smiles, protein_sequence in iter_reactzyme_pairs(split_path):
            reaction_id = stable_id("rxn", reaction_smiles)
            protein_id = stable_id("prot", protein_sequence)
            pair_key = (reaction_id, protein_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            reaction_rows[reaction_id] = {
                "reaction_id": reaction_id,
                "reaction_smiles": reaction_smiles,
            }
            protein_sequences[protein_id] = protein_sequence
            split_records[split_name].append(
                {
                    "reaction_id": reaction_id,
                    "protein_id": protein_id,
                    "reaction_smiles": reaction_smiles,
                    "protein_sequence": protein_sequence,
                }
            )

    pair_fields = ["reaction_id", "protein_id", "reaction_smiles", "protein_sequence"]
    train_pairs = write_csv(output_dir / "train_pairs.csv", split_records["train"], pair_fields)
    test_pairs = write_csv(output_dir / "test_pairs.csv", split_records["test"], pair_fields)
    reactions = write_csv(
        output_dir / "reactions.csv",
        (reaction_rows[key] for key in sorted(reaction_rows)),
        ["reaction_id", "reaction_smiles"],
    )
    candidates = write_id_list(output_dir / "candidate_ids.txt", protein_sequences)
    write_fasta(output_dir / "proteins.fasta", protein_sequences)

    return {
        "output_dir": str(output_dir),
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "reactions": reactions,
        "candidate_proteins": candidates,
    }


def prepare_reactzyme(root: Path) -> dict[str, Any]:
    processed = root / "reactzyme" / "processed"
    output_root = root / "reactzyme" / "eval"
    split_files = {
        "time": (
            processed / "time" / "positive_train_val_time.pt",
            processed / "time" / "positive_test_time.pt",
        ),
        "enzyme_smi": (
            processed / "enzyme_smi" / "positive_train_val_seq_smi.pt",
            processed / "enzyme_smi" / "positive_test_seq_smi.pt",
        ),
        "reaction_smi": (
            processed / "reaction_smi" / "positive_train_val_mol_smi.pt",
            processed / "reaction_smi" / "positive_test_mol_smi.pt",
        ),
    }

    summary: dict[str, Any] = {}
    all_proteins: dict[str, str] = {}
    for split_name, (train_path, test_path) in split_files.items():
        if not train_path.exists() or not test_path.exists():
            missing = [str(path) for path in (train_path, test_path) if not path.exists()]
            raise FileNotFoundError(f"Missing ReactZyme split files: {missing}")
        split_summary = materialize_reactzyme_split(
            train_path=train_path,
            test_path=test_path,
            output_dir=output_root / split_name,
        )
        summary[split_name] = split_summary

        fasta_path = output_root / split_name / "proteins.fasta"
        current_id: str | None = None
        chunks: list[str] = []
        with fasta_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id is not None:
                        all_proteins[current_id] = "".join(chunks)
                    current_id = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line)
            if current_id is not None:
                all_proteins[current_id] = "".join(chunks)

    write_fasta(output_root / "all_proteins.fasta", all_proteins)
    write_id_list(output_root / "all_candidate_ids.txt", all_proteins)
    summary["all_candidate_proteins"] = len(all_proteins)
    return summary


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def parse_protein_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped == "[]" or stripped.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return []
        value = parsed
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return []


def reaction_smiles_from_enzymemap_record(record: dict[str, Any]) -> str:
    reactants = [str(item) for item in record.get("reactants", []) if item]
    products = [str(item) for item in record.get("products", []) if item]
    if not reactants or not products:
        raise ValueError("EnzymeMap record is missing reactants or products")
    return ".".join(reactants) + ">>" + ".".join(products)


def prepare_clipzyme(root: Path) -> dict[str, Any]:
    files_dir = root / "clipzyme" / "files"
    output_dir = root / "clipzyme" / "eval" / "enzymemap"
    enzymemap_path = files_dir / "enzymemap.json"
    screening_path = files_dir / "clipzyme_screening_set.p"
    sequences_path = files_dir / "uniprot2sequence.p"

    for path in (enzymemap_path, screening_path, sequences_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing CLIPZyme/EnzymeMap file: {path}")

    screening_set = load_pickle(screening_path)
    candidate_ids = [str(item) for item in screening_set["uniprots"]]
    candidate_set = set(candidate_ids)
    uniprot2sequence = load_pickle(sequences_path)
    protein_sequences = {
        protein_id: str(uniprot2sequence[protein_id])
        for protein_id in candidate_ids
        if protein_id in uniprot2sequence
    }
    missing_sequences = len(candidate_ids) - len(protein_sequences)

    with enzymemap_path.open("r", encoding="utf-8") as handle:
        enzymemap = json.load(handle)

    reaction_rows: dict[str, dict[str, str]] = {}
    pair_rows: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    rows_with_refs = 0
    skipped_refs = 0

    for row_index, record in enumerate(enzymemap):
        refs = parse_protein_refs(record.get("protein_refs"))
        refs = [protein_id for protein_id in refs if protein_id in candidate_set]
        if not refs:
            if parse_protein_refs(record.get("protein_refs")):
                skipped_refs += 1
            continue
        rows_with_refs += 1
        reaction_smiles = reaction_smiles_from_enzymemap_record(record)
        reaction_id = stable_id("rxn", reaction_smiles)
        reaction_rows[reaction_id] = {
            "reaction_id": reaction_id,
            "reaction_smiles": reaction_smiles,
            "rxnid": str(record.get("rxnid", "")),
            "ec": str(record.get("ec", "")),
        }
        for protein_id in refs:
            pair_key = (reaction_id, protein_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            pair_rows.append(
                {
                    "reaction_id": reaction_id,
                    "protein_id": protein_id,
                    "reaction_smiles": reaction_smiles,
                    "rxnid": str(record.get("rxnid", "")),
                    "ec": str(record.get("ec", "")),
                    "enzymemap_row": str(row_index),
                }
            )

    write_csv(
        output_dir / "pairs.csv",
        pair_rows,
        ["reaction_id", "protein_id", "reaction_smiles", "rxnid", "ec", "enzymemap_row"],
    )
    reactions = write_csv(
        output_dir / "reactions.csv",
        (reaction_rows[key] for key in sorted(reaction_rows)),
        ["reaction_id", "reaction_smiles", "rxnid", "ec"],
    )
    candidates = write_id_list(output_dir / "candidate_ids.txt", protein_sequences)
    write_fasta(output_dir / "proteins.fasta", protein_sequences)

    return {
        "output_dir": str(output_dir),
        "candidate_proteins": candidates,
        "missing_candidate_sequences": missing_sequences,
        "enzymemap_rows": len(enzymemap),
        "rows_with_candidate_refs": rows_with_refs,
        "rows_with_refs_outside_candidate_pool": skipped_refs,
        "pairs": len(pair_rows),
        "reactions": reactions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/paper", help="Paper data root")
    parser.add_argument(
        "--skip-reactzyme",
        action="store_true",
        help="Do not materialize ReactZyme CSV/FASTA views",
    )
    parser.add_argument(
        "--skip-clipzyme",
        action="store_true",
        help="Do not materialize CLIPZyme/EnzymeMap CSV/FASTA views",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional JSON summary path. Defaults to <root>/manifests/prepared.json",
    )
    args = parser.parse_args()

    root = Path(args.root)
    summary: dict[str, Any] = {}
    if not args.skip_reactzyme:
        summary["reactzyme"] = prepare_reactzyme(root)
    if not args.skip_clipzyme:
        summary["clipzyme"] = prepare_clipzyme(root)

    summary_path = Path(args.summary) if args.summary else root / "manifests" / "prepared.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved preparation summary to: {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
