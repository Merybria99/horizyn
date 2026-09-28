#!/usr/bin/env python3
"""Validate generated SLEEC stage-1 labels, FASTA, embeddings, and optional MSAs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels", default=Path("data/sleec_stage1/mcsa_residue_labels.csv"), type=Path)
    parser.add_argument(
        "--embeddings",
        default=Path("data/sleec_stage1/mcsa_cath_esm2_650m_residue.h5"),
        type=Path,
    )
    parser.add_argument("--fasta", default=Path("data/sleec_stage1/fasta/mcsa_reference.fasta"), type=Path)
    parser.add_argument("--msa-dir", default=None, type=Path)
    parser.add_argument("--msa-glob", default="**/*.a3m")
    parser.add_argument("--expected-dim", default=1280, type=int)
    return parser.parse_args()


def read_fasta_lengths(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    lengths[current_id] = len("".join(chunks))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        lengths[current_id] = len("".join(chunks))
    return lengths


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    fasta_lengths = read_fasta_lengths(args.fasta)

    label_count = 0
    positive_count = 0
    split_proteins: dict[str, set[str]] = {"train": set(), "val": set()}
    max_label_index: dict[str, int] = {}
    with args.labels.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"protein_id", "residue_index", "label", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{args.labels} missing required columns: {sorted(missing)}")
        for row in reader:
            protein_id = row["protein_id"]
            residue_index = int(row["residue_index"])
            label = int(float(row["label"]))
            split = row["split"]
            label_count += 1
            positive_count += label
            if split in split_proteins:
                split_proteins[split].add(protein_id)
            max_label_index[protein_id] = max(max_label_index.get(protein_id, -1), residue_index)
            if protein_id not in fasta_lengths:
                errors.append(f"Label protein missing from FASTA: {protein_id}")
            elif residue_index >= fasta_lengths[protein_id]:
                errors.append(
                    f"Label index out of FASTA range: {protein_id}:{residue_index} >= "
                    f"{fasta_lengths[protein_id]}"
                )

    with h5py.File(args.embeddings, "r") as h5_file:
        for dataset_name in ("ids", "offsets", "vectors"):
            if dataset_name not in h5_file:
                errors.append(f"Embeddings missing dataset: {dataset_name}")
        h5_ids = {
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in h5_file["ids"][:]
        }
        offsets = h5_file["offsets"][:]
        h5_lengths = {
            protein_id: int(offsets[index + 1] - offsets[index])
            for index, protein_id in enumerate(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in h5_file["ids"][:]
            )
        }
        vector_dim = int(h5_file["vectors"].shape[1])
        if vector_dim != args.expected_dim:
            errors.append(f"Embedding dim {vector_dim} != expected {args.expected_dim}")
        missing_h5 = sorted(set(max_label_index) - h5_ids)
        errors.extend(f"Label protein missing from HDF5: {protein_id}" for protein_id in missing_h5[:20])
        if len(missing_h5) > 20:
            errors.append(f"...and {len(missing_h5) - 20} more HDF5-missing proteins")
        for protein_id, residue_index in max_label_index.items():
            if protein_id in h5_lengths and residue_index >= h5_lengths[protein_id]:
                errors.append(
                    f"Label index out of HDF5 range: {protein_id}:{residue_index} >= "
                    f"{h5_lengths[protein_id]}"
                )

    msa_summary = None
    if args.msa_dir is not None:
        a3m_paths = sorted(args.msa_dir.glob(args.msa_glob))
        zero_byte = [path for path in a3m_paths if path.stat().st_size == 0]
        msa_summary = {"a3m_files": len(a3m_paths), "zero_byte_a3m_files": len(zero_byte)}
        if zero_byte:
            errors.append(f"Found {len(zero_byte)} zero-byte A3M files")

    train_val_overlap = split_proteins["train"] & split_proteins["val"]
    if train_val_overlap:
        errors.append(f"Train/val protein overlap: {sorted(train_val_overlap)[:20]}")

    summary = {
        "labels": {
            "rows": label_count,
            "positives": positive_count,
            "proteins": len(max_label_index),
            "train_proteins": len(split_proteins["train"]),
            "val_proteins": len(split_proteins["val"]),
        },
        "fasta": {"records": len(fasta_lengths), "residues": sum(fasta_lengths.values())},
        "embeddings": {
            "ids": len(h5_ids),
            "residues": int(offsets[-1]),
            "vector_dim": vector_dim,
        },
        "msa": msa_summary,
        "errors": errors,
        "ok": not errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
