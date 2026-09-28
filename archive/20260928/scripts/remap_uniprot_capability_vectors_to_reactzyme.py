#!/usr/bin/env python3
"""Remap UniProt-keyed enzyme capability vectors to ReactZyme protein IDs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-vectors",
        required=True,
        help="Input NPZ with UniProt-keyed ids and vectors arrays.",
    )
    parser.add_argument(
        "--reactzyme-uniprot-tsv",
        default="data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv",
        help="ReactZyme UniProt/Rhea TSV containing Entry and Sequence columns.",
    )
    parser.add_argument(
        "--id-file",
        action="append",
        default=[],
        help="Text file with one requested protein ID per line. May be repeated.",
    )
    parser.add_argument(
        "--pair-file",
        action="append",
        default=[],
        help="CSV pair file with a protein_id column. May be repeated.",
    )
    parser.add_argument(
        "--fasta",
        action="append",
        default=[],
        help="FASTA file whose record IDs should be included. May be repeated.",
    )
    parser.add_argument("--output", required=True, help="Output ReactZyme-keyed NPZ.")
    parser.add_argument(
        "--mapping-output",
        default=None,
        help="Optional CSV report for every requested protein ID.",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional JSON coverage report.",
    )
    return parser.parse_args()


def reactzyme_hash(sequence: str) -> str:
    return "prot_" + hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]


def read_id_file(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_pair_protein_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "protein_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a protein_id column")
        for row in reader:
            protein_id = str(row.get("protein_id", "")).strip()
            if protein_id:
                ids.append(protein_id)
    return ids


def read_fasta_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                record_id = line[1:].strip().split()[0]
                if record_id:
                    ids.append(record_id)
    return ids


def load_reactzyme_hash_map(path: Path) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Entry", "Sequence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            entry = str(row.get("Entry", "")).strip()
            sequence = "".join(str(row.get("Sequence", "")).split())
            if not entry or not sequence:
                continue
            mapping.setdefault(reactzyme_hash(sequence), set()).add(entry.split("-")[0])
    return mapping


def load_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if "ids" not in payload or "vectors" not in payload:
        raise KeyError(f"{path} must contain 'ids' and 'vectors' arrays")
    ids = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in payload["ids"]
    ]
    vectors = np.asarray(payload["vectors"], dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"{path}: vectors must be rank-2, got {vectors.shape}")
    if len(ids) != vectors.shape[0]:
        raise ValueError(
            f"{path}: ids length {len(ids)} does not match vector rows {vectors.shape[0]}"
        )
    return ids, vectors


def collect_requested_ids(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    for value in args.id_file:
        requested.extend(read_id_file(Path(value)))
    for value in args.pair_file:
        requested.extend(read_pair_protein_ids(Path(value)))
    for value in args.fasta:
        requested.extend(read_fasta_ids(Path(value)))
    return sorted(set(requested))


def main() -> None:
    args = parse_args()
    source_ids, source_vectors = load_vectors(Path(args.source_vectors))
    source_index = {protein_id: idx for idx, protein_id in enumerate(source_ids)}
    hash_map = load_reactzyme_hash_map(Path(args.reactzyme_uniprot_tsv))
    requested_ids = collect_requested_ids(args)
    if not requested_ids:
        raise ValueError("No requested protein IDs were provided")

    output_ids: list[str] = []
    output_vectors: list[np.ndarray] = []
    mapping_rows: list[dict[str, str]] = []
    stats = {
        "requested_ids": len(requested_ids),
        "source_vector_ids": len(source_ids),
        "reactzyme_hash_mappings": len(hash_map),
        "direct_uniprot_requested": 0,
        "hashed_requested": 0,
        "mapped_ids": 0,
        "missing_accession_mapping": 0,
        "missing_source_vector": 0,
        "multi_accession_ids": 0,
    }

    for protein_id in requested_ids:
        if protein_id in source_index:
            accessions = {protein_id}
            stats["direct_uniprot_requested"] += 1
        elif "-" in protein_id and protein_id.split("-")[0] in source_index:
            accessions = {protein_id.split("-")[0]}
            stats["direct_uniprot_requested"] += 1
        else:
            accessions = set(hash_map.get(protein_id, set()))
            if protein_id.startswith("prot_"):
                stats["hashed_requested"] += 1

        vector_accessions = sorted(
            accession for accession in accessions if accession in source_index
        )
        if len(accessions) > 1:
            stats["multi_accession_ids"] += 1
        status = "mapped"
        if not accessions:
            status = "missing_accession_mapping"
            stats["missing_accession_mapping"] += 1
        elif not vector_accessions:
            status = "missing_source_vector"
            stats["missing_source_vector"] += 1
        else:
            rows = source_vectors[[source_index[accession] for accession in vector_accessions]]
            output_ids.append(protein_id)
            output_vectors.append(rows.mean(axis=0).astype(np.float32, copy=False))
            stats["mapped_ids"] += 1

        mapping_rows.append(
            {
                "protein_id": protein_id,
                "accessions": ";".join(sorted(accessions)),
                "vector_accessions": ";".join(vector_accessions),
                "status": status,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vectors = (
        np.stack(output_vectors).astype(np.float32, copy=False)
        if output_vectors
        else np.zeros((0, source_vectors.shape[1]), dtype=np.float32)
    )
    np.savez_compressed(
        output_path,
        ids=np.asarray(output_ids, dtype=object),
        vectors=vectors,
    )

    stats["output_vectors"] = int(vectors.shape[0])
    stats["vector_dim"] = int(vectors.shape[1]) if vectors.ndim == 2 else 0
    stats["coverage"] = float(stats["mapped_ids"] / stats["requested_ids"])

    if args.mapping_output:
        mapping_path = Path(args.mapping_output)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["protein_id", "accessions", "vector_accessions", "status"],
            )
            writer.writeheader()
            writer.writerows(mapping_rows)

    if args.metadata_output:
        metadata_path = Path(args.metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
