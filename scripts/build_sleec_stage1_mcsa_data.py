#!/usr/bin/env python3
"""
Build mCSA-supervised data for SLEEC stage-1 residue classification.

The SLEEC paper reports 820 mCSA proteins, 3716 functional residues, and a
403/417 train/validation split. The local mCSA export may not match that exact
curation, so this script records both raw and generated counts in a manifest
instead of silently claiming paper-exact reproduction. By default, generated
data uses a deterministic protein-level 50/50 split over the fetched local
mCSA reference proteins.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAPER_MCSA_TRAIN_PROTEINS = 403
PAPER_MCSA_VAL_PROTEINS = 417
PAPER_MCSA_FUNCTIONAL_RESIDUES = 3716
DEFAULT_MAX_SEQUENCE_LENGTH = 1022
AMINO_ACID_PATTERN = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYX]")


@dataclass(frozen=True)
class FastaRecord:
    protein_id: str
    sequence: str


@dataclass(frozen=True)
class BuildStats:
    raw_reference_proteins: int
    raw_reference_positive_residues: int
    raw_all_homologue_uniprot_ids: int
    raw_empty_uniprot_entries: int
    raw_invalid_residue_entries: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mcsa-json",
        default=Path("data/SLEEC/catalytic_residues_homologues.json"),
        type=Path,
        help="Local mCSA catalytic residue homologues JSON",
    )
    parser.add_argument(
        "--curated-csv",
        default=Path("data/SLEEC/curated_data.csv"),
        type=Path,
        help="Optional curated mCSA CSV used only for manifest count comparison",
    )
    parser.add_argument("--output-dir", default=Path("data/sleec_stage1"), type=Path)
    parser.add_argument(
        "--uniprot-cache",
        default=None,
        type=Path,
        help="TSV cache with protein_id,sequence. Defaults to <output-dir>/raw/uniprot_reference_sequences.tsv",
    )
    parser.add_argument(
        "--split-csv",
        default=None,
        type=Path,
        help="Optional CSV with protein_id,split columns. Omit for deterministic 50/50 split.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--uniprot-batch-size", default=100, type=int)
    parser.add_argument("--request-timeout", default=60, type=int)
    parser.add_argument("--request-retries", default=3, type=int)
    parser.add_argument(
        "--skip-uniprot-fetch",
        action="store_true",
        help="Use only cached UniProt sequences and do not call the UniProt API",
    )
    parser.add_argument(
        "--max-sequence-length",
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        type=int,
        help=(
            "ESM2-ready sequence length. Use 0 to disable truncation; the existing "
            "ESM2 extraction script defaults to 1022."
        ),
    )
    parser.add_argument(
        "--sequence-truncation",
        choices=("ends_center",),
        default="ends_center",
        help="Truncation strategy matching scripts/extract_esm2_residue_embeddings.py",
    )
    parser.add_argument(
        "--strict-paper-counts",
        action="store_true",
        help="Fail if generated train/val/positive counts do not match the paper counts",
    )
    return parser.parse_args()


def canonical_protein_id(raw_id: str) -> str:
    token = raw_id.strip().split()[0]
    pieces = token.split("|")
    if len(pieces) >= 3 and pieces[1]:
        return pieces[1]
    return token.replace(".cif", "").replace(".pdb", "")


def normalize_sequence(sequence: str) -> str:
    sequence = re.sub(r"\s+", "", sequence.upper())
    sequence = re.sub(r"[UZOB]", "X", sequence)
    return AMINO_ACID_PATTERN.sub("X", sequence)


def retained_index_map(
    sequence_length: int,
    *,
    max_sequence_length: int,
    strategy: str = "ends_center",
) -> dict[int, int]:
    """Map original residue indices to indices retained after ESM2-style truncation."""
    if max_sequence_length <= 0 or sequence_length <= max_sequence_length:
        return {idx: idx for idx in range(sequence_length)}
    if strategy != "ends_center":
        raise ValueError(f"Unsupported truncation strategy: {strategy}")

    first_count = max_sequence_length // 4
    last_count = max_sequence_length // 4
    middle_count = max_sequence_length - first_count - last_count
    middle_start = max((sequence_length - middle_count) // 2, first_count)
    middle_end = min(middle_start + middle_count, sequence_length - last_count)
    middle_start = max(middle_end - middle_count, first_count)

    retained = (
        list(range(first_count))
        + list(range(middle_start, middle_end))
        + list(range(sequence_length - last_count, sequence_length))
    )
    return {original_idx: truncated_idx for truncated_idx, original_idx in enumerate(retained)}


def truncate_sequence(
    sequence: str,
    *,
    max_sequence_length: int,
    strategy: str = "ends_center",
) -> tuple[str, dict[int, int]]:
    index_map = retained_index_map(
        len(sequence),
        max_sequence_length=max_sequence_length,
        strategy=strategy,
    )
    if len(index_map) == len(sequence):
        return sequence, index_map
    retained = sorted(index_map, key=index_map.get)
    return "".join(sequence[idx] for idx in retained), index_map


def load_mcsa_reference_positives(path: Path) -> tuple[dict[str, set[int]], BuildStats]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level list in {path}")

    positives: dict[str, set[int]] = {}
    all_uniprot_ids: set[str] = set()
    empty_uniprot_entries = 0
    invalid_residue_entries = 0

    for record in payload:
        if not isinstance(record, dict):
            continue
        for seq_residue in record.get("residue_sequences", []):
            if not isinstance(seq_residue, dict):
                continue
            raw_uniprot_id = seq_residue.get("uniprot_id")
            if not raw_uniprot_id:
                empty_uniprot_entries += 1
                continue
            protein_id = canonical_protein_id(str(raw_uniprot_id))
            all_uniprot_ids.add(protein_id)
            if not seq_residue.get("is_reference", False):
                continue
            try:
                residue_index = int(seq_residue["resid"]) - 1
            except (KeyError, TypeError, ValueError):
                invalid_residue_entries += 1
                continue
            if residue_index < 0:
                invalid_residue_entries += 1
                continue
            positives.setdefault(protein_id, set()).add(residue_index)

    positive_count = sum(len(values) for values in positives.values())
    stats = BuildStats(
        raw_reference_proteins=len(positives),
        raw_reference_positive_residues=positive_count,
        raw_all_homologue_uniprot_ids=len(all_uniprot_ids),
        raw_empty_uniprot_entries=empty_uniprot_entries,
        raw_invalid_residue_entries=invalid_residue_entries,
    )
    return positives, stats


def read_sequence_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    sequences: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"protein_id", "sequence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            protein_id = canonical_protein_id(row["protein_id"])
            sequence = normalize_sequence(row["sequence"])
            if protein_id and sequence:
                sequences[protein_id] = sequence
    return sequences


def write_sequence_cache(path: Path, sequences: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "sequence"], delimiter="\t")
        writer.writeheader()
        for protein_id in sorted(sequences):
            writer.writerow({"protein_id": protein_id, "sequence": sequences[protein_id]})


def chunked(values: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [values[index : index + size] for index in range(0, len(values), size)]


def fetch_uniprot_batch(
    protein_ids: list[str],
    *,
    timeout: int,
    retries: int,
) -> dict[str, str]:
    if not protein_ids:
        return {}
    query = " OR ".join(f"accession:{protein_id}" for protein_id in protein_ids)
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
        {"query": query, "fields": "accession,sequence", "format": "tsv", "size": len(protein_ids)}
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                text = response.read().decode("utf-8")
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= retries:
                raise RuntimeError(f"UniProt batch request failed after {retries} attempts") from exc
            time.sleep(2**attempt)
    else:
        raise RuntimeError("UniProt batch request failed") from last_error

    sequences: dict[str, str] = {}
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    for row in reader:
        protein_id = canonical_protein_id(row.get("Entry", ""))
        sequence = normalize_sequence(row.get("Sequence", ""))
        if protein_id and sequence:
            sequences[protein_id] = sequence
    return sequences


def fetch_uniprot_sequences(
    protein_ids: set[str],
    *,
    cache_path: Path,
    batch_size: int,
    timeout: int,
    retries: int,
    skip_fetch: bool,
) -> tuple[dict[str, str], list[str]]:
    sequences = read_sequence_cache(cache_path)
    missing = sorted(protein_ids - set(sequences))
    if missing and not skip_fetch:
        for batch_number, batch in enumerate(chunked(missing, batch_size), start=1):
            fetched = fetch_uniprot_batch(batch, timeout=timeout, retries=retries)
            sequences.update(fetched)
            print(
                f"Fetched UniProt batch {batch_number}: "
                f"{len(fetched)}/{len(batch)} sequences",
                flush=True,
            )
        write_sequence_cache(cache_path, sequences)
    missing_after_fetch = sorted(protein_ids - set(sequences))
    return {protein_id: sequences[protein_id] for protein_id in protein_ids if protein_id in sequences}, missing_after_fetch


def load_split_csv(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    split_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"protein_id", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            protein_id = canonical_protein_id(row["protein_id"])
            split = row["split"].strip().lower()
            if split == "valid":
                split = "val"
            if split not in {"train", "val"}:
                raise ValueError(f"Unsupported split '{row['split']}' for {protein_id}")
            split_map[protein_id] = split
    return split_map


def deterministic_split(protein_ids: list[str], *, seed: int) -> dict[str, str]:
    ids = sorted(set(protein_ids))
    train_count = max(1, int(round(0.5 * len(ids))))
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    train_ids = set(shuffled[:train_count])
    return {protein_id: "train" if protein_id in train_ids else "val" for protein_id in ids}


def write_fasta(records: list[FastaRecord], path: Path, *, line_width: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.protein_id}\n")
            sequence = record.sequence
            for index in range(0, len(sequence), line_width):
                handle.write(sequence[index : index + line_width] + "\n")


def curated_uniprot_counts(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    raw_tokens: set[str] = set()
    split_ids: set[str] = set()
    complex_tokens = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = (row.get("Uniprot IDs", "") or "").strip()
            for token in re.split(r"[;,\s]+", value):
                if not token:
                    continue
                raw_tokens.add(token)
                parts = [part for part in token.split(":") if part]
                if len(parts) > 1:
                    complex_tokens += 1
                split_ids.update(parts)
    return {
        "raw_uniprot_tokens": len(raw_tokens),
        "colon_split_uniprot_ids": len(split_ids),
        "complex_tokens": complex_tokens,
    }


def write_missing_tsv(path: Path, missing: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["protein_id", "reason", "positive_residue_count", "details"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in missing:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_labels(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["protein_id", "residue_index", "label", "split", "source", "entropy"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    fasta_dir = output_dir / "fasta"
    manifest_dir = output_dir / "manifests"
    cache_path = args.uniprot_cache or raw_dir / "uniprot_reference_sequences.tsv"

    positives_by_protein, raw_stats = load_mcsa_reference_positives(args.mcsa_json)
    protein_ids = set(positives_by_protein)
    sequences, missing_sequences = fetch_uniprot_sequences(
        protein_ids,
        cache_path=cache_path,
        batch_size=args.uniprot_batch_size,
        timeout=args.request_timeout,
        retries=args.request_retries,
        skip_fetch=args.skip_uniprot_fetch,
    )

    split_map = load_split_csv(args.split_csv)
    split_source = (
        "provided_csv" if split_map else f"deterministic_50_50_seed_{args.seed}"
    )
    if not split_map:
        split_map = deterministic_split(sorted(sequences), seed=args.seed)
    else:
        missing_split = sorted(set(sequences) - set(split_map))
        if missing_split:
            raise ValueError(
                f"{args.split_csv} does not assign {len(missing_split)} fetched proteins; "
                f"first missing IDs: {missing_split[:10]}"
            )

    full_fasta_records: list[FastaRecord] = []
    esm_ready_fasta_records: list[FastaRecord] = []
    label_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    dropped_out_of_range = 0
    dropped_by_truncation = 0
    retained_positive_count = 0

    for protein_id in sorted(protein_ids):
        protein_positives = positives_by_protein[protein_id]
        if protein_id not in sequences:
            missing_rows.append(
                {
                    "protein_id": protein_id,
                    "reason": "missing_uniprot_sequence",
                    "positive_residue_count": len(protein_positives),
                }
            )
            continue

        full_sequence = sequences[protein_id]
        full_fasta_records.append(FastaRecord(protein_id, full_sequence))
        truncated_sequence, index_map = truncate_sequence(
            full_sequence,
            max_sequence_length=args.max_sequence_length,
            strategy=args.sequence_truncation,
        )
        esm_ready_fasta_records.append(FastaRecord(protein_id, truncated_sequence))

        valid_positive_indices = {
            residue_index
            for residue_index in protein_positives
            if 0 <= residue_index < len(full_sequence)
        }
        dropped_out_of_range += len(protein_positives) - len(valid_positive_indices)
        retained_positives = {
            index_map[residue_index]
            for residue_index in valid_positive_indices
            if residue_index in index_map
        }
        dropped_by_truncation += len(valid_positive_indices) - len(retained_positives)
        retained_positive_count += len(retained_positives)
        split = split_map.get(protein_id, "")
        for residue_index in range(len(truncated_sequence)):
            label_rows.append(
                {
                    "protein_id": protein_id,
                    "residue_index": residue_index,
                    "label": 1 if residue_index in retained_positives else 0,
                    "split": split,
                    "source": "mcsa_reference",
                    "entropy": "",
                }
            )

    write_fasta(full_fasta_records, fasta_dir / "mcsa_reference.full.fasta")
    write_fasta(esm_ready_fasta_records, fasta_dir / "mcsa_reference.fasta")
    write_labels(output_dir / "mcsa_residue_labels.csv", label_rows)
    write_missing_tsv(manifest_dir / "mcsa_missing_or_invalid.tsv", missing_rows)

    train_proteins = {row["protein_id"] for row in label_rows if row["split"] == "train"}
    val_proteins = {row["protein_id"] for row in label_rows if row["split"] == "val"}
    train_positives = sum(1 for row in label_rows if row["split"] == "train" and row["label"] == 1)
    val_positives = sum(1 for row in label_rows if row["split"] == "val" and row["label"] == 1)
    positive_count = sum(1 for row in label_rows if row["label"] == 1)
    negative_count = len(label_rows) - positive_count
    paper_counts_match = (
        len(train_proteins) == PAPER_MCSA_TRAIN_PROTEINS
        and len(val_proteins) == PAPER_MCSA_VAL_PROTEINS
        and positive_count == PAPER_MCSA_FUNCTIONAL_RESIDUES
    )
    if args.strict_paper_counts and not paper_counts_match:
        raise ValueError(
            "Generated mCSA counts do not match paper target: "
            f"train={len(train_proteins)}, val={len(val_proteins)}, positives={positive_count}"
        )

    manifest = {
        "created_unix_time": time.time(),
        "stage": "SLEEC stage 1 mCSA supervised residue labels",
        "source_files": {
            "mcsa_json": str(args.mcsa_json),
            "curated_csv": str(args.curated_csv) if args.curated_csv else None,
            "split_csv": str(args.split_csv) if args.split_csv else None,
            "uniprot_cache": str(cache_path),
        },
        "outputs": {
            "full_fasta": str(fasta_dir / "mcsa_reference.full.fasta"),
            "esm_ready_fasta": str(fasta_dir / "mcsa_reference.fasta"),
            "labels_csv": str(output_dir / "mcsa_residue_labels.csv"),
            "missing_tsv": str(manifest_dir / "mcsa_missing_or_invalid.tsv"),
        },
        "paper_expected": {
            "train_proteins": PAPER_MCSA_TRAIN_PROTEINS,
            "val_proteins": PAPER_MCSA_VAL_PROTEINS,
            "functional_residues": PAPER_MCSA_FUNCTIONAL_RESIDUES,
        },
        "raw_mcsa": {
            "reference_proteins": raw_stats.raw_reference_proteins,
            "reference_positive_residues": raw_stats.raw_reference_positive_residues,
            "all_homologue_uniprot_ids": raw_stats.raw_all_homologue_uniprot_ids,
            "empty_uniprot_entries": raw_stats.raw_empty_uniprot_entries,
            "invalid_residue_entries": raw_stats.raw_invalid_residue_entries,
            "curated_csv_counts": curated_uniprot_counts(args.curated_csv),
        },
        "sequence_processing": {
            "max_sequence_length": args.max_sequence_length,
            "sequence_truncation": args.sequence_truncation,
            "full_sequence_proteins": len(full_fasta_records),
            "missing_sequence_proteins": len(missing_sequences),
            "dropped_positive_residues_out_of_range": dropped_out_of_range,
            "dropped_positive_residues_by_truncation": dropped_by_truncation,
        },
        "generated": {
            "split_source": split_source,
            "seed": args.seed,
            "train_proteins": len(train_proteins),
            "val_proteins": len(val_proteins),
            "total_proteins": len(train_proteins | val_proteins),
            "train_positive_residues": train_positives,
            "val_positive_residues": val_positives,
            "positive_residues": retained_positive_count,
            "negative_residues": negative_count,
            "total_residue_labels": len(label_rows),
            "paper_counts_match": paper_counts_match,
        },
    }
    manifest_path = manifest_dir / "mcsa_stage1_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_outputs(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
