#!/usr/bin/env python3
"""Build UniProt text rows for held-out benchmark candidate proteins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.build_uniprot_protein_text_table import (  # noqa: E402
    DEFAULT_FIELDS,
    build_text_rows,
    read_uniprot_cache,
    valid_uniprot_accession,
    write_uniprot_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidate-ids",
        default=(
            "data/standardized/retrieval_training_source_collapse/test/"
            "horizyn_reactzyme_shared_candidates/candidate_ids.txt"
        ),
        help="Candidate protein ID list, one ID per line.",
    )
    parser.add_argument(
        "--reactzyme-uniprot-tsv",
        default="data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv",
        help="ReactZyme UniProt/Rhea TSV with Entry and Sequence columns.",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/standardized/retrieval_training_source_collapse/test/"
            "horizyn_reactzyme_shared_candidates/proteins_pubmedbert_text_source.csv"
        ),
        help="Output CSV with protein_id, accessions, text columns.",
    )
    parser.add_argument(
        "--cache-tsv",
        default=None,
        help="UniProt TSV cache path. Defaults to <output>.uniprot.tsv.",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Metadata JSON path. Defaults to <output>.metadata.json.",
    )
    parser.add_argument("--fields", default=DEFAULT_FIELDS, help="UniProt return fields.")
    parser.add_argument("--batch-size", type=int, default=300, help="UniProt accessions per request.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per UniProt request.")
    parser.add_argument(
        "--max-accessions-per-protein",
        type=int,
        default=3,
        help="Maximum annotated accessions to concatenate per candidate.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse an existing UniProt TSV cache instead of downloading.",
    )
    return parser.parse_args()


def read_candidate_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def reactzyme_hash(sequence: str) -> str:
    return "prot_" + hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]


def load_reactzyme_hash_map(path: Path) -> dict[str, set[str]]:
    """Return mapping from ReactZyme prot_* sequence hashes to UniProt entries."""

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
            if not entry or not sequence or not valid_uniprot_accession(entry):
                continue
            mapping.setdefault(reactzyme_hash(sequence), set()).add(entry)
    return mapping


def build_candidate_accession_map(
    candidate_ids: list[str],
    reactzyme_mapping: dict[str, set[str]],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    protein_to_accessions: dict[str, set[str]] = {}
    direct_uniprot = 0
    hashed_mapped = 0
    missing = 0
    multi_accession_hash = 0
    for protein_id in candidate_ids:
        if valid_uniprot_accession(protein_id):
            protein_to_accessions[protein_id] = {protein_id.split("-")[0]}
            direct_uniprot += 1
            continue
        accessions = set(reactzyme_mapping.get(protein_id, set()))
        protein_to_accessions[protein_id] = accessions
        if accessions:
            hashed_mapped += 1
            if len(accessions) > 1:
                multi_accession_hash += 1
        else:
            missing += 1
    return protein_to_accessions, {
        "direct_uniprot_candidate_ids": direct_uniprot,
        "hashed_candidate_ids_mapped": hashed_mapped,
        "candidate_ids_without_accession": missing,
        "hashed_candidate_ids_with_multiple_accessions": multi_accession_hash,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_accessions_per_protein <= 0:
        raise ValueError("--max-accessions-per-protein must be positive")

    candidate_ids_path = Path(args.candidate_ids)
    reactzyme_tsv_path = Path(args.reactzyme_uniprot_tsv)
    output_path = Path(args.output)
    cache_path = Path(args.cache_tsv) if args.cache_tsv else output_path.with_suffix(".uniprot.tsv")
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output_path.with_suffix(".metadata.json")
    )

    candidate_ids = read_candidate_ids(candidate_ids_path)
    print(f"Loaded {len(candidate_ids):,} candidate IDs", flush=True)
    reactzyme_mapping = load_reactzyme_hash_map(reactzyme_tsv_path)
    print(f"Loaded {len(reactzyme_mapping):,} ReactZyme sequence-hash mappings", flush=True)
    protein_to_accessions, accession_stats = build_candidate_accession_map(
        candidate_ids,
        reactzyme_mapping,
    )
    accessions = sorted({acc for values in protein_to_accessions.values() for acc in values})
    print(
        f"Mapped candidates to {len(accessions):,} unique UniProt accessions "
        f"({accession_stats})",
        flush=True,
    )

    if args.reuse_cache and cache_path.exists():
        print(f"Reusing UniProt cache: {cache_path}", flush=True)
        cache_rows = None
        cache_batches = None
    else:
        print(f"Downloading UniProt annotations to {cache_path}", flush=True)
        cache_rows, cache_batches = write_uniprot_cache(
            accessions,
            cache_path,
            fields=args.fields,
            batch_size=args.batch_size,
            sleep=args.sleep,
            timeout=args.timeout,
            retries=args.retries,
        )

    annotations = read_uniprot_cache(cache_path)
    rows, row_stats = build_text_rows(
        protein_to_accessions,
        annotations,
        candidate_ids,
        max_accessions_per_protein=args.max_accessions_per_protein,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "accessions", "text"])
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "candidate_ids": str(candidate_ids_path),
        "reactzyme_uniprot_tsv": str(reactzyme_tsv_path),
        "output": str(output_path),
        "cache_tsv": str(cache_path),
        "metadata_output": str(metadata_path),
        "uniprot_endpoint": "https://rest.uniprot.org/uniprotkb/accessions",
        "fields": args.fields,
        "batch_size": int(args.batch_size),
        "unique_accessions": len(accessions),
        "annotations_found": len(annotations),
        "cache_rows_downloaded": cache_rows,
        "cache_batches_downloaded": cache_batches,
        **accession_stats,
        **row_stats,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {len(rows):,} protein text rows to {output_path}", flush=True)
    print(f"Wrote metadata to {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
