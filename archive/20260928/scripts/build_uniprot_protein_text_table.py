#!/usr/bin/env python3
"""Build a protein text table from source-collapsed training pairs and UniProt."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Iterable

import requests


DEFAULT_FIELDS = "accession,id,protein_name,organism_name,ec,cc_function"
UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:[A-Z0-9]{4})?|[OPQ][0-9][A-Z0-9]{3}[0-9])(?:-\d+)?$"
)
EVIDENCE_RE = re.compile(r"\s*\{[^{}]*\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs",
        default="data/standardized/retrieval_training_source_collapse/train_exact/train_pairs_valid_rxn_pseudo_nonempty_prott5.csv",
        help="Training pairs CSV with protein_id and source_entries columns.",
    )
    parser.add_argument(
        "--candidate-ids",
        default="data/standardized/retrieval_training_source_collapse/train_exact/candidate_ids.txt",
        help="Optional candidate ID list used for output ordering and missing-count metadata.",
    )
    parser.add_argument(
        "--output",
        default="data/standardized/retrieval_training_source_collapse/train_exact/fit_proteins_with_clipzyme_eval_pubmedbert_text_source.csv",
        help="Output CSV with protein_id,text columns.",
    )
    parser.add_argument(
        "--cache-tsv",
        default=None,
        help="UniProt TSV cache path. Defaults to <output>.uniprot.tsv.",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional metadata JSON path. Defaults to <output>.metadata.json.",
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
        help="Maximum annotated accessions to concatenate for a source-collapsed protein.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Use an existing UniProt TSV cache instead of downloading it.",
    )
    return parser.parse_args()


def valid_uniprot_accession(value: str) -> bool:
    return bool(UNIPROT_ACCESSION_RE.match(value)) and not value.startswith("prot_")


def parse_pair_accessions(path: Path) -> dict[str, set[str]]:
    protein_to_accessions: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"protein_id", "source_entries"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            protein_id = str(row["protein_id"]).strip()
            if not protein_id:
                continue
            for entry in str(row.get("source_entries") or "").split(";"):
                parts = entry.split(":")
                if len(parts) < 3:
                    continue
                accession = parts[2].strip()
                if valid_uniprot_accession(accession):
                    protein_to_accessions.setdefault(protein_id, set()).add(accession)
                else:
                    protein_to_accessions.setdefault(protein_id, set())
    return protein_to_accessions


def read_candidate_ids(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_uniprot_batch(
    session: requests.Session,
    accessions: list[str],
    *,
    fields: str,
    timeout: float,
    retries: int,
) -> str:
    url = "https://rest.uniprot.org/uniprotkb/accessions"
    params = {
        "accessions": ",".join(accessions),
        "fields": fields,
        "format": "tsv",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"UniProt request failed for batch starting {accessions[0]}") from last_error


def write_uniprot_cache(
    accessions: list[str],
    cache_path: Path,
    *,
    fields: str,
    batch_size: int,
    sleep: float,
    timeout: float,
    retries: int,
) -> tuple[int, int]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    wrote_header = False
    rows_written = 0
    batches_written = 0
    with cache_path.open("w", newline="", encoding="utf-8") as output:
        for batch in chunks(accessions, batch_size):
            text = fetch_uniprot_batch(
                session,
                batch,
                fields=fields,
                timeout=timeout,
                retries=retries,
            )
            lines = [line for line in text.splitlines() if line.strip()]
            if not lines:
                continue
            if not wrote_header:
                output.write(lines[0] + "\n")
                wrote_header = True
            for line in lines[1:]:
                output.write(line + "\n")
                rows_written += 1
            batches_written += 1
            if batches_written % 25 == 0:
                print(
                    f"Fetched {min(batches_written * batch_size, len(accessions)):,}/"
                    f"{len(accessions):,} accessions; rows={rows_written:,}",
                    flush=True,
                )
            if sleep > 0:
                time.sleep(sleep)
    return rows_written, batches_written


def read_uniprot_cache(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "Entry" not in reader.fieldnames:
            raise ValueError(f"UniProt cache has no Entry column: {path}")
        return {str(row["Entry"]).strip(): row for row in reader if str(row.get("Entry", "")).strip()}


def clean_field(value: str) -> str:
    value = EVIDENCE_RE.sub("", value or "")
    value = value.replace("FUNCTION:", "")
    value = " ".join(value.split())
    return value.strip(" ;")


def row_to_text(row: dict[str, str]) -> str:
    protein_names = clean_field(row.get("Protein names", ""))
    entry_name = clean_field(row.get("Entry Name", ""))
    organism = clean_field(row.get("Organism", ""))
    ec_number = clean_field(row.get("EC number", ""))
    function = clean_field(row.get("Function [CC]", ""))

    parts: list[str] = []
    if protein_names:
        parts.append(f"Protein names: {protein_names}.")
    if entry_name:
        parts.append(f"UniProt entry name: {entry_name}.")
    if organism:
        parts.append(f"Organism: {organism}.")
    if ec_number:
        parts.append(f"EC number: {ec_number}.")
    if function:
        parts.append(f"Function: {function}.")
    return " ".join(parts)


def build_text_rows(
    protein_to_accessions: dict[str, set[str]],
    annotations: dict[str, dict[str, str]],
    candidate_ids: list[str],
    *,
    max_accessions_per_protein: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    output_order = candidate_ids or sorted(protein_to_accessions)
    rows: list[dict[str, str]] = []
    missing_accession = 0
    missing_annotation = 0
    multi_accession = 0
    for protein_id in output_order:
        accessions = sorted(protein_to_accessions.get(protein_id, set()))
        if not accessions:
            missing_accession += 1
            continue
        if len(accessions) > 1:
            multi_accession += 1
        texts: list[str] = []
        used_accessions: list[str] = []
        for accession in accessions:
            annotation = annotations.get(accession)
            if annotation is None:
                continue
            text = row_to_text(annotation)
            if text:
                texts.append(text)
                used_accessions.append(accession)
            if len(texts) >= max_accessions_per_protein:
                break
        if not texts:
            missing_annotation += 1
            continue
        rows.append(
            {
                "protein_id": protein_id,
                "accessions": ";".join(used_accessions),
                "text": " ".join(texts),
            }
        )
    stats = {
        "candidate_ids": len(output_order),
        "proteins_with_source_accessions": sum(1 for values in protein_to_accessions.values() if values),
        "rows_written": len(rows),
        "missing_accession": missing_accession,
        "missing_annotation": missing_annotation,
        "multi_accession_proteins": multi_accession,
    }
    return rows, stats


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_accessions_per_protein <= 0:
        raise ValueError("--max-accessions-per-protein must be positive")

    pairs_path = Path(args.pairs)
    candidate_ids_path = Path(args.candidate_ids) if args.candidate_ids else None
    output_path = Path(args.output)
    cache_path = Path(args.cache_tsv) if args.cache_tsv else output_path.with_suffix(".uniprot.tsv")
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output_path.with_suffix(".metadata.json")
    )

    protein_to_accessions = parse_pair_accessions(pairs_path)
    candidate_ids = read_candidate_ids(candidate_ids_path)
    accessions = sorted({accession for values in protein_to_accessions.values() for accession in values})
    print(
        f"Parsed {len(protein_to_accessions):,} proteins and {len(accessions):,} unique UniProt accessions",
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
    rows, stats = build_text_rows(
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
        "pairs": str(pairs_path),
        "candidate_ids": None if candidate_ids_path is None else str(candidate_ids_path),
        "output": str(output_path),
        "cache_tsv": str(cache_path),
        "metadata_output": str(metadata_path),
        "uniprot_endpoint": "https://rest.uniprot.org/uniprotkb/accessions",
        "fields": args.fields,
        "batch_size": args.batch_size,
        "unique_accessions": len(accessions),
        "annotations_found": len(annotations),
        "cache_rows_downloaded": cache_rows,
        "cache_batches_downloaded": cache_batches,
        **stats,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {len(rows):,} protein text rows to {output_path}", flush=True)
    print(f"Wrote metadata to {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
