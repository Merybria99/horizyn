#!/usr/bin/env python3
"""Download and normalize the complete reviewed UniProtKB/Swiss-Prot FASTA."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from wet_lab.query import PROJECT_ROOT, _resolve_path


SWISSPROT_FASTA_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/uniprot_sprot.fasta.gz"
)
UNIPROT_RELEASE_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/reldate.txt"
)
SCHEMA_VERSION = "horizyn_wet_lab_swissprot_v1"
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    *,
    force: bool,
    retries: int = 5,
) -> Path:
    """Download a URL atomically with retries."""

    if destination.is_file() and not force:
        print(f"Reusing downloaded file: {destination}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Horizyn-WetLab/1.0"})
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url} (attempt {attempt}/{retries})...", flush=True)
            with urllib.request.urlopen(request, timeout=120) as response:
                with partial.open("wb") as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
            partial.replace(destination)
            return destination
        except (OSError, urllib.error.URLError) as error:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {error}") from error
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def _parse_header(header: str) -> tuple[str, str]:
    token = header.split(maxsplit=1)[0]
    fields = token.split("|")
    if len(fields) != 3 or fields[0] != "sp" or not fields[1] or not fields[2]:
        raise ValueError(f"Expected a Swiss-Prot FASTA header, got: >{header}")
    return fields[1], fields[2]


def _write_record(
    fasta_handle: TextIO,
    id_handle: TextIO,
    metadata_writer: csv.writer,
    *,
    accession: str,
    entry_name: str,
    sequence: str,
    seen: set[str],
    residue_counts: Counter[str],
) -> int:
    sequence = "".join(sequence.split()).upper()
    if not sequence:
        raise ValueError(f"Swiss-Prot entry {accession} has an empty sequence")
    if accession in seen:
        raise ValueError(f"Duplicate Swiss-Prot accession: {accession}")
    seen.add(accession)
    residue_counts.update(sequence)
    fasta_handle.write(f">{accession} swissprot_entry={entry_name}\n")
    for start in range(0, len(sequence), 80):
        fasta_handle.write(sequence[start : start + 80] + "\n")
    id_handle.write(accession + "\n")
    metadata_writer.writerow([accession, entry_name, len(sequence)])
    return len(sequence)


def normalize_swissprot_fasta(
    compressed_fasta: Path,
    output_fasta: Path,
    candidate_ids: Path,
    metadata_csv: Path,
) -> dict[str, object]:
    """Normalize official Swiss-Prot headers to unique canonical accessions."""

    for path in (output_fasta, candidate_ids, metadata_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    partial_fasta = output_fasta.with_suffix(output_fasta.suffix + ".partial")
    partial_ids = candidate_ids.with_suffix(candidate_ids.suffix + ".partial")
    partial_metadata = metadata_csv.with_suffix(metadata_csv.suffix + ".partial")
    for path in (partial_fasta, partial_ids, partial_metadata):
        path.unlink(missing_ok=True)

    seen: set[str] = set()
    residue_counts: Counter[str] = Counter()
    sequence_lengths: list[int] = []
    accession: str | None = None
    entry_name: str | None = None
    chunks: list[str] = []
    try:
        with (
            gzip.open(compressed_fasta, "rt", encoding="utf-8") as source,
            partial_fasta.open("w", encoding="utf-8") as fasta_handle,
            partial_ids.open("w", encoding="utf-8") as id_handle,
            partial_metadata.open("w", encoding="utf-8", newline="") as metadata_handle,
        ):
            metadata_writer = csv.writer(metadata_handle)
            metadata_writer.writerow(["protein_id", "entry_name", "sequence_length"])
            for raw_line in source:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if accession is not None and entry_name is not None:
                        sequence_lengths.append(
                            _write_record(
                                fasta_handle,
                                id_handle,
                                metadata_writer,
                                accession=accession,
                                entry_name=entry_name,
                                sequence="".join(chunks),
                                seen=seen,
                                residue_counts=residue_counts,
                            )
                        )
                    accession, entry_name = _parse_header(line[1:])
                    chunks = []
                else:
                    if accession is None:
                        raise ValueError("Swiss-Prot FASTA starts with sequence data")
                    chunks.append(line)
            if accession is not None and entry_name is not None:
                sequence_lengths.append(
                    _write_record(
                        fasta_handle,
                        id_handle,
                        metadata_writer,
                        accession=accession,
                        entry_name=entry_name,
                        sequence="".join(chunks),
                        seen=seen,
                        residue_counts=residue_counts,
                    )
                )
        if not sequence_lengths:
            raise ValueError(f"No Swiss-Prot entries found in {compressed_fasta}")
        partial_fasta.replace(output_fasta)
        partial_ids.replace(candidate_ids)
        partial_metadata.replace(metadata_csv)
    except Exception:
        for path in (partial_fasta, partial_ids, partial_metadata):
            path.unlink(missing_ok=True)
        raise

    nonstandard = {
        residue: count
        for residue, count in sorted(residue_counts.items())
        if residue not in STANDARD_AA
    }
    return {
        "protein_count": len(sequence_lengths),
        "total_residues": sum(sequence_lengths),
        "minimum_sequence_length": min(sequence_lengths),
        "maximum_sequence_length": max(sequence_lengths),
        "nonstandard_residue_counts": nonstandard,
    }


def prepare_swissprot(
    output_dir: str | Path,
    *,
    force_download: bool = False,
    force_normalize: bool = False,
) -> Path:
    output = _resolve_path(output_dir, must_exist=False)
    output.mkdir(parents=True, exist_ok=True)
    compressed_fasta = output / "uniprot_sprot.fasta.gz"
    release_path = output / "reldate.txt"
    normalized_fasta = output / "proteins.fasta"
    candidate_ids = output / "candidate_ids.txt"
    metadata_csv = output / "proteins.csv"
    manifest_path = output / "manifest.json"

    download_file(SWISSPROT_FASTA_URL, compressed_fasta, force=force_download)
    download_file(UNIPROT_RELEASE_URL, release_path, force=force_download)
    if (
        manifest_path.is_file()
        and normalized_fasta.is_file()
        and candidate_ids.is_file()
        and metadata_csv.is_file()
        and not force_normalize
        and not force_download
    ):
        print(f"Reusing normalized Swiss-Prot database: {manifest_path}", flush=True)
        return manifest_path

    print("Normalizing Swiss-Prot accessions and sequences...", flush=True)
    statistics = normalize_swissprot_fasta(
        compressed_fasta,
        normalized_fasta,
        candidate_ids,
        metadata_csv,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "fasta_url": SWISSPROT_FASTA_URL,
            "release_url": UNIPROT_RELEASE_URL,
            "release": release_path.read_text(encoding="utf-8").strip(),
            "canonical_sequences_only": True,
            "reviewed_only": True,
        },
        "files": {
            "compressed_fasta": str(compressed_fasta),
            "normalized_fasta": str(normalized_fasta),
            "candidate_ids": str(candidate_ids),
            "metadata_csv": str(metadata_csv),
        },
        "sha256": {
            "compressed_fasta": _sha256(compressed_fasta),
            "normalized_fasta": _sha256(normalized_fasta),
            "candidate_ids": _sha256(candidate_ids),
        },
        "statistics": statistics,
        "prott5": {
            "required_output": str(output / "proteins_prott5_residue.h5"),
            "model": "Rostlab/prot_t5_xl_half_uniref50-enc",
            "max_sequence_length": 1024,
            "sequence_truncation": "ends_center",
            "stored_dtype": "float16",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Prepared {statistics['protein_count']:,} reviewed Swiss-Prot proteins: "
        f"{manifest_path}",
        flush=True,
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="wet_lab/databases/swissprot/current",
        help="Repository-local Swiss-Prot database directory",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-normalize", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    prepare_swissprot(
        args.output_dir,
        force_download=args.force_download,
        force_normalize=args.force_normalize,
    )


if __name__ == "__main__":
    main()
