#!/usr/bin/env python3
"""Download and normalize a bounded NCBI RefSeq prokaryotic protein collection."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TextIO

from wet_lab.query import PROJECT_ROOT, _resolve_path


NCBI_REFSEQ_RELEASE_ROOT = "https://ftp.ncbi.nlm.nih.gov/refseq/release"
RELEASE_CATALOG_INDEX_URL = f"{NCBI_REFSEQ_RELEASE_ROOT}/release-catalog/"
RELEASE_NOTES_ROOT = f"{NCBI_REFSEQ_RELEASE_ROOT}/release-notes"
SUPPORTED_DIVISIONS = ("archaea", "bacteria")
SCHEMA_VERSION = "horizyn_wet_lab_refseq_prokaryotic_v1"
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
REFSEQ_ACCESSION = re.compile(r"^[A-Z]{2}_[A-Z0-9]+\.\d+$")
RELEASE_CATALOG_NAME = re.compile(r"release(?P<release>\d+)\.files\.installed$")
WP_FASTA_NAME = re.compile(
    r"^(?P<division>archaea|bacteria)\.wp_protein\." r"(?P<number>\d+)\.protein\.faa\.gz$"
)


@dataclass(frozen=True)
class RefSeqReleaseFile:
    division: str
    number: int
    filename: str
    md5: str
    url: str


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(
    url: str,
    destination: Path,
    *,
    force: bool,
    expected_md5: str | None = None,
    retries: int = 5,
) -> Path:
    """Download one file atomically and verify its NCBI catalog checksum."""

    if destination.is_file() and not force:
        if expected_md5 and _hash_file(destination, "md5") != expected_md5:
            raise RuntimeError(
                f"Existing download failed its MD5 checksum: {destination}. "
                "Use --force-download to replace it."
            )
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
            if expected_md5:
                observed_md5 = _hash_file(partial, "md5")
                if observed_md5 != expected_md5:
                    raise OSError(
                        f"MD5 mismatch for {url}: expected {expected_md5}, "
                        f"observed {observed_md5}"
                    )
            partial.replace(destination)
            return destination
        except (OSError, urllib.error.URLError) as error:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {error}") from error
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def discover_current_release(index_text: str) -> int:
    """Extract the newest RefSeq release number from the FTP catalog index."""

    releases = {
        int(match.group("release"))
        for token in re.findall(r'href=["\']([^"\']+)["\']', index_text)
        if (match := RELEASE_CATALOG_NAME.fullmatch(Path(token).name))
    }
    if not releases:
        raise ValueError("Could not discover an NCBI RefSeq release catalog")
    return max(releases)


def parse_release_files(
    catalog_text: str,
    *,
    divisions: Iterable[str],
    max_files_per_division: int | None,
) -> list[RefSeqReleaseFile]:
    """Select numerically ordered prokaryotic non-redundant protein FASTAs."""

    requested = tuple(dict.fromkeys(str(value).strip().lower() for value in divisions))
    invalid = sorted(set(requested) - set(SUPPORTED_DIVISIONS))
    if not requested:
        raise ValueError("At least one RefSeq division is required")
    if invalid:
        raise ValueError(f"Unsupported RefSeq division(s): {', '.join(invalid)}")
    if max_files_per_division is not None and max_files_per_division <= 0:
        raise ValueError("max_files_per_division must be positive or None")

    by_division: dict[str, list[RefSeqReleaseFile]] = {division: [] for division in requested}
    for raw_line in catalog_text.splitlines():
        fields = raw_line.split()
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{32}", fields[0]):
            continue
        filename = Path(fields[1]).name
        match = WP_FASTA_NAME.fullmatch(filename)
        if match is None:
            continue
        division = match.group("division")
        if division not in by_division:
            continue
        by_division[division].append(
            RefSeqReleaseFile(
                division=division,
                number=int(match.group("number")),
                filename=filename,
                md5=fields[0].lower(),
                url=f"{NCBI_REFSEQ_RELEASE_ROOT}/{division}/{filename}",
            )
        )

    selected: list[RefSeqReleaseFile] = []
    for division in requested:
        files = sorted(by_division[division], key=lambda item: item.number)
        if not files:
            raise ValueError(f"Release catalog contains no WP protein FASTAs for {division}")
        if max_files_per_division is not None:
            files = files[:max_files_per_division]
        selected.extend(files)
    return selected


def parse_refseq_header(header: str) -> tuple[str, str, str]:
    """Return versioned accession, protein description, and terminal organism."""

    accession, separator, remainder = header.strip().partition(" ")
    if not REFSEQ_ACCESSION.fullmatch(accession):
        raise ValueError(f"Expected a versioned RefSeq protein accession, got: >{header}")
    description = remainder.strip() if separator else ""
    organism = ""
    organism_match = re.search(r"\s+\[([^\[\]]+)\]\s*$", description)
    if organism_match:
        organism = organism_match.group(1).strip()
        description = description[: organism_match.start()].strip()
    return accession, description, organism


def _iter_fasta(path: Path) -> Iterable[tuple[str, str]]:
    header: str | None = None
    chunks: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                if header is None:
                    raise ValueError(f"RefSeq FASTA starts with sequence data: {path}")
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def _write_fasta_record(handle: TextIO, accession: str, sequence: str) -> None:
    handle.write(f">{accession}\n")
    for start in range(0, len(sequence), 80):
        handle.write(sequence[start : start + 80] + "\n")


def normalize_refseq_fastas(
    compressed_fastas: Iterable[Path],
    output_fasta: Path,
    candidate_ids: Path,
    metadata_csv: Path,
    *,
    min_length: int = 0,
    max_length: int | None = None,
) -> dict[str, object]:
    """Normalize RefSeq FASTAs to IDs shared by FASTA, metadata, and HDF5 keys."""

    source_paths = [Path(path) for path in compressed_fastas]
    if not source_paths:
        raise ValueError("At least one RefSeq FASTA is required")
    if min_length < 0:
        raise ValueError("min_length cannot be negative")
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive or None")
    if max_length is not None and max_length < min_length:
        raise ValueError("max_length cannot be smaller than min_length")

    for path in (output_fasta, candidate_ids, metadata_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    partial_fasta = output_fasta.with_suffix(output_fasta.suffix + ".partial")
    partial_ids = candidate_ids.with_suffix(candidate_ids.suffix + ".partial")
    partial_metadata = metadata_csv.with_suffix(metadata_csv.suffix + ".partial")
    partial_paths = (partial_fasta, partial_ids, partial_metadata)
    for path in partial_paths:
        path.unlink(missing_ok=True)

    seen: dict[str, bytes] = {}
    residue_counts: Counter[str] = Counter()
    sequence_lengths: list[int] = []
    read_count = 0
    duplicate_count = 0
    below_minimum = 0
    above_maximum = 0
    try:
        with (
            partial_fasta.open("w", encoding="utf-8") as fasta_handle,
            partial_ids.open("w", encoding="utf-8") as id_handle,
            partial_metadata.open("w", encoding="utf-8", newline="") as metadata_handle,
        ):
            metadata_writer = csv.writer(metadata_handle)
            metadata_writer.writerow(
                [
                    "protein_id",
                    "accession",
                    "description",
                    "organism",
                    "sequence_length",
                    "refseq_division",
                    "source_file",
                ]
            )
            for source_path in source_paths:
                if not source_path.is_file():
                    raise FileNotFoundError(f"RefSeq source FASTA does not exist: {source_path}")
                source_match = WP_FASTA_NAME.fullmatch(source_path.name)
                division = source_match.group("division") if source_match else ""
                for header, raw_sequence in _iter_fasta(source_path):
                    read_count += 1
                    accession, description, organism = parse_refseq_header(header)
                    sequence = "".join(raw_sequence.split()).upper()
                    if not sequence:
                        raise ValueError(f"RefSeq entry {accession} has an empty sequence")
                    sequence_digest = hashlib.sha256(sequence.encode("ascii")).digest()
                    previous_digest = seen.get(accession)
                    if previous_digest is not None:
                        if previous_digest != sequence_digest:
                            raise ValueError(
                                f"RefSeq accession {accession} has conflicting sequences"
                            )
                        duplicate_count += 1
                        continue
                    seen[accession] = sequence_digest

                    length = len(sequence)
                    if length < min_length:
                        below_minimum += 1
                        continue
                    if max_length is not None and length > max_length:
                        above_maximum += 1
                        continue

                    residue_counts.update(sequence)
                    sequence_lengths.append(length)
                    _write_fasta_record(fasta_handle, accession, sequence)
                    id_handle.write(accession + "\n")
                    metadata_writer.writerow(
                        [
                            accession,
                            accession.rsplit(".", 1)[0],
                            description,
                            organism,
                            length,
                            division,
                            source_path.name,
                        ]
                    )

        if not sequence_lengths:
            raise ValueError("No RefSeq proteins remain after normalization and length filtering")
        partial_fasta.replace(output_fasta)
        partial_ids.replace(candidate_ids)
        partial_metadata.replace(metadata_csv)
    except Exception:
        for path in partial_paths:
            path.unlink(missing_ok=True)
        raise

    nonstandard = {
        residue: count
        for residue, count in sorted(residue_counts.items())
        if residue not in STANDARD_AA
    }
    return {
        "source_record_count": read_count,
        "protein_count": len(sequence_lengths),
        "duplicate_accessions_skipped": duplicate_count,
        "below_minimum_length_skipped": below_minimum,
        "above_maximum_length_skipped": above_maximum,
        "total_residues": sum(sequence_lengths),
        "minimum_sequence_length": min(sequence_lengths),
        "maximum_sequence_length": max(sequence_lengths),
        "nonstandard_residue_counts": nonstandard,
    }


def _selection_settings(
    *,
    release: int,
    divisions: tuple[str, ...],
    max_files_per_division: int | None,
    min_length: int,
    max_length: int | None,
    selected_files: list[RefSeqReleaseFile],
) -> dict[str, object]:
    return {
        "release": release,
        "divisions": list(divisions),
        "max_files_per_division": max_files_per_division,
        "min_length": min_length,
        "max_length": max_length,
        "selected_filenames": [item.filename for item in selected_files],
    }


def prepare_refseq(
    output_dir: str | Path,
    *,
    divisions: Iterable[str] = SUPPORTED_DIVISIONS,
    max_files_per_division: int | None = 1,
    min_length: int = 50,
    max_length: int | None = None,
    force_download: bool = False,
    force_normalize: bool = False,
) -> Path:
    """Prepare a release-pinned RefSeq WP candidate database."""

    output = _resolve_path(output_dir, must_exist=False)
    output.mkdir(parents=True, exist_ok=True)
    requested_divisions = tuple(dict.fromkeys(str(value).strip().lower() for value in divisions))

    catalog_index = _download_file(
        RELEASE_CATALOG_INDEX_URL,
        output / "downloads/release-catalog-index.html",
        force=True,
    )
    release = discover_current_release(catalog_index.read_text(encoding="utf-8"))
    catalog_url = f"{NCBI_REFSEQ_RELEASE_ROOT}/release-catalog/" f"release{release}.files.installed"
    catalog_path = _download_file(
        catalog_url,
        output / f"downloads/release{release}.files.installed",
        force=force_download,
    )
    release_notes_url = f"{RELEASE_NOTES_ROOT}/RefSeq-release{release}.txt"
    release_notes_path = _download_file(
        release_notes_url,
        output / f"downloads/RefSeq-release{release}.txt",
        force=force_download,
    )
    selected_files = parse_release_files(
        catalog_path.read_text(encoding="utf-8"),
        divisions=requested_divisions,
        max_files_per_division=max_files_per_division,
    )
    settings = _selection_settings(
        release=release,
        divisions=requested_divisions,
        max_files_per_division=max_files_per_division,
        min_length=min_length,
        max_length=max_length,
        selected_files=selected_files,
    )

    normalized_fasta = output / "proteins.fasta"
    candidate_ids = output / "candidate_ids.txt"
    metadata_csv = output / "proteins.csv"
    manifest_path = output / "manifest.json"
    normalized_outputs = (normalized_fasta, candidate_ids, metadata_csv)
    existing_artifacts = manifest_path.is_file() or any(
        path.exists() for path in normalized_outputs
    )
    if manifest_path.is_file() and all(path.is_file() for path in normalized_outputs):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("selection") == settings and not force_download and not force_normalize:
            print(f"Reusing normalized NCBI RefSeq database: {manifest_path}", flush=True)
            return manifest_path
    if existing_artifacts and not (force_normalize or force_download):
        raise RuntimeError(
            f"RefSeq selection differs from or is incomplete in the existing database at "
            f"{output}. Use --force-normalize to replace it or choose a different "
            "--output-dir."
        )

    downloaded_paths: list[Path] = []
    for item in selected_files:
        downloaded_paths.append(
            _download_file(
                item.url,
                output / "downloads" / item.division / item.filename,
                force=force_download,
                expected_md5=item.md5,
            )
        )

    scope = "all available" if max_files_per_division is None else str(max_files_per_division)
    print(
        f"Normalizing RefSeq release {release}: {scope} WP FASTA file(s) per division...",
        flush=True,
    )
    statistics = normalize_refseq_fastas(
        downloaded_paths,
        normalized_fasta,
        candidate_ids,
        metadata_csv,
        min_length=min_length,
        max_length=max_length,
    )
    downloaded_manifest: list[dict[str, object]] = []
    for item, path in zip(selected_files, downloaded_paths):
        record = asdict(item)
        record.update({"path": str(path), "size_bytes": path.stat().st_size})
        downloaded_manifest.append(record)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "database": "NCBI RefSeq",
            "release": release,
            "release_catalog_url": catalog_url,
            "release_notes_url": release_notes_url,
            "release_notes_path": str(release_notes_path),
            "release_notes_sha256": _hash_file(release_notes_path, "sha256"),
            "nonredundant_wp_only": True,
        },
        "selection": settings,
        "downloaded_files": downloaded_manifest,
        "files": {
            "normalized_fasta": str(normalized_fasta),
            "candidate_ids": str(candidate_ids),
            "metadata_csv": str(metadata_csv),
        },
        "sha256": {
            "normalized_fasta": _hash_file(normalized_fasta, "sha256"),
            "candidate_ids": _hash_file(candidate_ids, "sha256"),
            "metadata_csv": _hash_file(metadata_csv, "sha256"),
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
        f"Prepared {statistics['protein_count']:,} NCBI RefSeq WP proteins: {manifest_path}",
        flush=True,
    )
    return manifest_path


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="wet_lab/databases/refseq/prokaryotes/current",
        help="Repository-local RefSeq database directory",
    )
    parser.add_argument(
        "--division",
        action="append",
        choices=SUPPORTED_DIVISIONS,
        dest="divisions",
        help="RefSeq division to include; repeat as needed (default: archaea and bacteria)",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--max-files-per-division",
        type=_positive_integer,
        default=1,
        help="Bounded pilot size (default: 1 WP FASTA shard per division)",
    )
    scope.add_argument(
        "--all-files",
        action="store_true",
        help="Download every WP FASTA shard in the selected divisions (very large)",
    )
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument(
        "--max-length",
        type=_positive_integer,
        help="Optionally exclude proteins longer than this many residues",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-normalize", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    prepare_refseq(
        args.output_dir,
        divisions=args.divisions or SUPPORTED_DIVISIONS,
        max_files_per_division=None if args.all_files else args.max_files_per_division,
        min_length=args.min_length,
        max_length=args.max_length,
        force_download=args.force_download,
        force_normalize=args.force_normalize,
    )


if __name__ == "__main__":
    main()
