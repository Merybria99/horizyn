"""Sequence-verified native annotations from a frozen UniProt release.

Only primary/secondary accession matches with an identical, case-normalized
sequence receive annotations. No sequence clustering, EC imputation, reaction
label transfer, or negative-label inference is performed here. Memory contains
target identifiers and SHA-256 digests, not all sequences or annotation rows.
Selected rows are spooled on disk so ambiguous accession matches can be masked
before publishing one final row per target. Uses the Python standard library.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, TextIO


SCHEMA_VERSION = "horizyn1_native_uniprot_annotations_v1"
PROGRESS_INTERVAL = 1_000_000
FIELDS = (
    "protein_id", "uniprot_accession", "annotation_status", "ec_numbers",
    "cofactor_names", "ec_evidence", "cofactor_evidence", "source_kind",
    "source_release", "rhea_ids", "sequence_sha256", "candidate_accessions",
)
LABEL_FIELDS = ("ec_numbers", "cofactor_names", "ec_evidence", "cofactor_evidence", "rhea_ids")
EC_COMPONENT = r"(?:[0-9]+|[nN][0-9]+|-)"
EC_RE = re.compile(r"(?:^|;)\s*(?:RecName:\s*|AltName:\s*|SubName:\s*)?EC\s*=\s*(" + EC_COMPONENT + r"(?:\s*\.\s*" + EC_COMPONENT + r"){3})(?=\s|;|\{|$)")
EVIDENCE_RE = re.compile(r"\bEvidence\s*=\s*\{([^{}]*)\}")
INLINE_EVIDENCE_RE = re.compile(r"\s*\{([^{}]*)\}")
RHEA_RE = re.compile(r"RHEA:(\d+)", re.IGNORECASE)
SEQUENCE_SPACE_RE = re.compile(r"[\s0-9]+")
SQ_LENGTH_RE = re.compile(r"\bSEQUENCE\s+(\d+)\s+AA\b")


def _source_iterator():
    # Importing horizyn.datasets would otherwise eagerly import torch/HDF5.
    name = "_horizyn1_native_annotation_reconstruction"
    if name not in sys.modules:
        path = Path(__file__).with_name("horizyn1_reconstruction.py")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load UniProt source reader from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].iter_uniprot_sources


def _progress(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", file=sys.stderr, flush=True)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _signature(path: Path) -> dict[str, int]:
    value = path.stat()
    return {"device": value.st_dev, "inode": value.st_ino, "size": value.st_size,
            "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_signatures() -> dict[str, str]:
    return {name: _sha256(Path(__file__).with_name(name)) for name in
            ("horizyn1_uniprot_annotations.py", "horizyn1_reconstruction.py")}


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and os.path.samefile(left, right)
    )


def _validate_paths(input_path: Path, target_fasta: Path, output_path: Path, manifest_path: Path) -> None:
    for path in (input_path, target_fasta):
        if not path.is_file():
            raise FileNotFoundError(f"Input is not a regular file: {path}")
    for output in (output_path, manifest_path):
        if output.is_symlink():
            raise ValueError(f"Refusing a symlink output: {output}")
        for source in (input_path, target_fasta):
            if _same_path(output, source):
                raise ValueError(f"Output aliases an input: {output} -> {source}")
    if _same_path(output_path, manifest_path):
        raise ValueError("Annotation output and manifest must be different files")


@dataclass(slots=True)
class TargetSignature:
    digest: bytes
    # 0 = not seen; 1 = exactly one native record; 2 = ambiguous native records.
    state: int = 0


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")


def index_target_fasta(path: Path) -> tuple[dict[str, TargetSignature], dict[str, object]]:
    targets: dict[str, TargetSignature] = {}
    protein_id: str | None = None
    sequence_digest = hashlib.sha256()
    dataset_digest = hashlib.sha256()
    length = residues = 0

    def finish() -> None:
        nonlocal residues
        if protein_id is None:
            return
        if not length:
            raise ValueError(f"Empty target sequence: {protein_id}")
        if protein_id in targets:
            raise ValueError(f"Duplicate target FASTA ID: {protein_id}")
        digest = sequence_digest.digest()
        targets[protein_id] = TargetSignature(digest)
        dataset_digest.update(protein_id.encode("utf-8") + b"\0" + digest + b"\n")
        residues += length
        if len(targets) % PROGRESS_INTERVAL == 0:
            _progress(f"Indexed {len(targets):,} target sequences")

    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith(">"):
                finish()
                parts = line[1:].split()
                if not parts:
                    raise ValueError(f"Empty target FASTA header at line {line_number}")
                protein_id = parts[0]
                if "\0" in protein_id:
                    raise ValueError("NUL character in target FASTA ID")
                sequence_digest = hashlib.sha256()
                length = 0
            elif line.strip():
                if protein_id is None:
                    raise ValueError(f"Sequence before a target FASTA header at line {line_number}")
                chunk = "".join(line.split())
                if not chunk.isascii() or not chunk.isalpha():
                    raise ValueError(f"Non-alphabetic target sequence at line {line_number}")
                sequence_digest.update(chunk.upper().encode("ascii"))
                length += len(chunk)
        finish()
    if not targets:
        raise ValueError("Target FASTA contains no sequences")
    return targets, {"target_sequences": len(targets), "target_residues": residues,
                     "target_id_sequence_sha256": dataset_digest.hexdigest(),
                     "target_digest_definition": "FASTA order: UTF-8 ID + NUL + binary SHA256(uppercase sequence) + LF"}


def _evidence_values(text: str) -> list[str]:
    return sorted({token.strip() for token in text.split(",") if token.strip()})


def _ec_annotations(text: str, context: str) -> list[dict[str, object]]:
    # A catalytic-activity block has one reaction and a block-level Evidence
    # field. DE evidence is attached only to its immediately preceding EC.
    shared = sorted({item for match in EVIDENCE_RE.finditer(text) for item in _evidence_values(match.group(1))}) if context == "CC CATALYTIC ACTIVITY" else []
    annotations = []
    for match in EC_RE.finditer(text):
        ec = re.sub(r"\s+", "", match.group(1))
        inline = INLINE_EVIDENCE_RE.match(text[match.end():])
        evidence = _evidence_values(inline.group(1)) if inline else shared
        annotations.append({"ec_number": ec, "evidence": evidence, "context": context})
    return annotations


def parse_native_annotations(de_lines: list[str], comments: list[tuple[str, str]], rhea_ids: set[str]) -> dict[str, object]:
    ec_records = _ec_annotations(" ".join(de_lines), "DE")
    cofactor_records: list[dict[str, object]] = []
    for topic, text in comments:
        if topic == "CATALYTIC ACTIVITY":
            ec_records.extend(_ec_annotations(text, "CC CATALYTIC ACTIVITY"))
            rhea_ids.update(RHEA_RE.findall(text))
        elif topic == "COFACTOR":
            text = re.split(r"(?:^|;)\s*Note\s*=", text, maxsplit=1)[0]
            names = list(re.finditer(r"(?:^|;)\s*Name\s*=\s*([^;]+)", text))
            for index, match in enumerate(names):
                name = " ".join(match.group(1).split())
                segment = text[match.end():names[index + 1].start() if index + 1 < len(names) else len(text)]
                # Evidence appearing inside a general free-text Note is not
                # automatically evidence for each cofactor in the block.
                segment = re.split(r"\bNote\s*=", segment, maxsplit=1)[0]
                evidence = sorted({item for evidence_match in EVIDENCE_RE.finditer(segment)
                                   for item in _evidence_values(evidence_match.group(1))})
                cofactor_records.append({"name": name, "evidence": evidence, "context": "CC COFACTOR"})
    ec_records = [json.loads(value) for value in sorted({_json(row) for row in ec_records})]
    cofactor_records = [json.loads(value) for value in sorted({_json(row) for row in cofactor_records})]
    return {"ec_numbers": sorted({str(row["ec_number"]) for row in ec_records}),
            "cofactor_names": sorted({str(row["name"]) for row in cofactor_records}),
            "ec_evidence": ec_records, "cofactor_evidence": cofactor_records,
            "rhea_ids": sorted(rhea_ids, key=int)}


def iter_selected_records(lines: Iterable[str], targets: dict[str, TargetSignature], stats: Counter) -> Iterator[dict[str, object]]:
    in_record = in_sequence = False
    primary = ""
    selected: list[str] = []
    de_lines: list[str] = []
    comments: list[tuple[str, str]] = []
    comment_topic = ""
    comment_parts: list[str] = []
    sequence_digest = hashlib.sha256()
    sequence_length = 0
    declared_length: int | None = None
    rhea_ids: set[str] = set()

    def finish_comment() -> None:
        if comment_topic in {"CATALYTIC ACTIVITY", "COFACTOR"}:
            comments.append((comment_topic, " ".join(comment_parts)))

    for line_number, line in enumerate(lines, 1):
        if line.startswith("ID   "):
            if in_record:
                raise ValueError(f"Truncated UniProt record before line {line_number}: missing //")
            in_record = True
            continue
        if line.startswith("//"):
            if not in_record:
                raise ValueError(f"Unexpected UniProt record terminator at line {line_number}")
            stats["records_scanned"] += 1
            if selected:
                if not in_sequence or not sequence_length:
                    raise ValueError(f"Selected UniProt record {primary} has no sequence")
                if declared_length is not None and declared_length != sequence_length:
                    raise ValueError(f"Sequence length mismatch in native record {primary}: SQ={declared_length}, observed={sequence_length}")
                finish_comment()
                stats["selected_native_records"] += 1
                yield {"primary": primary, "targets": selected, "digest": sequence_digest.digest(),
                       "annotations": parse_native_annotations(de_lines, comments, rhea_ids)}
            if stats["records_scanned"] % PROGRESS_INTERVAL == 0:
                _progress(f"Scanned {stats['records_scanned']:,} UniProt records; selected {stats['selected_native_records']:,}")
            in_record = in_sequence = False
            primary = ""
            selected, de_lines, comments, comment_parts = [], [], [], []
            comment_topic = ""
            sequence_digest = hashlib.sha256()
            sequence_length = 0
            declared_length = None
            rhea_ids = set()
            continue
        if not in_record:
            if line.strip():
                raise ValueError(f"Unexpected text outside UniProt record at line {line_number}")
            continue
        if line.startswith("AC   "):
            accessions = [value.strip() for value in line[5:].split(";") if value.strip()]
            if accessions and not primary:
                primary = accessions[0]
            selected.extend(value for value in accessions if value in targets and value not in selected)
            continue
        if not selected:
            continue
        if line.startswith("SQ   "):
            in_sequence = True
            match = SQ_LENGTH_RE.search(line)
            declared_length = int(match.group(1)) if match else None
        elif in_sequence:
            chunk = SEQUENCE_SPACE_RE.sub("", line)
            if chunk:
                if not chunk.isascii() or not chunk.isalpha():
                    raise ValueError(f"Invalid native sequence in {primary} at line {line_number}")
                sequence_digest.update(chunk.upper().encode("ascii"))
                sequence_length += len(chunk)
        elif line.startswith("DE   "):
            de_lines.append(line[5:].strip())
        elif line.startswith("CC   "):
            text = line[5:].strip()
            if text.startswith("-!-"):
                finish_comment()
                topic, separator, body = text[3:].strip().partition(":")
                comment_topic = topic if separator else ""
                comment_parts = [body.strip()] if comment_topic in {"CATALYTIC ACTIVITY", "COFACTOR"} else []
            elif comment_topic in {"CATALYTIC ACTIVITY", "COFACTOR"}:
                comment_parts.append(text)
        elif line.startswith("DR   "):
            rhea_ids.update(RHEA_RE.findall(line))
    if in_record:
        raise ValueError("Truncated UniProt flat file: final record has no // terminator")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_without_overwrite(temporary: Path, final: Path) -> None:
    # Atomic no-clobber publication even if another process races this writer.
    os.link(temporary, final)
    temporary.unlink()
    _fsync_directory(final.parent)


def _resume_if_valid(input_path: Path, target_fasta: Path, output_path: Path, manifest_path: Path, source_kind: str, release: str) -> dict[str, object] | None:
    if not output_path.exists() and not manifest_path.exists():
        return None
    if not output_path.is_file() or not manifest_path.is_file():
        raise FileExistsError("Output or manifest already exists without its completed counterpart; preserving it. Move the incomplete artifact before restarting.")
    manifest_before = _signature(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        saved = json.load(handle)
    expected = {"input_path": str(input_path.resolve()), "target_fasta": str(target_fasta.resolve()),
                "source_kind": source_kind, "release": release}
    if saved.get("schema_version") != SCHEMA_VERSION or saved.get("status") != "complete" or saved.get("request") != expected:
        raise FileExistsError("Existing annotation manifest does not match this completed extraction request; preserving outputs")
    if saved.get("input_signatures") != {"archive": _signature(input_path), "target_fasta": _signature(target_fasta)}:
        raise FileExistsError("Extraction inputs changed since the existing annotation manifest; preserving outputs")
    if saved.get("implementation_sha256") != _implementation_signatures():
        raise FileExistsError("Extractor implementation changed since the existing annotation manifest; preserving outputs")
    before = _signature(output_path)
    if saved.get("output", {}).get("path") != str(output_path.resolve()) or saved.get("output", {}).get("signature") != before:
        raise FileExistsError("Existing annotation output differs from its manifest; preserving outputs")
    if _sha256(output_path) != saved["output"].get("sha256") or before != _signature(output_path):
        raise FileExistsError("Existing annotation output checksum failed or changed while checking; preserving outputs")
    if (saved["input_signatures"] != {"archive": _signature(input_path), "target_fasta": _signature(target_fasta)}
            or manifest_before != _signature(manifest_path)
            or saved["implementation_sha256"] != _implementation_signatures()):
        raise FileExistsError("Extraction inputs, implementation, or manifest changed during resume verification; preserving outputs")
    _progress("Completed native annotation output verified; safely reusing it")
    return saved


def extract_uniprot_annotations(*, input_path: Path, target_fasta: Path, output_path: Path, manifest_path: Path,
                                source_kind: str = "auto", release: str = "2023_05") -> dict[str, object]:
    """Extract native labels for exact target IDs/sequences, publishing manifest last.

    Successful existing outputs are checksum-verified and reused. Incomplete or
    mismatched existing outputs are preserved and rejected, never overwritten.
    Native evidence objects are retained verbatim at the ECO-reference level;
    they are not converted to experimental-validation or confidence claims.
    """
    input_path, target_fasta, output_path, manifest_path = map(Path, (input_path, target_fasta, output_path, manifest_path))
    if source_kind not in {"auto", "sprot", "trembl"}:
        raise ValueError("source_kind must be auto, sprot, or trembl")
    if not release.strip():
        raise ValueError("release must not be empty")
    _validate_paths(input_path, target_fasta, output_path, manifest_path)
    resumed = _resume_if_valid(input_path, target_fasta, output_path, manifest_path, source_kind, release)
    if resumed is not None:
        return resumed
    signatures = {"archive": _signature(input_path), "target_fasta": _signature(target_fasta)}
    implementation_signatures = _implementation_signatures()
    started = datetime.now(timezone.utc).isoformat()
    targets, target_stats = index_target_fasta(target_fasta)
    if signatures["target_fasta"] != _signature(target_fasta):
        raise RuntimeError("Target FASTA changed while indexing")
    _progress(f"Indexed {len(targets):,} target sequences; streaming native annotations")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter({"records_scanned": 0, "selected_native_records": 0,
                              "rows_with_ec": 0, "rows_with_cofactor": 0, "rows_with_rhea": 0})
    status_counts: Counter = Counter()
    duplicate_accessions: dict[str, set[str]] = {}
    native_kinds: Counter = Counter()
    with tempfile.TemporaryDirectory(prefix=f".{output_path.name}.stage-", dir=output_path.parent) as staging:
        spool = Path(staging) / "selected_rows.tsv"
        temporary_output = Path(staging) / "annotations.partial"
        with spool.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for kind, source in _source_iterator()(input_path, source_kind=source_kind):
                # The shared archive iterator detects members; honor an explicit
                # source-kind filter here because its archive path is unfiltered.
                if source_kind != "auto" and kind != source_kind:
                    continue
                previous_records = stats["records_scanned"]
                for record in iter_selected_records(source, targets, stats):
                    primary = str(record["primary"])
                    annotations = record["annotations"]
                    for protein_id in record["targets"]:
                        target = targets[protein_id]
                        if target.state:
                            target.state = 2
                            duplicate_accessions.setdefault(protein_id, set()).add(primary)
                            continue
                        target.state = 1
                        matched = target.digest == record["digest"]
                        status = "sequence_mismatch"
                        if matched:
                            status = "matched" if annotations["ec_numbers"] or annotations["cofactor_names"] else "matched_unannotated"
                        row = {"protein_id": protein_id, "uniprot_accession": primary, "annotation_status": status,
                               "source_kind": kind, "source_release": release, "sequence_sha256": target.digest.hex(),
                               "candidate_accessions": _json([primary])}
                        for field in LABEL_FIELDS:
                            row[field] = _json(annotations[field] if matched else [])
                        writer.writerow(row)
                native_kinds[kind] += stats["records_scanned"] - previous_records
        if not stats["records_scanned"]:
            raise ValueError("No UniProt records scanned for the requested source kind")
        output_rows = 0

        def count_row(row: dict[str, str]) -> None:
            nonlocal output_rows
            output_rows += 1
            status_counts[row["annotation_status"]] += 1
            if row["ec_numbers"] != "[]":
                stats["rows_with_ec"] += 1
            if row["cofactor_names"] != "[]":
                stats["rows_with_cofactor"] += 1
            if row["rhea_ids"] != "[]":
                stats["rows_with_rhea"] += 1
            if output_rows % PROGRESS_INTERVAL == 0:
                _progress(f"Finalized {output_rows:,} / {len(targets):,} native annotation rows")

        with gzip.open(temporary_output, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            with spool.open("r", encoding="utf-8", newline="") as rows:
                for row in csv.DictReader(rows, delimiter="\t"):
                    protein_id = row["protein_id"]
                    if targets[protein_id].state == 2:
                        row["annotation_status"] = "ambiguous_accession"
                        row["candidate_accessions"] = _json(sorted(duplicate_accessions[protein_id] | {row["uniprot_accession"]}))
                        row["uniprot_accession"] = row["source_kind"] = ""
                        for field in LABEL_FIELDS:
                            row[field] = "[]"
                    count_row(row)
                    writer.writerow(row)
            for protein_id, target in targets.items():
                if target.state:
                    continue
                row = {field: "[]" for field in LABEL_FIELDS}
                row.update(protein_id=protein_id, uniprot_accession="", annotation_status="unresolved",
                           source_kind="", source_release=release, sequence_sha256=target.digest.hex(), candidate_accessions="[]")
                count_row(row)
                writer.writerow(row)
        if output_rows != len(targets):
            raise RuntimeError(f"Annotation row count mismatch: {output_rows} != {len(targets)}")
        with temporary_output.open("rb") as handle:
            os.fsync(handle.fileno())
        output_sha256 = _sha256(temporary_output)
        if signatures != {"archive": _signature(input_path), "target_fasta": _signature(target_fasta)}:
            raise RuntimeError("Extraction inputs changed while scanning; no final output published")
        if implementation_signatures != _implementation_signatures():
            raise RuntimeError("Extractor implementation changed while scanning; no final output published")
        _publish_without_overwrite(temporary_output, output_path)
        payload = {"schema_version": SCHEMA_VERSION, "status": "complete", "started_at": started,
                   "completed_at": datetime.now(timezone.utc).isoformat(),
                   "request": {"input_path": str(input_path.resolve()), "target_fasta": str(target_fasta.resolve()),
                               "source_kind": source_kind, "release": release},
                   "input_signatures": signatures, "implementation_sha256": implementation_signatures, "target_index": target_stats,
                   "counts": {**dict(stats), "output_rows": output_rows}, "annotation_status_counts": dict(status_counts),
                   "native_records_by_kind": dict(native_kinds), "columns": list(FIELDS),
                   "output": {"path": str(output_path.resolve()), "signature": _signature(output_path), "sha256": output_sha256},
                   "semantics": {"matching": "Exact primary/secondary accession AND uppercase sequence SHA-256; no prefix rewriting or cluster transfer",
                                 "unknowns": "Unresolved, mismatched, and ambiguous targets retain empty annotation lists; missing means unknown, never negative",
                                 "ec_sources": ["DE EC fields", "CC CATALYTIC ACTIVITY EC fields"],
                                 "cofactor_sources": ["CC COFACTOR Name fields"],
                                 "evidence": "Native ECO references retained; no experimental validation or confidence inferred",
                                 "archive_verification": "Archive stat stability recorded; no new whole-archive checksum claimed"}}
        descriptor, temporary_manifest_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.", suffix=".partial", dir=manifest_path.parent)
        temporary_manifest = Path(temporary_manifest_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _publish_without_overwrite(temporary_manifest, manifest_path)
        finally:
            if temporary_manifest.exists():
                temporary_manifest.unlink()
    _progress(f"Native annotation extraction complete: {output_rows:,} rows; {dict(status_counts)}")
    return payload
