#!/usr/bin/env python3
"""Build a deduplicated single-enzyme sequence catalog across local datasets.

The catalog is intentionally sequence-centric: every input record with an amino
acid sequence is retained in a source-record table, then records are collapsed by
exact cleaned sequence. EC labels are merged from direct dataset metadata and
from UniProt-accession caches. No EC is inferred for anonymized proteins unless
an exact duplicate sequence already has an EC label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "horizyn" / "results" / "single_enzyme_catalog"

UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|[A-Z0-9]{10})$"
)
EC_TOKEN_RE = re.compile(r"(?<![0-9])([1-7](?:\.(?:[0-9]+|-)){3})(?![0-9])")
VALID_AA_RE = re.compile(r"^[A-Z*.-]+$")


@dataclass
class SourceRecord:
    source: str
    dataset_split: str
    protein_id: str
    accession: str
    sequence: str
    ecs: set[str] = field(default_factory=set)
    path: str = ""


def clean_sequence(sequence: object) -> str:
    if sequence is None:
        return ""
    seq = str(sequence).strip().upper().replace(" ", "").replace("\n", "")
    seq = seq.replace("*", "")
    return seq


def is_sequence_like(sequence: str) -> bool:
    return bool(sequence) and len(sequence) >= 20 and bool(VALID_AA_RE.match(sequence))


def sequence_hash(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def detect_accession(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    token = text.split("|")[-1].split()[0]
    token = token.split(".")[0]
    return token if UNIPROT_ACCESSION_RE.match(token) else ""


def parse_ecs(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        values: Iterable[object] = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return set()
        text = text.replace("EC-", "")
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
                if isinstance(decoded, list):
                    values = decoded
                else:
                    values = [text]
            except json.JSONDecodeError:
                values = [text]
        else:
            values = [text]
    ecs: set[str] = set()
    for item in values:
        text = str(item or "").replace("EC-", "")
        for token in EC_TOKEN_RE.findall(text):
            ecs.add(token)
    return ecs


def complete_ecs(ecs: Iterable[str]) -> set[str]:
    return {ec for ec in ecs if re.fullmatch(r"[1-7]\.[0-9]+\.[0-9]+\.[0-9]+", ec)}


def read_csv_rows(path: Path, delimiter: str = ",") -> Iterator[dict[str, str]]:
    if not path.exists():
        return
    with path.open(newline="", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            yield row


def read_fasta(path: Path) -> Iterator[tuple[str, str]]:
    if not path.exists():
        return
    header = ""
    chunks: list[str] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if header:
        yield header, "".join(chunks)


def first_present(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        if name in row and str(row.get(name, "")).strip():
            return str(row.get(name, "")).strip()
    return ""


def add_record(
    records: list[SourceRecord],
    source: str,
    dataset_split: str,
    protein_id: object,
    sequence: object,
    *,
    accession: object = "",
    ecs: Iterable[str] | None = None,
    path: Path | str = "",
) -> None:
    seq = clean_sequence(sequence)
    if not is_sequence_like(seq):
        return
    protein = str(protein_id or "").strip()
    acc = str(accession or "").strip() or detect_accession(protein)
    records.append(
        SourceRecord(
            source=source,
            dataset_split=dataset_split,
            protein_id=protein,
            accession=acc,
            sequence=seq,
            ecs=set(ecs or []),
            path=str(path),
        )
    )


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_uniprot_ec_cache(path: Path) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    if not path.exists():
        return lookup
    for row in read_csv_rows(path):
        ids = {
            row.get("protein_id", "").strip(),
            row.get("uniprot_accession", "").strip(),
            row.get("Entry", "").strip(),
            row.get("accession", "").strip(),
        }
        ecs = parse_ecs(row.get("ec_number", "") or row.get("EC number", "") or row.get("ec", ""))
        if not ecs:
            continue
        for protein_id in ids:
            if protein_id:
                lookup[protein_id].update(ecs)
    return lookup


def load_ec_lookups(extra_caches: Iterable[Path]) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for path in [
        ROOT / "horizyn/data/sota/uniprot_all_ec_labels.csv",
        ROOT / "horizyn/data/sota/uniprot_reviewed_ec_labels.csv",
        ROOT / "horizyn/data/huggingface/AI4Protein_EC/uniprot_ai4protein_ec_labels.csv",
    ]:
        for key, ecs in load_uniprot_ec_cache(path).items():
            lookup[key].update(ecs)
    for path in extra_caches:
        for key, ecs in load_uniprot_ec_cache(path).items():
            lookup[key].update(ecs)

    uid_to_ec = ROOT / "EnzymeCAGE/dataset/others/uid_to_ec.pkl"
    if uid_to_ec.exists():
        data = load_pickle(uid_to_ec)
        for accession, ec_value in data.items():
            ecs = parse_ecs(ec_value)
            if ecs:
                lookup[str(accession)].update(ecs)

    clip_ec = ROOT / "horizyn/data/paper/clipzyme/files/ec2uniprot.p"
    if clip_ec.exists():
        data = load_pickle(clip_ec)
        for ec_value, accessions in data.items():
            ecs = parse_ecs(ec_value)
            if not ecs:
                continue
            for accession in accessions:
                lookup[str(accession)].update(ecs)
    return lookup


def collect_records() -> list[SourceRecord]:
    csv.field_size_limit(sys.maxsize)
    records: list[SourceRecord] = []

    for header, sequence in read_fasta(ROOT / "horizyn/data/sota/prots.fasta"):
        protein_id = header.split()[0]
        add_record(records, "horizyn_sota", "all", protein_id, sequence, path="horizyn/data/sota/prots.fasta")

    for header, sequence in read_fasta(ROOT / "horizyn/data/paper/reactzyme/eval/all_proteins.fasta"):
        protein_id = header.split()[0]
        add_record(
            records,
            "reactzyme",
            "eval_all",
            protein_id,
            sequence,
            path="horizyn/data/paper/reactzyme/eval/all_proteins.fasta",
        )

    clip_path = ROOT / "horizyn/data/paper/clipzyme/files/uniprot2sequence.p"
    if clip_path.exists():
        data = load_pickle(clip_path)
        for accession, sequence in data.items():
            add_record(records, "clipzyme", "all_uniprot2sequence", accession, sequence, path=clip_path)

    for split in ["novelty50", "novelty90"]:
        base = ROOT / f"horizyn/data/paper/sabio_rk/eval/{split}"
        ec_by_accession: dict[str, set[str]] = defaultdict(set)
        for row in read_csv_rows(base / "pairs.csv"):
            accession = row.get("protein_id", "").strip()
            ec_by_accession[accession].update(parse_ecs(row.get("ec_numbers_json", "")))
        for row in read_csv_rows(base / "sequences.tsv", delimiter="\t"):
            accession = row.get("accession", "").strip()
            add_record(
                records,
                "sabio_rk",
                split,
                accession,
                row.get("sequence", ""),
                accession=accession,
                ecs=ec_by_accession.get(accession, set()),
                path=base / "sequences.tsv",
            )

    for name in ["new_sequences.tsv", "old_sequences.tsv"]:
        path = ROOT / "sabio_rk_download/seqsim_fetch" / name
        split = name.replace("_sequences.tsv", "")
        for row in read_csv_rows(path, delimiter="\t"):
            accession = first_present(row, ["accession", "protein_id", "id"])
            sequence = first_present(row, ["sequence", "aa_seq"])
            add_record(records, "sabio_rk_raw_seqsim", split, accession, sequence, accession=accession, path=path)

    for split in ["train", "valid", "test"]:
        path = ROOT / f"horizyn/data/huggingface/AI4Protein_EC/{split}_with_ec.csv"
        for row in read_csv_rows(path):
            accession = row.get("uniprot_accession", "").strip()
            ecs = parse_ecs(row.get("complete_ec_numbers", "")) or parse_ecs(row.get("ec_number", ""))
            add_record(
                records,
                "ai4protein_ec",
                split,
                row.get("name", accession),
                row.get("aa_seq", ""),
                accession=accession,
                ecs=ecs,
                path=path,
            )

    for row in read_csv_rows(ROOT / "horizyn/data/SLEEC/New-392.csv", delimiter="\t"):
        accession = row.get("Entry", "").strip()
        add_record(
            records,
            "sleec_new392",
            "all",
            accession,
            row.get("Sequence", ""),
            accession=accession,
            ecs=parse_ecs(row.get("EC number", "")),
            path="horizyn/data/SLEEC/New-392.csv",
        )

    for row in read_csv_rows(ROOT / "horizyn/data/sleec_stage1/raw/uniprot_reference_sequences.tsv", delimiter="\t"):
        protein_id = row.get("protein_id", "").strip()
        add_record(
            records,
            "sleec_stage1_reference",
            "all",
            protein_id,
            row.get("sequence", ""),
            path="horizyn/data/sleec_stage1/raw/uniprot_reference_sequences.tsv",
        )

    for header, sequence in read_fasta(ROOT / "horizyn/data/sleec_stage1/fasta/mcsa_reference.full.fasta"):
        protein_id = header.split()[0]
        add_record(records, "sleec_mcsa_reference", "all", protein_id, sequence, path="horizyn/data/sleec_stage1/fasta/mcsa_reference.full.fasta")

    rhea_all = ROOT / "EnzymeCAGE/dataset/RHEA/2025-02-05/all_enzymes.csv"
    for row in read_csv_rows(rhea_all):
        accession = row.get("UniprotID", "").strip()
        add_record(records, "enzymecage_rhea_2025", "all", accession, row.get("sequence", ""), accession=accession, path=rhea_all)

    uid2seq = ROOT / "EnzymeCAGE/dataset/RHEA/2025-02-05/uid2seq.pkl"
    if uid2seq.exists():
        data = load_pickle(uid2seq)
        for accession, sequence in data.items():
            add_record(records, "enzymecage_uid2seq", "all", accession, sequence, accession=accession, path=uid2seq)

    for split in ["train", "valid"]:
        path = ROOT / f"EnzymeCAGE/dataset/training/{split}.csv"
        for row in read_csv_rows(path):
            accession = row.get("UniprotID", "").strip()
            add_record(
                records,
                "enzymecage_training",
                split,
                accession,
                row.get("sequence", ""),
                accession=accession,
                ecs=parse_ecs(row.get("EC number", "")),
                path=path,
            )

    for split in ["p450", "phosphatase", "terpene"]:
        path = ROOT / f"EnzymeCAGE/dataset/domain-specific-ft/{split}.csv"
        for row in read_csv_rows(path):
            accession = row.get("UniprotID", "").strip()
            add_record(records, "enzymecage_domain_ft", split, accession, row.get("sequence", ""), accession=accession, path=path)

    for split, file_name in [
        ("p450", "test_P450.csv"),
        ("phosphatase", "test_Phosphatase.csv"),
        ("terpene", "test_Terpene.csv"),
    ]:
        path = ROOT / f"EnzymeCAGE/dataset/external-test-set/{split}/{file_name}"
        for row in read_csv_rows(path):
            accession = row.get("UniprotID", "").strip()
            add_record(records, "enzymecage_external_test", split, accession, row.get("sequence", ""), accession=accession, path=path)

    for split, path in [
        ("enzyme_405", ROOT / "EnzymeCAGE/dataset/internal-test-set/Enzyme-405/Enzyme-405.csv"),
        ("orphan_335", ROOT / "EnzymeCAGE/dataset/internal-test-set/Orphan-335/Orphan-335.csv"),
    ]:
        for row in read_csv_rows(path):
            accession = row.get("UniprotID", "").strip()
            add_record(
                records,
                "enzymecage_internal_test",
                split,
                accession,
                row.get("sequence", ""),
                accession=accession,
                ecs=parse_ecs(row.get("EC number", "")),
                path=path,
            )

    for fasta_path, source, split in [
        (ROOT / "EnzymeCAGE/dataset/RHEA/2023-07-12/enzymes.fasta", "enzymecage_rhea_2023", "all"),
        (ROOT / "EnzymeCAGE/dataset/external-test-set/p450/proteins.fasta", "enzymecage_external_fasta", "p450"),
        (ROOT / "EnzymeCAGE/dataset/external-test-set/phosphatase/protein.fasta", "enzymecage_external_fasta", "phosphatase"),
        (ROOT / "EnzymeCAGE/dataset/external-test-set/terpene/proteins.fasta", "enzymecage_external_fasta", "terpene"),
        (ROOT / "EnzymeCAGE/dataset/case-study/withanolide/enzymes.fasta", "enzymecage_case_study", "withanolide"),
        (ROOT / "VenusRXN/data/enzymes.fasta", "venus_rxn", "all"),
        (ROOT / "horizyn/data/standardized/horizyn_reactzyme_eval/proteins.fasta", "horizyn_reactzyme_eval_standardized", "all"),
        (ROOT / "horizyn/data/paper/clipzyme/eval/enzymemap/proteins.fasta", "enzymemap_clipzyme_eval", "all"),
    ]:
        for header, sequence in read_fasta(fasta_path):
            protein_id = header.split()[0]
            add_record(records, source, split, protein_id, sequence, path=fasta_path)

    return records


def merge_catalog(records: list[SourceRecord], ec_lookup: dict[str, set[str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    catalog: dict[str, dict[str, object]] = {}
    record_rows: list[dict[str, str]] = []

    for idx, record in enumerate(records):
        ecs = set(record.ecs)
        for key in [record.accession, record.protein_id]:
            if key:
                ecs.update(ec_lookup.get(key, set()))
        seq_hash = sequence_hash(record.sequence)
        if seq_hash not in catalog:
            catalog[seq_hash] = {
                "seq_hash": seq_hash,
                "sequence": record.sequence,
                "length": len(record.sequence),
                "sources": set(),
                "dataset_splits": set(),
                "protein_ids": set(),
                "accessions": set(),
                "ecs": set(),
                "direct_ecs": set(),
                "record_count": 0,
            }
        entry = catalog[seq_hash]
        entry["sources"].add(record.source)  # type: ignore[index]
        entry["dataset_splits"].add(f"{record.source}:{record.dataset_split}")  # type: ignore[index]
        if record.protein_id:
            entry["protein_ids"].add(record.protein_id)  # type: ignore[index]
        if record.accession:
            entry["accessions"].add(record.accession)  # type: ignore[index]
        entry["ecs"].update(ecs)  # type: ignore[index]
        entry["direct_ecs"].update(record.ecs)  # type: ignore[index]
        entry["record_count"] = int(entry["record_count"]) + 1  # type: ignore[arg-type]

        record_rows.append(
            {
                "record_index": str(idx),
                "seq_hash": seq_hash,
                "source": record.source,
                "dataset_split": record.dataset_split,
                "protein_id": record.protein_id,
                "accession": record.accession,
                "length": str(len(record.sequence)),
                "ec_numbers": ";".join(sorted(ecs)),
                "complete_ec_numbers": ";".join(sorted(complete_ecs(ecs))),
                "direct_ec_numbers": ";".join(sorted(record.ecs)),
                "path": record.path,
            }
        )

    rows: list[dict[str, str]] = []
    for entry in catalog.values():
        ecs = sorted(entry["ecs"])  # type: ignore[arg-type]
        full_ecs = sorted(complete_ecs(ecs))
        rows.append(
            {
                "seq_hash": str(entry["seq_hash"]),
                "length": str(entry["length"]),
                "record_count": str(entry["record_count"]),
                "sources": ";".join(sorted(entry["sources"])),  # type: ignore[arg-type]
                "dataset_splits": ";".join(sorted(entry["dataset_splits"])),  # type: ignore[arg-type]
                "protein_ids": ";".join(sorted(entry["protein_ids"])),  # type: ignore[arg-type]
                "accessions": ";".join(sorted(entry["accessions"])),  # type: ignore[arg-type]
                "ec_numbers": ";".join(ecs),
                "complete_ec_numbers": ";".join(full_ecs),
                "has_any_ec": "1" if ecs else "0",
                "has_complete_ec": "1" if full_ecs else "0",
                "direct_ec_numbers": ";".join(sorted(entry["direct_ecs"])),  # type: ignore[arg-type]
                "sequence": str(entry["sequence"]),
            }
        )
    rows.sort(key=lambda row: (int(row["has_any_ec"]) == 0, int(row["length"]), row["seq_hash"]))
    return rows, record_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(catalog_rows: list[dict[str, str]], record_rows: list[dict[str, str]]) -> dict[str, object]:
    source_counts = Counter(row["source"] for row in record_rows)
    catalog_by_hash = {row["seq_hash"]: row for row in catalog_rows}
    unique_by_source: dict[str, int] = {}
    unique_with_ec_by_source: dict[str, int] = {}
    unique_with_direct_ec_by_source: dict[str, int] = {}
    for source in source_counts:
        hashes = {row["seq_hash"] for row in record_rows if row["source"] == source}
        with_ec_hashes = {
            seq_hash for seq_hash in hashes if catalog_by_hash.get(seq_hash, {}).get("has_any_ec") == "1"
        }
        with_direct_ec_hashes = {
            row["seq_hash"] for row in record_rows if row["source"] == source and row["direct_ec_numbers"]
        }
        unique_by_source[source] = len(hashes)
        unique_with_ec_by_source[source] = len(with_ec_hashes)
        unique_with_direct_ec_by_source[source] = len(with_direct_ec_hashes)

    with_any_ec = [row for row in catalog_rows if row["has_any_ec"] == "1"]
    with_complete_ec = [row for row in catalog_rows if row["has_complete_ec"] == "1"]
    missing_any_ec = [row for row in catalog_rows if row["has_any_ec"] == "0"]
    missing_with_accession = [row for row in missing_any_ec if row["accessions"]]
    missing_without_accession = [row for row in missing_any_ec if not row["accessions"]]
    direct_ec = [row for row in catalog_rows if row["direct_ec_numbers"]]
    rescued_by_lookup = [row for row in with_any_ec if not row["direct_ec_numbers"]]
    return {
        "source_record_count": len(record_rows),
        "unique_sequence_count": len(catalog_rows),
        "duplicate_records_removed_by_exact_sequence": len(record_rows) - len(catalog_rows),
        "unique_with_any_ec": len(with_any_ec),
        "unique_with_complete_ec": len(with_complete_ec),
        "unique_missing_any_ec": len(missing_any_ec),
        "unique_missing_any_ec_with_uniprot_accession": len(missing_with_accession),
        "unique_missing_any_ec_without_accession": len(missing_without_accession),
        "unique_with_direct_ec_from_source": len(direct_ec),
        "unique_ec_rescued_by_lookup_or_duplicate": len(rescued_by_lookup),
        "source_record_counts": dict(sorted(source_counts.items())),
        "unique_sequence_counts_by_source": dict(sorted(unique_by_source.items())),
        "unique_with_ec_counts_by_source": dict(sorted(unique_with_ec_by_source.items())),
        "unique_with_direct_ec_counts_by_source": dict(sorted(unique_with_direct_ec_by_source.items())),
    }


def write_markdown_summary(output_dir: Path, summary: dict[str, object]) -> None:
    source_counts = summary["source_record_counts"]  # type: ignore[assignment]
    unique_counts = summary["unique_sequence_counts_by_source"]  # type: ignore[assignment]
    ec_counts = summary["unique_with_ec_counts_by_source"]  # type: ignore[assignment]
    direct_ec_counts = summary["unique_with_direct_ec_counts_by_source"]  # type: ignore[assignment]
    lines = [
        "# Single-Enzyme Sequence Catalog",
        "",
        "This catalog considers only individual protein/enzyme sequences. Reaction pairs and reaction-only rows are not included.",
        "Duplicates are collapsed by exact cleaned amino-acid sequence.",
        "",
        "## Overall",
        "",
        f"- Source records with sequences: {summary['source_record_count']}",
        f"- Unique exact sequences: {summary['unique_sequence_count']}",
        f"- Duplicate source records removed: {summary['duplicate_records_removed_by_exact_sequence']}",
        f"- Unique sequences with any EC label: {summary['unique_with_any_ec']}",
        f"- Unique sequences with complete EC labels: {summary['unique_with_complete_ec']}",
        f"- Unique sequences still missing EC: {summary['unique_missing_any_ec']}",
        f"- Missing EC but carrying a UniProt-like accession: {summary['unique_missing_any_ec_with_uniprot_accession']}",
        f"- Missing EC and no usable accession: {summary['unique_missing_any_ec_without_accession']}",
        f"- EC rescued by lookup or exact-sequence duplicate rather than direct source label: {summary['unique_ec_rescued_by_lookup_or_duplicate']}",
        "",
        "## By Source",
        "",
        "| Source | Source records | Unique sequences | Direct source EC | EC after merge/rescue |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in sorted(source_counts):
        lines.append(
            f"| {source} | {source_counts[source]} | {unique_counts.get(source, 0)} | {direct_ec_counts.get(source, 0)} | {ec_counts.get(source, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `single_enzyme_catalog.csv`: exact-sequence deduplicated enzyme catalog.",
            "- `single_enzyme_source_records.csv`: one row per source sequence record, mapped to `seq_hash`.",
            "- `missing_ec_with_accessions.csv`: unresolved sequences that still have a UniProt-like accession.",
            "- `missing_ec_without_accessions.csv`: unresolved sequences without usable accessions; these require sequence search or manual curation.",
            "- `ec_coverage_summary.json`: machine-readable summary.",
            "",
            "EC rescue sources are direct dataset EC columns, exact-sequence duplicate propagation, and local UniProt/ClipZyme/EnzymeCAGE EC caches. No EC is assigned from reaction labels alone.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extra-uniprot-cache", type=Path, action="append", default=[])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print("Collecting source records...", flush=True)
    records = collect_records()
    print(f"Collected {len(records)} source records with sequences.", flush=True)
    print("Loading EC lookup caches...", flush=True)
    ec_lookup = load_ec_lookups(args.extra_uniprot_cache)
    print(f"Loaded EC mappings for {len(ec_lookup)} identifiers.", flush=True)
    print("Merging and deduplicating by exact sequence...", flush=True)
    catalog_rows, record_rows = merge_catalog(records, ec_lookup)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "single_enzyme_catalog.csv",
        catalog_rows,
        [
            "seq_hash",
            "length",
            "record_count",
            "sources",
            "dataset_splits",
            "protein_ids",
            "accessions",
            "ec_numbers",
            "complete_ec_numbers",
            "has_any_ec",
            "has_complete_ec",
            "direct_ec_numbers",
            "sequence",
        ],
    )
    write_csv(
        output_dir / "single_enzyme_source_records.csv",
        record_rows,
        [
            "record_index",
            "seq_hash",
            "source",
            "dataset_split",
            "protein_id",
            "accession",
            "length",
            "ec_numbers",
            "complete_ec_numbers",
            "direct_ec_numbers",
            "path",
        ],
    )
    missing_with_accessions = [row for row in catalog_rows if row["has_any_ec"] == "0" and row["accessions"]]
    missing_without_accessions = [row for row in catalog_rows if row["has_any_ec"] == "0" and not row["accessions"]]
    write_csv(output_dir / "missing_ec_with_accessions.csv", missing_with_accessions, list(catalog_rows[0].keys()))
    write_csv(output_dir / "missing_ec_without_accessions.csv", missing_without_accessions, list(catalog_rows[0].keys()))
    write_csv(
        output_dir / "missing_ec_accession_ids.csv",
        [{"protein_id": acc} for row in missing_with_accessions for acc in row["accessions"].split(";") if acc],
        ["protein_id"],
    )

    summary = summarize(catalog_rows, record_rows)
    (output_dir / "ec_coverage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown_summary(output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
