#!/usr/bin/env python3
"""Evaluate exact sequence/accession overlap for AI4Protein/EC.

The AI4Protein/EC rows are compared against the existing Horizyn-side source
datasets and standardized global retrieval splits. Exact sequence overlap is
the primary signal because several benchmark datasets anonymize protein IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AI_DIR = PROJECT_ROOT / "data/huggingface/AI4Protein_EC"
DEFAULT_OUT = PROJECT_ROOT / "results/sequence_overlap/ai4protein_existing_overlap"


def clean_sequence(sequence: str) -> str:
    return sequence.replace(" ", "").replace("\n", "").replace("\r", "").upper()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = clean_sequence("".join(chunks))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        records[current_id] = clean_sequence("".join(chunks))
    return records


def read_pair_ids(path: Path) -> set[str]:
    rows = read_csv_rows(path)
    ids = set()
    for row in rows:
        protein_id = (row.get("protein_id") or row.get("enzyme_id") or "").strip()
        if protein_id:
            ids.add(protein_id)
    return ids


def subset_sequences(id_to_sequence: dict[str, str], ids: Iterable[str]) -> dict[str, str]:
    return {protein_id: id_to_sequence[protein_id] for protein_id in ids if protein_id in id_to_sequence}


def read_sabio_sequences(path: Path, set_filter: str | None = None) -> dict[str, str]:
    records: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if set_filter is not None and row.get("set") != set_filter:
                continue
            accession = row.get("accession", "").strip()
            sequence = clean_sequence(row.get("sequence", ""))
            if accession and sequence:
                records[accession] = sequence
    return records


def load_clipzyme_sequences(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    records: dict[str, str] = {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict in {path}, got {type(raw)!r}")
    for key, value in raw.items():
        protein_id = str(key).strip()
        if isinstance(value, bytes):
            sequence = value.decode("utf-8")
        else:
            sequence = str(value)
        sequence = clean_sequence(sequence)
        if protein_id and sequence:
            records[protein_id] = sequence
    return records


def extract_uniprot_accession(name: str) -> str:
    value = (name or "").strip()
    return value.rsplit("-", 1)[-1].strip() if "-" in value else value


@dataclass
class AIDataset:
    rows: list[dict[str, str]]
    sequence_by_row_id: dict[str, str]
    accession_by_row_id: dict[str, str]
    rows_by_sequence: dict[str, list[str]]
    rows_by_accession: dict[str, list[str]]
    split_rows: Counter[str]


@dataclass
class ExistingDataset:
    name: str
    group: str
    split_role: str
    records: dict[str, str]
    accession_like_ids: bool
    path: str


def load_ai_dataset(ai_dir: Path) -> AIDataset:
    rows: list[dict[str, str]] = []
    sequence_by_row_id: dict[str, str] = {}
    accession_by_row_id: dict[str, str] = {}
    rows_by_sequence: dict[str, list[str]] = defaultdict(list)
    rows_by_accession: dict[str, list[str]] = defaultdict(list)
    split_rows: Counter[str] = Counter()
    for split in ("train", "valid", "test"):
        annotated = ai_dir / f"{split}_with_ec.csv"
        source = annotated if annotated.exists() else ai_dir / f"{split}.csv"
        for idx, row in enumerate(read_csv_rows(source)):
            row_id = f"{split}:{idx}"
            sequence = clean_sequence(row.get("aa_seq", ""))
            accession = row.get("uniprot_accession", "").strip() or extract_uniprot_accession(row.get("name", ""))
            enriched = dict(row)
            enriched["ai_split"] = split
            enriched["ai_row_id"] = row_id
            enriched["uniprot_accession"] = accession
            rows.append(enriched)
            split_rows[split] += 1
            sequence_by_row_id[row_id] = sequence
            accession_by_row_id[row_id] = accession
            if sequence:
                rows_by_sequence[sequence].append(row_id)
            if accession:
                rows_by_accession[accession].append(row_id)
    return AIDataset(
        rows=rows,
        sequence_by_row_id=sequence_by_row_id,
        accession_by_row_id=accession_by_row_id,
        rows_by_sequence=dict(rows_by_sequence),
        rows_by_accession=dict(rows_by_accession),
        split_rows=split_rows,
    )


def build_existing_datasets(project_root: Path) -> list[ExistingDataset]:
    datasets: list[ExistingDataset] = []

    sota_fasta = project_root / "data/sota/prots.fasta"
    if sota_fasta.exists():
        sota_records = read_fasta(sota_fasta)
        for split in ("train", "test"):
            pairs_path = project_root / f"data/sota/{split}_pairs.csv"
            if pairs_path.exists():
                ids = read_pair_ids(pairs_path)
                datasets.append(
                    ExistingDataset(
                        name=f"horizyn_sota_{split}",
                        group="source",
                        split_role=split,
                        records=subset_sequences(sota_records, ids),
                        accession_like_ids=True,
                        path=str(pairs_path),
                    )
                )
        datasets.append(
            ExistingDataset(
                name="horizyn_sota_all_proteins",
                group="source",
                split_role="all",
                records=sota_records,
                accession_like_ids=True,
                path=str(sota_fasta),
            )
        )

    clipzyme_pickle = project_root / "data/paper/clipzyme/files/uniprot2sequence.p"
    if clipzyme_pickle.exists():
        clipzyme_records = load_clipzyme_sequences(clipzyme_pickle)
        for split, split_role in (("train", "train"), ("eval", "test")):
            pairs_path = project_root / f"data/paper/clipzyme/{split}/enzymemap/pairs.csv"
            if pairs_path.exists():
                ids = read_pair_ids(pairs_path)
                datasets.append(
                    ExistingDataset(
                        name=f"clipzyme_enzymemap_{split_role}",
                        group="source",
                        split_role=split_role,
                        records=subset_sequences(clipzyme_records, ids),
                        accession_like_ids=True,
                        path=str(pairs_path),
                    )
                )

    reactzyme_root = project_root / "data/paper/reactzyme/eval"
    for benchmark in ("time", "enzyme_smi", "reaction_smi"):
        proteins_path = reactzyme_root / benchmark / "proteins.fasta"
        if not proteins_path.exists():
            continue
        records = read_fasta(proteins_path)
        for split in ("train", "test"):
            pairs_path = reactzyme_root / benchmark / f"{split}_pairs.csv"
            if pairs_path.exists():
                ids = read_pair_ids(pairs_path)
                datasets.append(
                    ExistingDataset(
                        name=f"reactzyme_{benchmark}_{split}",
                        group="source",
                        split_role=split,
                        records=subset_sequences(records, ids),
                        accession_like_ids=False,
                        path=str(pairs_path),
                    )
                )
    reactzyme_all = reactzyme_root / "all_proteins.fasta"
    if reactzyme_all.exists():
        datasets.append(
            ExistingDataset(
                name="reactzyme_all_proteins",
                group="source",
                split_role="all",
                records=read_fasta(reactzyme_all),
                accession_like_ids=False,
                path=str(reactzyme_all),
            )
        )

    sabio_root = project_root / "data/paper/sabio_rk/eval"
    for novelty in ("novelty90", "novelty50"):
        sequence_path = sabio_root / novelty / "sequences.tsv"
        if sequence_path.exists():
            for set_filter in (None, "new"):
                suffix = "all" if set_filter is None else set_filter
                datasets.append(
                    ExistingDataset(
                        name=f"sabio_rk_{novelty}_{suffix}",
                        group="source",
                        split_role="test",
                        records=read_sabio_sequences(sequence_path, set_filter=set_filter),
                        accession_like_ids=True,
                        path=str(sequence_path),
                    )
                )

    standardized_root = project_root / "data/standardized/global_unique_retrieval"
    for policy in ("exact", "nr90", "nr50"):
        for split in ("train", "test"):
            fasta_path = standardized_root / policy / f"{split}_proteins.fasta"
            if fasta_path.exists():
                datasets.append(
                    ExistingDataset(
                        name=f"standardized_global_unique_{policy}_{split}",
                        group="standardized",
                        split_role=split,
                        records=read_fasta(fasta_path),
                        accession_like_ids=False,
                        path=str(fasta_path),
                    )
                )

    mcsa_fasta = project_root / "data/sleec_stage1/fasta/mcsa_reference.full.fasta"
    if mcsa_fasta.exists():
        datasets.append(
            ExistingDataset(
                name="sleec_stage1_mcsa_reference",
                group="source",
                split_role="train",
                records=read_fasta(mcsa_fasta),
                accession_like_ids=False,
                path=str(mcsa_fasta),
            )
        )

    return datasets


def summarize_overlap(ai: AIDataset, existing: ExistingDataset) -> dict[str, object]:
    existing_seq_to_ids: dict[str, list[str]] = defaultdict(list)
    for protein_id, sequence in existing.records.items():
        if sequence:
            existing_seq_to_ids[sequence].append(protein_id)

    existing_sequences = set(existing_seq_to_ids)
    ai_sequences = set(ai.rows_by_sequence)
    overlap_sequences = ai_sequences & existing_sequences

    overlap_row_ids: set[str] = set()
    split_overlap_rows: Counter[str] = Counter()
    for sequence in overlap_sequences:
        for row_id in ai.rows_by_sequence[sequence]:
            overlap_row_ids.add(row_id)
            split_overlap_rows[row_id.split(":", 1)[0]] += 1

    accession_overlap: set[str] = set()
    if existing.accession_like_ids:
        accession_overlap = set(ai.rows_by_accession) & set(existing.records)

    return {
        "dataset": existing.name,
        "group": existing.group,
        "split_role": existing.split_role,
        "existing_path": existing.path,
        "existing_records": len(existing.records),
        "existing_unique_sequences": len(existing_sequences),
        "ai_unique_sequence_overlap": len(overlap_sequences),
        "ai_unique_sequence_overlap_fraction": len(overlap_sequences) / max(1, len(ai_sequences)),
        "ai_row_overlap": len(overlap_row_ids),
        "ai_train_row_overlap": split_overlap_rows["train"],
        "ai_valid_row_overlap": split_overlap_rows["valid"],
        "ai_test_row_overlap": split_overlap_rows["test"],
        "accession_overlap": len(accession_overlap) if existing.accession_like_ids else "",
        "accession_overlap_fraction": (
            len(accession_overlap) / max(1, len(ai.rows_by_accession))
            if existing.accession_like_ids
            else ""
        ),
    }


def write_examples(
    ai: AIDataset,
    existing_datasets: list[ExistingDataset],
    output_path: Path,
    max_examples_per_dataset: int,
) -> None:
    by_row_id = {row["ai_row_id"]: row for row in ai.rows}
    fieldnames = [
        "dataset",
        "ai_split",
        "ai_name",
        "uniprot_accession",
        "label",
        "complete_ec_numbers",
        "sequence_length",
        "existing_ids",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for existing in existing_datasets:
            existing_seq_to_ids: dict[str, list[str]] = defaultdict(list)
            for protein_id, sequence in existing.records.items():
                if sequence:
                    existing_seq_to_ids[sequence].append(protein_id)
            written = 0
            for sequence, row_ids in ai.rows_by_sequence.items():
                existing_ids = existing_seq_to_ids.get(sequence)
                if not existing_ids:
                    continue
                for row_id in row_ids:
                    row = by_row_id[row_id]
                    writer.writerow(
                        {
                            "dataset": existing.name,
                            "ai_split": row["ai_split"],
                            "ai_name": row.get("name", ""),
                            "uniprot_accession": row.get("uniprot_accession", ""),
                            "label": row.get("label", ""),
                            "complete_ec_numbers": row.get("complete_ec_numbers", ""),
                            "sequence_length": len(sequence),
                            "existing_ids": "|".join(existing_ids[:20]),
                        }
                    )
                    written += 1
                    if written >= max_examples_per_dataset:
                        break
                if written >= max_examples_per_dataset:
                    break


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai-dir", default=str(DEFAULT_AI_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max-examples-per-dataset", type=int, default=25)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ai_dir = Path(args.ai_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ai = load_ai_dataset(ai_dir)
    existing_datasets = build_existing_datasets(PROJECT_ROOT)
    rows = [summarize_overlap(ai, dataset) for dataset in existing_datasets]
    rows.sort(key=lambda row: (str(row["group"]), str(row["split_role"]), str(row["dataset"])))
    write_csv(rows, output_dir / "overlap_summary.csv")
    write_examples(ai, existing_datasets, output_dir / "overlap_examples.csv", args.max_examples_per_dataset)

    metadata = {
        "ai_dir": str(ai_dir),
        "ai_rows": len(ai.rows),
        "ai_split_rows": dict(ai.split_rows),
        "ai_unique_sequences": len(ai.rows_by_sequence),
        "ai_unique_accessions": len(ai.rows_by_accession),
        "datasets_compared": len(existing_datasets),
        "overlap_summary": str(output_dir / "overlap_summary.csv"),
        "overlap_examples": str(output_dir / "overlap_examples.csv"),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
