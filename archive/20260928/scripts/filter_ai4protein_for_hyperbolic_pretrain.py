#!/usr/bin/env python3
"""Build AI4Protein/EC subsets for hyperbolic enzyme pretraining.

The output keeps only AI4Protein proteins with complete EC-4 labels and removes
proteins that overlap a chosen existing training set. Two kinds of overlap are
handled:

* exact sequence overlap, which is robust for anonymized benchmark IDs
* UniProt accession overlap, which catches PDB-chain/fragments from the same
  source protein when the existing dataset has UniProt accessions
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_ai4protein_overlap import (  # noqa: E402
    ExistingDataset,
    build_existing_datasets,
    clean_sequence,
    load_ai_dataset,
)


DEFAULT_AI_DIR = PROJECT_ROOT / "data/huggingface/AI4Protein_EC"
DEFAULT_OUT = DEFAULT_AI_DIR / "hyperbolic_pretrain_subsets"


def split_semicolon(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def build_blocklist(
    existing_datasets: list[ExistingDataset],
    policy: str,
) -> tuple[set[str], set[str], list[str]]:
    blocked_sequences: set[str] = set()
    blocked_accessions: set[str] = set()
    sources: list[str] = []

    def include_dataset(dataset: ExistingDataset) -> bool:
        if policy in {"source_train_exact", "source_train_strict"}:
            return (
                dataset.group == "source"
                and dataset.split_role == "train"
                and not dataset.name.startswith("sleec_stage1")
            )
        if policy in {"source_train_or_test_exact", "source_train_or_test_strict"}:
            return (
                dataset.group == "source"
                and dataset.split_role in {"train", "test"}
                and not dataset.name.startswith("sleec_stage1")
            )
        if policy == "standardized_nr90_train_exact":
            return dataset.name == "standardized_global_unique_nr90_train"
        if policy == "standardized_exact_train_exact":
            return dataset.name == "standardized_global_unique_exact_train"
        if policy == "horizyn_sota_all_strict":
            return dataset.name == "horizyn_sota_all_proteins"
        raise ValueError(f"Unknown filter policy: {policy}")

    use_accessions = policy.endswith("_strict")
    for dataset in existing_datasets:
        if not include_dataset(dataset):
            continue
        sources.append(dataset.name)
        for protein_id, sequence in dataset.records.items():
            sequence = clean_sequence(sequence)
            if sequence:
                blocked_sequences.add(sequence)
            if use_accessions and dataset.accession_like_ids:
                blocked_accessions.add(protein_id)
    return blocked_sequences, blocked_accessions, sources


def write_fasta(records: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record['protein_id']}\n")
            sequence = str(record["sequence"])
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def write_csv(records: list[dict[str, object]], output_path: Path) -> None:
    if not records:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "protein_id",
        "uniprot_accessions",
        "ec_number",
        "source_ai_splits",
        "source_ai_names",
        "source_ai_labels",
        "num_source_rows",
        "sequence_length",
        "sequence",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def build_subset(ai_dir: Path, output_dir: Path, policy: str) -> dict[str, object]:
    ai = load_ai_dataset(ai_dir)
    existing_datasets = build_existing_datasets(PROJECT_ROOT)
    blocked_sequences, blocked_accessions, block_sources = build_blocklist(existing_datasets, policy)

    rows_with_complete_ec = 0
    dropped_no_complete_ec = 0
    dropped_exact_sequence = 0
    dropped_accession = 0
    kept_rows = 0

    by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ai.rows:
        sequence = clean_sequence(row.get("aa_seq", ""))
        accession = row.get("uniprot_accession", "")
        complete_ecs = split_semicolon(row.get("complete_ec_numbers", ""))
        if not complete_ecs:
            dropped_no_complete_ec += 1
            continue
        rows_with_complete_ec += 1
        if sequence in blocked_sequences:
            dropped_exact_sequence += 1
            continue
        if accession and accession in blocked_accessions:
            dropped_accession += 1
            continue
        kept_rows += 1
        by_sequence[sequence].append(row)

    records: list[dict[str, object]] = []
    duplicate_rows_collapsed = 0
    for idx, (sequence, rows) in enumerate(sorted(by_sequence.items(), key=lambda item: item[0])):
        duplicate_rows_collapsed += max(0, len(rows) - 1)
        accessions = sorted({row.get("uniprot_accession", "") for row in rows if row.get("uniprot_accession", "")})
        ecs = sorted({ec for row in rows for ec in split_semicolon(row.get("complete_ec_numbers", ""))})
        splits = sorted({row.get("ai_split", "") for row in rows if row.get("ai_split", "")})
        names = [row.get("name", "") for row in rows if row.get("name", "")]
        labels = sorted({label for row in rows for label in row.get("label", "").split(",") if label})
        prefix_acc = accessions[0] if accessions else "noacc"
        protein_id = f"AI4P_{idx:06d}_{prefix_acc}"
        records.append(
            {
                "protein_id": protein_id,
                "uniprot_accessions": "; ".join(accessions),
                "ec_number": "; ".join(ecs),
                "source_ai_splits": "; ".join(splits),
                "source_ai_names": "; ".join(names),
                "source_ai_labels": "; ".join(labels),
                "num_source_rows": len(rows),
                "sequence_length": len(sequence),
                "sequence": sequence,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ec_csv = output_dir / "ai4protein_hyperbolic_ec_labels.csv"
    fasta = output_dir / "ai4protein_hyperbolic_sequences.fasta"
    write_csv(records, ec_csv)
    write_fasta(records, fasta)

    split_counter: Counter[str] = Counter()
    for record in records:
        for split in split_semicolon(str(record["source_ai_splits"])):
            split_counter[split] += 1

    summary = {
        "policy": policy,
        "ai_dir": str(ai_dir),
        "output_dir": str(output_dir),
        "ec_labels_csv": str(ec_csv),
        "fasta": str(fasta),
        "block_sources": block_sources,
        "blocked_exact_sequences": len(blocked_sequences),
        "blocked_accessions": len(blocked_accessions),
        "ai_rows_total": len(ai.rows),
        "ai_unique_sequences_total": len(ai.rows_by_sequence),
        "ai_unique_accessions_total": len(ai.rows_by_accession),
        "rows_with_complete_ec_before_filter": rows_with_complete_ec,
        "dropped_no_complete_ec": dropped_no_complete_ec,
        "dropped_exact_sequence_overlap": dropped_exact_sequence,
        "dropped_accession_overlap": dropped_accession,
        "kept_rows_before_sequence_dedup": kept_rows,
        "kept_unique_sequences": len(records),
        "duplicate_rows_collapsed": duplicate_rows_collapsed,
        "kept_unique_sequences_by_original_ai_split": dict(split_counter),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai-dir", default=str(DEFAULT_AI_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        choices=[
            "source_train_exact",
            "source_train_strict",
            "source_train_or_test_exact",
            "source_train_or_test_strict",
            "standardized_nr90_train_exact",
            "standardized_exact_train_exact",
            "horizyn_sota_all_strict",
        ],
        help="Filter policy. Can be passed multiple times.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    policies = args.policy or [
        "source_train_exact",
        "source_train_strict",
        "standardized_nr90_train_exact",
        "horizyn_sota_all_strict",
    ]
    output_root = Path(args.output_root)
    summaries = []
    for policy in policies:
        summaries.append(build_subset(Path(args.ai_dir), output_root / policy, policy))

    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "summary.csv"
    fields = [
        "policy",
        "rows_with_complete_ec_before_filter",
        "dropped_exact_sequence_overlap",
        "dropped_accession_overlap",
        "kept_rows_before_sequence_dedup",
        "kept_unique_sequences",
        "duplicate_rows_collapsed",
        "blocked_exact_sequences",
        "blocked_accessions",
        "ec_labels_csv",
        "fasta",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in fields})
    print(json.dumps({"summary_csv": str(summary_csv), "policies": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
