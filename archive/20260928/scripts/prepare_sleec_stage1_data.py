#!/usr/bin/env python3
"""
Prepare the SLEEC stage-1 datasets from the paper.

Required paper datasets for stage 1:
  - mCSA functional residue labels for supervised training/validation.
  - MSA-derived pseudo-labels generated with Ligns/MMseqs against the chosen
    UniRef cluster database.

This script normalizes those sources into residue-label CSVs consumed by
``train_sleec_stage1.py``. It intentionally keeps the mCSA supervised labels
sequence-indexed. If raw mCSA API data are used, a FASTA containing the same
UniProt/reference sequences is required so negatives can be generated for every
non-functional residue, as described in the paper.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
import urllib.request
from urllib.parse import urljoin
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Ligns"))


def _load_sleec_stage1_module():
    """Load the SLEEC helpers without importing horizyn.__init__.

    The MSA preparation environment is intentionally lightweight. Importing the
    top-level package pulls in Lightning/TorchMetrics, which is unnecessary for
    pseudo-label generation and can fail when scientific binary wheels are not
    fully compatible in the MSA-only environment.
    """
    module_path = project_root / "horizyn" / "sleec_stage1.py"
    spec = importlib.util.spec_from_file_location("horizyn_sleec_stage1", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SLEEC stage-1 helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_sleec_stage1 = _load_sleec_stage1_module()
FastaRecord = _sleec_stage1.FastaRecord
ResidueLabelRecord = _sleec_stage1.ResidueLabelRecord
balance_binary_records = _sleec_stage1.balance_binary_records
load_residue_label_records = _sleec_stage1.load_residue_label_records
low_entropy_labels = _sleec_stage1.low_entropy_labels
msa_column_entropies = _sleec_stage1.msa_column_entropies
read_fasta = _sleec_stage1.read_fasta
write_residue_label_records = _sleec_stage1.write_residue_label_records


MCSA_ENTRIES_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json"
MCSA_RESIDUES_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/?format=json"
PAPER_MCSA_TRAIN_PROTEINS = 403
PAPER_MCSA_VAL_PROTEINS = 417


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", default="data/sleec_stage1/raw", type=Path)
    parser.add_argument("--output-dir", default="data/sleec_stage1", type=Path)
    parser.add_argument(
        "--mcsa-source",
        choices=("api", "labels-csv", "skip"),
        default="api",
        help="Use mCSA API, an already curated full residue-label CSV, or skip mCSA output.",
    )
    parser.add_argument(
        "--mcsa-fasta",
        type=Path,
        default=None,
        help=(
            "FASTA of mCSA proteins. Required with --mcsa-source api so every residue can "
            "receive a positive/negative supervised label."
        ),
    )
    parser.add_argument(
        "--mcsa-label-csv",
        type=Path,
        default=None,
        help=(
            "Curated sequence-indexed residue labels with columns protein_id,residue_index,label "
            "and optional split/source/entropy."
        ),
    )
    parser.add_argument(
        "--mcsa-index-base",
        type=int,
        default=0,
        help="Index base for --mcsa-label-csv residue_index values.",
    )
    parser.add_argument(
        "--mcsa-split-csv",
        type=Path,
        default=None,
        help="Optional CSV with protein_id,split columns. Omit for deterministic 50/50 split.",
    )
    parser.add_argument(
        "--mcsa-output",
        type=Path,
        default=None,
        help="Output mCSA residue-label CSV.",
    )
    parser.add_argument(
        "--msa-dir",
        type=Path,
        default=None,
        help="Directory containing Ligns/MMseqs A3M files.",
    )
    parser.add_argument("--msa-glob", default="**/*.a3m")
    parser.add_argument(
        "--pseudo-source",
        default="uniref50_msa_ligns",
        help="Source string written into generated pseudo-label rows.",
    )
    parser.add_argument(
        "--pseudo-output",
        type=Path,
        default=None,
        help="Output MSA pseudo-label CSV.",
    )
    parser.add_argument("--positive-fraction", type=float, default=0.10)
    parser.add_argument("--max-msas", type=int, default=None)
    parser.add_argument(
        "--query-id-source",
        choices=("first-record", "filename"),
        default="first-record",
        help="How to name the query protein for each generated pseudo-label set.",
    )
    parser.add_argument(
        "--balance-pseudo",
        action="store_true",
        help="Also write a balanced pseudo-label CSV beside the full pseudo labels.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch_paginated_json(url: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    next_url: str | None = url
    while next_url:
        payload = fetch_json(next_url)
        if isinstance(payload, list):
            records.extend(payload)
            break
        records.extend(payload.get("results", []))
        next_url = urljoin(next_url, payload["next"]) if payload.get("next") else None
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def canonical_protein_id(raw_id: str) -> str:
    token = raw_id.strip().split()[0]
    pieces = token.split("|")
    if len(pieces) >= 3 and pieces[1]:
        return pieces[1]
    return token.replace(".cif", "").replace(".pdb", "")


def load_split_csv(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"protein_id", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        return {canonical_protein_id(row["protein_id"]): row["split"] for row in reader}


def deterministic_half_split(protein_ids: list[str], *, seed: int) -> dict[str, str]:
    """Deterministic protein-level 50/50 split for local mCSA reproduction."""
    ids = sorted(set(protein_ids))
    train_count = max(1, int(round(0.5 * len(ids))))
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    train_ids = set(shuffled[:train_count])
    return {protein_id: "train" if protein_id in train_ids else "val" for protein_id in ids}


def extract_mcsa_positive_positions(entries: list[dict[str, Any]]) -> dict[str, set[int]]:
    positives: dict[str, set[int]] = {}
    for entry in entries:
        for residue in entry.get("residues", []):
            for seq_residue in residue.get("residue_sequences", []):
                if not seq_residue.get("is_reference", True):
                    continue
                uniprot_id = seq_residue.get("uniprot_id")
                resid = seq_residue.get("resid")
                if not uniprot_id or resid is None:
                    continue
                residue_index = int(resid) - 1
                if residue_index < 0:
                    continue
                positives.setdefault(canonical_protein_id(str(uniprot_id)), set()).add(
                    residue_index
                )
    return positives


def build_mcsa_labels_from_api(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.mcsa_fasta is None:
        raise ValueError("--mcsa-fasta is required with --mcsa-source api")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    entries_path = args.raw_dir / "mcsa_entries.json"
    residues_path = args.raw_dir / "mcsa_residues.json"

    print(f"Downloading mCSA entries from {MCSA_ENTRIES_URL}")
    entries = fetch_paginated_json(MCSA_ENTRIES_URL)
    write_json(entries_path, entries)
    print(f"Downloading mCSA residues from {MCSA_RESIDUES_URL}")
    residues = fetch_paginated_json(MCSA_RESIDUES_URL)
    write_json(residues_path, residues)

    positives = extract_mcsa_positive_positions(entries)
    fasta_records = read_fasta(args.mcsa_fasta)
    split_map = load_split_csv(args.mcsa_split_csv)
    split_is_paper_exact = bool(split_map)
    if not split_map:
        split_map = deterministic_half_split(
            [canonical_protein_id(record.protein_id) for record in fasta_records],
            seed=args.seed,
        )

    output_records: list[ResidueLabelRecord] = []
    skipped_positive = 0
    for record in fasta_records:
        protein_id = canonical_protein_id(record.protein_id)
        sequence = record.sequence
        protein_positives = positives.get(protein_id, set())
        for positive_idx in protein_positives:
            if positive_idx >= len(sequence):
                skipped_positive += 1
        valid_positives = {idx for idx in protein_positives if idx < len(sequence)}
        split = split_map.get(protein_id, "")
        for residue_index in range(len(sequence)):
            output_records.append(
                ResidueLabelRecord(
                    protein_id=protein_id,
                    residue_index=residue_index,
                    label=1 if residue_index in valid_positives else 0,
                    split=split,
                    source="mcsa",
                )
            )

    output_path = args.mcsa_output or args.output_dir / "mcsa_residue_labels.csv"
    count = write_residue_label_records(output_records, output_path)
    protein_ids = {record.protein_id for record in output_records}
    positive_count = sum(1 for record in output_records if record.label == 1)
    return count, {
        "output": str(output_path),
        "records": count,
        "proteins": len(protein_ids),
        "positives": positive_count,
        "skipped_positive_positions": skipped_positive,
        "entries_download": str(entries_path),
        "residues_download": str(residues_path),
        "paper_expected_proteins": 820,
        "paper_expected_functional_residues": 3716,
        "paper_exact_split": split_is_paper_exact,
        "fallback_split": f"deterministic_50_50_seed_{args.seed}",
    }


def normalize_curated_mcsa_csv(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.mcsa_label_csv is None:
        raise ValueError("--mcsa-label-csv is required with --mcsa-source labels-csv")
    records = load_residue_label_records(
        args.mcsa_label_csv,
        index_base=args.mcsa_index_base,
    )
    if not any(record.split for record in records):
        split_map = load_split_csv(args.mcsa_split_csv)
        if not split_map:
            split_map = deterministic_half_split(
                sorted({canonical_protein_id(record.protein_id) for record in records}),
                seed=args.seed,
            )
        records = [
            ResidueLabelRecord(
                protein_id=canonical_protein_id(record.protein_id),
                residue_index=record.residue_index,
                label=record.label,
                split=split_map.get(canonical_protein_id(record.protein_id), record.split),
                source=record.source or "mcsa",
                entropy=record.entropy,
            )
            for record in records
        ]
    output_path = args.mcsa_output or args.output_dir / "mcsa_residue_labels.csv"
    count = write_residue_label_records(records, output_path)
    return count, {
        "output": str(output_path),
        "records": count,
        "proteins": len({record.protein_id for record in records}),
        "positives": sum(1 for record in records if record.label == 1),
        "paper_expected_proteins": 820,
        "paper_expected_functional_residues": 3716,
        "paper_exact_split": args.mcsa_split_csv is not None,
    }


def read_ligns_a3m(msa_path: Path) -> list[FastaRecord]:
    try:
        from Ligns.utils.msa import read_msa as ligns_read_msa

        pairs = ligns_read_msa(msa_path, rm_insertions=True)
        return [
            FastaRecord(canonical_protein_id(description), sequence)
            for description, sequence in pairs
        ]
    except Exception:
        return [
            FastaRecord(canonical_protein_id(record.protein_id), record.sequence)
            for record in read_fasta(msa_path, a3m=True)
        ]


def protein_id_from_filename(msa_path: Path) -> str:
    stem = msa_path.stem
    if "_seqs" in stem:
        stem = stem.split("_seqs", 1)[0]
    if stem.endswith("_rm_ins"):
        stem = stem[: -len("_rm_ins")]
    return canonical_protein_id(stem)


def build_pseudo_labels(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.msa_dir is None:
        return 0, {"output": None, "records": 0, "msas": 0, "skipped": 0}
    msa_paths = sorted(args.msa_dir.glob(args.msa_glob))
    if args.max_msas is not None:
        msa_paths = msa_paths[: args.max_msas]
    output_records: list[ResidueLabelRecord] = []
    skipped: list[dict[str, str]] = []
    for msa_path in msa_paths:
        try:
            records = read_ligns_a3m(msa_path)
            if not records:
                raise ValueError("MSA contained no records")
            query_id = (
                records[0].protein_id
                if args.query_id_source == "first-record"
                else protein_id_from_filename(msa_path)
            )
            entropies = msa_column_entropies(records, query_id=records[0].protein_id)
            labels = low_entropy_labels(entropies, positive_fraction=args.positive_fraction)
            for residue_index, (entropy, label) in enumerate(zip(entropies, labels)):
                output_records.append(
                    ResidueLabelRecord(
                        protein_id=query_id,
                        residue_index=residue_index,
                        label=label,
                        split="train",
                        source=args.pseudo_source,
                        entropy=entropy,
                    )
                )
        except Exception as exc:
            skipped.append({"path": str(msa_path), "error": str(exc)})

    output_path = args.pseudo_output or args.output_dir / "msa_pseudo_residue_labels.csv"
    count = write_residue_label_records(output_records, output_path)
    balanced_output = None
    balanced_count = 0
    if args.balance_pseudo and output_records:
        balanced_records = balance_binary_records(output_records, seed=args.seed)
        balanced_output = output_path.with_name(f"{output_path.stem}.balanced{output_path.suffix}")
        balanced_count = write_residue_label_records(balanced_records, balanced_output)

    return count, {
        "output": str(output_path),
        "records": count,
        "proteins": len({record.protein_id for record in output_records}),
        "positives": sum(1 for record in output_records if record.label == 1),
        "msas_found": len(msa_paths),
        "msas_skipped": len(skipped),
        "skipped_examples": skipped[:20],
        "positive_fraction": args.positive_fraction,
        "balanced_output": str(balanced_output) if balanced_output is not None else None,
        "balanced_records": balanced_count,
        "paper_dataset": args.pseudo_source,
        "parser": "Ligns.utils.msa.read_msa(rm_insertions=True)",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "created_unix_time": time.time(),
        "paper": str((project_root.parent / "sources/2026.02.11.705200.full.pdf").resolve()),
        "stage": "SLEEC stage 1 functional residue classifier",
        "required_datasets_from_paper": {
            "supervised": (
                "mCSA residue labels. Paper reports 820 proteins, 3716 annotated "
                "functional residues, and a 403/417 train/val split; local reproduction "
                "uses a deterministic 50/50 split over available proteins unless a split "
                "CSV is provided."
            ),
            "pseudo": "MSAs; low-entropy 10% residues from Ligns/MMseqs alignments",
            "embeddings": "pre-generated ESM2 residue embeddings",
        },
    }

    if args.mcsa_source == "api":
        _, manifest["mcsa"] = build_mcsa_labels_from_api(args)
    elif args.mcsa_source == "labels-csv":
        _, manifest["mcsa"] = normalize_curated_mcsa_csv(args)
    else:
        manifest["mcsa"] = {"skipped": True}

    _, manifest["pseudo_labels"] = build_pseudo_labels(args)

    manifest_path = args.manifest or args.output_dir / "sleec_stage1_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
