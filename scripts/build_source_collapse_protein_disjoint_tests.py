#!/usr/bin/env python3
"""Build protein-disjoint held-out retrieval test splits.

The source-collapse retrieval runs train on NR90 representative proteins, while
the published held-out tests use original protein IDs. This script filters test
positives and candidate lists by sequence, not by ID, so the resulting suites
can test novel proteins more strictly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


TASKS = {
    "horizyn_sota": {
        "dataset_dir": "horizyn",
        "pairs_file": "test_pairs.csv",
        "reactions_file": "test_rxns.csv",
        "candidate_file": "horizyn_sota_candidate_ids.txt",
    },
    "reactzyme_time": {
        "dataset_dir": "reactzyme_time",
        "pairs_file": "test_pairs.csv",
        "reactions_file": "reactions.csv",
        "candidate_file": "reactzyme_time_candidate_ids.txt",
    },
    "reactzyme_enzyme_smi": {
        "dataset_dir": "reactzyme_enzyme_smi",
        "pairs_file": "test_pairs.csv",
        "reactions_file": "reactions.csv",
        "candidate_file": "reactzyme_enzyme_smi_candidate_ids.txt",
    },
    "reactzyme_reaction_smi": {
        "dataset_dir": "reactzyme_reaction_smi",
        "pairs_file": "test_pairs.csv",
        "reactions_file": "reactions.csv",
        "candidate_file": "reactzyme_reaction_smi_candidate_ids.txt",
    },
}


def clean_sequence(seq: str) -> str:
    return "".join(seq.split()).upper()


def seq_hash(seq: str) -> str:
    return hashlib.sha256(clean_sequence(seq).encode("utf-8")).hexdigest()


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current_id: str | None = None
    parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = clean_sequence("".join(parts))
                current_id = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if current_id is not None:
            records[current_id] = clean_sequence("".join(parts))
    return records


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein_id, sequence in records:
            handle.write(f">{protein_id}\n")
            for idx in range(0, len(sequence), 80):
                handle.write(sequence[idx : idx + 80] + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def symlink_or_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(source)
    except OSError:
        shutil.copy2(source, dest)


def used_train_records(train_pairs: Path, train_fasta: Path) -> tuple[dict[str, str], list[str]]:
    sequences = read_fasta(train_fasta)
    protein_ids = sorted({row["protein_id"] for row in read_csv(train_pairs)})
    missing = [protein_id for protein_id in protein_ids if protein_id not in sequences]
    if missing:
        raise ValueError(f"{len(missing)} train proteins are missing from {train_fasta}")
    return sequences, protein_ids


def run_mmseqs_search(
    *,
    mmseqs_bin: Path,
    query_fasta: Path,
    target_fasta: Path,
    output_tsv: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    max_seqs: int,
    reuse: bool,
) -> None:
    if reuse and output_tsv.exists():
        return
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if output_tsv.exists():
        output_tsv.unlink()
    cmd = [
        str(mmseqs_bin),
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(output_tsv),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
        "--max-seqs",
        str(max_seqs),
        "--format-output",
        "query,target,pident,qcov,tcov,alnlen,evalue,bits",
    ]
    print("Running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_mmseqs_blocked(path: Path) -> dict[str, dict[str, str]]:
    blocked: dict[str, dict[str, str]] = {}
    if not path.exists():
        return blocked
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 8:
                continue
            query, target, pident, qcov, tcov, alnlen, evalue, bits = row[:8]
            if query not in blocked or float(bits) > float(blocked[query]["bits"]):
                blocked[query] = {
                    "query": query,
                    "target": target,
                    "pident": pident,
                    "qcov": qcov,
                    "tcov": tcov,
                    "alnlen": alnlen,
                    "evalue": evalue,
                    "bits": bits,
                }
    return blocked


def filter_reactions(reactions_path: Path, kept_reaction_ids: set[str], output_path: Path) -> int:
    rows = read_csv(reactions_path)
    kept = [row for row in rows if row["reaction_id"] in kept_reaction_ids]
    write_csv(output_path, list(rows[0].keys()) if rows else ["reaction_id", "reaction_smiles"], kept)
    return len(kept)


def build_filtered_suite(
    *,
    root: Path,
    suite_name: str,
    blocked_ids: set[str],
    shared_sequences: dict[str, str],
) -> dict[str, Any]:
    source_test = root / "test"
    source_shared = source_test / "horizyn_reactzyme_shared_candidates"
    output_root = source_test / suite_name
    output_shared = output_root / "horizyn_reactzyme_shared_candidates"
    output_root.mkdir(parents=True, exist_ok=True)
    output_shared.mkdir(parents=True, exist_ok=True)

    # Reuse the large candidate embedding stores; filtered candidate ID files
    # control which rows are evaluable.
    for name in (
        "proteins.fasta",
        "proteins_esm2_650m_residue.h5",
        "proteins_esmc_6b_residue.h5",
        "proteins_prott5_residue.h5",
    ):
        source = source_shared / name
        if source.exists() or source.is_symlink():
            symlink_or_copy(Path("../..") / "horizyn_reactzyme_shared_candidates" / name, output_shared / name)

    all_candidate_ids = read_ids(source_shared / "candidate_ids.txt")
    filtered_all_candidates = [protein_id for protein_id in all_candidate_ids if protein_id not in blocked_ids]
    write_ids(output_shared / "candidate_ids.txt", filtered_all_candidates)

    summary: dict[str, Any] = {
        "suite": suite_name,
        "blocked_proteins": len(blocked_ids),
        "source_candidate_ids": len(all_candidate_ids),
        "filtered_candidate_ids": len(filtered_all_candidates),
        "tasks": {},
    }

    for task_name, task in TASKS.items():
        source_dir = source_test / task["dataset_dir"]
        out_dir = output_root / task["dataset_dir"]
        pairs = read_csv(source_dir / task["pairs_file"])
        pair_fields = list(pairs[0].keys()) if pairs else ["reaction_id", "protein_id"]
        kept_pairs = [row for row in pairs if row["protein_id"] not in blocked_ids]
        removed_pairs = len(pairs) - len(kept_pairs)
        kept_reactions = {row["reaction_id"] for row in kept_pairs}
        write_csv(out_dir / task["pairs_file"], pair_fields, kept_pairs)
        reaction_count = filter_reactions(
            source_dir / task["reactions_file"],
            kept_reactions,
            out_dir / task["reactions_file"],
        )

        task_candidates = read_ids(source_shared / task["candidate_file"])
        filtered_candidates = [
            protein_id
            for protein_id in task_candidates
            if protein_id not in blocked_ids and protein_id in shared_sequences
        ]
        write_ids(output_shared / task["candidate_file"], filtered_candidates)

        positive_proteins = {row["protein_id"] for row in kept_pairs}
        missing_positive_candidates = positive_proteins - set(filtered_candidates)
        if missing_positive_candidates:
            raise ValueError(
                f"{task_name}: {len(missing_positive_candidates)} kept positives are absent "
                f"from filtered candidates"
            )
        summary["tasks"][task_name] = {
            "source_pairs": len(pairs),
            "kept_pairs": len(kept_pairs),
            "removed_pairs": removed_pairs,
            "source_unique_proteins": len({row["protein_id"] for row in pairs}),
            "kept_unique_proteins": len(positive_proteins),
            "source_unique_reactions": len({row["reaction_id"] for row in pairs}),
            "kept_unique_reactions": len(kept_reactions),
            "written_reactions": reaction_count,
            "source_candidates": len(task_candidates),
            "kept_candidates": len(filtered_candidates),
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create protein-disjoint source-collapse test splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/standardized/retrieval_training_source_collapse"),
    )
    parser.add_argument(
        "--train-pairs",
        type=Path,
        default=Path("data/standardized/retrieval_training_source_collapse/train_nr90/train_pairs_valid_rxn_nonempty_esmc.csv"),
    )
    parser.add_argument(
        "--train-fasta",
        type=Path,
        default=Path("data/standardized/retrieval_training_source_collapse/train_nr90/train_proteins.fasta"),
    )
    parser.add_argument(
        "--shared-candidate-fasta",
        type=Path,
        default=Path("data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/proteins.fasta"),
    )
    parser.add_argument("--mmseqs-bin", type=Path, default=Path("tools/mmseqs/bin/mmseqs"))
    parser.add_argument("--min-seq-id", type=float, default=0.90)
    parser.add_argument("--coverage", type=float, default=0.85)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--max-seqs", type=int, default=1)
    parser.add_argument("--reuse-mmseqs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    out_base = root / "test" / "protein_disjoint_filter_artifacts"
    mmseqs_dir = out_base / "mmseqs"
    out_base.mkdir(parents=True, exist_ok=True)

    train_sequences, used_train_ids = used_train_records(args.train_pairs, args.train_fasta)
    used_train_records_list = [(protein_id, train_sequences[protein_id]) for protein_id in used_train_ids]
    used_train_fasta = out_base / "used_train_proteins.fasta"
    write_fasta(used_train_records_list, used_train_fasta)

    shared_sequences = read_fasta(args.shared_candidate_fasta)
    train_hashes = {seq_hash(train_sequences[protein_id]) for protein_id in used_train_ids}
    exact_blocked = {
        protein_id
        for protein_id, sequence in shared_sequences.items()
        if seq_hash(sequence) in train_hashes
    }
    write_ids(out_base / "exact_blocked_candidate_ids.txt", sorted(exact_blocked))

    mmseqs_tsv = mmseqs_dir / "shared_candidates_vs_used_train_nr90.tsv"
    run_mmseqs_search(
        mmseqs_bin=args.mmseqs_bin,
        query_fasta=args.shared_candidate_fasta,
        target_fasta=used_train_fasta,
        output_tsv=mmseqs_tsv,
        tmp_dir=mmseqs_dir / "tmp",
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        threads=args.threads,
        max_seqs=args.max_seqs,
        reuse=args.reuse_mmseqs,
    )
    mmseqs_hits = read_mmseqs_blocked(mmseqs_tsv)
    nr90_blocked = set(exact_blocked) | set(mmseqs_hits)
    write_ids(out_base / "nr90_blocked_candidate_ids.txt", sorted(nr90_blocked))
    write_csv(
        out_base / "nr90_mmseqs_hits.csv",
        ["query", "target", "pident", "qcov", "tcov", "alnlen", "evalue", "bits"],
        [mmseqs_hits[key] for key in sorted(mmseqs_hits)],
    )

    summaries = {
        "train_pairs": str(args.train_pairs),
        "train_fasta": str(args.train_fasta),
        "shared_candidate_fasta": str(args.shared_candidate_fasta),
        "used_train_proteins": len(used_train_ids),
        "shared_candidate_proteins": len(shared_sequences),
        "exact_blocked_candidate_proteins": len(exact_blocked),
        "nr90_mmseqs_hit_candidate_proteins": len(mmseqs_hits),
        "nr90_blocked_candidate_proteins": len(nr90_blocked),
        "mmseqs": {
            "path": str(mmseqs_tsv),
            "min_seq_id": args.min_seq_id,
            "coverage": args.coverage,
            "cov_mode": args.cov_mode,
            "threads": args.threads,
            "max_seqs": args.max_seqs,
        },
        "suites": {},
    }
    summaries["suites"]["protein_disjoint_exact"] = build_filtered_suite(
        root=root,
        suite_name="protein_disjoint_exact",
        blocked_ids=exact_blocked,
        shared_sequences=shared_sequences,
    )
    summaries["suites"]["protein_disjoint_nr90"] = build_filtered_suite(
        root=root,
        suite_name="protein_disjoint_nr90",
        blocked_ids=nr90_blocked,
        shared_sequences=shared_sequences,
    )
    summary_path = out_base / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_rows = []
    for suite_name, suite_summary in summaries["suites"].items():
        for task_name, task_summary in suite_summary["tasks"].items():
            csv_rows.append({"suite": suite_name, "task": task_name, **task_summary})
    write_csv(
        out_base / "summary.csv",
        [
            "suite",
            "task",
            "source_pairs",
            "kept_pairs",
            "removed_pairs",
            "source_unique_proteins",
            "kept_unique_proteins",
            "source_unique_reactions",
            "kept_unique_reactions",
            "written_reactions",
            "source_candidates",
            "kept_candidates",
        ],
        csv_rows,
    )
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
