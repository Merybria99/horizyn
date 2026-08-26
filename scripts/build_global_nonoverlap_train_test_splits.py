#!/usr/bin/env python3
"""Build global non-overlapping train/test splits respecting source splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIFIED = PROJECT_ROOT / "data/test/unified_nonoverlap_proteins"
DEFAULT_OUT = PROJECT_ROOT / "results/sequence_overlap/global_nonoverlap_train_test_splits"
DEFAULT_INDEX = PROJECT_ROOT / "data/test/global_nonoverlap_train_test_splits"


@dataclass(frozen=True)
class PairSpec:
    task: str
    source_dataset: str
    source_split: str
    split_role: str
    pairs_path: Path
    reactions_path: Path | None = None


def clean_sequence(sequence: str) -> str:
    return sequence.replace(" ", "").replace("\n", "").replace("\r", "").upper()


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


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def write_ids(ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_reaction_smiles(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    return {
        row["reaction_id"]: row.get("reaction_smiles", "")
        for row in read_csv_rows(path)
        if row.get("reaction_id")
    }


def split_pipe(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    return [item for item in value.split("|") if item]


def load_exact_catalog(
    unified_dir: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[tuple[str, str], str]]:
    table_path = unified_dir / "exact/unified_proteins_exact.tsv"
    fasta_path = unified_dir / "exact/unified_proteins_exact.fasta"
    sequences = read_fasta(fasta_path)
    exact_rows: dict[str, dict[str, str]] = {}
    source_id_to_uid: dict[tuple[str, str], str] = {}

    with table_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {table_path}")
        source_columns = [column for column in reader.fieldnames if column.endswith("_ids")]
        for row in reader:
            uid = row["protein_uid"]
            exact_rows[uid] = row
            for column in source_columns:
                source = column.removesuffix("_ids")
                for source_id in split_pipe(row.get(column, "")):
                    source_id_to_uid.setdefault((source, source_id), uid)
    return exact_rows, sequences, source_id_to_uid


def load_cluster_catalog(
    unified_dir: Path,
    policy: str,
) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]], dict[str, str], dict[str, str]]:
    cluster_table = unified_dir / f"{policy}/unified_proteins_{policy}.tsv"
    member_table = unified_dir / f"{policy}/unified_proteins_{policy}_members.tsv"
    cluster_fasta = unified_dir / f"{policy}/unified_proteins_{policy}.fasta"
    cluster_sequences = read_fasta(cluster_fasta)

    clusters: dict[str, dict[str, str]] = {}
    with cluster_table.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            clusters[row["cluster_uid"]] = row

    cluster_members: dict[str, list[dict[str, str]]] = defaultdict(list)
    exact_uid_to_cluster: dict[str, str] = {}
    with member_table.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cluster_uid = row["cluster_uid"]
            cluster_members[cluster_uid].append(row)
            exact_uid_to_cluster[row["member_uid"]] = cluster_uid
    return clusters, dict(cluster_members), exact_uid_to_cluster, cluster_sequences


def train_specs() -> list[PairSpec]:
    reactzyme_root = PROJECT_ROOT / "data/paper/reactzyme/eval"
    specs = [
        PairSpec(
            task="horizyn_sota",
            source_dataset="horizyn",
            source_split="train",
            split_role="train",
            pairs_path=PROJECT_ROOT / "data/sota/train_pairs.csv",
            reactions_path=PROJECT_ROOT / "data/sota/train_rxns.csv",
        ),
        PairSpec(
            task="clipzyme_enzymemap",
            source_dataset="clipzyme",
            source_split="upstream_train",
            split_role="train",
            pairs_path=PROJECT_ROOT / "data/paper/clipzyme/train/enzymemap/pairs.csv",
            reactions_path=PROJECT_ROOT / "data/paper/clipzyme/train/enzymemap/reactions.csv",
        ),
    ]
    for split in ["time", "enzyme_smi", "reaction_smi"]:
        specs.append(
            PairSpec(
                task=f"reactzyme_{split}",
                source_dataset="reactzyme",
                source_split="train",
                split_role="train",
                pairs_path=reactzyme_root / split / "train_pairs.csv",
                reactions_path=reactzyme_root / split / "reactions.csv",
            )
        )
    return specs


def test_specs() -> list[PairSpec]:
    reactzyme_root = PROJECT_ROOT / "data/paper/reactzyme/eval"
    sabio_root = PROJECT_ROOT / "data/paper/sabio_rk/eval"
    specs = [
        PairSpec(
            task="horizyn_sota",
            source_dataset="horizyn",
            source_split="published_test",
            split_role="test",
            pairs_path=PROJECT_ROOT / "data/sota/test_pairs.csv",
            reactions_path=PROJECT_ROOT / "data/sota/test_rxns.csv",
        ),
        PairSpec(
            task="clipzyme_enzymemap",
            source_dataset="clipzyme",
            source_split="published_eval",
            split_role="test",
            pairs_path=PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/pairs.csv",
            reactions_path=PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/reactions.csv",
        ),
        PairSpec(
            task="sabio_rk_novelty90",
            source_dataset="sabio_all",
            source_split="novelty90",
            split_role="test",
            pairs_path=sabio_root / "novelty90/pairs.csv",
            reactions_path=sabio_root / "novelty90/reactions.csv",
        ),
        PairSpec(
            task="sabio_rk_novelty50",
            source_dataset="sabio_all",
            source_split="novelty50",
            split_role="test",
            pairs_path=sabio_root / "novelty50/pairs.csv",
            reactions_path=sabio_root / "novelty50/reactions.csv",
        ),
    ]
    for split in ["time", "enzyme_smi", "reaction_smi"]:
        specs.append(
            PairSpec(
                task=f"reactzyme_{split}",
                source_dataset="reactzyme",
                source_split="published_test",
                split_role="test",
                pairs_path=reactzyme_root / split / "test_pairs.csv",
                reactions_path=reactzyme_root / split / "reactions.csv",
            )
        )
    return specs


PAIR_FIELDS = [
    "split",
    "source_task",
    "source_dataset",
    "source_split",
    "global_reaction_id",
    "reaction_id",
    "split_protein_id",
    "protein_uid",
    "cluster_uid",
    "source_protein_id",
    "reaction_smiles",
    "source_pair_index",
]


def collect_rows(
    specs: list[PairSpec],
    split_name: str,
    source_id_to_uid: dict[tuple[str, str], str],
    uid_to_split_id: dict[str, str],
    blocked_split_ids: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    blocked_split_ids = blocked_split_ids or set()
    rows_out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    by_source = []
    totals = {
        "input_pairs": 0,
        "kept_pairs": 0,
        "missing_protein": 0,
        "filtered_by_test_overlap": 0,
        "duplicate_after_normalization": 0,
    }

    for spec in specs:
        reaction_smiles_by_id = load_reaction_smiles(spec.reactions_path)
        rows = read_csv_rows(spec.pairs_path)
        spec_stats = {
            "task": spec.task,
            "source_dataset": spec.source_dataset,
            "source_split": spec.source_split,
            "split_role": spec.split_role,
            "pairs_path": str(spec.pairs_path),
            "input_pairs": len(rows),
            "kept_pairs": 0,
            "missing_protein": 0,
            "filtered_by_test_overlap": 0,
            "duplicate_after_normalization": 0,
        }
        totals["input_pairs"] += len(rows)
        for idx, row in enumerate(rows):
            source_protein_id = (row.get("protein_id") or "").strip()
            reaction_id = (row.get("reaction_id") or "").strip()
            protein_uid = source_id_to_uid.get((spec.source_dataset, source_protein_id))
            if protein_uid is None:
                totals["missing_protein"] += 1
                spec_stats["missing_protein"] += 1
                continue
            split_protein_id = uid_to_split_id[protein_uid]
            if split_protein_id in blocked_split_ids:
                totals["filtered_by_test_overlap"] += 1
                spec_stats["filtered_by_test_overlap"] += 1
                continue
            reaction_smiles = row.get("reaction_smiles") or reaction_smiles_by_id.get(reaction_id, "")
            global_reaction_id = f"{spec.task}:{spec.source_split}:{reaction_id}"
            key = (global_reaction_id, split_protein_id, split_name)
            if key in seen:
                totals["duplicate_after_normalization"] += 1
                spec_stats["duplicate_after_normalization"] += 1
                continue
            seen.add(key)
            cluster_uid = split_protein_id if split_protein_id != protein_uid else ""
            rows_out.append(
                {
                    "split": split_name,
                    "source_task": spec.task,
                    "source_dataset": spec.source_dataset,
                    "source_split": spec.source_split,
                    "global_reaction_id": global_reaction_id,
                    "reaction_id": reaction_id,
                    "split_protein_id": split_protein_id,
                    "protein_uid": protein_uid,
                    "cluster_uid": cluster_uid,
                    "source_protein_id": source_protein_id,
                    "reaction_smiles": reaction_smiles,
                    "source_pair_index": str(idx),
                }
            )
            totals["kept_pairs"] += 1
            spec_stats["kept_pairs"] += 1
        by_source.append(spec_stats)

    totals["by_source"] = by_source
    totals["unique_proteins"] = len({row["split_protein_id"] for row in rows_out})
    return rows_out, totals


def write_pairs(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_protein_files(
    rows: list[dict[str, str]],
    split_name: str,
    split_sequences: dict[str, str],
    out_dir: Path,
) -> dict[str, str | int]:
    protein_ids = sorted({row["split_protein_id"] for row in rows})
    fasta_path = out_dir / f"{split_name}_proteins.fasta"
    table_path = out_dir / f"{split_name}_proteins.tsv"
    ids_path = out_dir / f"{split_name}_candidate_ids.txt"
    write_fasta([(protein_id, split_sequences[protein_id]) for protein_id in protein_ids], fasta_path)
    write_ids(protein_ids, ids_path)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "split_protein_id"], delimiter="\t")
        writer.writeheader()
        for protein_id in protein_ids:
            writer.writerow({"split": split_name, "split_protein_id": protein_id})
    return {
        "count": len(protein_ids),
        "fasta": str(fasta_path),
        "candidate_ids": str(ids_path),
        "table": str(table_path),
    }


def choose_test_member(cluster_uid: str, test_uids: set[str], cluster_members: dict[str, list[dict[str, str]]]) -> str:
    candidates = [
        member["member_uid"]
        for member in cluster_members[cluster_uid]
        if member["member_uid"] in test_uids
    ]
    if not candidates:
        raise ValueError(f"Cluster has no test member: {cluster_uid}")
    return sorted(candidates)[0]


def build_policy(
    policy: str,
    out_dir: Path,
    exact_rows: dict[str, dict[str, str]],
    exact_sequences: dict[str, str],
    source_id_to_uid: dict[tuple[str, str], str],
    train_pair_specs: list[PairSpec],
    test_pair_specs: list[PairSpec],
    unified_dir: Path,
) -> dict[str, object]:
    if policy == "exact":
        uid_to_split_id = {uid: uid for uid in exact_rows}
        split_sequences = exact_sequences
        cluster_members: dict[str, list[dict[str, str]]] = {}
    else:
        _clusters, cluster_members, uid_to_split_id, cluster_sequences = load_cluster_catalog(unified_dir, policy)
        split_sequences = dict(cluster_sequences)

    raw_test_rows, raw_test_stats = collect_rows(
        test_pair_specs,
        "test",
        source_id_to_uid,
        uid_to_split_id,
    )
    test_split_ids = {row["split_protein_id"] for row in raw_test_rows}
    test_uids = {row["protein_uid"] for row in raw_test_rows}

    if policy != "exact":
        for split_id in list(test_split_ids):
            split_sequences[split_id] = exact_sequences[
                choose_test_member(split_id, test_uids, cluster_members)
            ]

    train_rows, train_stats = collect_rows(
        train_pair_specs,
        "train",
        source_id_to_uid,
        uid_to_split_id,
        blocked_split_ids=test_split_ids,
    )
    # Recollect test rows after split ids are known so the same duplicate rules are
    # applied, while preserving all original test/eval sources.
    test_rows, test_stats = collect_rows(
        test_pair_specs,
        "test",
        source_id_to_uid,
        uid_to_split_id,
    )

    train_ids = {row["split_protein_id"] for row in train_rows}
    test_ids = {row["split_protein_id"] for row in test_rows}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise RuntimeError(f"{policy} produced train/test overlap: {len(overlap)} ids")

    pairs_dir = out_dir / "pairs"
    proteins_dir = out_dir / "proteins"
    write_pairs(train_rows, pairs_dir / "train_pairs.csv")
    write_pairs(test_rows, pairs_dir / "test_pairs.csv")

    by_task_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        by_task_rows[row["source_task"]].append(row)
    per_test_task = {}
    for task, rows in sorted(by_task_rows.items()):
        safe_task = task.replace("/", "_")
        task_path = pairs_dir / f"test_{safe_task}_pairs.csv"
        write_pairs(rows, task_path)
        task_ids = sorted({row["split_protein_id"] for row in rows})
        write_ids(task_ids, pairs_dir / f"test_{safe_task}_candidate_ids.txt")
        write_fasta(
            [(protein_id, split_sequences[protein_id]) for protein_id in task_ids],
            pairs_dir / f"test_{safe_task}_candidate_proteins.fasta",
        )
        per_test_task[task] = {
            "pairs": len(rows),
            "candidate_proteins": len(task_ids),
            "pairs_path": str(task_path),
        }

    write_ids(sorted(train_ids), pairs_dir / "train_candidate_ids.txt")
    write_ids(sorted(test_ids), pairs_dir / "test_candidate_ids.txt")
    write_fasta(
        [(protein_id, split_sequences[protein_id]) for protein_id in sorted(train_ids)],
        pairs_dir / "train_candidate_proteins.fasta",
    )
    write_fasta(
        [(protein_id, split_sequences[protein_id]) for protein_id in sorted(test_ids)],
        pairs_dir / "test_candidate_proteins.fasta",
    )

    train_proteins = write_protein_files(train_rows, "train", split_sequences, proteins_dir)
    test_proteins = write_protein_files(test_rows, "test", split_sequences, proteins_dir)

    return {
        "policy": policy,
        "train": {
            **train_stats,
            "pairs_path": str(pairs_dir / "train_pairs.csv"),
            "candidate_ids": str(pairs_dir / "train_candidate_ids.txt"),
            "candidate_proteins_fasta": str(pairs_dir / "train_candidate_proteins.fasta"),
            "proteins": train_proteins,
        },
        "test": {
            **test_stats,
            "pairs_path": str(pairs_dir / "test_pairs.csv"),
            "candidate_ids": str(pairs_dir / "test_candidate_ids.txt"),
            "candidate_proteins_fasta": str(pairs_dir / "test_candidate_proteins.fasta"),
            "proteins": test_proteins,
            "by_task_output": per_test_task,
        },
        "raw_test_before_train_filter": raw_test_stats,
        "train_test_overlap_after_filter": 0,
    }


def write_summary_counts(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    policies = summary["policies"]  # type: ignore[index]
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "policy",
            "train_pairs",
            "train_proteins",
            "train_pairs_filtered_by_test_overlap",
            "test_pairs",
            "test_proteins",
            "missing_train_proteins",
            "missing_test_proteins",
            "duplicate_train_pairs",
            "duplicate_test_pairs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy, policy_summary in policies.items():  # type: ignore[union-attr]
            train = policy_summary["train"]
            test = policy_summary["test"]
            writer.writerow(
                {
                    "policy": policy,
                    "train_pairs": train["kept_pairs"],
                    "train_proteins": train["unique_proteins"],
                    "train_pairs_filtered_by_test_overlap": train["filtered_by_test_overlap"],
                    "test_pairs": test["kept_pairs"],
                    "test_proteins": test["unique_proteins"],
                    "missing_train_proteins": train["missing_protein"],
                    "missing_test_proteins": test["missing_protein"],
                    "duplicate_train_pairs": train["duplicate_after_normalization"],
                    "duplicate_test_pairs": test["duplicate_after_normalization"],
                }
            )


def symlink_tree(source_dir: Path, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(source_dir)
        target = index_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        target.symlink_to(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-dir", type=Path, default=DEFAULT_UNIFIED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--policies", nargs="+", default=["exact", "nr90", "nr50"])
    parser.add_argument("--no-index-symlinks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exact_rows, exact_sequences, source_id_to_uid = load_exact_catalog(args.unified_dir)
    train_pair_specs = train_specs()
    test_pair_specs = test_specs()

    summary: dict[str, object] = {
        "unified_dir": str(args.unified_dir),
        "split_definition": {
            "train": "Union of original train pair files only.",
            "test": "Union of original published test/eval pair files, including SABIO-RK.",
            "filtering": "Train proteins overlapping any test protein under the selected policy are removed from train.",
        },
        "train_sources": [spec.__dict__ | {"pairs_path": str(spec.pairs_path), "reactions_path": str(spec.reactions_path) if spec.reactions_path else None} for spec in train_pair_specs],
        "test_sources": [spec.__dict__ | {"pairs_path": str(spec.pairs_path), "reactions_path": str(spec.reactions_path) if spec.reactions_path else None} for spec in test_pair_specs],
        "policies": {},
    }
    for policy in args.policies:
        summary["policies"][policy] = build_policy(  # type: ignore[index]
            policy,
            args.out_dir / policy,
            exact_rows,
            exact_sequences,
            source_id_to_uid,
            train_pair_specs,
            test_pair_specs,
            args.unified_dir,
        )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_counts(summary, args.out_dir / "summary_counts.csv")
    if not args.no_index_symlinks:
        symlink_tree(args.out_dir, args.index_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
