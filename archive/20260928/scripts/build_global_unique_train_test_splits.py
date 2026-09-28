#!/usr/bin/env python3
"""Compare global train/test protein sets and keep only side-unique proteins."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIFIED = PROJECT_ROOT / "data/test/unified_nonoverlap_proteins"
DEFAULT_OUT = PROJECT_ROOT / "results/sequence_overlap/global_unique_train_test_splits"
DEFAULT_INDEX = PROJECT_ROOT / "data/test/global_unique_train_test_splits"


@dataclass(frozen=True)
class PairSpec:
    task: str
    source_dataset: str
    source_split: str
    split_role: str
    pairs_path: Path
    reactions_path: Path | None = None


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


def load_cluster_mapping(unified_dir: Path, policy: str) -> tuple[dict[str, str], dict[str, str]]:
    member_table = unified_dir / f"{policy}/unified_proteins_{policy}_members.tsv"
    cluster_fasta = unified_dir / f"{policy}/unified_proteins_{policy}.fasta"
    cluster_sequences = read_fasta(cluster_fasta)
    exact_uid_to_cluster: dict[str, str] = {}
    with member_table.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            exact_uid_to_cluster[row["member_uid"]] = row["cluster_uid"]
    return exact_uid_to_cluster, cluster_sequences


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


def collect_rows(
    specs: list[PairSpec],
    split_name: str,
    source_id_to_uid: dict[tuple[str, str], str],
    uid_to_split_id: dict[str, str],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    output_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    totals = {
        "input_pairs": 0,
        "kept_pairs": 0,
        "missing_protein": 0,
        "duplicate_after_normalization": 0,
    }
    by_source = []
    for spec in specs:
        rows = read_csv_rows(spec.pairs_path)
        reaction_smiles_by_id = load_reaction_smiles(spec.reactions_path)
        stats = {
            "task": spec.task,
            "source_dataset": spec.source_dataset,
            "source_split": spec.source_split,
            "split_role": spec.split_role,
            "pairs_path": str(spec.pairs_path),
            "input_pairs": len(rows),
            "kept_pairs": 0,
            "missing_protein": 0,
            "duplicate_after_normalization": 0,
        }
        totals["input_pairs"] += len(rows)
        for idx, row in enumerate(rows):
            source_protein_id = (row.get("protein_id") or "").strip()
            reaction_id = (row.get("reaction_id") or "").strip()
            protein_uid = source_id_to_uid.get((spec.source_dataset, source_protein_id))
            if protein_uid is None:
                totals["missing_protein"] += 1
                stats["missing_protein"] += 1
                continue
            split_protein_id = uid_to_split_id[protein_uid]
            key = (f"{spec.task}:{spec.source_split}:{reaction_id}", split_protein_id, split_name)
            if key in seen:
                totals["duplicate_after_normalization"] += 1
                stats["duplicate_after_normalization"] += 1
                continue
            seen.add(key)
            output_rows.append(
                {
                    "split": split_name,
                    "source_task": spec.task,
                    "source_dataset": spec.source_dataset,
                    "source_split": spec.source_split,
                    "global_reaction_id": key[0],
                    "reaction_id": reaction_id,
                    "split_protein_id": split_protein_id,
                    "protein_uid": protein_uid,
                    "cluster_uid": split_protein_id if split_protein_id != protein_uid else "",
                    "source_protein_id": source_protein_id,
                    "reaction_smiles": row.get("reaction_smiles") or reaction_smiles_by_id.get(reaction_id, ""),
                    "source_pair_index": str(idx),
                }
            )
            totals["kept_pairs"] += 1
            stats["kept_pairs"] += 1
        by_source.append(stats)
    totals["by_source"] = by_source
    totals["unique_proteins"] = len({row["split_protein_id"] for row in output_rows})
    return output_rows, totals


def filter_rows(rows: list[dict[str, str]], allowed_ids: set[str], split_name: str) -> list[dict[str, str]]:
    return [{**row, "split": split_name} for row in rows if row["split_protein_id"] in allowed_ids]


def write_pairs(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_protein_outputs(rows: list[dict[str, str]], split_name: str, sequences: dict[str, str], out_dir: Path) -> dict[str, object]:
    protein_ids = sorted({row["split_protein_id"] for row in rows})
    fasta = out_dir / f"{split_name}_proteins.fasta"
    ids = out_dir / f"{split_name}_candidate_ids.txt"
    table = out_dir / f"{split_name}_proteins.tsv"
    write_fasta([(protein_id, sequences[protein_id]) for protein_id in protein_ids], fasta)
    write_ids(protein_ids, ids)
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "split_protein_id"], delimiter="\t")
        writer.writeheader()
        for protein_id in protein_ids:
            writer.writerow({"split": split_name, "split_protein_id": protein_id})
    return {"count": len(protein_ids), "fasta": str(fasta), "candidate_ids": str(ids), "table": str(table)}


def write_task_outputs(rows: list[dict[str, str]], split_name: str, sequences: dict[str, str], out_dir: Path) -> dict[str, object]:
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[row["source_task"]].append(row)
    summary = {}
    for task, task_rows in sorted(by_task.items()):
        task_safe = task.replace("/", "_")
        pairs_path = out_dir / f"{split_name}_{task_safe}_pairs.csv"
        ids_path = out_dir / f"{split_name}_{task_safe}_candidate_ids.txt"
        fasta_path = out_dir / f"{split_name}_{task_safe}_candidate_proteins.fasta"
        task_ids = sorted({row["split_protein_id"] for row in task_rows})
        write_pairs(task_rows, pairs_path)
        write_ids(task_ids, ids_path)
        write_fasta([(protein_id, sequences[protein_id]) for protein_id in task_ids], fasta_path)
        summary[task] = {
            "pairs": len(task_rows),
            "candidate_proteins": len(task_ids),
            "pairs_path": str(pairs_path),
            "candidate_ids": str(ids_path),
            "candidate_proteins_fasta": str(fasta_path),
        }
    return summary


def write_split_outputs(
    split_name: str,
    rows: list[dict[str, str]],
    sequences: dict[str, str],
    out_dir: Path,
) -> dict[str, object]:
    pairs_dir = out_dir / "pairs"
    proteins_dir = out_dir / "proteins"
    pairs_path = pairs_dir / f"{split_name}_pairs.csv"
    ids_path = pairs_dir / f"{split_name}_candidate_ids.txt"
    fasta_path = pairs_dir / f"{split_name}_candidate_proteins.fasta"
    protein_ids = sorted({row["split_protein_id"] for row in rows})
    write_pairs(rows, pairs_path)
    write_ids(protein_ids, ids_path)
    write_fasta([(protein_id, sequences[protein_id]) for protein_id in protein_ids], fasta_path)
    return {
        "pairs": len(rows),
        "proteins": len(protein_ids),
        "pairs_path": str(pairs_path),
        "candidate_ids": str(ids_path),
        "candidate_proteins_fasta": str(fasta_path),
        "protein_files": write_protein_outputs(rows, split_name, sequences, proteins_dir),
        "by_task": write_task_outputs(rows, split_name, sequences, pairs_dir),
    }


def build_policy(
    policy: str,
    out_dir: Path,
    unified_dir: Path,
    exact_sequences: dict[str, str],
    source_id_to_uid: dict[tuple[str, str], str],
) -> dict[str, object]:
    if policy == "exact":
        uid_to_split_id = {uid: uid for uid in exact_sequences}
        sequences = exact_sequences
    else:
        uid_to_split_id, sequences = load_cluster_mapping(unified_dir, policy)

    raw_train_rows, raw_train_stats = collect_rows(train_specs(), "raw_train", source_id_to_uid, uid_to_split_id)
    raw_test_rows, raw_test_stats = collect_rows(test_specs(), "raw_test", source_id_to_uid, uid_to_split_id)
    train_ids = {row["split_protein_id"] for row in raw_train_rows}
    test_ids = {row["split_protein_id"] for row in raw_test_rows}
    overlap_ids = train_ids & test_ids
    train_unique_ids = train_ids - test_ids
    test_unique_ids = test_ids - train_ids

    train_unique_rows = filter_rows(raw_train_rows, train_unique_ids, "train_unique")
    test_unique_rows = filter_rows(raw_test_rows, test_unique_ids, "test_unique")
    train_overlap_rows = filter_rows(raw_train_rows, overlap_ids, "train_overlap")
    test_overlap_rows = filter_rows(raw_test_rows, overlap_ids, "test_overlap")

    return {
        "policy": policy,
        "raw_train": raw_train_stats,
        "raw_test": raw_test_stats,
        "protein_set_counts": {
            "raw_train_proteins": len(train_ids),
            "raw_test_proteins": len(test_ids),
            "overlap_proteins": len(overlap_ids),
            "train_unique_proteins": len(train_unique_ids),
            "test_unique_proteins": len(test_unique_ids),
        },
        "train_unique": write_split_outputs("train_unique", train_unique_rows, sequences, out_dir),
        "test_unique": write_split_outputs("test_unique", test_unique_rows, sequences, out_dir),
        "train_overlap": write_split_outputs("train_overlap", train_overlap_rows, sequences, out_dir),
        "test_overlap": write_split_outputs("test_overlap", test_overlap_rows, sequences, out_dir),
        "unique_train_test_intersection": len(train_unique_ids & test_unique_ids),
    }


def write_summary_counts(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "policy",
            "raw_train_pairs",
            "raw_train_proteins",
            "raw_test_pairs",
            "raw_test_proteins",
            "overlap_proteins",
            "train_unique_pairs",
            "train_unique_proteins",
            "test_unique_pairs",
            "test_unique_proteins",
            "train_overlap_pairs",
            "test_overlap_pairs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy, policy_summary in summary["policies"].items():  # type: ignore[index, union-attr]
            counts = policy_summary["protein_set_counts"]
            writer.writerow(
                {
                    "policy": policy,
                    "raw_train_pairs": policy_summary["raw_train"]["kept_pairs"],
                    "raw_train_proteins": counts["raw_train_proteins"],
                    "raw_test_pairs": policy_summary["raw_test"]["kept_pairs"],
                    "raw_test_proteins": counts["raw_test_proteins"],
                    "overlap_proteins": counts["overlap_proteins"],
                    "train_unique_pairs": policy_summary["train_unique"]["pairs"],
                    "train_unique_proteins": policy_summary["train_unique"]["proteins"],
                    "test_unique_pairs": policy_summary["test_unique"]["pairs"],
                    "test_unique_proteins": policy_summary["test_unique"]["proteins"],
                    "train_overlap_pairs": policy_summary["train_overlap"]["pairs"],
                    "test_overlap_pairs": policy_summary["test_overlap"]["pairs"],
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
    _exact_rows, exact_sequences, source_id_to_uid = load_exact_catalog(args.unified_dir)
    summary: dict[str, object] = {
        "unified_dir": str(args.unified_dir),
        "definition": {
            "raw_train": "Union of original train split proteins/pairs.",
            "raw_test": "Union of original test/evaluation split proteins/pairs.",
            "train_unique": "Raw train minus any protein also present in raw test under the policy.",
            "test_unique": "Raw test minus any protein also present in raw train under the policy.",
            "overlap": "Proteins/pairs whose normalized protein ID appears in both raw train and raw test.",
        },
        "train_sources": [
            spec.__dict__ | {
                "pairs_path": str(spec.pairs_path),
                "reactions_path": str(spec.reactions_path) if spec.reactions_path else None,
            }
            for spec in train_specs()
        ],
        "test_sources": [
            spec.__dict__ | {
                "pairs_path": str(spec.pairs_path),
                "reactions_path": str(spec.reactions_path) if spec.reactions_path else None,
            }
            for spec in test_specs()
        ],
        "policies": {},
    }
    for policy in args.policies:
        summary["policies"][policy] = build_policy(  # type: ignore[index]
            policy,
            args.out_dir / policy,
            args.unified_dir,
            exact_sequences,
            source_id_to_uid,
        )
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_counts(summary, args.out_dir / "summary_counts.csv")
    if not args.no_index_symlinks:
        symlink_tree(args.out_dir, args.index_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
