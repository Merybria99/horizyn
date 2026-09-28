#!/usr/bin/env python3
"""Build train/test splits with SABIO-RK held out from non-SABIO proteins."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIFIED = PROJECT_ROOT / "data/test/unified_nonoverlap_proteins"
DEFAULT_OUT = PROJECT_ROOT / "results/sequence_overlap/sabio_holdout_nonoverlap_splits"
DEFAULT_INDEX = PROJECT_ROOT / "data/test/sabio_holdout_nonoverlap_splits"


@dataclass(frozen=True)
class PairSpec:
    task: str
    source_dataset: str
    source_split: str
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


def write_candidate_ids(ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_reaction_smiles(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = read_csv_rows(path)
    return {
        row["reaction_id"]: row.get("reaction_smiles", "")
        for row in rows
        if row.get("reaction_id")
    }


def split_pipe(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    return [item for item in value.split("|") if item]


def load_exact_catalog(
    unified_dir: Path,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, str],
    dict[tuple[str, str], str],
]:
    table_path = unified_dir / "exact/unified_proteins_exact.tsv"
    fasta_path = unified_dir / "exact/unified_proteins_exact.fasta"
    sequences = read_fasta(fasta_path)
    rows: dict[str, dict[str, str]] = {}
    source_id_to_uid: dict[tuple[str, str], str] = {}

    with table_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {table_path}")
        source_columns = [column for column in reader.fieldnames if column.endswith("_ids")]
        for row in reader:
            uid = row["protein_uid"]
            rows[uid] = row
            for column in source_columns:
                source = column.removesuffix("_ids")
                for source_id in split_pipe(row.get(column, "")):
                    source_id_to_uid.setdefault((source, source_id), uid)
    return rows, sequences, source_id_to_uid


def uid_has_sabio(row: dict[str, str]) -> bool:
    return "sabio_all" in split_pipe(row.get("sources", ""))


def write_exact_protein_split(
    exact_rows: dict[str, dict[str, str]],
    exact_sequences: dict[str, str],
    out_dir: Path,
) -> dict[str, object]:
    train_uids = sorted(uid for uid, row in exact_rows.items() if not uid_has_sabio(row))
    test_uids = sorted(uid for uid, row in exact_rows.items() if uid_has_sabio(row))

    train_fasta = out_dir / "proteins/train_proteins.fasta"
    test_fasta = out_dir / "proteins/test_sabio_proteins.fasta"
    train_table = out_dir / "proteins/train_proteins.tsv"
    test_table = out_dir / "proteins/test_sabio_proteins.tsv"

    write_fasta([(uid, exact_sequences[uid]) for uid in train_uids], train_fasta)
    write_fasta([(uid, exact_sequences[uid]) for uid in test_uids], test_fasta)
    write_candidate_ids(train_uids, out_dir / "proteins/train_candidate_ids.txt")
    write_candidate_ids(test_uids, out_dir / "proteins/test_sabio_candidate_ids.txt")

    def write_table(uids: list[str], split_name: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "split",
                "split_protein_id",
                "protein_uid",
                "sequence_sha1",
                "length",
                "sources",
                "horizyn_ids",
                "reactzyme_ids",
                "clipzyme_ids",
                "sabio_all_ids",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for uid in uids:
                row = exact_rows[uid]
                writer.writerow(
                    {
                        "split": split_name,
                        "split_protein_id": uid,
                        "protein_uid": uid,
                        "sequence_sha1": row["sequence_sha1"],
                        "length": row["length"],
                        "sources": row["sources"],
                        "horizyn_ids": row.get("horizyn_ids", ""),
                        "reactzyme_ids": row.get("reactzyme_ids", ""),
                        "clipzyme_ids": row.get("clipzyme_ids", ""),
                        "sabio_all_ids": row.get("sabio_all_ids", ""),
                    }
                )

    write_table(train_uids, "train", train_table)
    write_table(test_uids, "test_sabio", test_table)
    return {
        "train_proteins": len(train_uids),
        "test_sabio_proteins": len(test_uids),
        "train_fasta": str(train_fasta),
        "test_sabio_fasta": str(test_fasta),
        "train_table": str(train_table),
        "test_sabio_table": str(test_table),
    }


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


def cluster_has_sabio(members: list[dict[str, str]]) -> bool:
    return any("sabio_all" in split_pipe(member.get("member_sources", "")) for member in members)


def choose_sabio_member(cluster_members: list[dict[str, str]]) -> str:
    sabio_uids = [
        member["member_uid"]
        for member in cluster_members
        if "sabio_all" in split_pipe(member.get("member_sources", ""))
    ]
    if not sabio_uids:
        raise ValueError("Cannot choose SABIO member from a non-SABIO cluster")
    return sorted(sabio_uids)[0]


def write_cluster_protein_split(
    policy: str,
    clusters: dict[str, dict[str, str]],
    cluster_members: dict[str, list[dict[str, str]]],
    cluster_sequences: dict[str, str],
    exact_rows: dict[str, dict[str, str]],
    exact_sequences: dict[str, str],
    out_dir: Path,
) -> dict[str, object]:
    train_clusters = sorted(
        cluster_uid
        for cluster_uid, members in cluster_members.items()
        if not cluster_has_sabio(members)
    )
    test_clusters = sorted(
        cluster_uid
        for cluster_uid, members in cluster_members.items()
        if cluster_has_sabio(members)
    )

    train_fasta = out_dir / "proteins/train_proteins.fasta"
    test_fasta = out_dir / "proteins/test_sabio_proteins.fasta"
    train_table = out_dir / "proteins/train_proteins.tsv"
    test_table = out_dir / "proteins/test_sabio_proteins.tsv"

    write_fasta([(cluster_uid, cluster_sequences[cluster_uid]) for cluster_uid in train_clusters], train_fasta)
    test_records = []
    for cluster_uid in test_clusters:
        sabio_uid = choose_sabio_member(cluster_members[cluster_uid])
        test_records.append((cluster_uid, exact_sequences[sabio_uid]))
    write_fasta(test_records, test_fasta)
    write_candidate_ids(train_clusters, out_dir / "proteins/train_candidate_ids.txt")
    write_candidate_ids(test_clusters, out_dir / "proteins/test_sabio_candidate_ids.txt")

    def write_table(cluster_uids: list[str], split_name: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "split",
                "split_protein_id",
                "cluster_uid",
                "original_representative_uid",
                "split_representative_uid",
                "split_representative_sources",
                "num_exact_sequences",
                "num_source_members",
                "sources",
                "min_length",
                "max_length",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for cluster_uid in cluster_uids:
                cluster = clusters[cluster_uid]
                if split_name == "test_sabio":
                    split_rep = choose_sabio_member(cluster_members[cluster_uid])
                else:
                    split_rep = cluster["representative_uid"]
                writer.writerow(
                    {
                        "split": split_name,
                        "split_protein_id": cluster_uid,
                        "cluster_uid": cluster_uid,
                        "original_representative_uid": cluster["representative_uid"],
                        "split_representative_uid": split_rep,
                        "split_representative_sources": exact_rows[split_rep]["sources"],
                        "num_exact_sequences": cluster["num_exact_sequences"],
                        "num_source_members": cluster["num_source_members"],
                        "sources": cluster["sources"],
                        "min_length": cluster["min_length"],
                        "max_length": cluster["max_length"],
                    }
                )

    write_table(train_clusters, "train", train_table)
    write_table(test_clusters, "test_sabio", test_table)
    return {
        "train_proteins": len(train_clusters),
        "test_sabio_proteins": len(test_clusters),
        "train_fasta": str(train_fasta),
        "test_sabio_fasta": str(test_fasta),
        "train_table": str(train_table),
        "test_sabio_table": str(test_table),
        "policy": policy,
    }


def default_train_pair_specs() -> list[PairSpec]:
    reactzyme_root = PROJECT_ROOT / "data/paper/reactzyme/eval"
    specs = [
        PairSpec(
            "horizyn_sota",
            "horizyn",
            "train",
            PROJECT_ROOT / "data/sota/train_pairs.csv",
            PROJECT_ROOT / "data/sota/train_rxns.csv",
        ),
        PairSpec(
            "horizyn_sota",
            "horizyn",
            "published_test",
            PROJECT_ROOT / "data/sota/test_pairs.csv",
            PROJECT_ROOT / "data/sota/test_rxns.csv",
        ),
        PairSpec(
            "clipzyme_enzymemap",
            "clipzyme",
            "all",
            PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/pairs.csv",
            PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/reactions.csv",
        ),
        PairSpec(
            "clipzyme_enzymemap",
            "clipzyme",
            "upstream_train",
            PROJECT_ROOT / "data/paper/clipzyme/train/enzymemap/pairs.csv",
            PROJECT_ROOT / "data/paper/clipzyme/train/enzymemap/reactions.csv",
        ),
    ]
    for split in ["time", "enzyme_smi", "reaction_smi"]:
        specs.extend(
            [
                PairSpec(
                    f"reactzyme_{split}",
                    "reactzyme",
                    "train",
                    reactzyme_root / split / "train_pairs.csv",
                    reactzyme_root / split / "reactions.csv",
                ),
                PairSpec(
                    f"reactzyme_{split}",
                    "reactzyme",
                    "published_test",
                    reactzyme_root / split / "test_pairs.csv",
                    reactzyme_root / split / "reactions.csv",
                ),
            ]
        )
    return specs


def default_test_pair_specs() -> list[PairSpec]:
    sabio_root = PROJECT_ROOT / "data/paper/sabio_rk/eval"
    return [
        PairSpec(
            "sabio_rk_novelty90",
            "sabio_all",
            "test",
            sabio_root / "novelty90/pairs.csv",
            sabio_root / "novelty90/reactions.csv",
        ),
        PairSpec(
            "sabio_rk_novelty50",
            "sabio_all",
            "test",
            sabio_root / "novelty50/pairs.csv",
            sabio_root / "novelty50/reactions.csv",
        ),
    ]


PAIR_FIELDNAMES = [
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


def normalize_pair_rows(
    specs: list[PairSpec],
    split_name: str,
    source_id_to_uid: dict[tuple[str, str], str],
    uid_to_split_protein_id: dict[str, str],
    allowed_split_ids: set[str],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    output_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    stats_by_spec = []
    total_rows = 0
    missing_protein = 0
    filtered_overlap = 0
    duplicates = 0

    for spec in specs:
        if not spec.pairs_path.exists():
            stats_by_spec.append(
                {
                    "task": spec.task,
                    "source_split": spec.source_split,
                    "pairs_path": str(spec.pairs_path),
                    "missing": True,
                }
            )
            continue
        reaction_smiles_by_id = load_reaction_smiles(spec.reactions_path)
        rows = read_csv_rows(spec.pairs_path)
        kept_for_spec = 0
        missing_for_spec = 0
        filtered_for_spec = 0
        duplicate_for_spec = 0
        for idx, row in enumerate(rows):
            total_rows += 1
            source_protein_id = (row.get("protein_id") or "").strip()
            reaction_id = (row.get("reaction_id") or "").strip()
            protein_uid = source_id_to_uid.get((spec.source_dataset, source_protein_id))
            if protein_uid is None:
                missing_protein += 1
                missing_for_spec += 1
                continue
            split_protein_id = uid_to_split_protein_id.get(protein_uid)
            if split_protein_id is None or split_protein_id not in allowed_split_ids:
                filtered_overlap += 1
                filtered_for_spec += 1
                continue
            reaction_smiles = row.get("reaction_smiles") or reaction_smiles_by_id.get(reaction_id, "")
            global_reaction_id = f"{spec.task}:{spec.source_split}:{reaction_id}"
            key = (global_reaction_id, split_protein_id, split_name)
            if key in seen:
                duplicates += 1
                duplicate_for_spec += 1
                continue
            seen.add(key)
            output_rows.append(
                {
                    "split": split_name,
                    "source_task": spec.task,
                    "source_dataset": spec.source_dataset,
                    "source_split": spec.source_split,
                    "global_reaction_id": global_reaction_id,
                    "reaction_id": reaction_id,
                    "split_protein_id": split_protein_id,
                    "protein_uid": protein_uid,
                    "cluster_uid": split_protein_id if split_protein_id != protein_uid else "",
                    "source_protein_id": source_protein_id,
                    "reaction_smiles": reaction_smiles,
                    "source_pair_index": str(idx),
                }
            )
            kept_for_spec += 1
        stats_by_spec.append(
            {
                "task": spec.task,
                "source_dataset": spec.source_dataset,
                "source_split": spec.source_split,
                "pairs_path": str(spec.pairs_path),
                "input_pairs": len(rows),
                "kept_pairs": kept_for_spec,
                "missing_protein": missing_for_spec,
                "filtered_by_sabio_holdout_overlap": filtered_for_spec,
                "duplicate_after_normalization": duplicate_for_spec,
            }
        )

    return output_rows, {
        "input_pairs": total_rows,
        "kept_pairs": len(output_rows),
        "missing_protein": missing_protein,
        "filtered_by_sabio_holdout_overlap": filtered_overlap,
        "duplicate_after_normalization": duplicates,
        "by_source": stats_by_spec,
    }


def write_pairs(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_pair_candidate_fasta(
    rows: list[dict[str, str]],
    protein_sequences: dict[str, str],
    path: Path,
) -> None:
    ids = sorted({row["split_protein_id"] for row in rows})
    write_fasta([(split_id, protein_sequences[split_id]) for split_id in ids], path)


def build_policy_pairs(
    policy: str,
    out_dir: Path,
    train_specs: list[PairSpec],
    test_specs: list[PairSpec],
    source_id_to_uid: dict[tuple[str, str], str],
    uid_to_split_protein_id: dict[str, str],
    train_split_ids: set[str],
    test_split_ids: set[str],
    split_sequences: dict[str, str],
) -> dict[str, object]:
    pairs_dir = out_dir / "pairs"
    train_rows, train_stats = normalize_pair_rows(
        train_specs,
        "train",
        source_id_to_uid,
        uid_to_split_protein_id,
        train_split_ids,
    )
    train_pairs_path = pairs_dir / "train_pairs.csv"
    write_pairs(train_rows, train_pairs_path)
    train_candidate_ids = sorted({row["split_protein_id"] for row in train_rows})
    write_candidate_ids(train_candidate_ids, pairs_dir / "train_candidate_ids.txt")
    write_pair_candidate_fasta(train_rows, split_sequences, pairs_dir / "train_candidate_proteins.fasta")

    test_summaries = {}
    for spec in test_specs:
        test_rows, test_stats = normalize_pair_rows(
            [spec],
            "test",
            source_id_to_uid,
            uid_to_split_protein_id,
            test_split_ids,
        )
        safe_task = spec.task.replace("/", "_")
        test_pairs_path = pairs_dir / f"test_{safe_task}_pairs.csv"
        write_pairs(test_rows, test_pairs_path)
        candidate_ids = sorted({row["split_protein_id"] for row in test_rows})
        write_candidate_ids(candidate_ids, pairs_dir / f"test_{safe_task}_candidate_ids.txt")
        write_pair_candidate_fasta(test_rows, split_sequences, pairs_dir / f"test_{safe_task}_candidate_proteins.fasta")
        test_summaries[spec.task] = {
            **test_stats,
            "pairs": str(test_pairs_path),
            "candidate_ids": str(pairs_dir / f"test_{safe_task}_candidate_ids.txt"),
            "candidate_proteins_fasta": str(pairs_dir / f"test_{safe_task}_candidate_proteins.fasta"),
            "unique_candidate_proteins": len(candidate_ids),
        }

    return {
        "policy": policy,
        "train_pairs": str(train_pairs_path),
        "train_candidate_ids": str(pairs_dir / "train_candidate_ids.txt"),
        "train_candidate_proteins_fasta": str(pairs_dir / "train_candidate_proteins.fasta"),
        "train": {
            **train_stats,
            "unique_candidate_proteins": len(train_candidate_ids),
        },
        "tests": test_summaries,
    }


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


def write_summary_counts(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    policies = summary["policies"]  # type: ignore[index]
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "policy",
            "train_proteins",
            "test_sabio_proteins",
            "train_pairs",
            "train_candidate_proteins",
            "train_pairs_filtered_by_sabio_overlap",
            "test_novelty90_pairs",
            "test_novelty90_candidate_proteins",
            "test_novelty50_pairs",
            "test_novelty50_candidate_proteins",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy, policy_summary in policies.items():  # type: ignore[union-attr]
            proteins = policy_summary["proteins"]
            pairs = policy_summary["pairs"]
            tests = pairs["tests"]
            writer.writerow(
                {
                    "policy": policy,
                    "train_proteins": proteins["train_proteins"],
                    "test_sabio_proteins": proteins["test_sabio_proteins"],
                    "train_pairs": pairs["train"]["kept_pairs"],
                    "train_candidate_proteins": pairs["train"]["unique_candidate_proteins"],
                    "train_pairs_filtered_by_sabio_overlap": pairs["train"][
                        "filtered_by_sabio_holdout_overlap"
                    ],
                    "test_novelty90_pairs": tests["sabio_rk_novelty90"]["kept_pairs"],
                    "test_novelty90_candidate_proteins": tests["sabio_rk_novelty90"][
                        "unique_candidate_proteins"
                    ],
                    "test_novelty50_pairs": tests["sabio_rk_novelty50"]["kept_pairs"],
                    "test_novelty50_candidate_proteins": tests["sabio_rk_novelty50"][
                        "unique_candidate_proteins"
                    ],
                }
            )


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
    train_specs = default_train_pair_specs()
    test_specs = default_test_pair_specs()

    summary: dict[str, object] = {
        "unified_dir": str(args.unified_dir),
        "split_definition": {
            "test": "SABIO-RK proteins/pairs",
            "train": "non-SABIO source proteins/pairs after removing SABIO exact or cluster overlap",
        },
        "policies": {},
    }

    for policy in args.policies:
        policy_out = args.out_dir / policy
        if policy == "exact":
            protein_summary = write_exact_protein_split(exact_rows, exact_sequences, policy_out)
            uid_to_split_protein_id = {uid: uid for uid in exact_rows}
            train_split_ids = {
                uid for uid, row in exact_rows.items() if not uid_has_sabio(row)
            }
            test_split_ids = {
                uid for uid, row in exact_rows.items() if uid_has_sabio(row)
            }
            split_sequences = exact_sequences
        else:
            clusters, cluster_members, uid_to_cluster, cluster_sequences = load_cluster_catalog(
                args.unified_dir,
                policy,
            )
            protein_summary = write_cluster_protein_split(
                policy,
                clusters,
                cluster_members,
                cluster_sequences,
                exact_rows,
                exact_sequences,
                policy_out,
            )
            uid_to_split_protein_id = uid_to_cluster
            train_split_ids = {
                cluster_uid
                for cluster_uid, members in cluster_members.items()
                if not cluster_has_sabio(members)
            }
            test_split_ids = {
                cluster_uid
                for cluster_uid, members in cluster_members.items()
                if cluster_has_sabio(members)
            }
            train_sequences = {
                cluster_uid: cluster_sequences[cluster_uid]
                for cluster_uid in train_split_ids
            }
            test_sequences = {
                cluster_uid: exact_sequences[choose_sabio_member(cluster_members[cluster_uid])]
                for cluster_uid in test_split_ids
            }
            split_sequences = {**train_sequences, **test_sequences}

        pair_summary = build_policy_pairs(
            policy,
            policy_out,
            train_specs,
            test_specs,
            source_id_to_uid,
            uid_to_split_protein_id,
            train_split_ids,
            test_split_ids,
            split_sequences,
        )
        summary["policies"][policy] = {  # type: ignore[index]
            "proteins": protein_summary,
            "pairs": pair_summary,
        }

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_counts(summary, args.out_dir / "summary_counts.csv")

    if not args.no_index_symlinks:
        symlink_tree(args.out_dir, args.index_dir)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
