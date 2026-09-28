#!/usr/bin/env python3
"""Build a unified nonredundant protein catalog across benchmark datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENZYME_DISCOVERY_ROOT = PROJECT_ROOT.parent

DEFAULT_HORIZYN = PROJECT_ROOT / "data/sota/prots.fasta"
DEFAULT_REACTZYME = PROJECT_ROOT / "data/paper/reactzyme/eval/all_proteins.fasta"
DEFAULT_CLIPZYME = PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/proteins.fasta"
DEFAULT_SABIO_ALL = ENZYME_DISCOVERY_ROOT / "sabio_rk_download/seqsim_fetch/new_sequences.tsv"
DEFAULT_OUT = PROJECT_ROOT / "results/sequence_overlap/unified_nonoverlap_proteins"
DEFAULT_INDEX = PROJECT_ROOT / "data/test/unified_nonoverlap_proteins"
DEFAULT_MMSEQS = PROJECT_ROOT / "tools/mmseqs/bin/mmseqs"


@dataclass(frozen=True)
class SourceEntry:
    source: str
    record_id: str


@dataclass
class ExactProtein:
    uid: str
    sha1: str
    sequence: str
    members: list[SourceEntry]


def clean_sequence(sequence: str) -> str:
    return sequence.replace(" ", "").replace("\n", "").replace("\r", "").upper()


def sequence_sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, clean_sequence("".join(chunks))))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            records.append((current_id, clean_sequence("".join(chunks))))
    return records


def read_sabio_tsv(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        missing = {"accession", "sequence"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            accession = (row.get("accession") or "").strip()
            sequence = clean_sequence(row.get("sequence") or "")
            if accession and sequence:
                records.append((accession, sequence))
    return records


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def dataset_stats(records: list[tuple[str, str]]) -> dict[str, int | float]:
    nonempty = [(record_id, sequence) for record_id, sequence in records if sequence]
    lengths = [len(sequence) for _record_id, sequence in nonempty]
    return {
        "records": len(records),
        "nonempty_sequences": len(nonempty),
        "unique_ids": len({record_id for record_id, _sequence in nonempty}),
        "unique_sequences": len({sequence for _record_id, sequence in nonempty}),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
    }


def load_sources(args: argparse.Namespace) -> dict[str, list[tuple[str, str]]]:
    datasets = {
        "horizyn": read_fasta(args.horizyn_fasta),
        "reactzyme": read_fasta(args.reactzyme_fasta),
        "clipzyme": read_fasta(args.clipzyme_fasta),
    }
    if args.sabio_all_tsv:
        datasets["sabio_all"] = read_sabio_tsv(args.sabio_all_tsv)
    return datasets


def choose_representative(
    members: list[SourceEntry],
    source_priority: dict[str, int],
) -> SourceEntry:
    return sorted(
        members,
        key=lambda member: (source_priority.get(member.source, 999), member.source, member.record_id),
    )[0]


def build_exact_catalog(
    datasets: dict[str, list[tuple[str, str]]],
    source_priority: list[str],
) -> list[ExactProtein]:
    sha_to_sequence: dict[str, str] = {}
    sha_to_members: dict[str, list[SourceEntry]] = defaultdict(list)
    seen_source_ids: set[tuple[str, str, str]] = set()

    for source, records in datasets.items():
        for record_id, sequence in records:
            if not sequence:
                continue
            sha = sequence_sha1(sequence)
            key = (sha, source, record_id)
            if key in seen_source_ids:
                continue
            seen_source_ids.add(key)
            sha_to_sequence[sha] = sequence
            sha_to_members[sha].append(SourceEntry(source=source, record_id=record_id))

    priority = {source: idx for idx, source in enumerate(source_priority)}
    exact: list[ExactProtein] = []
    for sha in sorted(sha_to_sequence):
        members = sha_to_members[sha]
        representative = choose_representative(members, priority)
        uid = f"uprot_{sha[:16]}"
        # The uid is intentionally based on the sequence hash, not the chosen
        # representative, so it stays stable if source priority changes.
        exact.append(
            ExactProtein(
                uid=uid,
                sha1=sha,
                sequence=sha_to_sequence[sha],
                members=members,
            )
        )
        if representative not in members:
            raise RuntimeError("Representative selection produced a non-member")
    return exact


def source_id_columns(sources: list[str], members: list[SourceEntry]) -> dict[str, str]:
    by_source: dict[str, list[str]] = {source: [] for source in sources}
    for member in members:
        by_source.setdefault(member.source, []).append(member.record_id)
    return {
        f"{source}_ids": "|".join(sorted(set(ids)))
        for source, ids in by_source.items()
    }


def write_exact_outputs(
    exact: list[ExactProtein],
    datasets: dict[str, list[tuple[str, str]]],
    source_priority: list[str],
    out_dir: Path,
) -> dict[str, object]:
    exact_dir = out_dir / "exact"
    exact_dir.mkdir(parents=True, exist_ok=True)
    sources = list(datasets)
    priority = {source: idx for idx, source in enumerate(source_priority)}

    fasta_path = exact_dir / "unified_proteins_exact.fasta"
    table_path = exact_dir / "unified_proteins_exact.tsv"
    members_path = exact_dir / "unified_proteins_exact_members.tsv"
    source_private_fasta_path = exact_dir / "unified_proteins_exact_source_private.fasta"
    source_private_table_path = exact_dir / "unified_proteins_exact_source_private.tsv"
    stats_path = exact_dir / "source_dataset_stats.csv"

    write_fasta([(protein.uid, protein.sequence) for protein in exact], fasta_path)

    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "protein_uid",
            "sequence_sha1",
            "length",
            "representative_source",
            "representative_id",
            "sources",
            "source_count",
            "member_count",
        ] + [f"{source}_ids" for source in sources]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for protein in exact:
            representative = choose_representative(protein.members, priority)
            present_sources = sorted({member.source for member in protein.members})
            row = {
                "protein_uid": protein.uid,
                "sequence_sha1": protein.sha1,
                "length": len(protein.sequence),
                "representative_source": representative.source,
                "representative_id": representative.record_id,
                "sources": "|".join(present_sources),
                "source_count": len(present_sources),
                "member_count": len(protein.members),
            }
            row.update(source_id_columns(sources, protein.members))
            writer.writerow(row)

    source_private = [
        protein
        for protein in exact
        if len({member.source for member in protein.members}) == 1
    ]
    write_fasta(
        [(protein.uid, protein.sequence) for protein in source_private],
        source_private_fasta_path,
    )
    with source_private_table_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "protein_uid",
            "sequence_sha1",
            "length",
            "source",
            "source_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for protein in source_private:
            member = protein.members[0]
            writer.writerow(
                {
                    "protein_uid": protein.uid,
                    "sequence_sha1": protein.sha1,
                    "length": len(protein.sequence),
                    "source": member.source,
                    "source_id": member.record_id,
                }
            )

    with members_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_uid", "sequence_sha1", "source", "source_id", "length"],
            delimiter="\t",
        )
        writer.writeheader()
        for protein in exact:
            for member in sorted(protein.members, key=lambda item: (item.source, item.record_id)):
                writer.writerow(
                    {
                        "protein_uid": protein.uid,
                        "sequence_sha1": protein.sha1,
                        "source": member.source,
                        "source_id": member.record_id,
                        "length": len(protein.sequence),
                    }
                )

    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "source",
            "records",
            "nonempty_sequences",
            "unique_ids",
            "unique_sequences",
            "min_length",
            "max_length",
            "mean_length",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source, records in datasets.items():
            row = {"source": source}
            row.update(dataset_stats(records))
            writer.writerow(row)

    shared_by_source_count = defaultdict(int)
    for protein in exact:
        shared_by_source_count[len({member.source for member in protein.members})] += 1

    return {
        "fasta": str(fasta_path),
        "table": str(table_path),
        "members": str(members_path),
        "source_private_fasta": str(source_private_fasta_path),
        "source_private_table": str(source_private_table_path),
        "source_stats": str(stats_path),
        "exact_unique_sequences": len(exact),
        "exact_source_private_sequences": len(source_private),
        "total_source_records": sum(len(records) for records in datasets.values()),
        "shared_by_source_count": dict(sorted(shared_by_source_count.items())),
    }


def find_mmseqs(path: Path) -> Path:
    if path.exists():
        return path
    found = shutil.which("mmseqs")
    if found:
        return Path(found)
    raise FileNotFoundError("Could not find MMseqs; pass --mmseqs-bin")


def run_linclust(
    mmseqs: Path,
    input_fasta: Path,
    cluster_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    reuse: bool,
) -> Path:
    cluster_tsv = Path(str(cluster_prefix) + "_cluster.tsv")
    if reuse and cluster_tsv.exists():
        print(f"reusing {cluster_tsv}", flush=True)
        return cluster_tsv
    cluster_prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(mmseqs),
        "easy-linclust",
        str(input_fasta),
        str(cluster_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
    ]
    print("running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"MMseqs did not create expected cluster TSV: {cluster_tsv}")
    return cluster_tsv


def read_cluster_tsv(cluster_tsv: Path, all_uids: set[str]) -> dict[str, list[str]]:
    clusters: dict[str, set[str]] = defaultdict(set)
    with cluster_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            representative, member = row[0], row[1]
            if representative not in all_uids or member not in all_uids:
                continue
            clusters[representative].add(member)

    assigned = set()
    for representative, members in clusters.items():
        members.add(representative)
        assigned.update(members)
    for uid in sorted(all_uids - assigned):
        clusters[uid].add(uid)
    return {representative: sorted(members) for representative, members in sorted(clusters.items())}


def write_cluster_outputs(
    exact: list[ExactProtein],
    clusters: dict[str, list[str]],
    threshold: float,
    out_dir: Path,
) -> dict[str, object]:
    threshold_name = f"nr{int(threshold)}"
    cluster_dir = out_dir / threshold_name
    cluster_dir.mkdir(parents=True, exist_ok=True)
    protein_by_uid = {protein.uid: protein for protein in exact}

    fasta_path = cluster_dir / f"unified_proteins_{threshold_name}.fasta"
    cluster_table_path = cluster_dir / f"unified_proteins_{threshold_name}.tsv"
    members_path = cluster_dir / f"unified_proteins_{threshold_name}_members.tsv"

    representative_records = []
    for representative_uid in sorted(clusters):
        representative = protein_by_uid[representative_uid]
        cluster_uid = f"{threshold_name}_{representative_uid.removeprefix('uprot_')}"
        representative_records.append((cluster_uid, representative.sequence))
    write_fasta(representative_records, fasta_path)

    with cluster_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_uid",
                "representative_uid",
                "representative_sha1",
                "representative_length",
                "num_exact_sequences",
                "num_source_members",
                "sources",
                "min_length",
                "max_length",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for representative_uid, member_uids in sorted(clusters.items()):
            representative = protein_by_uid[representative_uid]
            member_proteins = [protein_by_uid[uid] for uid in member_uids]
            source_members = [member for protein in member_proteins for member in protein.members]
            lengths = [len(protein.sequence) for protein in member_proteins]
            cluster_uid = f"{threshold_name}_{representative_uid.removeprefix('uprot_')}"
            writer.writerow(
                {
                    "cluster_uid": cluster_uid,
                    "representative_uid": representative_uid,
                    "representative_sha1": representative.sha1,
                    "representative_length": len(representative.sequence),
                    "num_exact_sequences": len(member_uids),
                    "num_source_members": len(source_members),
                    "sources": "|".join(sorted({member.source for member in source_members})),
                    "min_length": min(lengths),
                    "max_length": max(lengths),
                }
            )

    with members_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_uid",
                "representative_uid",
                "member_uid",
                "member_sha1",
                "member_length",
                "member_sources",
                "source_member_count",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for representative_uid, member_uids in sorted(clusters.items()):
            cluster_uid = f"{threshold_name}_{representative_uid.removeprefix('uprot_')}"
            for member_uid in member_uids:
                protein = protein_by_uid[member_uid]
                source_members = protein.members
                writer.writerow(
                    {
                        "cluster_uid": cluster_uid,
                        "representative_uid": representative_uid,
                        "member_uid": member_uid,
                        "member_sha1": protein.sha1,
                        "member_length": len(protein.sequence),
                        "member_sources": "|".join(sorted({member.source for member in source_members})),
                        "source_member_count": len(source_members),
                    }
                )

    cluster_sizes = [len(members) for members in clusters.values()]
    return {
        "threshold": threshold,
        "name": threshold_name,
        "representative_fasta": str(fasta_path),
        "cluster_table": str(cluster_table_path),
        "cluster_members": str(members_path),
        "num_clusters": len(clusters),
        "num_exact_sequences_clustered": sum(cluster_sizes),
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "mean_cluster_size": round(sum(cluster_sizes) / len(cluster_sizes), 3) if cluster_sizes else 0.0,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizyn-fasta", type=Path, default=DEFAULT_HORIZYN)
    parser.add_argument("--reactzyme-fasta", type=Path, default=DEFAULT_REACTZYME)
    parser.add_argument("--clipzyme-fasta", type=Path, default=DEFAULT_CLIPZYME)
    parser.add_argument("--sabio-all-tsv", type=Path, default=DEFAULT_SABIO_ALL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mmseqs-bin", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--cluster-thresholds", type=float, nargs="*", default=[])
    parser.add_argument("--coverage", type=float, default=0.85)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--reuse-mmseqs", action="store_true")
    parser.add_argument(
        "--source-priority",
        nargs="+",
        default=["horizyn", "clipzyme", "reactzyme", "sabio_all"],
        help="Priority for choosing a representative source ID among exact duplicates.",
    )
    parser.add_argument("--no-index-symlinks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_sources(args)
    exact = build_exact_catalog(datasets, args.source_priority)
    summary: dict[str, object] = {
        "inputs": {source: str(path) for source, path in {
            "horizyn": args.horizyn_fasta,
            "reactzyme": args.reactzyme_fasta,
            "clipzyme": args.clipzyme_fasta,
            "sabio_all": args.sabio_all_tsv,
        }.items() if path},
        "sources": list(datasets),
        "source_priority": args.source_priority,
        "dataset_stats": {source: dataset_stats(records) for source, records in datasets.items()},
        "exact": write_exact_outputs(exact, datasets, args.source_priority, args.out_dir),
        "clusters": [],
    }

    if args.cluster_thresholds:
        mmseqs = find_mmseqs(args.mmseqs_bin)
        exact_fasta = Path(summary["exact"]["fasta"])  # type: ignore[index]
        all_uids = {protein.uid for protein in exact}
        for threshold in args.cluster_thresholds:
            threshold_name = f"nr{int(threshold)}"
            cluster_prefix = args.out_dir / threshold_name / "mmseqs" / threshold_name
            tmp_dir = args.out_dir / threshold_name / "mmseqs" / "tmp"
            cluster_tsv = run_linclust(
                mmseqs=mmseqs,
                input_fasta=exact_fasta,
                cluster_prefix=cluster_prefix,
                tmp_dir=tmp_dir,
                min_seq_id=threshold / 100.0,
                coverage=args.coverage,
                cov_mode=args.cov_mode,
                threads=args.threads,
                reuse=args.reuse_mmseqs,
            )
            clusters = read_cluster_tsv(cluster_tsv, all_uids)
            cluster_summary = write_cluster_outputs(exact, clusters, threshold, args.out_dir)
            cluster_summary["mmseqs_cluster_tsv"] = str(cluster_tsv)
            summary["clusters"].append(cluster_summary)  # type: ignore[union-attr]

    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if not args.no_index_symlinks:
        symlink_tree(args.out_dir, args.index_dir)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
