#!/usr/bin/env python3
"""Evaluate sequence overlap and build common protein bases across benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HORIZYN = PROJECT_ROOT / "data/sota/prots.fasta"
DEFAULT_REACTZYME = PROJECT_ROOT / "data/paper/reactzyme/eval/all_proteins.fasta"
DEFAULT_CLIPZYME = PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/proteins.fasta"
DEFAULT_OUT = PROJECT_ROOT / "results/sequence_overlap/common_base"
DEFAULT_MMSEQS = PROJECT_ROOT / "tools/mmseqs/bin/mmseqs"


def clean_sequence(sequence: str) -> str:
    return sequence.replace(" ", "").replace("\n", "").replace("\r", "").upper()


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


def sequence_sha(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def write_dataset_fasta(records: list[tuple[str, str]], dataset: str, path: Path) -> None:
    prefixed = [(f"{dataset}|{record_id}", sequence) for record_id, sequence in records if sequence]
    write_fasta(prefixed, path)


def load_datasets(args: argparse.Namespace) -> dict[str, list[tuple[str, str]]]:
    return {
        "horizyn": read_fasta(args.horizyn_fasta),
        "reactzyme": read_fasta(args.reactzyme_fasta),
        "clipzyme": read_fasta(args.clipzyme_fasta),
    }


def dataset_stats(records: list[tuple[str, str]]) -> dict[str, int | float]:
    nonempty = [(record_id, sequence) for record_id, sequence in records if sequence]
    lengths = [len(sequence) for _record_id, sequence in nonempty]
    return {
        "records": len(records),
        "nonempty_sequences": len(nonempty),
        "unique_ids": len({record_id for record_id, sequence in nonempty}),
        "unique_sequences": len({sequence for _record_id, sequence in nonempty}),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": round(sum(lengths) / len(lengths), 3) if lengths else 0,
    }


def exact_overlap_summary(
    datasets: dict[str, list[tuple[str, str]]],
    out_dir: Path,
) -> dict[str, object]:
    ids = {
        name: {record_id for record_id, sequence in records if sequence}
        for name, records in datasets.items()
    }
    seqs = {
        name: {sequence for _record_id, sequence in records if sequence}
        for name, records in datasets.items()
    }
    names = list(datasets)

    pairwise_rows = []
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            pairwise_rows.append(
                {
                    "left": left,
                    "right": right,
                    "id_overlap": len(ids[left] & ids[right]),
                    "exact_sequence_overlap": len(seqs[left] & seqs[right]),
                    "left_unique_sequences": len(seqs[left]),
                    "right_unique_sequences": len(seqs[right]),
                }
            )

    sequence_to_ids: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    sequence_by_sha: dict[str, str] = {}
    for dataset, records in datasets.items():
        for record_id, sequence in records:
            if not sequence:
                continue
            sha = sequence_sha(sequence)
            sequence_by_sha[sha] = sequence
            sequence_to_ids[sha][dataset].append(record_id)

    all_dataset_names = set(names)
    all3_shas = sorted(
        sha
        for sha, per_dataset in sequence_to_ids.items()
        if set(per_dataset) == all_dataset_names
    )
    all3_records = [(f"common_exact_all3|{sha}", sequence_by_sha[sha]) for sha in all3_shas]
    write_fasta(all3_records, out_dir / "fasta/common_exact_all3.fasta")

    mapping_path = out_dir / "summary/common_exact_all3_mapping.csv"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_sha1", "length", "horizyn_ids", "reactzyme_ids", "clipzyme_ids"])
        for sha in all3_shas:
            per_dataset = sequence_to_ids[sha]
            writer.writerow(
                [
                    sha,
                    len(sequence_by_sha[sha]),
                    "|".join(sorted(per_dataset["horizyn"])),
                    "|".join(sorted(per_dataset["reactzyme"])),
                    "|".join(sorted(per_dataset["clipzyme"])),
                ]
            )

    return {
        "dataset_stats": {name: dataset_stats(records) for name, records in datasets.items()},
        "pairwise_exact": pairwise_rows,
        "all3_exact_sequence_overlap": len(all3_shas),
        "common_exact_all3_fasta": str(out_dir / "fasta/common_exact_all3.fasta"),
        "common_exact_all3_mapping": str(mapping_path),
    }


def find_mmseqs(path: Path) -> Path:
    if path.exists():
        return path
    found = shutil.which("mmseqs")
    if found:
        return Path(found)
    raise FileNotFoundError("Could not find MMseqs; pass --mmseqs-bin")


def run_mmseqs_search(
    mmseqs: Path,
    query_fasta: Path,
    target_fasta: Path,
    output_tsv: Path,
    tmp_dir: Path,
    coverage: float,
    cov_mode: int,
    threads: int,
    max_seqs: int,
) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if output_tsv.exists():
        output_tsv.unlink()
    cmd = [
        str(mmseqs),
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(output_tsv),
        str(tmp_dir),
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
    print("running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def best_hits_from_tsv(path: Path) -> dict[str, dict[str, str | float]]:
    best: dict[str, dict[str, str | float]] = {}
    if not path.exists():
        return best
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 8:
                continue
            query, target, pident, qcov, tcov, alnlen, evalue, bits = row[:8]
            try:
                score = float(bits)
                pident_value = float(pident)
                qcov_value = float(qcov)
                tcov_value = float(tcov)
            except ValueError:
                continue
            current = best.get(query)
            if current is None or score > float(current["bits"]):
                best[query] = {
                    "target": target,
                    "pident": pident_value,
                    "qcov": qcov_value,
                    "tcov": tcov_value,
                    "alnlen": float(alnlen),
                    "evalue": evalue,
                    "bits": score,
                }
    return best


def mmseqs_common_base(
    datasets: dict[str, list[tuple[str, str]]],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, object]:
    mmseqs = find_mmseqs(args.mmseqs_bin)
    fasta_dir = out_dir / "fasta"
    mmseqs_dir = out_dir / "mmseqs"

    for dataset, records in datasets.items():
        write_dataset_fasta(records, dataset, fasta_dir / f"{dataset}.fasta")

    searches = {
        "horizyn_vs_reactzyme": ("horizyn", "reactzyme"),
        "horizyn_vs_clipzyme": ("horizyn", "clipzyme"),
    }
    for name, (query, target) in searches.items():
        output_tsv = mmseqs_dir / f"{name}.tsv"
        if args.reuse_mmseqs and output_tsv.exists():
            print(f"reusing {output_tsv}", flush=True)
            continue
        run_mmseqs_search(
            mmseqs=mmseqs,
            query_fasta=fasta_dir / f"{query}.fasta",
            target_fasta=fasta_dir / f"{target}.fasta",
            output_tsv=output_tsv,
            tmp_dir=mmseqs_dir / f"{name}.tmp",
            coverage=args.coverage,
            cov_mode=args.cov_mode,
            threads=args.threads,
            max_seqs=args.max_seqs,
        )

    reactzyme_hits = best_hits_from_tsv(mmseqs_dir / "horizyn_vs_reactzyme.tsv")
    clipzyme_hits = best_hits_from_tsv(mmseqs_dir / "horizyn_vs_clipzyme.tsv")
    horizyn_sequences = {
        f"horizyn|{record_id}": sequence
        for record_id, sequence in datasets["horizyn"]
        if sequence
    }

    thresholds = sorted(set(float(value) for value in args.thresholds), reverse=True)
    summary_rows = []
    for threshold in thresholds:
        common_ids = []
        mapping_path = out_dir / "summary" / f"common_horizyn_anchor_ge{threshold:g}_mapping.csv"
        with mapping_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "horizyn_id",
                    "reactzyme_best_id",
                    "reactzyme_pident",
                    "reactzyme_qcov",
                    "reactzyme_tcov",
                    "clipzyme_best_id",
                    "clipzyme_pident",
                    "clipzyme_qcov",
                    "clipzyme_tcov",
                ]
            )
            for query_id in sorted(horizyn_sequences):
                reactzyme_hit = reactzyme_hits.get(query_id)
                clipzyme_hit = clipzyme_hits.get(query_id)
                if reactzyme_hit is None or clipzyme_hit is None:
                    continue
                if (
                    float(reactzyme_hit["pident"]) >= threshold
                    and float(clipzyme_hit["pident"]) >= threshold
                ):
                    common_ids.append(query_id)
                    writer.writerow(
                        [
                            query_id.split("|", 1)[1],
                            str(reactzyme_hit["target"]).split("|", 1)[1],
                            reactzyme_hit["pident"],
                            reactzyme_hit["qcov"],
                            reactzyme_hit["tcov"],
                            str(clipzyme_hit["target"]).split("|", 1)[1],
                            clipzyme_hit["pident"],
                            clipzyme_hit["qcov"],
                            clipzyme_hit["tcov"],
                        ]
                    )
        fasta_path = out_dir / "fasta" / f"common_horizyn_anchor_ge{threshold:g}.fasta"
        write_fasta(
            [(query_id.split("|", 1)[1], horizyn_sequences[query_id]) for query_id in common_ids],
            fasta_path,
        )
        summary_rows.append(
            {
                "threshold": threshold,
                "horizyn_anchor_common_count": len(common_ids),
                "mapping": str(mapping_path),
                "fasta": str(fasta_path),
            }
        )

    return {
        "mmseqs_bin": str(mmseqs),
        "coverage": args.coverage,
        "cov_mode": args.cov_mode,
        "threshold_common_bases": summary_rows,
        "horizyn_with_reactzyme_hit": len(reactzyme_hits),
        "horizyn_with_clipzyme_hit": len(clipzyme_hits),
    }


def write_summary_files(summary: dict[str, object], out_dir: Path) -> None:
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "sequence_overlap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    dataset_stats = summary["dataset_stats"]
    with (summary_dir / "dataset_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "records",
                "nonempty_sequences",
                "unique_ids",
                "unique_sequences",
                "min_length",
                "max_length",
                "mean_length",
            ],
        )
        writer.writeheader()
        for dataset, stats in dataset_stats.items():
            writer.writerow({"dataset": dataset, **stats})

    with (summary_dir / "pairwise_exact_overlap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "left",
                "right",
                "id_overlap",
                "exact_sequence_overlap",
                "left_unique_sequences",
                "right_unique_sequences",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["pairwise_exact"])

    if "mmseqs" in summary:
        rows = summary["mmseqs"]["threshold_common_bases"]
        with (summary_dir / "mmseqs_common_base_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["threshold", "horizyn_anchor_common_count", "mapping", "fasta"],
            )
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ReactZyme, CLIPZyme, and Horizyn protein sequence overlap."
    )
    parser.add_argument("--horizyn-fasta", type=Path, default=DEFAULT_HORIZYN)
    parser.add_argument("--reactzyme-fasta", type=Path, default=DEFAULT_REACTZYME)
    parser.add_argument("--clipzyme-fasta", type=Path, default=DEFAULT_CLIPZYME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-mmseqs", action="store_true")
    parser.add_argument("--reuse-mmseqs", action="store_true")
    parser.add_argument("--mmseqs-bin", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--coverage", type=float, default=0.85)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-seqs", type=int, default=1)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[100.0, 90.0, 50.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(args)
    summary = exact_overlap_summary(datasets, args.out_dir)
    if args.run_mmseqs:
        summary["mmseqs"] = mmseqs_common_base(datasets, args, args.out_dir)
    write_summary_files(summary, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
