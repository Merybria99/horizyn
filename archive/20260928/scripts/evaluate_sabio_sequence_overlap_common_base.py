#!/usr/bin/env python3
"""Add SABIO-RK to the Horizyn/ReactZyme/CLIPZyme common-base analysis."""

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
ENZYME_DISCOVERY_ROOT = PROJECT_ROOT.parent

DEFAULT_OUT = PROJECT_ROOT / "results/sequence_overlap/common_base_with_sabio"
DEFAULT_BASE_COMMON = PROJECT_ROOT / "results/sequence_overlap/common_base"
DEFAULT_MMSEQS = PROJECT_ROOT / "tools/mmseqs/bin/mmseqs"
DEFAULT_HORIZYN_FASTA = PROJECT_ROOT / "data/sota/prots.fasta"
DEFAULT_REACTZYME_FASTA = PROJECT_ROOT / "data/paper/reactzyme/eval/all_proteins.fasta"
DEFAULT_CLIPZYME_FASTA = PROJECT_ROOT / "data/paper/clipzyme/eval/enzymemap/proteins.fasta"
DEFAULT_SABIO_ALL = (
    ENZYME_DISCOVERY_ROOT / "sabio_rk_download/seqsim_fetch/new_sequences.tsv"
)
DEFAULT_SABIO_NOVELTY90 = PROJECT_ROOT / "data/paper/sabio_rk/eval/novelty90/sequences.tsv"
DEFAULT_SABIO_NOVELTY50 = PROJECT_ROOT / "data/paper/sabio_rk/eval/novelty50/sequences.tsv"


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


def read_sequence_tsv(path: Path) -> list[tuple[str, str]]:
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


def sequence_sha(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def write_prefixed_fasta(records: list[tuple[str, str]], prefix: str, path: Path) -> None:
    write_fasta([(f"{prefix}|{record_id}", sequence) for record_id, sequence in records], path)


def stats(records: list[tuple[str, str]]) -> dict[str, int | float]:
    lengths = [len(sequence) for _record_id, sequence in records if sequence]
    return {
        "records": len(records),
        "nonempty_sequences": len(lengths),
        "unique_ids": len({record_id for record_id, sequence in records if sequence}),
        "unique_sequences": len({sequence for _record_id, sequence in records if sequence}),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
    }


def exact_summary(
    datasets: dict[str, list[tuple[str, str]]],
    sabio_name: str,
    out_dir: Path,
) -> dict[str, object]:
    seq_sets = {
        name: {sequence for _record_id, sequence in records if sequence}
        for name, records in datasets.items()
    }
    id_sets = {
        name: {record_id for record_id, sequence in records if sequence}
        for name, records in datasets.items()
    }

    pairwise_rows = []
    names = list(datasets)
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            pairwise_rows.append(
                {
                    "left": left,
                    "right": right,
                    "id_overlap": len(id_sets[left] & id_sets[right]),
                    "exact_sequence_overlap": len(seq_sets[left] & seq_sets[right]),
                    "left_unique_sequences": len(seq_sets[left]),
                    "right_unique_sequences": len(seq_sets[right]),
                }
            )

    seq_to_ids: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    sha_to_seq: dict[str, str] = {}
    for dataset, records in datasets.items():
        for record_id, sequence in records:
            if not sequence:
                continue
            sha = sequence_sha(sequence)
            sha_to_seq[sha] = sequence
            seq_to_ids[sha][dataset].append(record_id)

    all_names = set(datasets)
    all4_shas = sorted(
        sha
        for sha, per_dataset in seq_to_ids.items()
        if set(per_dataset) == all_names
    )
    fasta_path = out_dir / "fasta" / f"common_exact_all4_{sabio_name}.fasta"
    mapping_path = out_dir / "summary" / f"common_exact_all4_{sabio_name}_mapping.csv"
    write_fasta([(f"common_exact_all4_{sabio_name}|{sha}", sha_to_seq[sha]) for sha in all4_shas], fasta_path)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sequence_sha1", "length"] + [f"{name}_ids" for name in names]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sha in all4_shas:
            row = {
                "sequence_sha1": sha,
                "length": len(sha_to_seq[sha]),
            }
            for name in names:
                row[f"{name}_ids"] = "|".join(sorted(seq_to_ids[sha].get(name, [])))
            writer.writerow(row)

    return {
        "dataset_stats": {name: stats(records) for name, records in datasets.items()},
        "pairwise_exact": pairwise_rows,
        "all4_exact_sequence_overlap": len(all4_shas),
        "common_exact_all4_fasta": str(fasta_path),
        "common_exact_all4_mapping": str(mapping_path),
    }


def best_hits(path: Path) -> dict[str, dict[str, str | float]]:
    best: dict[str, dict[str, str | float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 8:
                continue
            query, target, pident, qcov, tcov, alnlen, evalue, bits = row[:8]
            try:
                bits_value = float(bits)
                pident_value = float(pident)
                qcov_value = float(qcov)
                tcov_value = float(tcov)
            except ValueError:
                continue
            current = best.get(query)
            if current is None or bits_value > float(current["bits"]):
                best[query] = {
                    "target": target,
                    "pident": pident_value,
                    "qcov": qcov_value,
                    "tcov": tcov_value,
                    "alnlen": float(alnlen),
                    "evalue": evalue,
                    "bits": bits_value,
                }
    return best


def find_mmseqs(path: Path) -> Path:
    if path.exists():
        return path
    found = shutil.which("mmseqs")
    if found:
        return Path(found)
    raise FileNotFoundError("Could not find MMseqs; pass --mmseqs-bin")


def run_mmseqs(
    mmseqs: Path,
    query_fasta: Path,
    target_fasta: Path,
    result_tsv: Path,
    tmp_dir: Path,
    coverage: float,
    cov_mode: int,
    threads: int,
    max_seqs: int,
    reuse: bool,
) -> None:
    if reuse and result_tsv.exists():
        print(f"reusing {result_tsv}", flush=True)
        return
    result_tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if result_tsv.exists():
        result_tsv.unlink()
    cmd = [
        str(mmseqs),
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(result_tsv),
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


def mmseqs_summary(
    datasets: dict[str, list[tuple[str, str]]],
    sabio_name: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, object]:
    base_mmseqs_dir = args.base_common_dir / "mmseqs"
    reactzyme_hits_path = base_mmseqs_dir / "horizyn_vs_reactzyme.tsv"
    clipzyme_hits_path = base_mmseqs_dir / "horizyn_vs_clipzyme.tsv"
    if not reactzyme_hits_path.exists() or not clipzyme_hits_path.exists():
        raise FileNotFoundError(
            "Missing base MMseqs hits. Run scripts/evaluate_sequence_overlap_common_base.py --run-mmseqs first."
        )

    fasta_dir = out_dir / "fasta"
    mmseqs_dir = out_dir / "mmseqs"
    horizyn_fasta = fasta_dir / "horizyn.fasta"
    sabio_fasta = fasta_dir / f"{sabio_name}.fasta"
    if not horizyn_fasta.exists():
        write_prefixed_fasta(datasets["horizyn"], "horizyn", horizyn_fasta)
    write_prefixed_fasta(datasets[sabio_name], sabio_name, sabio_fasta)

    sabio_hits_path = mmseqs_dir / f"horizyn_vs_{sabio_name}.tsv"
    run_mmseqs(
        find_mmseqs(args.mmseqs_bin),
        horizyn_fasta,
        sabio_fasta,
        sabio_hits_path,
        mmseqs_dir / f"horizyn_vs_{sabio_name}.tmp",
        args.coverage,
        args.cov_mode,
        args.threads,
        args.max_seqs,
        args.reuse_mmseqs,
    )

    reactzyme_hits = best_hits(reactzyme_hits_path)
    clipzyme_hits = best_hits(clipzyme_hits_path)
    sabio_hits = best_hits(sabio_hits_path)
    horizyn_sequences = {
        f"horizyn|{record_id}": sequence
        for record_id, sequence in datasets["horizyn"]
        if sequence
    }

    threshold_rows = []
    for threshold in sorted({float(value) for value in args.thresholds}, reverse=True):
        common_query_ids = []
        mapping_path = out_dir / "summary" / f"common_horizyn_anchor_with_{sabio_name}_ge{threshold:g}_mapping.csv"
        with mapping_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "horizyn_id",
                    "reactzyme_best_id",
                    "reactzyme_pident",
                    "clipzyme_best_id",
                    "clipzyme_pident",
                    f"{sabio_name}_best_id",
                    f"{sabio_name}_pident",
                ]
            )
            for query_id in sorted(horizyn_sequences):
                reactzyme_hit = reactzyme_hits.get(query_id)
                clipzyme_hit = clipzyme_hits.get(query_id)
                sabio_hit = sabio_hits.get(query_id)
                if reactzyme_hit is None or clipzyme_hit is None or sabio_hit is None:
                    continue
                if (
                    float(reactzyme_hit["pident"]) >= threshold
                    and float(clipzyme_hit["pident"]) >= threshold
                    and float(sabio_hit["pident"]) >= threshold
                ):
                    common_query_ids.append(query_id)
                    writer.writerow(
                        [
                            query_id.split("|", 1)[1],
                            str(reactzyme_hit["target"]).split("|", 1)[1],
                            reactzyme_hit["pident"],
                            str(clipzyme_hit["target"]).split("|", 1)[1],
                            clipzyme_hit["pident"],
                            str(sabio_hit["target"]).split("|", 1)[1],
                            sabio_hit["pident"],
                        ]
                    )
        fasta_path = out_dir / "fasta" / f"common_horizyn_anchor_with_{sabio_name}_ge{threshold:g}.fasta"
        write_fasta(
            [(query_id.split("|", 1)[1], horizyn_sequences[query_id]) for query_id in common_query_ids],
            fasta_path,
        )
        threshold_rows.append(
            {
                "sabio_set": sabio_name,
                "threshold": threshold,
                "horizyn_anchor_common_count": len(common_query_ids),
                "mapping": str(mapping_path),
                "fasta": str(fasta_path),
            }
        )

    return {
        "sabio_set": sabio_name,
        "coverage": args.coverage,
        "cov_mode": args.cov_mode,
        "horizyn_with_reactzyme_hit": len(reactzyme_hits),
        "horizyn_with_clipzyme_hit": len(clipzyme_hits),
        "horizyn_with_sabio_hit": len(sabio_hits),
        "threshold_common_bases": threshold_rows,
        "sabio_hits_tsv": str(sabio_hits_path),
    }


def write_summary_files(summary: dict[str, object], out_dir: Path, sabio_name: str) -> None:
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / f"sequence_overlap_with_{sabio_name}.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    with (summary_dir / f"dataset_stats_with_{sabio_name}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
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
        for dataset, row in summary["exact"]["dataset_stats"].items():
            writer.writerow({"dataset": dataset, **row})
    with (summary_dir / f"pairwise_exact_with_{sabio_name}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
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
        writer.writerows(summary["exact"]["pairwise_exact"])
    if "mmseqs" in summary:
        with (summary_dir / f"mmseqs_common_base_with_{sabio_name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sabio_set", "threshold", "horizyn_anchor_common_count", "mapping", "fasta"],
            )
            writer.writeheader()
            writer.writerows(summary["mmseqs"]["threshold_common_bases"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SABIO-RK overlap with the existing Horizyn/ReactZyme/CLIPZyme common base."
    )
    parser.add_argument("--horizyn-fasta", type=Path, default=DEFAULT_HORIZYN_FASTA)
    parser.add_argument("--reactzyme-fasta", type=Path, default=DEFAULT_REACTZYME_FASTA)
    parser.add_argument("--clipzyme-fasta", type=Path, default=DEFAULT_CLIPZYME_FASTA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-common-dir", type=Path, default=DEFAULT_BASE_COMMON)
    parser.add_argument("--mmseqs-bin", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--run-mmseqs", action="store_true")
    parser.add_argument("--reuse-mmseqs", action="store_true")
    parser.add_argument("--coverage", type=float, default=0.85)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-seqs", type=int, default=1)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[100.0, 90.0, 50.0])
    parser.add_argument(
        "--sabio-set",
        choices=["all", "novelty90", "novelty50"],
        action="append",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "horizyn": read_fasta(args.horizyn_fasta),
        "reactzyme": read_fasta(args.reactzyme_fasta),
        "clipzyme": read_fasta(args.clipzyme_fasta),
    }
    sabio_paths = {
        "sabio_all": DEFAULT_SABIO_ALL,
        "sabio_novelty90": DEFAULT_SABIO_NOVELTY90,
        "sabio_novelty50": DEFAULT_SABIO_NOVELTY50,
    }
    requested = args.sabio_set or ["all", "novelty90", "novelty50"]
    summaries = []
    for requested_name in requested:
        sabio_name = f"sabio_{requested_name}"
        datasets = dict(base)
        datasets[sabio_name] = read_sequence_tsv(sabio_paths[sabio_name])
        summary: dict[str, object] = {
            "sabio_set": sabio_name,
            "exact": exact_summary(datasets, sabio_name, args.out_dir),
        }
        if args.run_mmseqs:
            summary["mmseqs"] = mmseqs_summary(datasets, sabio_name, args, args.out_dir)
        write_summary_files(summary, args.out_dir, sabio_name)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    combined_path = args.out_dir / "summary/sabio_overlap_combined_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    rows = []
    for summary in summaries:
        exact = summary["exact"]
        row = {
            "sabio_set": summary["sabio_set"],
            "all4_exact_sequence_overlap": exact["all4_exact_sequence_overlap"],
        }
        if "mmseqs" in summary:
            for item in summary["mmseqs"]["threshold_common_bases"]:
                row[f"mmseqs_ge{item['threshold']:g}_common_count"] = item[
                    "horizyn_anchor_common_count"
                ]
        rows.append(row)
    if rows:
        columns = sorted({column for row in rows for column in row})
        with (args.out_dir / "summary/sabio_overlap_counts.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
