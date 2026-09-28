#!/usr/bin/env python3
"""Propose EC labels for EC-missing sequences by homology transfer.

This script does not overwrite curated EC labels. It searches EC-missing query
sequences against the EC-annotated sequences already present in the deduplicated
single-enzyme catalog. Candidate neighbors are found with a rare k-mer index and
then scored with Smith-Waterman alignments via parasail.

The output is a proposed annotation table with explicit confidence tiers. Only
the strict "high_full_length" tier should be treated as high-confidence EC
transfer without additional manual review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "horizyn/results/single_enzyme_catalog/single_enzyme_catalog.csv"
DEFAULT_OUTPUT_DIR = ROOT / "horizyn/results/single_enzyme_catalog/homology_ec_annotation"

TARGETS: list[dict[str, str]] = []
PARASAIL = None


def read_catalog(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    targets: list[dict[str, str]] = []
    queries: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            trimmed = {
                "seq_hash": row.get("seq_hash", ""),
                "length": row.get("length", ""),
                "sources": row.get("sources", ""),
                "accessions": row.get("accessions", ""),
                "complete_ec_numbers": row.get("complete_ec_numbers", ""),
                "sequence": row.get("sequence", ""),
            }
            if row.get("has_complete_ec") == "1" and row.get("complete_ec_numbers"):
                targets.append(trimmed)
            elif row.get("has_any_ec") == "0":
                queries.append(trimmed)
    return targets, queries


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("ascii"), digest_size=8).digest(), "little")


def seq_kmers(sequence: str, k: int, stride: int = 1, max_kmers: int | None = None) -> list[str]:
    if len(sequence) < k:
        return []
    kmers = {sequence[i : i + k] for i in range(0, len(sequence) - k + 1, stride)}
    if max_kmers is not None and len(kmers) > max_kmers:
        ranked = sorted(kmers, key=stable_hash)
        return ranked[:max_kmers]
    return list(kmers)


def build_kmer_index(
    targets: list[dict[str, str]],
    *,
    k: int,
    target_stride: int,
    max_kmers_per_target: int,
    max_kmer_df: int,
) -> dict[str, list[int]]:
    counts: Counter[str] = Counter()
    target_kmers: list[list[str]] = []
    for row in targets:
        kmers = seq_kmers(row["sequence"], k, stride=target_stride, max_kmers=max_kmers_per_target)
        target_kmers.append(kmers)
        counts.update(kmers)
    index: dict[str, list[int]] = defaultdict(list)
    for target_idx, kmers in enumerate(target_kmers):
        for kmer in kmers:
            if counts[kmer] <= max_kmer_df:
                index[kmer].append(target_idx)
    return dict(index)


def candidate_targets(
    sequence: str,
    index: dict[str, list[int]],
    *,
    k: int,
    max_candidates: int,
    min_shared_kmers: int,
) -> list[int]:
    counts: Counter[int] = Counter()
    for kmer in seq_kmers(sequence, k, stride=1, max_kmers=None):
        for target_idx in index.get(kmer, []):
            counts[target_idx] += 1
    if not counts:
        return []
    return [
        target_idx
        for target_idx, shared in counts.most_common(max_candidates)
        if shared >= min_shared_kmers
    ]


def init_worker(targets: list[dict[str, str]]) -> None:
    global TARGETS, PARASAIL
    TARGETS = targets
    import parasail  # type: ignore

    PARASAIL = parasail


def alignment_metrics(query: str, target: str) -> dict[str, float]:
    if PARASAIL is None:
        raise RuntimeError("parasail was not initialized")
    result = PARASAIL.sw_trace_striped_16(query, target, 10, 1, PARASAIL.blosum62)
    aligned_q = result.traceback.query
    aligned_t = result.traceback.ref
    q_aligned = sum(1 for char in aligned_q if char != "-")
    t_aligned = sum(1 for char in aligned_t if char != "-")
    aligned_pairs = sum(1 for q, t in zip(aligned_q, aligned_t) if q != "-" and t != "-")
    identities = sum(1 for q, t in zip(aligned_q, aligned_t) if q != "-" and t != "-" and q == t)
    pident = identities / aligned_pairs if aligned_pairs else 0.0
    qcov = q_aligned / len(query) if query else 0.0
    tcov = t_aligned / len(target) if target else 0.0
    return {
        "score": float(result.score),
        "pident": pident,
        "qcov": qcov,
        "tcov": tcov,
        "aligned_pairs": float(aligned_pairs),
        "identities": float(identities),
    }


def confidence_tier(metrics: dict[str, float], query_len: int) -> str:
    pident = metrics["pident"]
    qcov = metrics["qcov"]
    tcov = metrics["tcov"]
    if query_len < 80:
        if pident >= 0.98 and qcov >= 0.95 and tcov >= 0.95:
            return "high_full_length"
        return "below_threshold"
    if pident >= 0.90 and qcov >= 0.80 and tcov >= 0.80:
        return "high_full_length"
    if pident >= 0.95 and qcov >= 0.85 and tcov >= 0.30:
        return "high_query_fragment"
    if pident >= 0.70 and qcov >= 0.70 and tcov >= 0.70:
        return "medium_full_length"
    if pident >= 0.50 and qcov >= 0.50 and tcov >= 0.50:
        return "low_full_length"
    return "below_threshold"


def score_one(task: tuple[dict[str, str], list[int]]) -> dict[str, str]:
    query_row, candidate_ids = task
    query_seq = query_row["sequence"]
    best: dict[str, str] | None = None
    best_metric: dict[str, float] | None = None
    top_ecs: list[str] = []
    for target_idx in candidate_ids:
        target = TARGETS[target_idx]
        metrics = alignment_metrics(query_seq, target["sequence"])
        if best_metric is None or (
            metrics["score"],
            metrics["pident"],
            metrics["qcov"],
            metrics["tcov"],
        ) > (
            best_metric["score"],
            best_metric["pident"],
            best_metric["qcov"],
            best_metric["tcov"],
        ):
            best_metric = metrics
            best = target
        if len(top_ecs) < 10 and metrics["pident"] >= 0.70 and metrics["qcov"] >= 0.60:
            top_ecs.extend(target["complete_ec_numbers"].split(";"))

    if best is None or best_metric is None:
        return {
            "query_seq_hash": query_row["seq_hash"],
            "query_length": query_row["length"],
            "query_sources": query_row["sources"],
            "query_accessions": query_row["accessions"],
            "proposed_ec_numbers": "",
            "confidence": "no_candidate",
            "target_seq_hash": "",
            "target_length": "",
            "target_sources": "",
            "target_accessions": "",
            "target_ec_numbers": "",
            "alignment_score": "",
            "pident": "",
            "qcov": "",
            "tcov": "",
            "aligned_pairs": "",
            "candidate_count": "0",
            "top_ec_consensus": "",
        }

    tier = confidence_tier(best_metric, len(query_seq))
    proposed = best["complete_ec_numbers"] if tier != "below_threshold" else ""
    ec_counter = Counter(top_ecs)
    consensus = ""
    if ec_counter:
        ec, count = ec_counter.most_common(1)[0]
        consensus = f"{ec}:{count}/{sum(ec_counter.values())}"
    return {
        "query_seq_hash": query_row["seq_hash"],
        "query_length": query_row["length"],
        "query_sources": query_row["sources"],
        "query_accessions": query_row["accessions"],
        "proposed_ec_numbers": proposed,
        "confidence": tier,
        "target_seq_hash": best["seq_hash"],
        "target_length": best["length"],
        "target_sources": best["sources"],
        "target_accessions": best["accessions"],
        "target_ec_numbers": best["complete_ec_numbers"],
        "alignment_score": f"{best_metric['score']:.1f}",
        "pident": f"{best_metric['pident']:.4f}",
        "qcov": f"{best_metric['qcov']:.4f}",
        "tcov": f"{best_metric['tcov']:.4f}",
        "aligned_pairs": f"{best_metric['aligned_pairs']:.0f}",
        "candidate_count": str(len(candidate_ids)),
        "top_ec_consensus": consensus,
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    confidence_counts = Counter(row["confidence"] for row in rows)
    proposed = [row for row in rows if row["proposed_ec_numbers"]]
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for source in row["query_sources"].split(";"):
            by_source[source][row["confidence"]] += 1
    return {
        "query_count": len(rows),
        "proposed_annotation_count": len(proposed),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "proposed_counts_by_confidence": dict(sorted(Counter(row["confidence"] for row in proposed).items())),
        "by_source": {source: dict(counter) for source, counter in sorted(by_source.items())},
    }


def write_readme(output_dir: Path, summary: dict[str, object], args: argparse.Namespace) -> None:
    lines = [
        "# Missing-EC Homology Annotation",
        "",
        "This directory contains proposed EC annotations for sequences that were still missing EC labels in the single-enzyme catalog.",
        "The original catalog is not overwritten.",
        "",
        "## Method",
        "",
        f"- Reference set: catalog sequences with complete EC labels from `{args.catalog}`.",
        "- Query set: catalog sequences with no EC label.",
        f"- Candidate search: rare {args.k}-mer index over the reference sequences.",
        "- Scoring: Smith-Waterman local alignment with BLOSUM62 using `parasail`.",
        "- Proposed ECs are transferred from the best aligned reference hit only when identity and coverage pass a confidence threshold.",
        "",
        "## Confidence Tiers",
        "",
        "- `high_full_length`: >=90% identity, >=80% query coverage, >=80% target coverage. For queries shorter than 80 aa this is stricter: >=98% identity and >=95% coverage on both sides.",
        "- `high_query_fragment`: >=95% identity, >=85% query coverage, >=30% target coverage. This should be manually reviewed because the query may be a fragment.",
        "- `medium_full_length`: >=70% identity and >=70% coverage on both sides.",
        "- `low_full_length`: >=50% identity and >=50% coverage on both sides.",
        "- `below_threshold` and `no_candidate`: no EC transferred.",
        "",
        "## Summary",
        "",
        f"- Queries scored: {summary['query_count']}",
        f"- Queries with proposed ECs: {summary['proposed_annotation_count']}",
        "",
        "| Confidence | Count |",
        "|---|---:|",
    ]
    for name, count in summary["confidence_counts"].items():  # type: ignore[union-attr]
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `missing_ec_homology_annotations.csv`: all query sequences with best hit metrics and proposed EC if above threshold.",
            "- `missing_ec_homology_summary.json`: machine-readable summary.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--target-stride", type=int, default=4)
    parser.add_argument("--max-kmers-per-target", type=int, default=256)
    parser.add_argument("--max-kmer-df", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--min-shared-kmers", type=int, default=2)
    parser.add_argument("--workers", type=int, default=max(1, min(16, mp.cpu_count() // 2)))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(f"Loading catalog: {args.catalog}", flush=True)
    targets, queries = read_catalog(args.catalog)
    print(f"Targets with complete EC: {len(targets)}", flush=True)
    print(f"Queries missing EC: {len(queries)}", flush=True)
    print("Building k-mer index...", flush=True)
    index = build_kmer_index(
        targets,
        k=args.k,
        target_stride=args.target_stride,
        max_kmers_per_target=args.max_kmers_per_target,
        max_kmer_df=args.max_kmer_df,
    )
    print(f"Indexed {len(index)} rare k-mers.", flush=True)
    print("Collecting candidate targets...", flush=True)
    tasks: list[tuple[dict[str, str], list[int]]] = []
    no_candidate = 0
    for idx, query in enumerate(queries, start=1):
        candidates = candidate_targets(
            query["sequence"],
            index,
            k=args.k,
            max_candidates=args.max_candidates,
            min_shared_kmers=args.min_shared_kmers,
        )
        if not candidates:
            no_candidate += 1
        tasks.append((query, candidates))
        if idx % 5000 == 0:
            print(f"Prepared candidates for {idx}/{len(queries)} queries", flush=True)
    print(f"Queries with no candidates before alignment: {no_candidate}", flush=True)

    print(f"Scoring alignments with {args.workers} workers...", flush=True)
    rows: list[dict[str, str]] = []
    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(targets,)) as pool:
        for idx, row in enumerate(pool.imap_unordered(score_one, tasks, chunksize=16), start=1):
            rows.append(row)
            if idx % 1000 == 0:
                print(f"Scored {idx}/{len(tasks)} queries", flush=True)
    rows.sort(key=lambda row: (row["query_seq_hash"], row["confidence"]))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_seq_hash",
        "query_length",
        "query_sources",
        "query_accessions",
        "proposed_ec_numbers",
        "confidence",
        "target_seq_hash",
        "target_length",
        "target_sources",
        "target_accessions",
        "target_ec_numbers",
        "alignment_score",
        "pident",
        "qcov",
        "tcov",
        "aligned_pairs",
        "candidate_count",
        "top_ec_consensus",
    ]
    write_csv(output_dir / "missing_ec_homology_annotations.csv", rows, fieldnames)
    summary = summarize(rows)
    (output_dir / "missing_ec_homology_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_readme(output_dir, summary, args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
