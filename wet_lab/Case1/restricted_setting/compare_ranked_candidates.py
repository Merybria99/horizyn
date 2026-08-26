#!/usr/bin/env python3
"""Compare restricted-setting F3 rankings with the Case1 Ranked Candidates sheet."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR.parent / "sequence_pool/final_entry_sequences.csv"
RANK_PATTERN = re.compile(r"^RANK_(\d+)$")
EVIDENCE_ID_PATTERN = re.compile(r"\b(?:H|P)\d{1,3}\b", flags=re.IGNORECASE)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _ranked_position(entry_id: str) -> int:
    match = RANK_PATTERN.fullmatch(entry_id)
    if not match:
        raise ValueError(f"Unexpected Ranked Candidates entry ID: {entry_id}")
    return int(match.group(1))


def _evidence_ids(value: str) -> set[str]:
    return {match.upper() for match in EVIDENCE_ID_PATTERN.findall(value or "")}


def _join(values: Iterable[object]) -> str:
    return ";".join(str(value) for value in values)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty comparison table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_rankings(
    results_path: Path,
    source_path: Path = DEFAULT_SOURCE,
    output_dir: Path | None = None,
    *,
    top_k: int = 25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Cross-reference the model top-k against ranked evidence and sequence identity."""

    results_path = results_path.resolve()
    source_path = source_path.resolve()
    output_dir = (output_dir or results_path.parent / "ranked_candidates_comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    rankings = list(results.get("rankings", []))
    if len(rankings) < top_k:
        raise ValueError(f"Query has only {len(rankings)} rankings; top {top_k} requested")

    source_rows = _load_csv(source_path)
    homolog_rows = [
        row for row in source_rows if row.get("sheet") == "Homologs" and row.get("sequence")
    ]
    ranked_rows = [row for row in source_rows if row.get("sheet") == "Ranked Candidates"]
    resolved_ranked = [row for row in ranked_rows if row.get("sequence")]

    ranked_by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    ranked_by_evidence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in resolved_ranked:
        ranked_by_sha[row["sha256"]].append(row)
        for evidence_id in _evidence_ids(row.get("evidence_row", "")):
            ranked_by_evidence[evidence_id].append(row)

    output_rows: list[dict[str, Any]] = []
    for model_row in rankings[:top_k]:
        protein_id = str(model_row["protein_id"]).upper()
        sha256 = str(model_row.get("sha256", ""))
        evidence_matches = ranked_by_evidence.get(protein_id, [])
        sequence_matches = ranked_by_sha.get(sha256, []) if sha256 else []
        matches_by_id = {row["entry_id"]: row for row in evidence_matches + sequence_matches}
        matches = sorted(matches_by_id.values(), key=lambda row: _ranked_position(row["entry_id"]))
        evidence_match_ids = {row["entry_id"] for row in evidence_matches}
        sequence_match_ids = {row["entry_id"] for row in sequence_matches}
        if evidence_match_ids and sequence_match_ids:
            basis = "evidence_id+exact_sequence"
        elif evidence_match_ids:
            basis = "evidence_id"
        elif sequence_match_ids:
            basis = "exact_sequence"
        else:
            basis = ""

        output_rows.append(
            {
                "model_rank": int(model_row["rank"]),
                "entry_id": model_row["protein_id"],
                "name": model_row.get("name", ""),
                "cosine_similarity": float(model_row["cosine_similarity"]),
                "sha256": sha256,
                "sequence_duplicate_count": int(model_row.get("sequence_duplicate_count", 1)),
                "in_ranked_candidates": bool(matches),
                "match_basis": basis,
                "ranked_candidate_positions": _join(
                    _ranked_position(row["entry_id"]) for row in matches
                ),
                "ranked_candidate_ids": _join(row["entry_id"] for row in matches),
                "ranked_candidate_names": _join(row["name"] for row in matches),
                "evidence_match_ids": _join(sorted(evidence_match_ids)),
                "exact_sequence_match_ids": _join(sorted(sequence_match_ids)),
            }
        )

    homolog_shas = {row["sha256"] for row in homolog_rows}
    eligible_ranked_shas = {row["sha256"] for row in resolved_ranked}.intersection(homolog_shas)
    top_shas = {str(row["sha256"]) for row in output_rows}
    recovered_shas = eligible_ranked_shas.intersection(top_shas)
    matched_rows = [row for row in output_rows if row["in_ranked_candidates"]]

    summary: dict[str, Any] = {
        "schema_version": "case1_restricted_ranked_comparison_v1",
        "reaction": results.get("reaction", {}),
        "model": results.get("model", {}),
        "candidate_pool_rows": results.get("candidate_pool", {}).get("candidate_count"),
        "top_k": top_k,
        "ranked_candidate_rows_total": len(ranked_rows),
        "ranked_candidate_rows_resolved": len(resolved_ranked),
        "ranked_candidate_unique_sequences_resolved": len(
            {row["sha256"] for row in resolved_ranked}
        ),
        "ranked_candidate_unique_sequences_eligible_in_homolog_pool": len(eligible_ranked_shas),
        "top_k_rows_matching_ranked_candidates": len(matched_rows),
        "top_k_rows_with_direct_evidence_match": sum(
            bool(row["evidence_match_ids"]) for row in output_rows
        ),
        "top_k_rows_with_exact_sequence_match": sum(
            bool(row["exact_sequence_match_ids"]) for row in output_rows
        ),
        "top_k_unique_sequences": len(top_shas),
        "eligible_ranked_unique_sequences_recovered": len(recovered_shas),
        "eligible_ranked_unique_sequence_recall": (
            len(recovered_shas) / len(eligible_ranked_shas) if eligible_ranked_shas else None
        ),
        "comparison_semantics": {
            "evidence_id": "Homolog entry_id occurs in Ranked Candidates evidence_row",
            "exact_sequence": "Homolog and resolved Ranked Candidate SHA-256 values match",
            "note": "Ranked Candidates are a comparison set, not binary ground truth.",
        },
        "inputs": {
            "query_results": str(results_path),
            "entry_table": str(source_path),
        },
    }

    csv_path = output_dir / "top25_vs_ranked_candidates.csv"
    _write_csv(csv_path, output_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    table_lines = [
        "| F3 rank | Homolog | Name | Score | Ranked position(s) | Match |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in output_rows:
        safe_name = str(row["name"]).replace("|", "\\|")
        table_lines.append(
            f"| {row['model_rank']} | {row['entry_id']} | {safe_name} | "
            f"{row['cosine_similarity']:.6f} | "
            f"{row['ranked_candidate_positions'] or '-'} | {row['match_basis'] or '-'} |"
        )
    report = "\n".join(
        [
            "# Restricted Setting Results",
            "",
            "## Summary",
            "",
            f"- Candidate rows ranked: **{summary['candidate_pool_rows']}**",
            f"- Top-{top_k} rows matching Ranked Candidates: "
            f"**{summary['top_k_rows_matching_ranked_candidates']}**",
            f"- Direct evidence-row matches: "
            f"**{summary['top_k_rows_with_direct_evidence_match']}**",
            f"- Exact-sequence matches: **{summary['top_k_rows_with_exact_sequence_match']}**",
            f"- Eligible Ranked Candidate sequence recovery: "
            f"**{summary['eligible_ranked_unique_sequences_recovered']}/"
            f"{summary['ranked_candidate_unique_sequences_eligible_in_homolog_pool']}** "
            f"({summary['eligible_ranked_unique_sequence_recall']:.1%})",
            "",
            "Duplicate Homolog rows are retained as separate candidates. The sequence-recovery "
            "statistic is deduplicated by SHA-256.",
            "",
            "## Top 25",
            "",
            *table_lines,
            "",
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary, output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, _ = compare_rankings(
        args.results,
        args.source,
        args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
