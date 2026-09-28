#!/usr/bin/env python3
"""Merge exact top-k results from disjoint wet-lab query shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "horizyn_wet_lab_query_shard_merge_v1"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _top_csv_path(result: dict[str, Any], cutoff: int) -> Path:
    artifacts = result.get("artifacts") or {}
    top_k_csv = artifacts.get("top_k_csv") or {}
    value = top_k_csv.get(str(cutoff))
    if not value:
        raise ValueError(f"Shard result does not provide its top-{cutoff} CSV")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Shard top-k CSV does not exist: {path}")
    return path


def _sum_stat(results: list[dict[str, Any]], key: str) -> int:
    return sum(int((result.get("candidate_pool") or {}).get(key, 0)) for result in results)


def merge_sharded_results(result_paths: list[Path], output_dir: Path) -> Path:
    """Merge per-shard retained rankings for one model and reaction.

    If each disjoint shard retains at least K candidates, sorting the union of
    those retained candidates yields the exact global top K.
    """

    if len(result_paths) < 2:
        raise ValueError("At least two shard results are required")
    paths = [path.expanduser().resolve() for path in result_paths]
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    model = dict(results[0].get("model") or {})
    reaction = dict(results[0].get("reaction") or {})
    scoring = dict(results[0].get("scoring") or {})
    if not model.get("name") or not model.get("checkpoint"):
        raise ValueError("Shard result is missing model provenance")
    if not reaction.get("id") or not reaction.get("smiles"):
        raise ValueError("Shard result is missing reaction provenance")
    for result in results[1:]:
        if result.get("model") != model:
            raise ValueError("All shard results must use the same model")
        other_reaction = result.get("reaction") or {}
        if (other_reaction.get("id"), other_reaction.get("smiles")) != (
            reaction["id"],
            reaction["smiles"],
        ):
            raise ValueError("All shard results must use the same reaction")
        if result.get("scoring") != scoring:
            raise ValueError("All shard results must use the same scoring configuration")

    top_k_values = sorted({int(value) for value in scoring.get("top_k", [])})
    if not top_k_values or top_k_values[0] <= 0:
        raise ValueError("Shard results do not define positive top-k cutoffs")
    max_k = max(top_k_values)

    retained_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        with _top_csv_path(result, max_k).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < max_k:
            raise ValueError(f"A shard retained only {len(rows)} candidates; expected {max_k}")
        for row in rows:
            protein_id = str(row["protein_id"])
            if not protein_id:
                raise ValueError("Shard ranking contains an empty protein ID")
            if protein_id in seen:
                raise ValueError(f"Protein {protein_id} occurs in more than one shard")
            seen.add(protein_id)
            normalized = dict(row)
            normalized["rank"] = int(normalized["rank"])
            normalized["cosine_similarity"] = float(normalized["cosine_similarity"])
            if not math.isfinite(normalized["cosine_similarity"]):
                raise ValueError(f"Shard score for {protein_id} is not finite")
            if normalized.get("sequence_length") not in {None, ""}:
                normalized["sequence_length"] = int(normalized["sequence_length"])
            retained_rows.append(normalized)

    retained_rows.sort(key=lambda row: (-float(row["cosine_similarity"]), str(row["protein_id"])))
    merged_rows = retained_rows[:max_k]
    for rank, row in enumerate(merged_rows, start=1):
        row["rank"] = rank

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "rankings.csv", merged_rows)
    for cutoff in top_k_values:
        _write_csv(output_dir / f"top_{cutoff}.csv", merged_rows[:cutoff])

    fasta_path = output_dir / f"top_{max_k}.fasta"
    with fasta_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            sequence = str(row.get("sequence") or "")
            if not sequence:
                raise ValueError(f"Merged candidate {row['protein_id']} has no sequence")
            handle.write(
                f">{row['protein_id']} rank={row['rank']} "
                f"cosine_similarity={float(row['cosine_similarity']):.8f}\n"
            )
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")

    result = {
        "schema_version": SCHEMA_VERSION,
        "reaction": reaction,
        "model": model,
        "scoring": scoring,
        "candidate_pool": {
            "requested_candidate_count": _sum_stat(results, "requested_candidate_count"),
            "candidate_count": _sum_stat(results, "candidate_count"),
            "missing_candidate_id_count": _sum_stat(results, "missing_candidate_id_count"),
            "zero_length_candidate_count": _sum_stat(results, "zero_length_candidate_count"),
            "shard_count": len(results),
            "shards": [
                {
                    "result": str(path),
                    "candidate_pool": shard_result.get("candidate_pool"),
                }
                for path, shard_result in zip(paths, results)
            ],
        },
        "rankings": [
            {key: value for key, value in row.items() if key != "sequence"} for row in merged_rows
        ],
        "artifacts": {
            "rankings_csv": str(output_dir / "rankings.csv"),
            "top_k_csv": {
                str(cutoff): str(output_dir / f"top_{cutoff}.csv") for cutoff in top_k_values
            },
            "fasta": str(fasta_path),
            "source_results": [str(path) for path in paths],
        },
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved merged query results to: {result_path}", flush=True)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    merge_sharded_results(args.result, args.output_dir)


if __name__ == "__main__":
    main()
