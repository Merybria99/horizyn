#!/usr/bin/env python3
"""Run and compare all three split-specific F3 checkpoints on Case1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

from compare_ranked_candidates import compare_rankings


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SOURCE_TABLE = SCRIPT_DIR.parent / "sequence_pool/final_entry_sequences.csv"
SPLIT_CONFIGS = {
    "time": SCRIPT_DIR / "query_f3_time.yaml",
    "enzyme_smi": SCRIPT_DIR / "query_f3_enzyme_smi.yaml",
    "reaction_smi": SCRIPT_DIR / "query_f3.yaml",
}
OUTPUT_ROOTS = {
    "time": SCRIPT_DIR / "runs/time",
    "enzyme_smi": SCRIPT_DIR / "runs/enzyme_smi",
    "reaction_smi": SCRIPT_DIR / "runs",
}


def _runtime_environment(split: str, gpu: int) -> dict[str, str]:
    runtime = SCRIPT_DIR / f"runtime/all_splits/{split}"
    directories = {
        "HOME": runtime / "home",
        "HF_HOME": runtime / "cache/huggingface",
        "TORCH_HOME": runtime / "cache/torch",
        "XDG_CACHE_HOME": runtime / "cache/xdg",
        "XDG_CONFIG_HOME": runtime / "config/xdg",
        "XDG_STATE_HOME": runtime / "state/xdg",
        "MPLCONFIGDIR": runtime / "config/matplotlib",
        "TMPDIR": runtime / "tmp",
    }
    environment = os.environ.copy()
    for variable, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(directory)
    environment["TRANSFORMERS_CACHE"] = str(directories["HF_HOME"] / "transformers")
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return environment


def _latest_result(split: str) -> Path | None:
    candidates = sorted(
        OUTPUT_ROOTS[split].glob("tagatose_4_epimerase_*/results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _run_split(split: str, gpu: int, *, force_features: bool) -> Path:
    python = REPOSITORY_ROOT / ".capability-run-py/bin/python"
    command = [
        str(python),
        "-m",
        "wet_lab.query",
        "--config",
        str(SPLIT_CONFIGS[split]),
    ]
    if force_features:
        command.append("--force-features")
    log_path = SCRIPT_DIR / f"logs/all_splits/{split}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_runtime_environment(split, gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"{split} query failed with exit code {completed.returncode}; {log_path}\n"
            + "\n".join(tail)
        )
    result = _latest_result(split)
    if result is None:
        raise RuntimeError(f"{split} finished without a results.json; inspect {log_path}")
    return result


def _load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    rankings = result.get("rankings", [])
    if len(rankings) != 144:
        raise ValueError(f"Expected 144 rankings in {path}, found {len(rankings)}")
    if len({row["protein_id"] for row in rankings}) != 144:
        raise ValueError(f"Candidate IDs are not unique in {path}")
    return result


def _spearman(first: dict[str, int], second: dict[str, int]) -> float:
    identifiers = sorted(first)
    if set(identifiers) != set(second):
        raise ValueError("Cannot correlate rankings with different candidate sets")
    count = len(identifiers)
    squared_difference = sum((first[key] - second[key]) ** 2 for key in identifiers)
    return 1.0 - 6.0 * squared_difference / (count * (count * count - 1))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(result_paths: dict[str, Path]) -> Path:
    output_dir = SCRIPT_DIR / "all_splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {split: _load_result(path) for split, path in result_paths.items()}
    rankings = {split: result["rankings"] for split, result in results.items()}
    rows_by_id = {
        split: {str(row["protein_id"]): row for row in rows} for split, rows in rankings.items()
    }
    ranks = {
        split: {protein_id: int(row["rank"]) for protein_id, row in split_rows.items()}
        for split, split_rows in rows_by_id.items()
    }

    comparisons: dict[str, dict[str, Any]] = {}
    for split, path in result_paths.items():
        summary, _ = compare_rankings(
            path,
            SOURCE_TABLE,
            output_dir / f"ranked_candidates/{split}",
            top_k=25,
        )
        comparisons[split] = summary

    pairwise: list[dict[str, Any]] = []
    for first, second in combinations(SPLIT_CONFIGS, 2):
        row: dict[str, Any] = {
            "first_split": first,
            "second_split": second,
            "spearman_all_144": _spearman(ranks[first], ranks[second]),
        }
        for cutoff in (5, 10, 25, 50):
            first_ids = {str(item["protein_id"]) for item in rankings[first][:cutoff]}
            second_ids = {str(item["protein_id"]) for item in rankings[second][:cutoff]}
            overlap = len(first_ids.intersection(second_ids))
            row[f"top_{cutoff}_overlap"] = overlap
            row[f"top_{cutoff}_jaccard"] = overlap / len(first_ids.union(second_ids))
        pairwise.append(row)
    _write_csv(output_dir / "pairwise_rank_agreement.csv", pairwise)

    reference_split = "reaction_smi"
    consensus_rows: list[dict[str, Any]] = []
    for protein_id, reference in rows_by_id[reference_split].items():
        split_ranks = {split: ranks[split][protein_id] for split in SPLIT_CONFIGS}
        split_scores = {
            split: float(rows_by_id[split][protein_id]["cosine_similarity"])
            for split in SPLIT_CONFIGS
        }
        consensus_rows.append(
            {
                "protein_id": protein_id,
                "name": reference.get("name", ""),
                "sha256": reference.get("sha256", ""),
                "sequence_duplicate_count": int(reference.get("sequence_duplicate_count", 1)),
                "rrf_score": sum(1.0 / (60 + rank) for rank in split_ranks.values()),
                "mean_rank": sum(split_ranks.values()) / len(split_ranks),
                "best_rank": min(split_ranks.values()),
                "worst_rank": max(split_ranks.values()),
                **{f"{split}_rank": split_ranks[split] for split in SPLIT_CONFIGS},
                **{f"{split}_score": split_scores[split] for split in SPLIT_CONFIGS},
            }
        )
    consensus_rows.sort(key=lambda row: (-row["rrf_score"], row["mean_rank"], row["protein_id"]))
    for rank, row in enumerate(consensus_rows, start=1):
        row["rank"] = rank
        row["cosine_similarity"] = row["rrf_score"]
    ordered_consensus = [
        {
            "rank": row["rank"],
            **{key: value for key, value in row.items() if key != "rank"},
        }
        for row in consensus_rows
    ]
    _write_csv(output_dir / "consensus_rankings.csv", ordered_consensus)

    consensus_result = {
        "schema_version": "case1_restricted_f3_split_consensus_v1",
        "reaction": results[reference_split]["reaction"],
        "model": {
            "name": "F3 reciprocal-rank fusion across time, enzyme_smi, reaction_smi",
            "checkpoint": "three split-specific F3 checkpoints",
            "config": "run_all_f3_splits.py",
        },
        "candidate_pool": {"candidate_count": 144},
        "scoring": {
            "type": "reciprocal_rank_fusion",
            "rrf_k": 60,
            "note": "Cosine scores are retained per split but never averaged across checkpoints.",
        },
        "rankings": ordered_consensus,
    }
    consensus_path = output_dir / "consensus_results.json"
    consensus_path.write_text(json.dumps(consensus_result, indent=2) + "\n", encoding="utf-8")
    consensus_summary, _ = compare_rankings(
        consensus_path,
        SOURCE_TABLE,
        output_dir / "ranked_candidates/consensus",
        top_k=25,
    )

    split_summary_rows = []
    for split in SPLIT_CONFIGS:
        summary = comparisons[split]
        split_summary_rows.append(
            {
                "split": split,
                "checkpoint": results[split]["model"]["checkpoint"],
                "top25_ranked_candidate_rows": summary["top_k_rows_matching_ranked_candidates"],
                "top25_direct_evidence_rows": summary["top_k_rows_with_direct_evidence_match"],
                "top25_exact_sequence_rows": summary["top_k_rows_with_exact_sequence_match"],
                "eligible_unique_sequences_recovered": summary[
                    "eligible_ranked_unique_sequences_recovered"
                ],
                "eligible_unique_sequence_recall": summary[
                    "eligible_ranked_unique_sequence_recall"
                ],
            }
        )
    _write_csv(output_dir / "split_summary.csv", split_summary_rows)

    summary_table = [
        "| Checkpoint split | Top-25 matching rows | Direct evidence | Exact sequence | "
        "Eligible unique recovery |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in split_summary_rows:
        summary_table.append(
            f"| {row['split']} | {row['top25_ranked_candidate_rows']} | "
            f"{row['top25_direct_evidence_rows']} | {row['top25_exact_sequence_rows']} | "
            f"{row['eligible_unique_sequences_recovered']}/25 "
            f"({float(row['eligible_unique_sequence_recall']):.1%}) |"
        )
    summary_table.append(
        f"| RRF consensus | {consensus_summary['top_k_rows_matching_ranked_candidates']} | "
        f"{consensus_summary['top_k_rows_with_direct_evidence_match']} | "
        f"{consensus_summary['top_k_rows_with_exact_sequence_match']} | "
        f"{consensus_summary['eligible_ranked_unique_sequences_recovered']}/25 "
        f"({float(consensus_summary['eligible_ranked_unique_sequence_recall']):.1%}) |"
    )

    agreement_table = [
        "| Pair | Spearman (144) | Top-10 overlap | Top-25 overlap |",
        "|---|---:|---:|---:|",
    ]
    for row in pairwise:
        agreement_table.append(
            f"| {row['first_split']} vs {row['second_split']} | "
            f"{row['spearman_all_144']:.3f} | {row['top_10_overlap']}/10 | "
            f"{row['top_25_overlap']}/25 |"
        )

    consensus_table = [
        "| Consensus rank | Candidate | Time | Enzyme-Sim | Reaction-Sim | Mean rank |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in ordered_consensus[:25]:
        consensus_table.append(
            f"| {row['rank']} | {row['protein_id']} | {row['time_rank']} | "
            f"{row['enzyme_smi_rank']} | {row['reaction_smi_rank']} | "
            f"{row['mean_rank']:.1f} |"
        )

    report = "\n".join(
        [
            "# F3 Restricted Setting Across All Splits",
            "",
            "The same D-fructose -> D-tagatose reaction and the same 144 Homolog rows were "
            "scored independently with the validation-selected F3 checkpoint from each "
            "ReactZyme split.",
            "",
            "Cosine values are checkpoint-specific and are not averaged. The consensus uses "
            "reciprocal-rank fusion with k=60 and is diagnostic rather than a trained model.",
            "",
            "## Ranked Candidates overlap",
            "",
            *summary_table,
            "",
            "## Rank agreement",
            "",
            *agreement_table,
            "",
            "## Consensus top 25",
            "",
            *consensus_table,
            "",
        ]
    )
    report_path = output_dir / "README.md"
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "schema_version": "case1_restricted_all_f3_splits_v1",
        "result_paths": {split: str(path) for split, path in result_paths.items()},
        "comparisons": comparisons,
        "pairwise_rank_agreement": pairwise,
        "consensus_summary": consensus_summary,
        "report": str(report_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def run(gpus: list[int], *, force_features: bool, rerun_reaction_smi: bool) -> Path:
    splits_to_run = ["time", "enzyme_smi"]
    if rerun_reaction_smi or _latest_result("reaction_smi") is None:
        splits_to_run.append("reaction_smi")
    if len(gpus) < len(splits_to_run):
        raise ValueError(
            f"Need at least {len(splits_to_run)} GPU IDs for concurrent execution, got {len(gpus)}"
        )

    result_paths: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=len(splits_to_run)) as executor:
        futures = {
            executor.submit(
                _run_split,
                split,
                gpu,
                force_features=force_features,
            ): split
            for split, gpu in zip(splits_to_run, gpus)
        }
        for future in as_completed(futures):
            split = futures[future]
            result_paths[split] = future.result()
            print(f"Completed {split}: {result_paths[split]}", flush=True)

    for split in SPLIT_CONFIGS:
        if split not in result_paths:
            result = _latest_result(split)
            if result is None:
                raise RuntimeError(f"No result is available for {split}")
            result_paths[split] = result
    return aggregate(result_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--rerun-reaction-smi", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(
        args.gpus,
        force_features=args.force_features,
        rerun_reaction_smi=args.rerun_reaction_smi,
    )
    print(f"All-split manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
