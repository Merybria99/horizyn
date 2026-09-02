#!/usr/bin/env python3
"""Audit released Level-1 split and candidate identities without loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.benchmarks.retrieval import (  # noqa: E402
    load_benchmark_suite,
    read_id_list,
    read_pairs,
    validate_level1_benchmark_sources,
    validate_level1_candidate_pool,
)


DEFAULT_SUITES = (
    PROJECT_ROOT / "configs/benchmarks/horizyn_reactzyme_eval.yaml",
    PROJECT_ROOT / "configs/benchmarks/horizyn_f3_f4_paper_test.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        type=Path,
        help="Suite YAML to audit; repeat for multiple suites.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suites = args.suite or list(DEFAULT_SUITES)
    report: dict[str, object] = {"schema_version": 1, "suites": {}}
    for suite_path in suites:
        suite_path = suite_path if suite_path.is_absolute() else PROJECT_ROOT / suite_path
        task_reports: dict[str, object] = {}
        for task in load_benchmark_suite(suite_path, PROJECT_ROOT):
            if task.benchmark_protocol == "custom":
                continue
            raw_pairs = read_pairs(task.pairs, task.reaction_id_col, task.protein_id_col)
            source_stats = validate_level1_benchmark_sources(task, raw_pairs)
            if task.candidates_from_test_positives:
                candidate_ids = sorted({protein_id for _reaction_id, protein_id in raw_pairs})
            else:
                candidate_ids = read_id_list(task.candidate_ids)
            candidate_stats = (
                {"release_candidate_pool_compliance": "deferred_to_embedding_store"}
                if candidate_ids is None
                else validate_level1_candidate_pool(task, candidate_ids)
            )
            task_reports[task.name] = {**source_stats, **candidate_stats}
        report["suites"][str(suite_path)] = task_reports

    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
