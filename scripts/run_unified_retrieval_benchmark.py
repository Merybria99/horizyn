#!/usr/bin/env python3
"""Run the unified enzyme-reaction retrieval benchmark suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from horizyn.benchmarks.retrieval import (
    load_benchmark_suite,
    run_benchmark_task,
    task_to_dict,
    write_grouped_summary_tables,
    write_json,
    write_summary_csv,
    write_summary_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manifest-driven enzyme-reaction retrieval benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None, help="Repo Lightning checkpoint")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--pipeline", choices=["v4", "f3", "circev2"], default="v4")
    parser.add_argument(
        "--suite",
        default="configs/benchmarks/enzyme_retrieval_unified.yaml",
        help="Benchmark suite YAML",
    )
    parser.add_argument("--tasks", nargs="+", default=["all"], help="Task names or 'all'")
    parser.add_argument(
        "--protein-embedding",
        choices=["prott5", "esm2", "esmc"],
        default="prott5",
        help="Residue HDF5 family to use for residue-pooling checkpoints",
    )
    parser.add_argument(
        "--score-protein-embedding",
        choices=["prott5", "esm2", "esmc"],
        default="esm2",
        help="Residue HDF5 family to use for external SLEEC score embeddings",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark artifacts")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=512)
    parser.add_argument("--store-targets-on-cpu", action="store_true")
    parser.add_argument(
        "--target-cache-dir",
        default=None,
        help=(
            "Optional directory for checkpoint-specific encoded target embeddings. "
            "Matching tasks reuse exact caches, and compatible subset tasks can slice "
            "from a cached superset."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def select_tasks(tasks, requested_names: list[str]):
    if requested_names == ["all"] or "all" in requested_names:
        return tasks
    by_name = {task.name: task for task in tasks}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown benchmark task(s): {missing}; available={sorted(by_name)}")
    return [by_name[name] for name in requested_names]


def main() -> None:
    args = parse_args()
    from horizyn.config import load_config
    from horizyn.pipelines.registry import validate_pipeline

    validate_pipeline(load_config(args.config), args.pipeline)
    if not args.validate_only and args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --validate-only is set")

    project_root = Path.cwd()
    suite_path = Path(args.suite)
    tasks = select_tasks(load_benchmark_suite(suite_path, project_root), args.tasks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_suite = {
        "suite": str(suite_path),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "protein_embedding": args.protein_embedding,
        "score_protein_embedding": args.score_protein_embedding,
        "device": args.device,
        "query_batch_size": args.query_batch_size,
        "target_batch_size": args.target_batch_size,
        "store_targets_on_cpu": bool(args.store_targets_on_cpu),
        "target_cache_dir": args.target_cache_dir,
        "validate_only": bool(args.validate_only),
        "tasks": [task_to_dict(task) for task in tasks],
    }
    write_json(output_dir / "resolved_suite.json", resolved_suite)

    results = []
    for task in tasks:
        print(f"Running benchmark task: {task.name}", flush=True)
        result = run_benchmark_task(
            task,
            checkpoint=args.checkpoint or "",
            config_path=args.config,
            protein_embedding=args.protein_embedding,
            score_protein_embedding=args.score_protein_embedding,
            device=args.device,
            query_batch_size=args.query_batch_size,
            target_batch_size=args.target_batch_size,
            store_targets_on_cpu=args.store_targets_on_cpu,
            target_cache_dir=args.target_cache_dir,
            validate_only=args.validate_only,
        )
        results.append(result)
        write_json(output_dir / f"{task.name}.json", result)
        print(json.dumps(result, indent=2)[:4000], flush=True)

    write_summary_csv(results, output_dir / "summary.csv")
    write_summary_markdown(results, output_dir / "summary.md")
    write_grouped_summary_tables(results, output_dir)
    print(f"Saved benchmark outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
