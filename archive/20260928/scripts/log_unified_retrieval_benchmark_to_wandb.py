#!/usr/bin/env python3
"""Log unified retrieval benchmark outputs to one Weights & Biases run."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from horizyn.benchmarks.retrieval import (
    flatten_result_metrics,
    write_grouped_summary_tables,
    write_json,
    write_summary_csv,
    write_summary_markdown,
)


def configure_wandb_env_defaults() -> None:
    root = Path(os.environ.get("WANDB_ROOT", "/datastor2/deep-proteins/EnzymeDiscovery/wandb"))
    os.environ.setdefault("WANDB_DIR", str(root))
    os.environ.setdefault("WANDB_DATA_DIR", str(root / "data"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(root / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(root / "config"))
    for key in ("WANDB_DIR", "WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark task JSON files and log them as one W&B run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output_root", help="Benchmark output root containing task JSON files")
    parser.add_argument(
        "--project",
        default="horizyn-unified-retrieval-benchmark",
        help="W&B project for benchmark test runs",
    )
    parser.add_argument("--entity", default=None, help="Optional W&B entity/team")
    parser.add_argument("--run-name", default=None, help="Optional W&B run name")
    parser.add_argument(
        "--mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode",
    )
    parser.add_argument(
        "--expected-tasks",
        nargs="*",
        default=None,
        help="Task names that must be present before logging",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until expected task JSON files are present",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval when --wait is set",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Maximum wait time; 0 means no timeout",
    )
    parser.add_argument(
        "--defer-wandb-init",
        action="store_true",
        help="Create the W&B run only after all expected results are present",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_task_results(output_root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(output_root.rglob("*.json")):
        if path.name in {"resolved_suite.json", "combined_results.json"}:
            continue
        payload = read_json(path)
        if isinstance(payload, dict) and payload.get("task"):
            payload["_source_json"] = str(path)
            results.append(payload)
    return sorted(results, key=lambda item: str(item.get("task", "")))


def wait_for_results(
    output_root: Path,
    expected_tasks: set[str] | None,
    poll_seconds: float,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    start = time.monotonic()
    while True:
        results = find_task_results(output_root)
        present_tasks = {str(result.get("task")) for result in results}
        if expected_tasks is None:
            if results:
                return results
        elif expected_tasks.issubset(present_tasks):
            return results

        if timeout_seconds > 0 and time.monotonic() - start > timeout_seconds:
            missing = sorted((expected_tasks or set()) - present_tasks)
            raise TimeoutError(f"Timed out waiting for benchmark tasks: {missing}")
        time.sleep(poll_seconds)


def load_resolved_suites(output_root: Path) -> list[dict[str, Any]]:
    suites = []
    for path in sorted(output_root.rglob("resolved_suite.json")):
        payload = read_json(path)
        payload["_source_json"] = str(path)
        suites.append(payload)
    return suites


def sanitize_artifact_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "unified-retrieval-benchmark"


def benchmark_config(
    output_root: Path,
    results: list[dict[str, Any]],
    suites: list[dict[str, Any]],
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "output_root": str(output_root),
        "tasks": [result.get("task") for result in results],
        "num_tasks": len(results),
    }
    for key in (
        "checkpoint",
        "config",
        "suite",
        "protein_embedding",
        "score_protein_embedding",
        "query_batch_size",
        "target_batch_size",
        "store_targets_on_cpu",
    ):
        values = sorted(
            {str(suite[key]) for suite in suites if key in suite and suite.get(key) is not None}
        )
        if len(values) == 1:
            config[key] = values[0]
        elif values:
            config[key] = values
    return config


def write_combined_outputs(output_root: Path, results: list[dict[str, Any]]) -> None:
    write_json(output_root / "combined_results.json", {"results": results})
    write_summary_csv(results, output_root / "summary.csv")
    write_summary_markdown(results, output_root / "summary.md")
    write_grouped_summary_tables(results, output_root)


def init_wandb_run(
    output_root: Path,
    results: list[dict[str, Any]],
    suites: list[dict[str, Any]],
    project: str,
    entity: str | None,
    run_name: str | None,
    mode: str,
    expected_tasks: set[str] | None = None,
):
    configure_wandb_env_defaults()
    import wandb

    if run_name is None:
        run_name = output_root.name
    config = benchmark_config(output_root, results, suites)
    if expected_tasks is not None:
        config["expected_tasks"] = sorted(expected_tasks)
        config["expected_task_count"] = len(expected_tasks)
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        mode=mode,
        job_type="benchmark",
        config=config,
    )
    if run is not None:
        run.log(
            {
                "benchmark/status": "waiting_for_results",
                "benchmark/num_tasks": len(results),
                "benchmark/expected_task_count": len(expected_tasks or []),
            }
        )
    return run


def log_to_wandb(
    output_root: Path,
    results: list[dict[str, Any]],
    suites: list[dict[str, Any]],
    project: str,
    entity: str | None,
    run_name: str | None,
    mode: str,
    run: Any | None = None,
) -> str | None:
    configure_wandb_env_defaults()
    import wandb

    rows = [row for result in results for row in flatten_result_metrics(result)]
    config = benchmark_config(output_root, results, suites)
    if run is None:
        run = init_wandb_run(
            output_root,
            results,
            suites,
            project=project,
            entity=entity,
            run_name=run_name,
            mode=mode,
        )
    else:
        run.config.update(config, allow_val_change=True)
    if run is None:
        return None
    if run_name is None:
        run_name = output_root.name

    table_columns = [
        "dataset",
        "task",
        "task_label",
        "setting",
        "split",
        "direction",
        "metric",
        "value",
        "candidate_pool_size",
    ]
    table = wandb.Table(columns=table_columns)
    scalar_metrics: dict[str, float] = {}
    for row in rows:
        table.add_data(*(row.get(column, "") for column in table_columns))
        value = row.get("value")
        if isinstance(value, (int, float)):
            key = (
                f"{row.get('dataset')}/{row.get('task')}/"
                f"{row.get('direction')}/{row.get('metric')}"
            )
            scalar_metrics[key] = float(value)

    grouped_tables: dict[str, Any] = {}
    for dataset in sorted({str(row.get("dataset", "")) for row in rows}):
        dataset_rows = [row for row in rows if str(row.get("dataset", "")) == dataset]
        dataset_table = wandb.Table(columns=table_columns)
        for row in dataset_rows:
            dataset_table.add_data(*(row.get(column, "") for column in table_columns))
        grouped_tables[f"benchmark/by_dataset/{dataset}"] = dataset_table
    for task in sorted({str(row.get("task", "")) for row in rows}):
        task_rows = [row for row in rows if str(row.get("task", "")) == task]
        task_table = wandb.Table(columns=table_columns)
        for row in task_rows:
            task_table.add_data(*(row.get(column, "") for column in table_columns))
        grouped_tables[f"benchmark/by_task/{task}"] = task_table

    run.log(
        {
            "benchmark/summary": table,
            "benchmark/status": "complete",
            "benchmark/num_tasks": len(results),
            "benchmark/num_metric_rows": len(rows),
            **grouped_tables,
            **scalar_metrics,
        }
    )

    artifact = wandb.Artifact(
        sanitize_artifact_name(f"{run_name}-results"),
        type="benchmark-results",
    )
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            artifact.add_file(str(path), name=str(path.relative_to(output_root)))
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    return url


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"Benchmark output root not found: {output_root}")

    expected_tasks = set(args.expected_tasks) if args.expected_tasks else None
    suites = load_resolved_suites(output_root)
    run = None
    if args.wait and not args.defer_wandb_init:
        run = init_wandb_run(
            output_root,
            [],
            suites,
            project=args.project,
            entity=args.entity,
            run_name=args.run_name,
            mode=args.mode,
            expected_tasks=expected_tasks,
        )
        if run is not None and getattr(run, "url", None):
            print(f"W&B run initialized: {run.url}", flush=True)

    if args.wait:
        results = wait_for_results(
            output_root,
            expected_tasks,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        results = find_task_results(output_root)
        if expected_tasks is not None:
            present_tasks = {str(result.get("task")) for result in results}
            missing = sorted(expected_tasks - present_tasks)
            if missing:
                raise FileNotFoundError(f"Missing benchmark task results: {missing}")

    if not results:
        raise FileNotFoundError(f"No task result JSON files found under: {output_root}")

    write_combined_outputs(output_root, results)
    url = log_to_wandb(
        output_root,
        results,
        suites,
        project=args.project,
        entity=args.entity,
        run_name=args.run_name,
        mode=args.mode,
        run=run,
    )
    print(f"Logged {len(results)} benchmark task results to W&B project: {args.project}")
    if url:
        print(f"W&B run: {url}")


if __name__ == "__main__":
    main()
