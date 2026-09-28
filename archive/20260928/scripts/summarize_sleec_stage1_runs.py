#!/usr/bin/env python3
"""Summarize SLEEC stage-1 training runs from checkpoint directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run label and directory as label=path. May be repeated.",
    )
    parser.add_argument(
        "--output-json",
        default=Path("checkpoints/SLEEC/sleec_stage1_esm2_650m_comparison.json"),
        type=Path,
    )
    parser.add_argument(
        "--output-csv",
        default=Path("checkpoints/SLEEC/sleec_stage1_esm2_650m_comparison.csv"),
        type=Path,
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def summarize_run(label: str, run_dir: Path) -> dict[str, Any]:
    best = load_checkpoint(run_dir / "best.ckpt")
    last = load_checkpoint(run_dir / "last.ckpt")
    manifest = load_json(run_dir / "run_manifest.json")
    best_metrics = (best or {}).get("metrics", {})
    row = {
        "label": label,
        "run_dir": str(run_dir),
        "model_variant": manifest.get("model_variant"),
        "best_step": (best or {}).get("step"),
        "last_step": (last or {}).get("step"),
        "best_metric_name": (best or {}).get("best_metric_name", "f1"),
        "best_metric_value": (best or {}).get("best_metric_value"),
        "best_val_f1": (best or {}).get("best_val_f1"),
        "val_loss": best_metrics.get("loss"),
        "val_precision": best_metrics.get("precision"),
        "val_recall": best_metrics.get("recall"),
        "val_accuracy": best_metrics.get("accuracy"),
        "checkpoint_exists": best is not None,
    }
    return row


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"--run must have form label=path, got: {item}")
        label, path = item.split("=", 1)
        rows.append(summarize_run(label, Path(path)))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
