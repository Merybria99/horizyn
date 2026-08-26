#!/usr/bin/env python3
"""
Evaluate selected experiment checkpoints with comparable retrieval JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.evaluate_protein_pooling import (
    EVALUATION_PROTOCOLS,
    LEGACY_ALL_CANDIDATES,
    evaluate_checkpoint as evaluate_protein_pooling,
)
from scripts.evaluate_reaction_conditioned import evaluate_checkpoint as evaluate_reaction_conditioned


def parse_labeled_checkpoint(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.stem.replace("protein-pooling-", "").replace("reaction-conditioned-", ""), path


def find_epoch_checkpoint(checkpoint_dir: Path, epoch: str) -> Path:
    epoch_int = int(epoch)
    candidates = sorted(checkpoint_dir.glob(f"*epoch={epoch_int:02d}.ckpt"))
    if not candidates:
        candidates = sorted(checkpoint_dir.glob(f"*epoch={epoch_int}.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint for epoch {epoch_int:02d} in {checkpoint_dir}")
    return candidates[0]


def collect_checkpoints(args: argparse.Namespace) -> list[tuple[str, Path]]:
    checkpoint_dir = Path(args.checkpoint_dir)
    selected: list[tuple[str, Path]] = []

    for epoch in args.epochs:
        path = find_epoch_checkpoint(checkpoint_dir, epoch)
        selected.append((f"epoch{int(epoch):02d}", path))

    if args.include_last:
        last_path = checkpoint_dir / "last.ckpt"
        if last_path.exists():
            selected.append(("last", last_path))
        else:
            raise FileNotFoundError(f"Requested last checkpoint but not found: {last_path}")

    for value in args.checkpoint:
        selected.append(parse_labeled_checkpoint(value))

    seen = set()
    deduped = []
    for label, path in selected:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append((label, path))
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kind",
        choices=("protein_pooling", "reaction_conditioned"),
        required=True,
        help="Checkpoint family to evaluate",
    )
    parser.add_argument("--config", required=True, help="Config used for the run")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing checkpoints")
    parser.add_argument("--output-dir", required=True, help="Directory for eval JSON files")
    parser.add_argument("--epochs", nargs="*", default=[], help="Epoch numbers to evaluate")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Extra checkpoint path, optionally label=path. Can be passed multiple times.",
    )
    parser.add_argument("--include-last", action="store_true", help="Also evaluate last.ckpt")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Query batch size")
    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=512,
        help="Protein-pooling target encoding batch size",
    )
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=64,
        help="Reaction-conditioned target scoring chunk size",
    )
    parser.add_argument(
        "--store-targets-on-cpu",
        action="store_true",
        help="Protein-pooling only: store encoded targets on CPU before scoring",
    )
    parser.add_argument(
        "--evaluation-protocol",
        choices=EVALUATION_PROTOCOLS,
        default=LEGACY_ALL_CANDIDATES,
        help="Protein-pooling only: retrieval candidate/query protocol",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = collect_checkpoints(args)
    if not checkpoints:
        raise ValueError("No checkpoints selected; pass --epochs, --include-last, or --checkpoint")

    summary = {}
    for label, checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if args.kind == "protein_pooling":
            results = evaluate_protein_pooling(
                checkpoint_path=str(checkpoint_path),
                config_path=args.config,
                device=args.device,
                batch_size=args.batch_size,
                target_batch_size=args.target_batch_size,
                store_targets_on_device=not args.store_targets_on_cpu,
                evaluation_protocol=args.evaluation_protocol,
            )
        else:
            results = evaluate_reaction_conditioned(
                checkpoint_path=str(checkpoint_path),
                config_path=args.config,
                device=args.device,
                batch_size=args.batch_size,
                target_chunk_size=args.target_chunk_size,
            )

        output_path = output_dir / f"eval_{label}.json"
        with output_path.open("w") as handle:
            json.dump(results, handle, indent=2)
        summary[label] = {
            key: results.get(key)
            for key in ("num_queries", "top_1", "top_10", "r_precision", "avg_precision")
        }
        print(f"Wrote {output_path}")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
