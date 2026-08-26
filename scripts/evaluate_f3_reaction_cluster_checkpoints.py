#!/usr/bin/env python3
"""Evaluate all available F3 epochs on reaction-cluster-held-out validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from horizyn.benchmarks.reactzyme_cluster_evaluation import (
    build_f3_panel_config,
    discover_epoch_checkpoints,
    write_chemistry_subset,
    write_panel_report,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=ROOT / "data/revised_protocols/reactzyme_reaction_cluster_validation_v1",
    )
    parser.add_argument("--panels", nargs="+", default=["similarity_0p85"])
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT
        / "runs/reactzyme_reaction_features_v1/checkpoints/reaction_smi/F3_set_chemistry",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT
        / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/test.yaml",
    )
    parser.add_argument(
        "--chemistry-npz",
        type=Path,
        nargs="+",
        default=[
            ROOT
            / "runs/reactzyme_reaction_features_v1/data/reaction_smi/reaction_set"
            / "train_reaction_set_features.npz",
            ROOT
            / "runs/reactzyme_reaction_features_v1/data/reaction_smi/reaction_set"
            / "validation_reaction_set_features.npz",
        ],
    )
    parser.add_argument(
        "--official-train-feature-dir",
        type=Path,
        default=ROOT
        / "data/revised_protocols/reactzyme_official/features/reaction_smi/train",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs/reactzyme_f3_cluster_validation_v1",
    )
    parser.add_argument("--minimum-epoch", type=int, default=10)
    parser.add_argument("--maximum-epoch", type=int, default=30)
    parser.add_argument("--baseline-epoch", type=int, default=28)
    parser.add_argument("--r2e-tolerance", type=float, default=0.015)
    parser.add_argument(
        "--checkpoint-training-pairs",
        type=Path,
        default=ROOT
        / "data/revised_protocols/reactzyme_paper/reaction_smi/train_pairs.csv",
        help="Actual pair table used to train the retrospective F3 checkpoints",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=512)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoints, missing = discover_epoch_checkpoints(
        args.checkpoint_dir,
        minimum_epoch=args.minimum_epoch,
        maximum_epoch=args.maximum_epoch,
    )
    if not checkpoints:
        raise SystemExit(f"No requested checkpoints found in {args.checkpoint_dir}")
    print(f"Available checkpoint epochs: {list(checkpoints)}", flush=True)
    print(f"Missing checkpoint epochs: {missing}", flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    panel_reports = {}
    for panel in args.panels:
        panel_dir = args.protocol_root / panel
        if not panel_dir.is_dir():
            raise SystemExit(f"Unknown panel directory: {panel_dir}")
        asset_dir = args.output_root / "assets" / panel
        chemistry_path = asset_dir / "validation_f3_checkpoint_chemistry.npz"
        chemistry_manifest = write_chemistry_subset(
            source_npz_paths=args.chemistry_npz,
            reactions_path=panel_dir / "validation_rxns.csv",
            output_path=chemistry_path,
        )
        config_path = args.output_root / "configs" / f"{panel}.yaml"
        build_f3_panel_config(
            base_config_path=args.base_config,
            panel_dir=panel_dir,
            chemistry_path=chemistry_path,
            official_train_feature_dir=args.official_train_feature_dir,
            output_path=config_path,
        )
        (asset_dir / "manifest.json").write_text(
            json.dumps(chemistry_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result_dir = args.output_root / "results" / panel
        result_dir.mkdir(parents=True, exist_ok=True)
        if not args.prepare_only:
            for epoch, checkpoint in checkpoints.items():
                output_path = result_dir / f"epoch_{epoch}.json"
                log_path = args.output_root / "logs" / panel / f"epoch_{epoch}.log"
                if output_path.exists() and not args.force:
                    print(f"Skipping existing {panel} epoch {epoch}", flush=True)
                    continue
                log_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(ROOT / "scripts/evaluate_protein_pooling.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--config",
                    str(config_path),
                    "--device",
                    "cuda",
                    "--direction",
                    "both",
                    "--evaluation-protocol",
                    "configured_forward_candidates",
                    "--batch-size",
                    str(args.batch_size),
                    "--target-batch-size",
                    str(args.target_batch_size),
                    "--output",
                    str(output_path),
                ]
                print(f"Evaluating {panel} epoch {epoch}", flush=True)
                with log_path.open("w", encoding="utf-8") as log_handle:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode != 0:
                    tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
                    raise SystemExit(
                        f"Evaluation failed for {panel} epoch {epoch}:\n{tail}"
                    )
        panel_reports[panel] = write_panel_report(
            panel=panel,
            result_dir=result_dir,
            output_dir=args.output_root / "reports",
            available_epochs=list(checkpoints),
            missing_epochs=missing,
            minimum_epoch=args.minimum_epoch,
            maximum_epoch=args.maximum_epoch,
            baseline_epoch=args.baseline_epoch,
            r2e_tolerance=args.r2e_tolerance,
            panel_pairs_path=panel_dir / "validation_pairs.csv",
            checkpoint_training_pairs_path=args.checkpoint_training_pairs,
        )
    print(json.dumps(panel_reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
