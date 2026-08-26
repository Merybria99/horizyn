#!/usr/bin/env python3
"""Create the clean F3 Reaction-Sim cluster-proxy training configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from horizyn.benchmarks.reactzyme_cluster_evaluation import (
    build_f3_cluster_training_config,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT
        / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/train.yaml",
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=ROOT
        / "data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85",
    )
    parser.add_argument(
        "--chemistry-dir",
        type=Path,
        default=ROOT
        / "data/revised_protocols/reactzyme_reaction_cluster_validation_v1"
        / "similarity_0p85/features/reaction_set",
    )
    parser.add_argument(
        "--official-train-feature-dir",
        type=Path,
        default=ROOT
        / "data/revised_protocols/reactzyme_official/features/reaction_smi/train",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs/reactzyme_f3_cluster_proxy_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs/benchmarks/reactzyme_f3_cluster_proxy_v1.yaml",
    )
    args = parser.parse_args()
    build_f3_cluster_training_config(
        base_config_path=args.base_config,
        panel_dir=args.panel_dir,
        chemistry_dir=args.chemistry_dir,
        official_train_feature_dir=args.official_train_feature_dir,
        run_root=args.run_root,
        output_path=args.output,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
