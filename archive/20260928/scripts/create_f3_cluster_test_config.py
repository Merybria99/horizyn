#!/usr/bin/env python3
"""Create the released Reaction-Sim test config for clean cluster-trained F3."""

from __future__ import annotations

import argparse
from pathlib import Path

from horizyn.benchmarks.reactzyme_cluster_evaluation import (
    build_f3_cluster_test_config,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT
        / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/test.yaml",
    )
    parser.add_argument(
        "--test-protocol-dir",
        type=Path,
        default=ROOT
        / "data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85",
    )
    parser.add_argument(
        "--chemistry-path",
        type=Path,
        default=ROOT
        / "data/revised_protocols/reactzyme_reaction_cluster_validation_v1"
        / "similarity_0p85/features/reaction_set/test_reaction_set_features.npz",
    )
    parser.add_argument(
        "--official-test-feature-dir",
        type=Path,
        default=ROOT
        / "data/revised_protocols/reactzyme_official/features/reaction_smi/test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/reactzyme_f3_cluster_proxy_v1/configs/test.yaml",
    )
    args = parser.parse_args()
    build_f3_cluster_test_config(
        base_config_path=args.base_config,
        test_protocol_dir=args.test_protocol_dir,
        chemistry_path=args.chemistry_path,
        official_test_feature_dir=args.official_test_feature_dir,
        output_path=args.output,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
