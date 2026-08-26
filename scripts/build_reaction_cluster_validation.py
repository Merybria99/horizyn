#!/usr/bin/env python3
"""Build reaction-cluster-held-out ReactZyme Reaction-Sim validation panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from horizyn.benchmarks.reactzyme_cluster_validation import (
    DEFAULT_PRIMARY_THRESHOLD,
    DEFAULT_THRESHOLDS,
    materialize_reaction_cluster_validation,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data/revised_protocols/reactzyme_official/reaction_smi",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/revised_protocols/reactzyme_reaction_cluster_validation_v1",
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
        "--reactiont5-h5",
        type=Path,
        default=ROOT
        / "data/revised_protocols/reactzyme_official/features/reaction_smi/train"
        / "reactiont5v2.h5",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--primary-threshold", type=float, default=DEFAULT_PRIMARY_THRESHOLD)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--morgan-dims", type=int, default=512)
    parser.add_argument("--morgan-weight", type=float, default=0.70)
    parser.add_argument("--reactiont5-weight", type=float, default=0.30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = materialize_reaction_cluster_validation(
        source_dir=args.source_dir,
        out_root=args.output_root,
        chemistry_npz_paths=args.chemistry_npz,
        reactiont5_h5_path=args.reactiont5_h5,
        thresholds=args.thresholds,
        primary_threshold=args.primary_threshold,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        morgan_dims=args.morgan_dims,
        morgan_weight=args.morgan_weight,
        reactiont5_weight=args.reactiont5_weight,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output_root": str(args.output_root), "panels": manifest["panels"]}, indent=2))


if __name__ == "__main__":
    main()
