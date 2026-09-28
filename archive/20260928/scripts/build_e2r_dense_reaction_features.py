#!/usr/bin/env python3
"""Build train-fitted dense reaction transformation vectors for Q3/Q5."""

from __future__ import annotations

import argparse
import json

from horizyn.capability.reaction_dense_features import (
    build_dense_reaction_feature_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}-reactions", required=True)
        parser.add_argument(f"--{split}-center-features", default=None)
        parser.add_argument(f"--{split}-directional-reactions", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    report = build_dense_reaction_feature_splits(
        train_reactions_path=args.train_reactions,
        validation_reactions_path=args.validation_reactions,
        test_reactions_path=args.test_reactions,
        train_center_features_path=args.train_center_features,
        validation_center_features_path=args.validation_center_features,
        test_center_features_path=args.test_center_features,
        train_directional_reactions_path=args.train_directional_reactions,
        validation_directional_reactions_path=args.validation_directional_reactions,
        test_directional_reactions_path=args.test_directional_reactions,
        out_dir=args.out_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
