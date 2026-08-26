#!/usr/bin/env python3
"""Build train-only enzyme capability labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.capability.enzyme_labels import build_enzyme_capability_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--reaction-features", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--validation-pairs", default=None)
    parser.add_argument("--test-pairs", default=None)
    parser.add_argument("--sequence-cluster-map", default=None)
    args = parser.parse_args()
    build_enzyme_capability_labels(
        train_pairs_path=args.train_pairs,
        reaction_features_path=args.reaction_features,
        out_dir=args.out_dir,
        validation_pairs_path=args.validation_pairs,
        test_pairs_path=args.test_pairs,
        sequence_cluster_map_path=args.sequence_cluster_map,
    )


if __name__ == "__main__":
    main()
