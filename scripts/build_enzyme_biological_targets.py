#!/usr/bin/env python3
"""Build train-only minimal mechanism and cofactor enzyme targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.capability.biological_targets import build_minimal_targets, write_minimal_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", required=True, type=Path)
    parser.add_argument("--matched-rhea-members", required=True, type=Path)
    parser.add_argument("--directional-reaction-features", required=True, type=Path)
    parser.add_argument("--enzyme-cofactor-labels", type=Path, default=None)
    parser.add_argument("--out-npz", required=True, type=Path)
    parser.add_argument("--out-vocab", required=True, type=Path)
    args = parser.parse_args()

    ids, arrays, metadata = build_minimal_targets(
        train_pairs_path=args.train_pairs,
        matched_members_path=args.matched_rhea_members,
        directional_features_path=args.directional_reaction_features,
        enzyme_cofactor_labels_path=args.enzyme_cofactor_labels,
    )
    write_minimal_targets(args.out_npz, args.out_vocab, ids, arrays, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
