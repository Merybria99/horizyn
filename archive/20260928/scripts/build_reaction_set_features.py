#!/usr/bin/env python3
"""Build train-fitted, paper-compatible ReactZyme molecule-set features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horizyn.capability.reaction_set_features import build_reaction_set_feature_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-reactions", required=True)
    parser.add_argument("--validation-reactions", required=True)
    parser.add_argument("--test-reactions", required=True)
    parser.add_argument("--cofactor-dictionary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--morgan-bits", type=int, default=512)
    args = parser.parse_args()
    report = build_reaction_set_feature_splits(
        train_reactions_path=args.train_reactions,
        validation_reactions_path=args.validation_reactions,
        test_reactions_path=args.test_reactions,
        cofactor_dictionary_path=args.cofactor_dictionary,
        out_dir=args.out_dir,
        morgan_bits=args.morgan_bits,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
