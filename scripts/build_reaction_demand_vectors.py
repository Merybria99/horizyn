#!/usr/bin/env python3
"""Build reaction demand vectors for capability pretraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.capability.reaction_demand import build_reaction_demand_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-features", required=True)
    parser.add_argument("--reaction-drfp", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    build_reaction_demand_vectors(
        reaction_features_path=args.reaction_features,
        reaction_drfp_path=args.reaction_drfp,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
