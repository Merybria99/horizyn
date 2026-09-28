#!/usr/bin/env python3
"""Assemble masked F5/F6 directional vectors from extracted Rhea embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horizyn.capability.reaction_directional_features import build_directional_vector_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}-reactions", required=True)
        parser.add_argument(f"--{split}-mapping", required=True)
        parser.add_argument(f"--{split}-reaction-t5", required=True)
        parser.add_argument(f"--{split}-unimol2", required=True)
        parser.add_argument(f"--{split}-chiro", required=True)
        parser.add_argument(f"--{split}-center-features", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    inputs = {}
    for split in ("train", "validation", "test"):
        prefix = split.replace("-", "_")
        inputs[split] = {
            "reactions": getattr(args, f"{prefix}_reactions"),
            "mapping": getattr(args, f"{prefix}_mapping"),
            "reaction_t5": getattr(args, f"{prefix}_reaction_t5"),
            "unimol2": getattr(args, f"{prefix}_unimol2"),
            "chiro": getattr(args, f"{prefix}_chiro"),
            "center_features": getattr(args, f"{prefix}_center_features"),
        }
    report = build_directional_vector_splits(split_inputs=inputs, out_dir=args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
