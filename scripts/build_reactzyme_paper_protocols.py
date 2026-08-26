#!/usr/bin/env python3
"""Materialize reproducible ReactZyme paper train/validation/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from horizyn.benchmarks.reactzyme_protocol import (
    REACTZYME_SPLITS,
    build_protocol,
    materialize_paper_protocols,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data/revised_protocols/reactzyme_official"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/revised_protocols/reactzyme_paper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize_paper_protocols(
        source_root=args.source_root,
        out_root=args.out_root,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
        protocols=REACTZYME_SPLITS,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
