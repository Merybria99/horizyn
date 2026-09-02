#!/usr/bin/env python3
"""Build the audited ReactZyme unseen-reaction development protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from horizyn.benchmarks.reactzyme_protocol import (
    REACTZYME_SPLITS,
    materialize_unseen_reaction_protocols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "data/revised_protocols/reactzyme_official",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=project_root / "data/revised_protocols/reactzyme_unseen_reaction",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=REACTZYME_SPLITS,
        default=list(REACTZYME_SPLITS),
        help="ReactZyme protocols to materialize.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = materialize_unseen_reaction_protocols(
        source_root=args.source_root,
        out_root=args.out_root,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
        protocols=args.protocols,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
