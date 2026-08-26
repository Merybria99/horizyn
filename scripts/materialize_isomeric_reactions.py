#!/usr/bin/env python3
"""Materialize a strict canonical-isomeric reaction CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horizyn.chemistry.isomeric_smiles import materialize_isomeric_reaction_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance")
    args = parser.parse_args()
    report = materialize_isomeric_reaction_csv(
        args.input,
        args.output,
        provenance_path=args.provenance,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

