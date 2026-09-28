#!/usr/bin/env python3
"""Stamp or validate canonical-isomeric provenance on a reaction HDF5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horizyn.chemistry.feature_provenance import (
    stamp_h5_reaction_feature_provenance,
    validate_h5_reaction_feature_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--extractor-name")
    parser.add_argument("--extractor-version")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--minimum-coverage", type=float, default=1.0)
    args = parser.parse_args()
    if args.validate:
        report = validate_h5_reaction_feature_provenance(
            args.artifact,
            args.source_csv,
            minimum_coverage=args.minimum_coverage,
        )
    else:
        if not args.extractor_name or not args.extractor_version:
            parser.error("stamping requires --extractor-name and --extractor-version")
        report = stamp_h5_reaction_feature_provenance(
            args.artifact,
            args.source_csv,
            extractor_name=args.extractor_name,
            extractor_version=args.extractor_version,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

