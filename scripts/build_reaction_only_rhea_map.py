#!/usr/bin/env python3
"""Map ReactZyme molecule sets to Rhea using reaction participants only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horizyn.capability.reaction_directional_features import build_reaction_only_rhea_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactions", required=True)
    parser.add_argument("--rhea-molecules", required=True)
    parser.add_argument("--out-mapping", required=True)
    parser.add_argument("--out-directional-reactions", required=True)
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()
    report = build_reaction_only_rhea_map(
        reactions_path=args.reactions,
        rhea_molecules_path=args.rhea_molecules,
        output_mapping_path=args.out_mapping,
        output_directional_reactions_path=args.out_directional_reactions,
        output_report_path=args.out_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
