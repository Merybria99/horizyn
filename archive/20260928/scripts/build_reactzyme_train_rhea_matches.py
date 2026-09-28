#!/usr/bin/env python3
"""Write row-specific directional Rhea matches for one ReactZyme train split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import RDLogger

from horizyn.capability.reactzyme_rhea import reconstruct_reactzyme_train_rhea

RDLogger.DisableLog("rdApp.warning")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", required=True, type=Path)
    parser.add_argument(
        "--split",
        required=True,
        choices=("time", "enzyme_smi", "reaction_smi"),
    )
    parser.add_argument("--cleaned-uniprot-rhea", required=True, type=Path)
    parser.add_argument("--rhea-molecules", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    pairs, reactions, members, report = reconstruct_reactzyme_train_rhea(
        reactzyme_eval_root=args.protocol_root,
        cleaned_uniprot_rhea_path=args.cleaned_uniprot_rhea,
        rhea_molecules_path=args.rhea_molecules,
        splits=(args.split,),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out_dir / "matched_pairs.csv", index=False)
    reactions.to_csv(args.out_dir / "matched_reactions.csv", index=False)
    members.to_csv(args.out_dir / "matched_members.csv", index=False)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
