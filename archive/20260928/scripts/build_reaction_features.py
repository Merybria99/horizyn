#!/usr/bin/env python3
"""Build reaction-level capability features."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
extra_site = os.environ.get("HORIZYN_CAPABILITY_SITE_PACKAGES", "").strip()
if extra_site and extra_site not in sys.path:
    # Keep the invoking environment's torch/transformers stack authoritative.
    sys.path.append(extra_site)

from horizyn.capability.reaction_features import build_reaction_features


def _bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-smiles", required=True)
    parser.add_argument("--train-pairs", default=None)
    parser.add_argument("--rhea2ec", default=None)
    parser.add_argument("--chebi-names", default=None)
    parser.add_argument("--rhea-chebi-smiles", default=None)
    parser.add_argument("--cofactor-dictionary", default=None)
    parser.add_argument("--atom-mapped-reaction-smiles", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--drfp-bits", type=int, default=2048)
    parser.add_argument("--use-rxnmapper", type=_bool, default=False)
    parser.add_argument("--rxnmapper-batch-size", type=int, default=64)
    args = parser.parse_args()
    build_reaction_features(
        reaction_smiles_path=args.reaction_smiles,
        train_pairs_path=args.train_pairs,
        rhea2ec_path=args.rhea2ec,
        chebi_names_path=args.chebi_names,
        rhea_chebi_smiles_path=args.rhea_chebi_smiles,
        cofactor_dictionary_path=args.cofactor_dictionary,
        atom_mapped_reaction_smiles_path=args.atom_mapped_reaction_smiles,
        out_dir=args.out_dir,
        drfp_bits=args.drfp_bits,
        use_rxnmapper=args.use_rxnmapper,
        rxnmapper_batch_size=args.rxnmapper_batch_size,
    )


if __name__ == "__main__":
    main()
