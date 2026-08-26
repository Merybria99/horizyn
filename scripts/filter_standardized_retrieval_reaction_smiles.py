#!/usr/bin/env python3
"""Filter standardized retrieval files to reactions valid for Horizyn fingerprints."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from horizyn.chemistry.standardizer import Standardizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create *_valid_rxn.csv files by keeping only reactions accepted by "
            "Horizyn's RDKit reaction standardizer."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default="data/standardized/global_unique_retrieval",
        help="Root containing exact/nr90/nr50 standardized policy folders",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["exact", "nr90", "nr50"],
        help="Policies to filter",
    )
    parser.add_argument(
        "--policy-dir-template",
        default="{policy}",
        help=(
            "Directory template below --root for each policy. Use '{policy}' for the "
            "policy name, e.g. 'train_{policy}' for source-collapse folders."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Split prefixes to filter, e.g. train test",
    )
    parser.add_argument(
        "--allow-missing-splits",
        action="store_true",
        help="Skip split files that are absent instead of failing.",
    )
    parser.add_argument("--suffix", default="_valid_rxn", help="Output filename suffix")
    parser.add_argument(
        "--min-gt",
        type=int,
        default=2,
        help="Minimum number of '>' separators required before RDKit validation",
    )
    parser.add_argument(
        "--allow-pseudo-reactions",
        action="store_true",
        help=(
            "Accept molecule-set SMILES without reaction arrows by validating them as "
            "pseudo self-reactions, e.g. 'A.B' -> 'A.B>>A.B'."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_valid_reaction(
    smiles: str,
    standardizer: Standardizer,
    min_gt: int,
    allow_pseudo_reactions: bool,
) -> tuple[bool, str]:
    if smiles.count(">") < min_gt:
        if not allow_pseudo_reactions or ">" in smiles:
            return False, "missing_reaction_separators"
        smiles = f"{smiles}>>{smiles}"
    try:
        standardizer.standardize_reaction(smiles)
    except Exception as exc:
        return False, type(exc).__name__
    return True, "valid"


def filter_split(
    policy_root: Path,
    split: str,
    suffix: str,
    standardizer: Standardizer,
    min_gt: int,
    allow_pseudo_reactions: bool,
):
    rxn_fields, rxn_rows = read_csv(policy_root / f"{split}_rxns.csv")
    pair_fields, pair_rows = read_csv(policy_root / f"{split}_pairs.csv")

    valid_rxn_rows: list[dict[str, str]] = []
    invalid_rxn_rows: list[dict[str, str]] = []
    invalid_reasons: Counter[str] = Counter()
    invalid_by_source: Counter[str] = Counter()

    for row in rxn_rows:
        smiles = row.get("reaction_smiles", "")
        ok, reason = is_valid_reaction(
            smiles,
            standardizer=standardizer,
            min_gt=min_gt,
            allow_pseudo_reactions=allow_pseudo_reactions,
        )
        if ok:
            valid_rxn_rows.append(row)
        else:
            invalid_rxn_rows.append(row)
            invalid_reasons[reason] += 1
            source = (
                row.get("source_dataset")
                or row.get("source_entries")
                or row.get("source_reaction_ids")
                or ""
            )
            invalid_by_source[source] += 1

    valid_rxn_ids = {row["reaction_id"] for row in valid_rxn_rows}
    valid_pair_rows = [row for row in pair_rows if row.get("reaction_id") in valid_rxn_ids]
    invalid_pair_rows = [row for row in pair_rows if row.get("reaction_id") not in valid_rxn_ids]

    write_csv(policy_root / f"{split}_rxns{suffix}.csv", rxn_fields, valid_rxn_rows)
    write_csv(policy_root / f"{split}_pairs{suffix}.csv", pair_fields, valid_pair_rows)

    return {
        "input_reactions": len(rxn_rows),
        "valid_reactions": len(valid_rxn_rows),
        "invalid_reactions": len(invalid_rxn_rows),
        "input_pairs": len(pair_rows),
        "valid_pairs": len(valid_pair_rows),
        "invalid_pairs": len(invalid_pair_rows),
        "invalid_reaction_reasons": dict(sorted(invalid_reasons.items())),
        "invalid_reactions_by_source": dict(sorted(invalid_by_source.items())),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    standardizer = Standardizer(
        standardize_hypervalent=True,
        standardize_remove_hs=True,
        standardize_kekulize=False,
        standardize_uncharge=True,
        standardize_metals=True,
    )

    all_metadata = {}
    for policy in args.policies:
        policy_root = root / args.policy_dir_template.format(policy=policy)
        if not policy_root.exists():
            raise FileNotFoundError(policy_root)

        metadata = {
            "policy": policy,
            "policy_root": str(policy_root),
            "suffix": args.suffix,
            "validation": "Horizyn Standardizer.standardize_reaction",
            "allow_pseudo_reactions": args.allow_pseudo_reactions,
            "splits": {},
        }
        for split in args.splits:
            rxn_path = policy_root / f"{split}_rxns.csv"
            pair_path = policy_root / f"{split}_pairs.csv"
            if args.allow_missing_splits and (not rxn_path.exists() or not pair_path.exists()):
                metadata["splits"][split] = {
                    "skipped": True,
                    "reason": "missing_split_files",
                    "rxn_path": str(rxn_path),
                    "pair_path": str(pair_path),
                }
                continue
            metadata["splits"][split] = filter_split(
                policy_root=policy_root,
                split=split,
                suffix=args.suffix,
                standardizer=standardizer,
                min_gt=args.min_gt,
                allow_pseudo_reactions=args.allow_pseudo_reactions,
            )
        metadata_path = policy_root / f"metadata{args.suffix}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        all_metadata[policy] = metadata

    print(json.dumps(all_metadata, indent=2))


if __name__ == "__main__":
    main()
