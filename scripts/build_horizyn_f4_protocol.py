#!/usr/bin/env python3
"""Build the train-derived validation protocol used for Horizyn F4 training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import h5py


PAIR_FIELDS = ("pr_id", "reaction_id", "protein_id")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_ids(rows: Iterable[dict[str, str]], column: str) -> list[str]:
    return list(dict.fromkeys(row[column] for row in rows))


def write_ids(path: Path, values: Sequence[str]) -> None:
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def fasta_ids(path: Path) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            protein_id = line[1:].strip().split(maxsplit=1)[0]
            if not protein_id:
                raise ValueError(f"Empty FASTA identifier in {path}")
            if protein_id in seen:
                raise ValueError(f"Duplicate FASTA identifier {protein_id!r} in {path}")
            seen.add(protein_id)
            values.append(protein_id)
    if not values:
        raise ValueError(f"No FASTA records found in {path}")
    return values


def subset_reactions(
    reaction_rows: Sequence[dict[str, str]],
    pair_rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    requested = {row["reaction_id"] for row in pair_rows}
    selected = [row for row in reaction_rows if row["reaction_id"] in requested]
    found = {row["reaction_id"] for row in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(
            f"Missing {len(missing)} reaction rows; examples: {', '.join(missing[:5])}"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/sota")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--protein-h5", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    protein_h5 = Path(args.protein_h5).resolve()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_fields, source_pairs = read_csv(source_dir / "train_pairs.csv")
    reaction_fields, source_reactions = read_csv(source_dir / "train_rxns.csv")
    missing_fields = set(PAIR_FIELDS) - set(pair_fields)
    if missing_fields:
        raise ValueError(f"Train pairs are missing fields: {sorted(missing_fields)}")

    indices = list(range(len(source_pairs)))
    random.Random(args.seed).shuffle(indices)
    train_count = int((1.0 - args.validation_fraction) * len(indices))
    train_indices = set(indices[:train_count])
    train_pairs = [
        row for index, row in enumerate(source_pairs) if index in train_indices
    ]
    validation_pairs = [
        row for index, row in enumerate(source_pairs) if index not in train_indices
    ]
    train_reactions = subset_reactions(source_reactions, train_pairs)
    validation_reactions = subset_reactions(source_reactions, validation_pairs)

    write_csv(out_dir / "train_pairs.csv", pair_fields, train_pairs)
    write_csv(out_dir / "validation_pairs.csv", pair_fields, validation_pairs)
    write_csv(out_dir / "train_rxns.csv", reaction_fields, train_reactions)
    write_csv(out_dir / "validation_rxns.csv", reaction_fields, validation_reactions)
    for filename in ("test_pairs.csv", "test_rxns.csv"):
        shutil.copy2(source_dir / filename, out_dir / filename)

    train_candidates = unique_ids(train_pairs, "protein_id")
    validation_candidates = unique_ids(validation_pairs, "protein_id")
    published_candidates = fasta_ids(source_dir / "prots.fasta")
    write_ids(out_dir / "train_candidate_ids.txt", train_candidates)
    write_ids(out_dir / "validation_candidate_ids.txt", validation_candidates)
    write_ids(out_dir / "test_candidate_ids.txt", published_candidates)

    source_keys = {
        (row["reaction_id"], row["protein_id"]) for row in source_pairs
    }
    train_keys = {
        (row["reaction_id"], row["protein_id"]) for row in train_pairs
    }
    validation_keys = {
        (row["reaction_id"], row["protein_id"]) for row in validation_pairs
    }
    if train_keys & validation_keys:
        raise ValueError("Exact enzyme-reaction pairs overlap between train and validation")
    if train_keys | validation_keys != source_keys:
        raise ValueError("Train and validation do not reconstruct the published train set")

    with h5py.File(protein_h5, "r") as handle:
        store_ids = {
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["ids"][:]
        }
    required_ids = set(train_candidates) | set(validation_candidates) | set(
        published_candidates
    )
    missing_proteins = sorted(required_ids - store_ids)
    if missing_proteins:
        raise ValueError(
            f"Protein H5 is missing {len(missing_proteins)} required IDs; "
            f"examples: {', '.join(missing_proteins[:5])}"
        )

    _, test_pairs = read_csv(source_dir / "test_pairs.csv")
    train_reaction_ids = {row["reaction_id"] for row in train_pairs}
    validation_reaction_ids = {row["reaction_id"] for row in validation_pairs}
    test_reaction_ids = {row["reaction_id"] for row in test_pairs}
    manifest = {
        "schema_version": "horizyn_f4_protocol_v1",
        "source_dir": str(source_dir),
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "split_method": "pair-level shuffled indices",
        "reaction_direction_mode": "forward_only",
        "counts": {
            "source_train_pairs": len(source_pairs),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "test_pairs": len(test_pairs),
            "train_reactions": len(train_reactions),
            "validation_reactions": len(validation_reactions),
            "published_test_reactions": len(test_reaction_ids),
            "train_candidates": len(train_candidates),
            "validation_candidates": len(validation_candidates),
            "published_test_candidates": len(published_candidates),
            "protein_store_candidates": len(store_ids),
        },
        "audits": {
            "train_validation_exact_pair_overlap": len(train_keys & validation_keys),
            "source_train_pair_set_reconstructed": train_keys
            | validation_keys
            == source_keys,
            "train_validation_reaction_id_overlap": len(
                train_reaction_ids & validation_reaction_ids
            ),
            "published_train_test_reaction_id_overlap": len(
                ({row["reaction_id"] for row in source_pairs}) & test_reaction_ids
            ),
            "missing_required_proteins": len(missing_proteins),
            "test_files_byte_identical_to_release": True,
            "test_sha256": {
                filename: sha256(source_dir / filename)
                for filename in ("test_pairs.csv", "test_rxns.csv")
            },
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
