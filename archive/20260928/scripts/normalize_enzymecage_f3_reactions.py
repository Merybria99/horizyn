#!/usr/bin/env python3
"""Normalize EnzymeCAGE reactions without neutralizing charge-sensitive species."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from rdkit import Chem
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizyn.chemistry.standardizer import Standardizer


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def normalize(source: Path, target: Path, audit: Path, standardizer: Standardizer) -> dict:
    counts = {"reactions": 0, "changed": 0, "invalid_source": 0, "invalid_normalized": 0}
    with source.open(newline="", encoding="utf-8") as input_file, \
            target.open("w", newline="", encoding="utf-8") as output_file, \
            audit.open("w", newline="", encoding="utf-8") as audit_file:
        reader = csv.DictReader(input_file)
        if not {"reaction_id", "reaction_smiles"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid reaction CSV: {source}")
        writer = csv.writer(output_file)
        audit_writer = csv.writer(audit_file)
        writer.writerow(("reaction_id", "reaction_smiles"))
        audit_writer.writerow(("reaction_id", "source_smiles", "normalized_smiles"))
        for row in reader:
            original = row["reaction_smiles"]
            components = [part for side in original.split(">>") for part in side.split(".")]
            if len(original.split(">>")) != 2 or any(Chem.MolFromSmiles(part) is None for part in components):
                counts["invalid_source"] += 1
                raise ValueError(f"Invalid source reaction {row['reaction_id']}: {original}")
            try:
                value = standardizer.standardize_reaction(original)
            except Exception as exc:
                raise ValueError(f"Cannot safely normalize reaction {row['reaction_id']}: {original}") from exc
            normalized_components = [part for side in value.split(">>") for part in side.split(".")]
            if len(value.split(">>")) != 2 or any(Chem.MolFromSmiles(part) is None for part in normalized_components):
                counts["invalid_normalized"] += 1
                raise ValueError(f"Invalid normalized reaction {row['reaction_id']}: {value}")
            writer.writerow((row["reaction_id"], value))
            if value != original:
                counts["changed"] += 1
                audit_writer.writerow((row["reaction_id"], original, value))
            counts["reactions"] += 1
    return {**counts, "source_sha256": digest(source), "normalized_sha256": digest(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.data_dir / "normalized"
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing normalization: {output}")
    output.mkdir(parents=True, exist_ok=True)
    standardizer = Standardizer(standardize_uncharge=False)
    report = {"policy": "CIRCE F3 standardization with uncharging disabled; preserve valid formal charges"}
    for split in ("train", "validation"):
        report[split] = normalize(args.data_dir / f"{split}_rxns.csv",
                                  output / f"{split}_rxns.csv",
                                  output / f"{split}_changes.csv", standardizer)
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
