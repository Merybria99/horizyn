"""Audit exact esterase exposure in the checksum-verified Horizyn-1 training release."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "runs/generalization_20260919_2251/esterase_audit"
TRAIN = ROOT / "data/sota"


def rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, expected_md5: str):
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "md5").hexdigest()
    if actual != expected_md5:
        raise ValueError(f"Official training asset checksum mismatch: {path}")
    return {"sha256": sha256(path), "published_md5": expected_md5}


def fasta(path: Path):
    sequences = {}
    key, parts = None, []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if key is not None:
                    sequences[key] = "".join(parts)
                key, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if key is not None:
        sequences[key] = "".join(parts)
    return sequences


def main():
    assets = {
        "train_pairs.csv": "d77c894783a2d3552b90b26eb253633b",
        "train_rxns.csv": "7b0335ac694e4afee87e7a0a970f56e4",
        "prots.fasta": "b0946eddf6d3b89047ac6b1b1ca374ae",
    }
    receipts = {name: verified(TRAIN / name, md5) for name, md5 in assets.items()}
    panel_proteins = rows(PANEL / "inputs/proteins.csv")
    panel_reactions = rows(PANEL / "inputs/reactions.csv")
    train_ids = {row["protein_id"] for row in rows(TRAIN / "train_pairs.csv")}
    train_fasta = fasta(TRAIN / "prots.fasta")
    if not train_ids <= train_fasta.keys():
        raise ValueError("Training protein ID missing from released FASTA")
    train_sequences = {train_fasta[pid] for pid in train_ids}
    exact_proteins = [row["protein_id"] for row in panel_proteins if row["sequence"] in train_sequences]
    train_reactions = {row["reaction_smiles"] for row in rows(TRAIN / "train_rxns.csv")}
    exact_reactions = [row["reaction_id"] for row in panel_reactions if row["reaction_smiles"] in train_reactions]
    if (len(panel_proteins), len(panel_reactions)) != (145, 86):
        raise ValueError("Panel changed")
    report = {
        "schema": "esterase_horizyn1_exact_training_exposure_v1",
        "released_source": "https://zenodo.org/records/17957034",
        "assets": receipts,
        "panel_protein_count": len(panel_proteins),
        "panel_reaction_count": len(panel_reactions),
        "training_protein_id_count": len(train_ids),
        "training_unique_sequence_count": len(train_sequences),
        "panel_exact_sequence_matches_to_training": exact_proteins,
        "panel_exact_physical_reaction_string_matches_to_training": exact_reactions,
        "limits": "Exact sequence and literal reaction-string checks only; homologs, equivalent chemistry and backbone pretraining exposure are not excluded.",
        "implementation_sha256": sha256(Path(__file__)),
    }
    target = PANEL / "public_horizyn1_dev/training_exposure.json"
    if target.exists():
        raise ValueError("Preserve existing exposure report")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"exact_proteins": len(exact_proteins), "exact_reactions": len(exact_reactions), "output": str(target)}))


if __name__ == "__main__":
    main()
