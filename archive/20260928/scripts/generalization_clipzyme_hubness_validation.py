#!/usr/bin/env python3
"""Choose a training-reference hubness correction using EnzymeMap validation only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from horizyn.benchmarks.retrieval import (
    BenchmarkTask, build_reaction_inputs, encode_reactions, encode_residue_targets)
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset

from generalization_clipzyme_f3_screen import model_from_checkpoint


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ids(path: Path, column: str) -> list[str]:
    with path.open(newline="") as handle:
        return [row[column] for row in csv.DictReader(handle)]


def reaction_vectors(model, config, catalog: Path, split: str, keys: list[str],
                     device: str) -> torch.Tensor:
    features = catalog / "features"
    config.data.reaction_chemistry_vectors_path = str(
        catalog / f"chemistry/{split}_reaction_set_features.npz")
    task = BenchmarkTask(name=f"clipzyme_{split}_hubness", task_type="retrieval",
        dataset="EnzymeMap", task_label="rule_split", split=split,
        pairs=catalog / f"{split}_pairs.csv", reactions=catalog / f"{split}_rxns.csv",
        reaction_model_embeds_h5=features / "reactiont5v2.h5",
        reaction_unimol2_embeds_h5=features / "unimol2.h5",
        reaction_chiro_embeds_h5=features / "chiro.h5")
    dataset = build_reaction_inputs(task, config)
    if not set(keys).issubset(dataset.keys):
        raise ValueError(f"Missing {split} reaction features")
    return F.normalize(encode_reactions(model, dataset, keys, device, 512).float(), dim=-1)


def mrr(scores: np.ndarray, query_ids: list[str], candidate_ids: list[str],
        positives: dict[str, set[str]]) -> float:
    positions = {key: i for i, key in enumerate(candidate_ids)}
    rank = np.argsort(np.argsort(-scores, axis=1, kind="stable"), axis=1,
                      kind="stable") + 1
    values = []
    for i, key in enumerate(query_ids):
        positive_indices = [positions[x] for x in positives[key] if x in positions]
        if positive_indices:
            values.append(1.0 / int(rank[i, positive_indices].min()))
    if len(values) != len(query_ids):
        raise ValueError("Validation query with no positive candidate")
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--protein-batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    model, config = model_from_checkpoint(args.config, args.checkpoint, args.device)
    train_reactions = ids(args.catalog / "train_rxns.csv", "reaction_id")
    val_reactions = ids(args.catalog / "validation_rxns.csv", "reaction_id")
    val_proteins = (args.catalog / "validation_candidate_ids.txt").read_text().splitlines()
    with (args.catalog / "train_pairs.csv").open(newline="") as handle:
        train_proteins = sorted({row["protein_id"] for row in csv.DictReader(handle)})
    train_rxn = reaction_vectors(model, config, args.catalog, "train",
                                 train_reactions, args.device)
    val_rxn = reaction_vectors(model, config, args.catalog, "validation",
                               val_reactions, args.device)
    residue = ResidueEmbedDataset(str(args.catalog / "features/proteins_prott5_residue.h5"),
                                  max_tokens=config.data.max_protein_tokens,
                                  truncation=config.data.protein_truncation)
    try:
        if not (set(train_proteins) | set(val_proteins)).issubset(residue.keys):
            raise ValueError("Missing validation or training protein features")
        train_protein = encode_residue_targets(model, residue, train_proteins,
            args.device, args.protein_batch_size, True)
        val_protein = encode_residue_targets(model, residue, val_proteins,
            args.device, args.protein_batch_size, True)
    finally:
        residue.close()
    train_protein = F.normalize(train_protein.float(), dim=-1)
    val_protein = F.normalize(val_protein.float(), dim=-1)
    with torch.inference_mode():
        base = (val_rxn @ val_protein.T).cpu().numpy()
        enzyme_neighborhood = val_protein @ train_rxn.T
        reaction_neighborhood = val_rxn @ train_protein.T
        enzyme_penalties = {k: enzyme_neighborhood.topk(k, dim=1).values.mean(1).cpu().numpy()
                            for k in (10, 50, 200)}
        reaction_penalties = {k: reaction_neighborhood.topk(k, dim=1).values.mean(1).cpu().numpy()
                             for k in (10, 50, 200)}
    r2e = {key: set() for key in val_reactions}
    e2r = {key: set() for key in val_proteins}
    with (args.catalog / "validation_pairs.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            r2e[row["reaction_id"]].add(row["protein_id"])
            e2r[row["protein_id"]].add(row["reaction_id"])
    grid = []
    for k in (10, 50, 200):
        for strength in (0.0, 0.25, 0.5, 1.0):
            reaction_to_enzyme = mrr(base - strength * enzyme_penalties[k][None, :],
                                     val_reactions, val_proteins, r2e)
            enzyme_to_reaction = mrr((base - strength * reaction_penalties[k][:, None]).T,
                                     val_proteins, val_reactions, e2r)
            grid.append({"k": k, "strength": strength,
                         "reaction_to_enzyme_mrr": reaction_to_enzyme,
                         "enzyme_to_reaction_mrr": enzyme_to_reaction,
                         "balanced_mrr": (reaction_to_enzyme + enzyme_to_reaction) / 2})
    baseline = next(row for row in grid if row["k"] == 10 and row["strength"] == 0)
    eligible = [row for row in grid
                if row["reaction_to_enzyme_mrr"] >= baseline["reaction_to_enzyme_mrr"]
                and row["enzyme_to_reaction_mrr"] >= baseline["enzyme_to_reaction_mrr"]]
    selected = max(eligible, key=lambda row: row["balanced_mrr"])
    args.output.mkdir(parents=True)
    receipt = {"schema": "clipzyme_f3_hubness_validation_v1",
               "checkpoint_sha256": sha256(args.checkpoint),
               "config_sha256": sha256(args.config),
               "train_pairs_sha256": sha256(args.catalog / "train_pairs.csv"),
               "validation_pairs_sha256": sha256(args.catalog / "validation_pairs.csv"),
               "test_inputs_read": False, "test_labels_read": False,
               "reference_train_reactions": len(train_reactions),
               "reference_train_proteins": len(train_proteins),
               "validation_reactions": len(val_reactions),
               "validation_proteins": len(val_proteins),
               "grid": grid, "selected": selected,
               "source_sha256": sha256(Path(__file__))}
    (args.output / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"baseline": baseline, "selected": selected}), flush=True)


if __name__ == "__main__":
    main()
