#!/usr/bin/env python3
"""Validation-only local hubness correction for frozen ReactZyme F3 vectors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generalization_metrics import evaluate_scores


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def density(candidates: torch.Tensor, references: torch.Tensor,
            k_values: tuple[int, ...], chunk: int) -> dict[int, torch.Tensor]:
    parts = {k: [] for k in k_values}
    with torch.inference_mode():
        for start in range(0, len(candidates), chunk):
            scores = candidates[start:start + chunk] @ references.T
            nearest = scores.topk(max(k_values), dim=1).values
            for k in k_values:
                parts[k].append(nearest[:, :k].mean(dim=1))
    return {k: torch.cat(values) for k, values in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed-k", type=int)
    parser.add_argument("--fixed-strength", type=float)
    args = parser.parse_args()
    if (args.fixed_k is None) != (args.fixed_strength is None):
        parser.error("--fixed-k and --fixed-strength must be specified together")
    if args.fixed_k is not None and (args.fixed_k < 1 or args.fixed_strength <= 0):
        parser.error("Fixed k and strength must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    catalog_path = args.features / "catalog.json"
    pairs_path = args.features / "pairs.npz"
    vectors_path = args.features / "f3_features.npz"
    catalog = json.loads(catalog_path.read_text())
    with np.load(pairs_path) as data:
        val_edges = data["validation"].copy()
    reaction_index = {key: i for i, key in enumerate(catalog["reactions"])}
    protein_index = {key: i for i, key in enumerate(catalog["proteins"])}
    qrows = np.array([reaction_index[key] for key in catalog["validation_reactions"]])
    erows = np.array([protein_index[key] for key in catalog["validation_candidates"]])
    qmap = np.full(len(reaction_index), -1)
    emap = np.full(len(protein_index), -1)
    qmap[qrows], emap[erows] = np.arange(len(qrows)), np.arange(len(erows))
    if (qmap[val_edges[:, 0]] < 0).any() or (emap[val_edges[:, 1]] < 0).any():
        raise ValueError("Validation positive outside fixed candidate axes")
    train_reaction_set = set(catalog["train_reactions"])
    train_protein_set = set(catalog["train_proteins"])
    truth = {"reaction_index": qmap[val_edges[:, 0]],
             "enzyme_index": emap[val_edges[:, 1]],
             "reaction_seen": [x in train_reaction_set
                               for x in catalog["validation_reactions"]],
             "enzyme_seen": [x in train_protein_set
                             for x in catalog["validation_candidates"]]}
    with np.load(vectors_path) as data:
        all_proteins = F.normalize(torch.as_tensor(data["proteins"], device=args.device), dim=1)
        val_queries = F.normalize(torch.as_tensor(data["reactions"][qrows], device=args.device), dim=1)
        train_queries = F.normalize(torch.as_tensor(data["train_reactions"], device=args.device), dim=1)
    val_enzymes = all_proteins[torch.as_tensor(erows, device=args.device)]
    train_enzymes = all_proteins[torch.as_tensor(
        [protein_index[key] for key in catalog["train_proteins"]], device=args.device)]
    k_values = (args.fixed_k,) if args.fixed_k is not None else (10, 50, 200)
    strengths = (0.0, args.fixed_strength) if args.fixed_strength is not None else (0.0, 0.25, 0.5, 1.0)
    enzyme_density = density(val_enzymes, train_queries, k_values, 512)
    reaction_density = density(val_queries, train_enzymes, k_values, 128)
    base = val_queries @ val_enzymes.T
    baseline = evaluate_scores(base, truth, batch_size=512)["summary"]
    values = []
    for k in k_values:
        for strength in strengths:
            if strength == 0:
                summary = baseline
            else:
                r2e = evaluate_scores(base - strength * enzyme_density[k][None, :],
                                      truth, batch_size=512,
                                      directions=["reaction_to_enzyme"])["summary"]
                e2r = evaluate_scores(base - strength * reaction_density[k][:, None],
                                      truth, batch_size=512,
                                      directions=["enzyme_to_reaction"])["summary"]
                summary = {**r2e, **e2r}
            def metric(direction: str, stratum: str) -> float:
                return summary[direction][stratum]["reactzyme_mrr"]
            row = {"k": k, "strength": strength,
                   "r2e_all": metric("reaction_to_enzyme", "all"),
                   "e2r_all": metric("enzyme_to_reaction", "all"),
                   "r2e_unseen_reaction": metric("reaction_to_enzyme", "unseen_reaction"),
                   "e2r_unseen_reaction": metric("enzyme_to_reaction", "unseen_reaction")}
            row["balanced_all"] = (row["r2e_all"] + row["e2r_all"]) / 2
            values.append(row)
            print(json.dumps(row), flush=True)
    baseline_row = values[0]
    guard_keys = ("r2e_all", "e2r_all", "r2e_unseen_reaction", "e2r_unseen_reaction")
    eligible = [row for row in values
                if all(row[key] >= baseline_row[key] - 0.005 for key in guard_keys)]
    selected = (values[-1] if args.fixed_k is not None
                else max(eligible, key=lambda row: row["balanced_all"]))
    args.output.mkdir(parents=True)
    receipt = {"schema": "reactzyme_f3_local_hubness_validation_v1",
               "inputs": {p.name: sha256(p) for p in (catalog_path, pairs_path, vectors_path)},
               "baseline": baseline_row, "selected": selected, "grid": values,
               "selection_mode": "fixed_reaction_smi_transfer" if args.fixed_k is not None else "validation_grid",
               "test_inputs_read": False, "test_labels_read": False,
               "case1_inputs_read": False, "case1_labels_read": False,
               "source_sha256": sha256(Path(__file__))}
    (args.output / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"baseline": baseline_row, "selected": selected}), flush=True)


if __name__ == "__main__":
    main()
