#!/usr/bin/env python3
"""Evaluate the validation-frozen local hubness correction on ReactZyme tests."""
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
from scripts.generalization_reactzyme_local_hubness import density


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evaluate(split: str, run_root: Path, device: str, k: int, strength: float) -> dict:
    train_name = {"reaction_smi": "features", "enzyme_smi": "features_enzyme_smi",
                  "time": "features_time"}[split]
    train_root = run_root / train_name
    test_root = run_root / f"features_test_{split}"
    train_catalog = json.loads((train_root / "catalog.json").read_text())
    test_catalog = json.loads((test_root / "catalog.json").read_text())
    with np.load(train_root / "f3_features.npz") as data:
        train_queries = F.normalize(torch.as_tensor(data["train_reactions"], device=device), dim=1)
        train_all_proteins = torch.as_tensor(data["proteins"], device=device)
        train_protein_index = {key: i for i, key in enumerate(train_catalog["proteins"])}
        train_proteins = F.normalize(train_all_proteins[torch.as_tensor(
            [train_protein_index[key] for key in train_catalog["train_proteins"]], device=device)], dim=1)
    with np.load(test_root / "f3_features.npz") as data:
        test_queries = F.normalize(torch.as_tensor(data["reactions"], device=device), dim=1)
        test_proteins = F.normalize(torch.as_tensor(data["proteins"], device=device), dim=1)
    if (test_catalog["test_reactions"] != test_catalog["reactions"]
            or test_catalog["test_candidates"] != test_catalog["proteins"]):
        raise ValueError(f"Unexpected fixed ReactZyme {split} candidate axes")
    with np.load(test_root / "pairs.npz") as data:
        edge = data["test"].copy()
    truth = {"reaction_index": edge[:, 0], "enzyme_index": edge[:, 1]}
    enzyme_penalty = density(test_proteins, train_queries, (k,), 512)[k]
    reaction_penalty = density(test_queries, train_proteins, (k,), 128)[k]
    base = test_queries @ test_proteins.T
    original = evaluate_scores(base, truth, batch_size=512)
    r2e = evaluate_scores(base - strength * enzyme_penalty[None, :], truth,
                          batch_size=512, directions=["reaction_to_enzyme"])
    e2r = evaluate_scores(base - strength * reaction_penalty[:, None], truth,
                          batch_size=512, directions=["enzyme_to_reaction"])
    changed = {"summary": {**r2e["summary"], **e2r["summary"]},
               "per_query": {**r2e["per_query"], **e2r["per_query"]}}
    result = {"baseline": {direction: original["summary"][direction]["all"]
                           for direction in ("reaction_to_enzyme", "enzyme_to_reaction")},
              "corrected": {direction: changed["summary"][direction]["all"]
                            for direction in ("reaction_to_enzyme", "enzyme_to_reaction")},
              "test_reactions": len(test_queries), "test_candidate_proteins": len(test_proteins),
              "test_positive_edges": len(edge),
              "train_reference_reactions": len(train_queries),
              "train_reference_proteins": len(train_proteins),
              "inputs": {str(path.relative_to(run_root)): sha256(path)
                         for path in (train_root / "catalog.json", train_root / "f3_features.npz",
                                      test_root / "catalog.json", test_root / "f3_features.npz",
                                      test_root / "pairs.npz")}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    selection = json.loads(args.selection.read_text())
    if (selection["schema"] != "reactzyme_f3_local_hubness_validation_v1"
            or selection["test_inputs_read"] or selection["test_labels_read"]):
        raise ValueError("The local hubness rule was not frozen on validation only")
    chosen = selection["selected"]
    if chosen["k"] != 10 or chosen["strength"] != 0.5:
        raise ValueError("Unexpected validated local hubness setting")
    args.output.mkdir(parents=True)
    splits = {}
    for split in ("reaction_smi", "enzyme_smi", "time"):
        splits[split] = evaluate(split, args.run_root, args.device,
                                 chosen["k"], chosen["strength"])
        print(json.dumps({"split": split,
            "baseline_r2e": splits[split]["baseline"]["reaction_to_enzyme"]["reactzyme_mrr"],
            "corrected_r2e": splits[split]["corrected"]["reaction_to_enzyme"]["reactzyme_mrr"],
            "baseline_e2r": splits[split]["baseline"]["enzyme_to_reaction"]["reactzyme_mrr"],
            "corrected_e2r": splits[split]["corrected"]["enzyme_to_reaction"]["reactzyme_mrr"]}), flush=True)
    mean = {}
    for variant in ("baseline", "corrected"):
        mean[variant] = float(np.mean([
            splits[split][variant][direction]["reactzyme_mrr"]
            for split in splits for direction in ("reaction_to_enzyme", "enzyme_to_reaction")]))
    receipt = {"schema": "reactzyme_f3_local_hubness_official_test_v1",
               "selection_sha256": sha256(args.selection),
               "selection_validation_only": True,
               "retrospective_prior_test_exposure": True,
               "k": chosen["k"], "strength": chosen["strength"],
               "splits": splits, "six_cell_mean": mean,
               "case1_inputs_read": False, "case1_labels_read": False,
               "source_sha256": sha256(Path(__file__))}
    (args.output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"six_cell_mean": mean}), flush=True)


if __name__ == "__main__":
    main()
