#!/usr/bin/env python3
"""Evaluate the frozen target-trained phase-2 validation score matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from generalization_full_graph import validation_data
from generalization_metrics import evaluate_scores


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads((args.scores / "receipt.json").read_text())
    if (receipt["scope"] != "validation" or receipt["labels_read"] is not False
            or sha256(args.scores / "scores.npz") != receipt["scores_sha256"]):
        raise ValueError("Validation score export is incomplete or changed")
    catalog_path = args.features / "catalog.json"
    pairs_path = args.features / "pairs.npz"
    if sha256(catalog_path) != receipt["inputs"]["catalog"]["sha256"]:
        raise ValueError("Validation catalog/score mismatch")
    catalog = json.loads(catalog_path.read_text())
    with np.load(pairs_path) as source:
        pairs = {name: source[name] for name in source.files}
    _, _, truth = validation_data(catalog, pairs)
    result = {}
    with np.load(args.scores / "scores.npz", allow_pickle=False) as source:
        if (source["reaction_ids"].tolist() != catalog["validation_reactions"]
                or source["protein_ids"].tolist() != catalog["validation_candidates"]):
            raise ValueError("Validation score axes changed")
        for name in ("baseline", "baseline_fp64", "selected"):
            evaluation = evaluate_scores(torch.as_tensor(source[name]), truth)["summary"]
            result[name] = {direction: {stratum: float(evaluation[direction][stratum]["reactzyme_mrr"])
                                        for stratum in ("all", "unseen_reaction")}
                            for direction in ("reaction_to_enzyme", "enzyme_to_reaction")}
    args.output.mkdir(parents=True)
    summary = {"schema": "enzymecage_phase2_target_validation_v1",
               "score_receipt_sha256": sha256(args.scores / "receipt.json"),
               "pairs_sha256": sha256(pairs_path),
               "results": result, "external_labels_used": False,
               "source_sha256": sha256(Path(__file__))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
