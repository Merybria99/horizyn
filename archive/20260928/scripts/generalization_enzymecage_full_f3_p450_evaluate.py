#!/usr/bin/env python3
"""Evaluate held-out P450 scores after label-free F3 BCE export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "runs/enzymecage_p450_reproduction_20260918"
sys.path.insert(0, str(PILOT))
from p450_protocol import DATA_SHA, KEYS, align_scores, digest as panel_digest, score_metrics, validate_panel  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def ranks_for(scores: np.ndarray, reactions: list[str], proteins: list[str], positives):
    index = {key: i for i, key in enumerate(proteins)}
    order = np.argsort(-scores, axis=1, kind="stable")
    inverse = np.empty_like(order, dtype=np.int32)
    np.put_along_axis(inverse, order,
                       np.broadcast_to(np.arange(1, len(proteins) + 1), order.shape), axis=1)
    return np.asarray([min(inverse[i, index[key]] for key in positives[reaction])
                       for i, reaction in enumerate(reactions)], dtype=np.int32)


def summary_metrics(ranks):
    return {"top4": float(np.mean(ranks <= 4)), "top14": float(np.mean(ranks <= 14)),
            "top24": float(np.mean(ranks <= 24)),
            "first_positive_mrr": float(np.mean(1.0 / ranks))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads((args.scores / "receipt.json").read_text())
    path = args.scores / "scores.npz"
    if (receipt["schema"] != "enzymecage_full_f3_p450_scores_v1"
            or receipt["labels_read"] is not False or receipt["scope"] != "external"
            or receipt["score_sha256"] != digest(path)):
        raise ValueError("External score provenance failed")
    catalog_path = ROOT / "runs/generalization_20260919_2251/p450_audit/features/catalog.json"
    labels_path = ROOT / "runs/generalization_20260919_2251/p450_audit/labels.json"
    if digest(catalog_path) != receipt["catalog_sha256"]:
        raise ValueError("P450 catalog changed")
    catalog = json.loads(catalog_path.read_text())
    labels = json.loads(labels_path.read_text())
    if labels["catalog_sha256"] != digest(catalog_path):
        raise ValueError("P450 labels/catalog mismatch")
    with np.load(path, allow_pickle=False) as data:
        reactions = data["reaction_ids"].tolist()
        proteins = data["protein_ids"].tolist()
        scores = data["scores"].astype(np.float64)
    if (reactions != catalog["reactions"] or proteins != catalog["proteins"]
            or scores.shape != (191, 490) or not np.isfinite(scores).all()):
        raise ValueError("P450 score matrix or axes changed")
    panel_path = PILOT / "data/test_P450.csv"
    if panel_digest(panel_path) != DATA_SHA:
        raise ValueError("Official P450 panel changed")
    panel = pd.read_csv(panel_path)
    validate_panel(panel, released=True)
    with (PILOT / "f3_features/reaction_mapping.csv").open(newline="") as handle:
        mapping = list(csv.DictReader(handle))
    if [row["reaction_id"] for row in mapping] != reactions:
        raise ValueError("P450 reaction mapping changed")
    ranks = ranks_for(scores, reactions, proteins, labels["known_positive_ids"])
    raw = summary_metrics(ranks)
    frame = pd.DataFrame({KEYS[0]: np.repeat([row["raw_reaction"] for row in mapping], len(proteins)),
                          KEYS[1]: np.tile(proteins, len(reactions)),
                          "pred": ((scores + 2.0) / 3.0).reshape(-1)})
    official = score_metrics(align_scores(panel, frame), panel,
                             ROOT / ".deps/enzymecage_p450_official")
    for key, name in (("Top 1.0%", "top4"), ("Top 3.0%", "top14"),
                      ("Top 5.0%", "top24")):
        if abs(float(official[key]) - raw[name]) > 1e-12:
            raise ValueError("Official ranking evaluator disagrees")
    baselines = {}
    with (PILOT / "comparison_per_query.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["evaluation"] == "raw_diagnostic" and row["method"] in ("pretrained", "p450_finetuned"):
                baselines.setdefault(row["method"], {})[row["reaction_id"]] = int(row["best_positive_rank"])
    random = np.random.default_rng(42)
    draws = random.integers(0, len(reactions), size=(10000, len(reactions)))
    paired = {}
    for method, values in baselines.items():
        other = np.asarray([values[key] for key in reactions])
        paired[method] = {}
        for name, cutoff in (("top4", 4), ("top14", 14), ("top24", 24)):
            diff = (ranks <= cutoff).astype(float) - (other <= cutoff).astype(float)
            paired[method][name] = {"difference": float(diff.mean()),
                                     "paired_query_bootstrap_95pct":
                                        np.quantile(diff[draws].mean(1), [0.025, 0.975]).tolist()}
    args.output.mkdir(parents=True)
    result = {"schema": "enzymecage_full_f3_p450_evaluation_v1",
              "score_receipt_sha256": digest(args.scores / "receipt.json"),
              "scores_sha256": digest(path), "labels_sha256": digest(labels_path),
              "official_panel_sha256": digest(panel_path),
              "query_count": len(reactions), "candidate_count": len(proteins),
              "raw": raw, "official": official, "per_query_ranks": ranks.tolist(),
              "paired_comparisons": paired,
              "retrospective_panel_previously_opened": True,
              "test_used_for_model_selection": False,
              "method": "Full F3 dual encoder trained on all labeled EnzymeCAGE pairs with BCE",
              "training_initialization": receipt["training_initialization"],
              "source_sha256": digest(Path(__file__))}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"raw": raw, "official": official}))


if __name__ == "__main__":
    main()
