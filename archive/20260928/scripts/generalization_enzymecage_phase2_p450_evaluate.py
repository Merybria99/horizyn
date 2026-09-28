#!/usr/bin/env python3
"""Evaluate frozen target-trained phase 2 on the official EnzymeCAGE P450 pool."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "runs/enzymecage_p450_reproduction_20260918"
sys.path.insert(0, str(PILOT))
from p450_protocol import (DATA_SHA, KEYS, align_scores, digest as panel_digest,
                            score_metrics, validate_panel)  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def positive_ranks(scores: np.ndarray, reactions: list[str], proteins: list[str],
                   positives: dict[str, list[str]]) -> np.ndarray:
    index = {key: i for i, key in enumerate(proteins)}
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    np.put_along_axis(ranks, order,
                       np.broadcast_to(np.arange(1, len(proteins) + 1), order.shape), axis=1)
    return np.asarray([min(ranks[i, index[protein]] for protein in positives[reaction])
                       for i, reaction in enumerate(reactions)], dtype=np.int32)


def metrics(ranks: np.ndarray) -> dict[str, float]:
    return {"top4": float(np.mean(ranks <= 4)),
            "top14": float(np.mean(ranks <= 14)),
            "top24": float(np.mean(ranks <= 24)),
            "first_positive_mrr": float(np.mean(1 / ranks))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads((args.scores / "receipt.json").read_text())
    score_path = args.scores / "scores.npz"
    if (receipt["schema"] != "enzymecage_phase2_target_score_v1"
            or receipt["scope"] != "external" or receipt["seed"] != 42
            or receipt["labels_read"] is not False
            or sha256(score_path) != receipt["scores_sha256"]):
        raise ValueError("Target phase-2 score provenance is invalid")
    freeze = json.loads(Path(receipt["freeze"]["path"]).read_text())
    if (sha256(Path(receipt["freeze"]["path"])) != receipt["freeze"]["sha256"]
            or freeze["comparison_is_training_controlled"] is not False):
        raise ValueError("Frozen target recipe changed")
    catalog_path = ROOT / "runs/generalization_20260919_2251/p450_audit/features/catalog.json"
    label_path = ROOT / "runs/generalization_20260919_2251/p450_audit/labels.json"
    if sha256(catalog_path) != receipt["inputs"]["catalog"]["sha256"]:
        raise ValueError("P450 catalog changed")
    catalog = json.loads(catalog_path.read_text())
    labels = json.loads(label_path.read_text())
    if labels["catalog_sha256"] != sha256(catalog_path):
        raise ValueError("P450 labels/catalog mismatch")
    with np.load(score_path, allow_pickle=False) as source:
        reactions, proteins = source["reaction_ids"].tolist(), source["protein_ids"].tolist()
        matrices = {key: source[key].astype(np.float64)
                    for key in ("selected", "baseline", "baseline_fp64")}
    if (reactions != catalog["reactions"] or proteins != catalog["proteins"]
            or any(values.shape != (191, 490) or not np.isfinite(values).all()
                   for values in matrices.values())):
        raise ValueError("P450 score matrix or axes changed")
    panel_path = PILOT / "data/test_P450.csv"
    if panel_digest(panel_path) != DATA_SHA:
        raise ValueError("Official P450 panel changed")
    panel = pd.read_csv(panel_path)
    validate_panel(panel, released=True)
    with (PILOT / "f3_features/reaction_mapping.csv").open(newline="") as handle:
        mapping = list(csv.DictReader(handle))
    if [row["reaction_id"] for row in mapping] != reactions:
        raise ValueError("P450 physical reaction mapping changed")
    raw_reactions = [row["raw_reaction"] for row in mapping]
    comparisons = {}
    baselines = {}
    with (PILOT / "comparison_per_query.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["evaluation"] == "raw_diagnostic" and row["method"] in ("pretrained", "p450_finetuned"):
                baselines.setdefault(row["method"], {})[row["reaction_id"]] = int(row["best_positive_rank"])
    comparator_ranks = {name: np.asarray([values[q] for q in reactions], dtype=np.int32)
                        for name, values in baselines.items()}
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(reactions), (10000, len(reactions)))
    result = {}
    for name, scores in matrices.items():
        ranks = positive_ranks(scores, reactions, proteins, labels["known_positive_ids"])
        raw = metrics(ranks)
        table = pd.DataFrame({KEYS[0]: np.repeat(raw_reactions, len(proteins)),
                              KEYS[1]: np.tile(proteins, len(reactions)),
                              "pred": ((scores + 2) / 3).reshape(-1)})
        official = score_metrics(align_scores(panel, table), panel,
                                 ROOT / ".deps/enzymecage_p450_official")
        for key, cutoff in (("Top 1.0%", "top4"), ("Top 3.0%", "top14"),
                            ("Top 5.0%", "top24")):
            if abs(float(official[key]) - raw[cutoff]) > 1e-12:
                raise ValueError(f"Official evaluator disagrees for {name}/{key}")
        result[name] = {"raw": raw, "official": official, "per_query_ranks": ranks.tolist()}
        for comparator, other in comparator_ranks.items():
            comparisons.setdefault(name, {})[comparator] = {}
            for key, cutoff in (("top4", 4), ("top14", 14), ("top24", 24)):
                delta = (ranks <= cutoff).astype(float) - (other <= cutoff).astype(float)
                comparisons[name][comparator][key] = {
                    "difference": float(delta.mean()),
                    "paired_query_bootstrap_95pct": np.quantile(delta[draws].mean(1),
                                                                [0.025, 0.975]).tolist()}
    args.output.mkdir(parents=True)
    summary = {"schema": "enzymecage_phase2_target_p450_evaluation_v1",
               "score_receipt_sha256": sha256(args.scores / "receipt.json"),
               "score_sha256": sha256(score_path),
               "labels_sha256": sha256(label_path), "official_panel_sha256": sha256(panel_path),
               "query_count": len(reactions), "candidate_count": len(proteins),
               "results": result, "paired_comparisons": comparisons,
               "comparison_is_training_controlled": False,
               "reason": "F3 and its phase-2 extension used positive pairs only; EnzymeCAGE official recipe also uses explicit negative labels",
               "retrospective_panel_previously_opened": True,
               "test_used_for_model_selection": False,
               "source_sha256": sha256(Path(__file__))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({name: value["raw"] for name, value in result.items()}))


if __name__ == "__main__":
    main()
