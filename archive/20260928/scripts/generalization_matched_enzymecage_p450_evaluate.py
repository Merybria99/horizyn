#!/usr/bin/env python3
"""Evaluate fresh target-trained F3 on the unchanged official P450 IDs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "runs/enzymecage_f3_combined_20260918"
sys.path.insert(0, str(PILOT))
from p450_protocol import (DATA_SHA, KEYS, align_scores, apply_official_prior,
                            digest, score_metrics, validate_panel)  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    export = json.loads((args.export / "export.json").read_text())
    score_path = args.export / "scores.npz"
    if (export["schema"] != "fresh_matched_enzymecage_f3_p450_score_export_v1"
            or export["test_labels_read"] is not False
            or sha256(score_path) != export["score_matrix_sha256"]):
        raise ValueError("Invalid frozen score export")
    catalog_path = ROOT / "runs/generalization_20260919_2251/p450_audit/features/catalog.json"
    labels_path = ROOT / "runs/generalization_20260919_2251/p450_audit/labels.json"
    catalog = json.loads(catalog_path.read_text())
    labels = json.loads(labels_path.read_text())
    if (sha256(catalog_path) != export["catalog_sha256"]
            or labels["catalog_sha256"] != export["catalog_sha256"]):
        raise ValueError("P450 catalog/label lineage changed")
    with np.load(score_path, allow_pickle=False) as data:
        reactions = data["reaction_ids"].tolist()
        proteins = data["protein_ids"].tolist()
        scores = data["scores"].astype(np.float64)
    if (reactions != catalog["reactions"] or proteins != catalog["proteins"]
            or scores.shape != (191, 490) or not np.isfinite(scores).all()):
        raise ValueError("Score axes or values changed")
    pindex = {protein: i for i, protein in enumerate(proteins)}
    ranks = np.empty_like(scores, dtype=np.int32)
    order = np.argsort(-scores, axis=1, kind="stable")
    np.put_along_axis(ranks, order,
                       np.broadcast_to(np.arange(1, 491), ranks.shape), axis=1)
    best = np.array([min(ranks[i, pindex[protein]]
                         for protein in labels["known_positive_ids"][reaction])
                     for i, reaction in enumerate(reactions)])
    raw = {"top4": float(np.mean(best <= 4)),
           "top14": float(np.mean(best <= 14)),
           "top24": float(np.mean(best <= 24)),
           "first_positive_mrr": float(np.mean(1 / best)),
           "positive_queries": int(len(best))}
    panel_path = ROOT / "runs/enzymecage_p450_reproduction_20260918/data/test_P450.csv"
    if digest(panel_path) != DATA_SHA:
        raise ValueError("Official P450 panel changed")
    panel = pd.read_csv(panel_path)
    validate_panel(panel, released=True)
    mapping_path = ROOT / "runs/enzymecage_p450_reproduction_20260918/f3_features/reaction_mapping.csv"
    with mapping_path.open(newline="") as handle:
        mapping = list(csv.DictReader(handle))
    if [row["reaction_id"] for row in mapping] != reactions:
        raise ValueError("Raw reaction mapping changed")
    raw_rxn = [row["raw_reaction"] for row in mapping]
    flat = pd.DataFrame({KEYS[0]: np.repeat(raw_rxn, len(proteins)),
                         KEYS[1]: np.tile(proteins, len(reactions)),
                         "pred": ((scores + 2) / 3).reshape(-1)})
    aligned = align_scores(panel, flat)
    official = score_metrics(aligned, panel, ROOT / ".deps/enzymecage_p450_official")
    for key, cutoff in (("Top 1.0%", 4), ("Top 3.0%", 14), ("Top 5.0%", 24)):
        if abs(official[key] - raw[f"top{cutoff}"]) > 1e-12:
            raise ValueError(f"Official evaluator disagrees at {key}")
    prior_path = ROOT / "runs/enzymecage_p450_reproduction_20260918/data/corr_score_map.pkl"
    prior_receipt = ROOT / "runs/enzymecage_p450_reproduction_20260918/prior.receipt.json"
    if sha256(prior_path) != json.loads(prior_receipt.read_text())["prior_sha256"]:
        raise ValueError("Official homology/reaction prior changed")
    with prior_path.open("rb") as handle:
        prior = pickle.load(handle)
    official_prior = score_metrics(apply_official_prior(aligned, prior), panel,
                                   ROOT / ".deps/enzymecage_p450_official")
    args.output.mkdir(parents=True)
    output = {
        "schema": "fresh_matched_enzymecage_f3_p450_evaluation_v1",
        "scope": "Fresh fixed-F3 EnzymeCAGE-target-trained base, P450 external test; extension not yet retrained",
        "score_export_sha256": sha256(args.export / "export.json"),
        "score_matrix_sha256": sha256(score_path),
        "labels_sha256": sha256(labels_path),
        "official_panel_sha256": sha256(panel_path),
        "raw_metrics": raw,
        "official_raw_metrics": official,
        "official_prior_metrics": official_prior,
        "official_prior_sha256": sha256(prior_path),
        "prior_is_external_information_not_used_by_raw_f3": True,
        "candidate_count": len(proteins),
        "query_count": len(reactions),
        "test_scores_used_for_model_selection": False,
        "source_code_sha256": sha256(Path(__file__)),
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(raw), flush=True)


if __name__ == "__main__":
    main()
