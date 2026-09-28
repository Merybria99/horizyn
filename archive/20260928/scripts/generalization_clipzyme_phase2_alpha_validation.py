#!/usr/bin/env python3
"""Choose the phase-2 mixture on EnzymeMap validation full-library BEDROC85."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generalization_clipzyme_screening_protocol import identifier
from scripts.generalization_clipzyme_screening_evaluate import notebook_metrics, sha256


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--alpha0", type=Path, required=True)
    p.add_argument("--alpha1", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads(a.manifest.read_text())
    train_path = Path(manifest["associations"]["train"]["path"])
    dev_path = Path(manifest["associations"]["dev"]["path"])
    for split, path in (("train", train_path), ("dev", dev_path)):
        if sha256(path) != manifest["associations"][split]["sha256"]:
            raise ValueError(f"Original {split} associations changed")
    train, dev = read_csv(train_path), read_csv(dev_path)
    train_reactions = {row["reaction"] for row in train}
    train_proteins = {row["protein_id"] for row in train}
    positives = defaultdict(set)
    for row in dev:
        if row["reaction"] not in train_reactions:
            positives[identifier(row["reaction"])].add(row["protein_id"])
    query_ids = (a.alpha0 / "query_ids.txt").read_text().splitlines()
    candidate_ids = (a.alpha0 / "candidate_ids.txt").read_text().splitlines()
    if (query_ids != (a.alpha1 / "query_ids.txt").read_text().splitlines() or
        candidate_ids != (a.alpha1 / "candidate_ids.txt").read_text().splitlines() or
        len(query_ids) != 2661 or len(candidate_ids) != 261907):
        raise ValueError("Validation score axes disagree")
    for directory, alpha in ((a.alpha0, 0.), (a.alpha1, 1.)):
        receipt = json.loads((directory / "score_receipt.json").read_text())
        if (receipt["scope"] != "validation" or receipt["alpha"] != alpha or
            receipt["test_labels_read"] is not False or
            sha256(directory / "scores.npy") != receipt["scores_sha256"]):
            raise ValueError("Validation score provenance mismatch")
    score0 = np.load(a.alpha0 / "scores.npy", mmap_mode="r")
    score1 = np.load(a.alpha1 / "scores.npy", mmap_mode="r")
    if score0.shape != score1.shape or score0.shape != (2661, 261907):
        raise ValueError("Invalid full-library score matrices")
    alphas = (0., .1, .25, .5, .75, 1.)
    index = {key: i for i, key in enumerate(candidate_ids)}
    kept = np.asarray([i for i, key in enumerate(candidate_ids) if key not in train_proteins], np.int32)
    rows = []
    for i, key in enumerate(query_ids):
        target = positives.get(key, set()) & index.keys()
        if not target:
            continue
        labels = np.zeros(len(candidate_ids), bool)
        labels[[index[x] for x in target]] = True
        values0, values1 = np.asarray(score0[i]), np.asarray(score1[i])
        record = dict(query_index=i, reaction_id=key, positives=len(target), metrics={})
        for alpha in alphas:
            values = (1 - alpha) * values0 + alpha * values1
            metrics = notebook_metrics(labels[np.argsort(-values)])
            reduced = labels[kept]
            other = notebook_metrics(reduced[np.argsort(-values[kept])]) if reduced.any() else None
            record["metrics"][str(alpha)] = dict(table1=metrics, table2=other)
        rows.append(record)
        if len(rows) % 250 == 0:
            print(json.dumps({"evaluated": len(rows)}), flush=True)
    summary = {}
    for alpha in alphas:
        key = str(alpha)
        summary[key] = {}
        for table in ("table1", "table2"):
            chosen = [row["metrics"][key][table] for row in rows
                      if row["metrics"][key][table] is not None]
            summary[key][table] = dict(queries=len(chosen),
                **{metric: float(np.mean([row[metric] for row in chosen]))
                   for metric in ("bedroc85", "bedroc20", "ef0.05", "ef0.1")})
    selected = max(alphas, key=lambda x: (summary[str(x)]["table1"]["bedroc85"],
                                           summary[str(x)]["table2"]["bedroc85"]))
    a.output.mkdir(parents=True)
    (a.output / "per_query.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = dict(schema="clipzyme_phase2_enzymemap_alpha_validation_v1",
        selection="Maximum validation Table-1 BEDROC85; Table-2 BEDROC85 tie-break",
        alpha_grid=list(alphas), selected_alpha=selected, summary=summary,
        test_labels_read=False, manifest_sha256=sha256(a.manifest),
        alpha0_receipt_sha256=sha256(a.alpha0 / "score_receipt.json"),
        alpha1_receipt_sha256=sha256(a.alpha1 / "score_receipt.json"),
        per_query_sha256=sha256(a.output / "per_query.jsonl"),
        source_sha256=sha256(Path(__file__)))
    (a.output / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"selected_alpha": selected, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
