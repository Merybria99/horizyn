#!/usr/bin/env python3
"""Compare F3 and phase 2 on held-out EnzymeMap rules and the full library."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generalization_clipzyme_screening_protocol import identifier
from scripts.generalization_clipzyme_screening_evaluate import notebook_metrics, sha256


def pairs(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--phase2", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    release = json.loads(a.manifest.read_text())
    train_path = Path(release["associations"]["train"]["path"])
    dev_path = Path(release["associations"]["dev"]["path"])
    if (sha256(train_path) != release["associations"]["train"]["sha256"] or
        sha256(dev_path) != release["associations"]["dev"]["sha256"]):
        raise ValueError("Original EnzymeMap association split changed")
    train = pairs(train_path)
    dev = pairs(dev_path)
    seen_reactions = {row["reaction"] for row in train}
    train_proteins = {row["protein_id"] for row in train}
    truth = defaultdict(set)
    for row in dev:
        if row["reaction"] not in seen_reactions:
            truth[identifier(row["reaction"])].add(row["protein_id"])
    query_ids = (a.phase2 / "query_ids.txt").read_text().splitlines()
    candidate_ids = (a.phase2 / "candidate_ids.txt").read_text().splitlines()
    if len(query_ids) != 2661 or len(candidate_ids) != 261907:
        raise ValueError("Validation full-screen axes changed")
    score = np.load(a.phase2 / "scores.npy", mmap_mode="r")
    if score.shape != (2661, 261907):
        raise ValueError("Phase-2 score matrix shape changed")
    feature_catalog = json.loads((a.features / "catalog.json").read_text())
    if query_ids != feature_catalog["validation_reactions"]:
        raise ValueError("Validation reaction order changed")
    with np.load(a.features / "f3_features.npz") as data:
        reaction_index = {key: i for i, key in enumerate(feature_catalog["reactions"])}
        reactions = data["reactions"][[reaction_index[key] for key in query_ids]]
    shard_ids, shards = [], []
    for rank in range(4):
        stem = f"protein_rank{rank:02d}_of_04"
        shard_ids.extend((a.screen / f"{stem}.ids.txt").read_text().splitlines())
        shards.append(np.load(a.screen / f"{stem}.npy", mmap_mode="r"))
    base_protein = np.concatenate(shards)
    # The F3 catalog maps each original UniProt ID to its sequence-hash ID.
    candidate_map = a.screen.parent / "clipzyme_f3_catalog_v1/screening_candidate_map.csv"
    with candidate_map.open(newline="") as stream:
        mapping = list(csv.DictReader(stream))
    if [row["uniprot_id"] for row in mapping] != candidate_ids:
        raise ValueError("F3 and phase-2 screening candidate IDs disagree")
    shard_index = {key: i for i, key in enumerate(shard_ids)}
    expanded = np.asarray([shard_index[row["protein_id"]] for row in mapping], np.int64)
    candidate = F.normalize(torch.from_numpy(base_protein).to(a.device), dim=1)
    index = {key: i for i, key in enumerate(candidate_ids)}
    kept = np.asarray([i for i, key in enumerate(candidate_ids) if key not in train_proteins], np.int32)
    per_query = []
    for start in range(0, len(query_ids), 32):
        query = F.normalize(torch.from_numpy(reactions[start:start + 32]).to(a.device), dim=1)
        baseline = (query @ candidate.T).float().cpu().numpy()[:, expanded]
        for j, base_values in enumerate(baseline):
            i = start + j
            key = query_ids[i]
            positives = truth.get(key, set()) & index.keys()
            if not positives:
                continue
            labels = np.zeros(len(candidate_ids), bool)
            labels[[index[p] for p in positives]] = True
            row = dict(query_index=i, reaction_id=key, positives=len(positives))
            for name, values in (("f3", base_values), ("phase2", np.asarray(score[i]))):
                row[name] = notebook_metrics(labels[np.argsort(-values)])
                reduced = labels[kept]
                row[name + "_exclude_train"] = (
                    notebook_metrics(reduced[np.argsort(-values[kept])])
                    if reduced.any() else None)
            per_query.append(row)
        if start % 320 == 0:
            print(json.dumps({"evaluated": min(start + 32, len(query_ids))}), flush=True)
    summary = {}
    for name in ("f3", "phase2"):
        summary[name] = {}
        for group, suffix in (("table1", ""), ("table2", "_exclude_train")):
            rows = [r[name + suffix] for r in per_query if r[name + suffix] is not None]
            summary[name][group] = dict(queries=len(rows), candidates=(
                len(candidate_ids) if not suffix else len(kept)),
                **{metric: float(np.mean([row[metric] for row in rows]))
                   for metric in ("bedroc85", "bedroc20", "ef0.05", "ef0.1")})
    a.output.mkdir(parents=True)
    (a.output / "per_query.jsonl").write_text("".join(json.dumps(row) + "\n" for row in per_query))
    receipt = dict(schema="clipzyme_enzymemap_phase2_validation_full_pool_v1",
                   validation_only=True, test_labels_read=False,
                   association_manifest_sha256=sha256(a.manifest),
                   phase2_score_sha256=sha256(a.phase2 / "scores.npy"),
                   per_query_sha256=sha256(a.output / "per_query.jsonl"),
                   summary=summary, source_sha256=sha256(Path(__file__)))
    (a.output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
