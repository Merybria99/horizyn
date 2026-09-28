#!/usr/bin/env python3
"""Select EnzymeMap F3 snapshots by full-library validation BEDROC85."""
from __future__ import annotations

import argparse
import hashlib
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizyn.screen_scoring import ScreenScorer

from generalization_clipzyme_screening_evaluate import notebook_metrics, sha256, evaluate_query
from generalization_clipzyme_screening_protocol import identifier


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--metric-workers", type=int, default=8)
    p.add_argument("--refiner", type=Path, help="Fixed train-only residual snapshot; select using full-library validation")
    p.add_argument("--fusion-calibration", type=Path, help="Explicit receipt for a verified inference-only fusion scalar change")
    p.add_argument("--calibrated-base-checkpoint", type=Path)
    a = p.parse_args()
    if a.metric_workers < 1:
        p.error('metric-workers must be positive')
    if a.output.exists():
        raise FileExistsError(a.output)
    release = json.loads(a.manifest.read_text())
    train_path = Path(release["associations"]["train"]["path"])
    dev_path = Path(release["associations"]["dev"]["path"])
    if (sha256(train_path) != release["associations"]["train"]["sha256"] or
            sha256(dev_path) != release["associations"]["dev"]["sha256"]):
        raise ValueError("EnzymeMap train/dev associations changed")
    train = read_csv(train_path)
    dev = read_csv(dev_path)
    train_reactions = {row["reaction"] for row in train}
    train_proteins = {row["protein_id"] for row in train}
    truth: dict[str, set[str]] = defaultdict(set)
    for row in dev:
        if row["reaction"] not in train_reactions:
            truth[identifier(row["reaction"])].add(row["protein_id"])

    reaction_receipt = json.loads((a.embeddings / "reaction_receipt.json").read_text())
    if reaction_receipt.get("scope") != "validation" or reaction_receipt["labels_read"]:
        raise ValueError("Expected label-free validation reaction export")
    query_ids = (a.embeddings / "query_ids.txt").read_text().splitlines()
    expected_queries = [row["reaction_id"] for row in read_csv(a.catalog / "validation_rxns.csv")]
    if query_ids != expected_queries or len(query_ids) != 2661:
        raise ValueError("Validation reaction axis changed")
    reactions = np.load(a.embeddings / "reaction_embeddings.npy")
    if reactions.shape != (2661, 512) or sha256(a.embeddings / "reaction_embeddings.npy") != reaction_receipt["embedding_sha256"]:
        raise ValueError("Reaction export does not match receipt")

    shard_ids, shards = [], []
    for rank in range(4):
        stem = f"protein_rank{rank:02d}_of_04"
        receipt = json.loads((a.embeddings / f"{stem}.receipt.json").read_text())
        ids_path = a.embeddings / f"{stem}.ids.txt"
        matrix_path = a.embeddings / f"{stem}.npy"
        if (receipt["checkpoint_sha256"] != reaction_receipt["checkpoint_sha256"] or
                sha256(ids_path) != receipt["ids_sha256"] or
                sha256(matrix_path) != receipt["embedding_sha256"]):
            raise ValueError("Protein export does not match reaction checkpoint")
        own = ids_path.read_text().splitlines()
        shard = np.load(matrix_path, mmap_mode="r")
        if shard.shape != (len(own), 512):
            raise ValueError("Protein export shape mismatch")
        shard_ids.extend(own)
        shards.append(np.asarray(shard))
    if len(shard_ids) != 222985 or len(set(shard_ids)) != len(shard_ids):
        raise ValueError("Expected complete deduplicated enzyme library")
    mapping = read_csv(a.catalog / "screening_candidate_map.csv")
    candidate_ids = [row["uniprot_id"] for row in mapping]
    if len(candidate_ids) != 261907 or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Expected official full screening library")
    protein_index = {key: i for i, key in enumerate(shard_ids)}
    expanded = np.asarray([protein_index[row["protein_id"]] for row in mapping], np.int32)
    candidate_index = {key: i for i, key in enumerate(candidate_ids)}
    kept = np.asarray([i for i, key in enumerate(candidate_ids)
                       if key not in train_proteins], np.int32)
    if len(kept) != 252113:
        raise ValueError("Training-enzyme exclusion library changed")
    protein = np.concatenate(shards, axis=0)
    candidate = torch.from_numpy(protein).to(a.device)
    scorer = ScreenScorer.from_export(candidate, a.embeddings, reaction_receipt)
    refiner_info = None
    if a.refiner:
        if reaction_receipt.get("scoring", {}).get("kind", "cosine") != "cosine":
            raise ValueError("Residual refinement currently requires a single-vector base")
        from horizyn.generalization_residual import FrozenGeometryResidual
        saved = torch.load(a.refiner, map_location=a.device, weights_only=False)
        features = Path(saved["registry"]["arguments"]["features"])
        base_manifest = json.loads((features / "manifest.json").read_text())
        if sha256(features / "manifest.json") != saved["registry"]["feature_manifest_sha256"]:
            raise ValueError("Refiner training feature manifest changed")
        from horizyn.inference_fusion_lineage import verify_refiner_base
        fusion_lineage = verify_refiner_base(base_manifest["checkpoint"]["sha256"],
            reaction_receipt["checkpoint_sha256"], a.fusion_calibration, a.calibrated_base_checkpoint)
        refiner = FrozenGeometryResidual(**saved["model_config"]).to(a.device).eval().requires_grad_(False)
        refiner.load_state_dict(saved["state_dict"], strict=True)
        with torch.inference_mode():
            candidate = torch.cat([refiner.encode_enzymes(b) for b in candidate.split(8192)])
            reactions = refiner.encode_reactions(torch.from_numpy(reactions).to(a.device)).cpu().numpy()
        scorer = ScreenScorer(candidate, already_normalized=True)
        refiner_info = {"path": str(a.refiner.resolve()), "sha256": sha256(a.refiner),
                        "base_checkpoint_sha256": reaction_receipt["checkpoint_sha256"],
                        "train_only_fit": True, "test_labels_read": False}
        if fusion_lineage:
            refiner_info["inference_fusion_calibration"] = fusion_lineage
    per_query: list[dict] = []
    positives = [np.asarray([candidate_index[key] for key in truth.get(query, set())
                            if key in candidate_index], dtype=np.int64) for query in query_ids]
    torch.set_num_threads(4)
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=a.metric_workers) as pool:
        for start in range(0, len(query_ids), a.batch_size):
            query = torch.from_numpy(reactions[start:start + a.batch_size]).to(a.device)
            scores = scorer(query).float().cpu().numpy()[:, expanded]
            arguments = [(start+i, query_ids[start+i], values, positives[start+i], kept)
                         for i, values in enumerate(scores)]
            per_query.extend(row for row in pool.map(evaluate_query, arguments) if row is not None)
            if (start // a.batch_size) % 10 == 0:
                print(json.dumps({"evaluated_queries": min(start + a.batch_size, len(query_ids))}), flush=True)
    summary = {}
    for table in ("table1", "table2"):
        rows = [row[table] for row in per_query if row[table] is not None]
        summary[table] = dict(queries=len(rows), candidate_ids=(
            len(candidate_ids) if table == "table1" else len(kept)),
            **{metric: float(np.mean([row[metric] for row in rows]))
               for metric in ("bedroc85", "bedroc20", "ef0.05", "ef0.1")})
    a.output.mkdir(parents=True)
    row_path = a.output / "per_query.jsonl"
    row_path.write_text("".join(json.dumps(row) + "\n" for row in per_query))
    receipt = dict(schema="clipzyme_f3_full_library_validation_v1",
                   validation_only=True, test_labels_read=False,
                   selection_metric="table1.bedroc85",
                   selection_value=summary["table1"]["bedroc85"],
                   association_manifest_sha256=sha256(a.manifest),
                   checkpoint_sha256=reaction_receipt["checkpoint_sha256"],
                   reaction_embeddings_sha256=reaction_receipt["embedding_sha256"],
                   scoring=reaction_receipt.get("scoring", {"kind": "cosine"}),
                   metric_workers=a.metric_workers,
                   chemistry_masked=reaction_receipt.get("chemistry_masked", False),
                   per_query_sha256=sha256(row_path), summary=summary,
                   source_sha256=sha256(Path(__file__)))
    if refiner_info:
        receipt["refiner"] = refiner_info
        receipt["checkpoint_sha256"] = hashlib.sha256(
            (refiner_info["base_checkpoint_sha256"] + refiner_info["sha256"]).encode()).hexdigest()
    (a.output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
