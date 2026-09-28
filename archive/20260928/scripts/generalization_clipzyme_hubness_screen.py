#!/usr/bin/env python3
"""Apply a validation-selected, train-reference hubness correction to F3 screening scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from generalization_clipzyme_f3_screen import model_from_checkpoint, sha256
from generalization_clipzyme_hubness_validation import ids, reaction_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-screen", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = json.loads(args.selection.read_text())
    selected = selection["selected"]
    if (selection["schema"] != "clipzyme_f3_hubness_validation_v1"
            or selection["checkpoint_sha256"] != sha256(args.checkpoint)
            or selection["config_sha256"] != sha256(args.config)
            or selection["train_pairs_sha256"] != sha256(args.catalog / "train_pairs.csv")
            or selection["validation_pairs_sha256"] != sha256(args.catalog / "validation_pairs.csv")
            or selected not in selection["grid"]
            or selected["k"] < 1 or selected["strength"] < 0):
        raise ValueError("Hubness selection does not match the checkpoint and validation run")
    raw_receipt = json.loads((args.raw_screen / "score_receipt.json").read_text())
    if (raw_receipt["schema"] != "clipzyme_f3_screen_scores_v1"
            or raw_receipt["checkpoint_sha256"] != sha256(args.checkpoint)
            or raw_receipt["config_sha256"] != sha256(args.config)
            or raw_receipt["scores_sha256"] != sha256(args.raw_screen / "scores.npy")):
        raise ValueError("Raw score matrix does not match the selected checkpoint")
    with (args.catalog / "screening_candidate_map.csv").open(newline="") as handle:
        candidates = list(csv.DictReader(handle))
    if (len(candidates) != 261907
            or [row["uniprot_id"] for row in candidates]
               != (args.raw_screen / "candidate_ids.txt").read_text().splitlines()):
        raise ValueError("Screening candidate axis changed")
    pieces = []
    protein_ids = []
    for rank in range(4):
        stem = f"protein_rank{rank:02d}_of_04"
        receipt = json.loads((args.raw_screen / f"{stem}.receipt.json").read_text())
        path = args.raw_screen / f"{stem}.npy"
        if (receipt["checkpoint_sha256"] != sha256(args.checkpoint)
                or receipt["embedding_sha256"] != sha256(path)):
            raise ValueError("Protein shard does not match checkpoint")
        piece_ids = (args.raw_screen / f"{stem}.ids.txt").read_text().splitlines()
        vectors = np.load(path)
        if vectors.shape != (len(piece_ids), 512):
            raise ValueError("Protein shard shape mismatch")
        protein_ids.extend(piece_ids)
        pieces.append(vectors)
    protein = np.concatenate(pieces)
    if len(protein_ids) != 222985 or len(set(protein_ids)) != len(protein_ids):
        raise ValueError("Unique screening protein inventory changed")
    model, config = model_from_checkpoint(args.config, args.checkpoint, args.device)
    train_ids = ids(args.catalog / "train_rxns.csv", "reaction_id")
    torch.set_num_threads(4)
    train_reactions = reaction_vectors(model, config, args.catalog, "train",
                                       train_ids, args.device)
    train_reactions = F.normalize(train_reactions.float(), dim=-1)
    del model
    penalty = np.empty(len(protein_ids), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(protein_ids), args.batch_size):
            end = min(start + args.batch_size, len(protein_ids))
            vectors = F.normalize(torch.from_numpy(protein[start:end]).to(args.device), dim=-1)
            penalty[start:end] = (vectors @ train_reactions.T).topk(
                selected["k"], dim=1).values.mean(dim=1).cpu().numpy()
            print(json.dumps({"reference_scored_proteins": end}), flush=True)
    protein_index = {key: i for i, key in enumerate(protein_ids)}
    candidate_penalty = penalty[[protein_index[row["protein_id"]] for row in candidates]]
    raw = np.load(args.raw_screen / "scores.npy", mmap_mode="r")
    if raw.shape != (1521, 261907) or raw.dtype != np.float32:
        raise ValueError("Raw screening score matrix changed")
    args.output.mkdir(parents=True)
    out_path = args.output / "scores.npy"
    corrected = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                          shape=raw.shape)
    for start in range(0, len(raw), 64):
        end = min(start + 64, len(raw))
        corrected[start:end] = raw[start:end] - selected["strength"] * candidate_penalty
        corrected.flush()
    del corrected
    for name in ("candidate_ids.txt", "query_ids.txt"):
        (args.output / name).write_bytes((args.raw_screen / name).read_bytes())
    receipt = {"schema": "clipzyme_f3_train_reference_hubness_scores_v1",
               "test_labels_read": False,
               "checkpoint_sha256": sha256(args.checkpoint),
               "selection_sha256": sha256(args.selection),
               "raw_score_receipt_sha256": sha256(args.raw_screen / "score_receipt.json"),
               "corrected_scores_sha256": sha256(out_path),
               "candidate_ids_sha256": sha256(args.output / "candidate_ids.txt"),
               "query_ids_sha256": sha256(args.output / "query_ids.txt"),
               "train_reaction_count": len(train_ids),
               "unique_protein_count": len(protein_ids),
               "selected": selected,
               "source_sha256": sha256(Path(__file__))}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"scores": str(out_path), "selected": selected}), flush=True)


if __name__ == "__main__":
    main()
