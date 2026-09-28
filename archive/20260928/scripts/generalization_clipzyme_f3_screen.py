#!/usr/bin/env python3
"""Encode and score the fixed CLIPZyme screening pool with target-trained F3."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess
from generalization_gpu_budget import free_memory_mib

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from horizyn.benchmarks.retrieval import (BenchmarkTask, build_reaction_inputs,
    encode_reactions, encode_residue_targets, load_repo_checkpoint)
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.screen_scoring import ScreenScorer, export_scoring_head
from horizyn.training_io import StoragePrecisionResidues


class ChemistryMaskedInputs:
    def __init__(self, source):
        self.source, self.keys = source, source.keys

    def __getitem__(self, key):
        sample = dict(self.source[key])
        if "has_reaction_chemistry" not in sample:
            raise ValueError("Chemistry masking requires an explicit chemistry availability flag")
        sample["has_reaction_chemistry"] = torch.zeros_like(
            torch.as_tensor(sample["has_reaction_chemistry"], dtype=torch.bool))
        return sample


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_from_checkpoint(config_path: Path, checkpoint: Path, device: str):
    config = load_config(str(config_path))
    # Full training snapshots include optimizer tensors. Stage them on the
    # host so loading does not transiently consume several models of VRAM.
    model, kind = load_repo_checkpoint(checkpoint, config, "cpu")
    if kind != "residue":
        raise ValueError("Expected locked F3 residue dual encoder")
    model.to(device).eval().requires_grad_(False)
    return model, config


@contextmanager
def export_device_lock(device):
    """Serialize large exports on each physical GPU across queue processes."""
    if not str(device).startswith("cuda"):
        yield
        return
    logical = int(str(device).partition(":")[2] or 0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = visible.split(",")[logical].strip() if visible else str(logical)
    if not physical or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in physical):
        raise ValueError("Invalid physical GPU identifier")
    path = Path("/tmp") / f"enzymediscovery_screen_export_gpu_{physical}.lock"
    with path.open("a+") as stream:
        print(json.dumps(dict(waiting_for_export_gpu=physical)), flush=True)
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def candidate_map(catalog: Path) -> tuple[list[str], set[str]]:
    with (catalog / "screening_candidate_map.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 261907 or [int(row["candidate_index"]) for row in rows] != list(range(261907)):
        raise ValueError("Official screening candidate pool changed")
    return [row["uniprot_id"] for row in rows], {row["protein_id"] for row in rows}


def protein_phase(args) -> None:
    _, required = candidate_map(args.catalog)
    model, config = model_from_checkpoint(args.config, args.checkpoint, args.device)
    residue_path = args.catalog / "features/proteins_prott5_residue.h5"
    cache_path = args.catalog / "features/proteins_prott5_residue.local.h5"
    cache_receipt = None
    if cache_path.is_file():
        cache_receipt_path = cache_path.with_suffix(".receipt.json")
        cache_receipt = json.loads(cache_receipt_path.read_text())
        stat = cache_path.stat()
        if (cache_receipt["source_sha256"] != sha256(residue_path) or
                cache_receipt.get("purpose") != "screening" or
                cache_receipt["protein_count"] != len(required) or
                cache_receipt["output_size_bytes"] != stat.st_size or
                cache_receipt["output_mtime_ns"] != stat.st_mtime_ns):
            raise ValueError("Local screening residue cache provenance mismatch")
        residue_path = cache_path
    residue = StoragePrecisionResidues(str(residue_path),
                                  max_tokens=config.data.max_protein_tokens,
                                  truncation=config.data.protein_truncation)
    try:
        keys = [key for key in residue.keys if key in required]
        if len(keys) != 222985 or len(set(keys)) != len(keys):
            raise ValueError("F3 residue cache does not cover the screening pool")
        if args.shard_block_size:
            # The virtual residue bank combines physical HDF5 sources with
            # different read costs. Distribute contiguous source-local blocks
            # over ranks so one slow source does not set the wall time.
            own = [key for block_start in range(0, len(keys), args.shard_block_size)
                   if (block_start // args.shard_block_size) % args.world_size == args.rank
                   for key in keys[block_start:block_start + args.shard_block_size]]
        else:
            left = len(keys) * args.rank // args.world_size
            right = len(keys) * (args.rank + 1) // args.world_size
            own = keys[left:right]
        torch.set_num_threads(4)
        vectors = encode_residue_targets(model, residue, own, args.device,
                                         args.batch_size, False,
                                         progress_every_batches=10).float().cpu().numpy()
    finally:
        residue.close()
    if vectors.shape != (len(own), 512) or not np.isfinite(vectors).all():
        raise ValueError("F3 candidate embedding export failed")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"protein_rank{args.rank:02d}_of_{args.world_size:02d}"
    np.save(args.output / f"{stem}.npy", vectors)
    (args.output / f"{stem}.ids.txt").write_text("\n".join(own) + "\n")
    receipt = {"schema": "clipzyme_f3_screen_protein_shard_v1", "rank": args.rank,
               "world_size": args.world_size, "shard_block_size": args.shard_block_size,
               "protein_count": len(own),
               "batch_size": args.batch_size,
               "residue_transport": "storage_precision_restored_to_model_dtype_on_device",
               "embedding_sha256": sha256(args.output / f"{stem}.npy"),
               "ids_sha256": sha256(args.output / f"{stem}.ids.txt"),
               "checkpoint_sha256": sha256(args.checkpoint),
               "config_sha256": sha256(args.config),
               "residue_store": str(residue_path.resolve()),
               "local_cache_sha256": cache_receipt["output_sha256"] if cache_receipt else None,
               "labels_read": False}
    (args.output / f"{stem}.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"rank": args.rank, "proteins": len(own)}), flush=True)


def reaction_phase(args) -> None:
    model, config = model_from_checkpoint(args.config, args.checkpoint, args.device)
    if args.scope == "test":
        input_path = args.protocol / "query_inputs.csv"
        protocol = json.loads((args.protocol / "receipt.json").read_text())
        if sha256(input_path) != protocol["query_inputs_sha256"]:
            raise ValueError("Label-free query catalog changed")
        with input_path.open(newline="") as handle:
            query_ids = [row["reaction_id"] for row in csv.DictReader(handle)]
        if len(query_ids) != 1521 or len(set(query_ids)) != len(query_ids):
            raise ValueError("Expected unique train-unseen screening queries")
        pairs_path = args.catalog / "test_pairs.csv"
        reactions_path = args.catalog / "test_rxns.csv"
        chemistry_path = args.catalog / "chemistry/test_reaction_set_features.npz"
    else:
        reactions_path = args.catalog / "validation_rxns.csv"
        with reactions_path.open(newline="") as handle:
            query_ids = [row["reaction_id"] for row in csv.DictReader(handle)]
        if len(query_ids) != 2661 or len(set(query_ids)) != len(query_ids):
            raise ValueError("Expected unique validation reactions")
        input_path = reactions_path
        pairs_path = args.catalog / "validation_pairs.csv"
        chemistry_path = args.catalog / "chemistry/validation_reaction_set_features.npz"
    features = args.catalog / "features"
    config.data.reaction_chemistry_vectors_path = str(chemistry_path)
    task = BenchmarkTask(name="clipzyme_f3_screening", task_type="retrieval",
        dataset="EnzymeMap", task_label="rule_split", split=args.scope,
        pairs=pairs_path, reactions=reactions_path,
        reaction_model_embeds_h5=features / "reactiont5v2.h5",
        reaction_unimol2_embeds_h5=features / "unimol2.h5",
        reaction_chiro_embeds_h5=features / "chiro.h5")
    inputs = build_reaction_inputs(task, config)
    if args.mask_chemistry:
        inputs = ChemistryMaskedInputs(inputs)
    if not set(query_ids).issubset(set(inputs.keys)):
        raise ValueError("Missing F3 reaction features for screening queries")
    torch.set_num_threads(4)
    vectors = encode_reactions(model, inputs, query_ids, args.device,
                               args.batch_size).float().cpu().numpy()
    if vectors.shape != (len(query_ids), 512) or not np.isfinite(vectors).all():
        raise ValueError("F3 reaction embedding export failed")
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "reaction_embeddings.npy", vectors)
    (args.output / "query_ids.txt").write_text("\n".join(query_ids) + "\n")
    receipt = {"schema": "clipzyme_f3_screen_reactions_v1", "labels_read": False,
               "scope": args.scope,
               "query_count": len(query_ids),
               "query_inputs_sha256": sha256(input_path),
               "embedding_sha256": sha256(args.output / "reaction_embeddings.npy"),
               "checkpoint_sha256": sha256(args.checkpoint),
               "config_sha256": sha256(args.config)}
    receipt["chemistry_masked"] = bool(args.mask_chemistry)
    receipt["scoring"] = export_scoring_head(model.model, args.output)
    (args.output / "reaction_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"reactions": len(query_ids)}), flush=True)


def score_phase(args) -> None:
    torch.set_num_threads(4)
    checkpoint_digest = sha256(args.checkpoint)
    candidate_ids, required = candidate_map(args.catalog)
    with (args.catalog / "screening_candidate_map.csv").open(newline="") as handle:
        protein_ids = [row["protein_id"] for row in csv.DictReader(handle)]
    pieces = []
    ids = []
    for rank in range(args.world_size):
        stem = f"protein_rank{rank:02d}_of_{args.world_size:02d}"
        receipt = json.loads((args.output / f"{stem}.receipt.json").read_text())
        path = args.output / f"{stem}.npy"
        id_path = args.output / f"{stem}.ids.txt"
        if (receipt["schema"] != "clipzyme_f3_screen_protein_shard_v1"
                or receipt["checkpoint_sha256"] != checkpoint_digest
                or receipt["embedding_sha256"] != sha256(path)
                or receipt["ids_sha256"] != sha256(id_path)):
            raise ValueError("Protein shard provenance mismatch")
        own = id_path.read_text().splitlines()
        vectors = np.load(path, mmap_mode="r")
        if vectors.shape != (len(own), 512):
            raise ValueError("Protein shard/ID mismatch")
        pieces.append(np.asarray(vectors))
        ids.extend(own)
    if len(ids) != len(required) or set(ids) != required or len(set(ids)) != len(ids):
        raise ValueError("Protein shard union does not match screening library")
    protein = np.concatenate(pieces, axis=0)
    index = {key: i for i, key in enumerate(ids)}
    candidate = protein[[index[key] for key in protein_ids]]
    reaction = np.load(args.output / "reaction_embeddings.npy")
    reaction_receipt = json.loads((args.output / "reaction_receipt.json").read_text())
    if (reaction_receipt["schema"] != "clipzyme_f3_screen_reactions_v1"
            or reaction_receipt["checkpoint_sha256"] != checkpoint_digest
            or reaction_receipt["embedding_sha256"] != sha256(args.output / "reaction_embeddings.npy")
            or reaction.shape != (1521, 512)):
        raise ValueError("Reaction embedding provenance mismatch")
    device = torch.device(args.device)
    target = torch.from_numpy(candidate.copy()).to(device)
    scorer = ScreenScorer.from_export(target, args.output, reaction_receipt)
    refiner_info = None
    if args.refiner:
        if reaction_receipt.get("scoring", {}).get("kind", "cosine") != "cosine":
            raise ValueError("Residual refinement currently requires a single-vector base")
        from horizyn.generalization_residual import FrozenGeometryResidual
        saved = torch.load(args.refiner, map_location=device, weights_only=False)
        features = Path(saved["registry"]["arguments"]["features"])
        manifest = json.loads((features / "manifest.json").read_text())
        if sha256(features / "manifest.json") != saved["registry"]["feature_manifest_sha256"]:
            raise ValueError("Refiner training feature manifest changed")
        from horizyn.inference_fusion_lineage import verify_refiner_base
        fusion_lineage = verify_refiner_base(manifest["checkpoint"]["sha256"],
            reaction_receipt["checkpoint_sha256"], args.fusion_calibration, args.checkpoint)
        refiner = FrozenGeometryResidual(**saved["model_config"]).to(device).eval().requires_grad_(False)
        refiner.load_state_dict(saved["state_dict"], strict=True)
        with torch.inference_mode():
            target = torch.cat([refiner.encode_enzymes(b) for b in target.split(8192)])
            reaction = refiner.encode_reactions(torch.from_numpy(reaction).to(device)).cpu().numpy()
        scorer = ScreenScorer(target, already_normalized=True)
        refiner_info = {"path": str(args.refiner.resolve()), "sha256": sha256(args.refiner),
                        "base_checkpoint_sha256": reaction_receipt["checkpoint_sha256"],
                        "train_only_fit": True}
        if fusion_lineage:
            refiner_info["inference_fusion_calibration"] = fusion_lineage
    scores_path = args.output / "scores.npy"
    if scores_path.exists():
        raise FileExistsError(scores_path)
    # The fixed matrix is 1.48 GiB. Buffering it avoids repeated NFS mmap
    # page faults and synchronous flushes for every query block.
    scores = np.empty((1521, 261907), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(reaction), args.batch_size):
            query = torch.from_numpy(reaction[start:start + args.batch_size]).to(device)
            scores[start:start + len(query)] = scorer(query).float().cpu().numpy()
            print(json.dumps({"scored_queries": start + len(query)}), flush=True)
    partial = scores_path.with_name("scores.partial.npy")
    with partial.open("wb") as stream:
        np.save(stream, scores)
    partial.replace(scores_path)
    del scores
    (args.output / "candidate_ids.txt").write_text("\n".join(candidate_ids) + "\n")
    receipt = {"schema": "clipzyme_f3_screen_scores_v1", "labels_read": False,
               "checkpoint_sha256": checkpoint_digest,
               "config_sha256": sha256(args.config),
               "candidate_map_sha256": sha256(args.catalog / "screening_candidate_map.csv"),
               "candidate_ids_sha256": sha256(args.output / "candidate_ids.txt"),
               "query_ids_sha256": sha256(args.output / "query_ids.txt"),
               "scores_sha256": sha256(scores_path),
               "score_shape": [1521, 261907],
               "score_write_mode": "ram_buffer_atomic_sequential_npy",
               "scoring": reaction_receipt.get("scoring", {"kind": "cosine"}),
               "chemistry_masked": reaction_receipt.get("chemistry_masked", False),
               "source_sha256": sha256(Path(__file__))}
    if refiner_info:
        receipt["refiner"] = refiner_info
        receipt["checkpoint_sha256"] = hashlib.sha256(
            (refiner_info["base_checkpoint_sha256"] + refiner_info["sha256"]).encode()).hexdigest()
    (args.output / "score_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"scores": str(scores_path), "shape": receipt["score_shape"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("proteins", "reactions", "scores"), required=True)
    parser.add_argument("--scope", choices=("test", "validation"), default="test")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--shard-block-size", type=int, default=0,
                        help="Round-robin contiguous blocks across ranks; zero retains contiguous shards")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mask-chemistry", action="store_true")
    parser.add_argument("--refiner", type=Path, help="Fixed train-only residual snapshot")
    parser.add_argument("--fusion-calibration", type=Path, help="Explicit receipt for a verified inference-only fusion scalar change")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size or args.batch_size < 1 or args.shard_block_size < 0:
        parser.error("Invalid rank, world size or batch size")
    if args.phase == "scores":
        score_phase(args)
    else:
        with export_device_lock(args.device):
            if args.phase == "proteins":
                # Keep room for the concurrent trainers. This changes export
                # microbatching only, never the candidate set or model.
                if str(args.device).startswith("cuda"):
                    while True:
                        # Query externally before initializing a CUDA context:
                        # mem_get_info itself can OOM when another job peaks.
                        logical = int(str(args.device).partition(":")[2] or 0)
                        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                        physical = visible.split(",")[logical].strip() if visible else str(logical)
                        free_gib = free_memory_mib(physical) / 1024
                        if free_gib >= 14:
                            break
                        print(json.dumps(dict(waiting_for_free_gpu_gib=free_gib)), flush=True)
                        time.sleep(5)
                    cap = 256 if free_gib >= 48 else 128 if free_gib >= 27 else 64
                    args.batch_size = min(args.batch_size, cap)
                    print(json.dumps(dict(export_batch_size=args.batch_size, free_gpu_gib=free_gib)), flush=True)
                protein_phase(args)
            else:
                reaction_phase(args)


if __name__ == "__main__":
    main()
