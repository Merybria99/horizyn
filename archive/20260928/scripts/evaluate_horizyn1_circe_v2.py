#!/usr/bin/env python3
"""Full-catalog CIRCE-v2 evaluation with separate cold query sets and exact ranks.

The gold file must contain ALL observed own-source associations for selected
queries, including positives in other split quadrants. No labels enter encoders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from horizyn.benchmarks.chunked_retrieval import chunked_positive_ranks, rank_metrics
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_features import build_reaction_feature_dataset
from scripts.evaluate_protein_pooling import KeySubsetDataset, encode_targets, encode_reactions


def file_signature(path):
    path = Path(path).resolve()
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_ids(path):
    if path is None:
        return None
    with open(path) as handle:
        ids = [line.strip() for line in handle if line.strip()]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError(f"ID file must be nonempty and unique: {path}")
    return ids


def implementation_signatures():
    """Invalidate cached embeddings if an encoder or input implementation changes."""
    paths = (
        "scripts/evaluate_horizyn1_circe_v2.py",
        "scripts/evaluate_protein_pooling.py",
        "horizyn/benchmarks/chunked_retrieval.py",
        "horizyn/model.py",
        "horizyn/protein_pooling_lightning_module.py",
        "horizyn/reaction_features.py",
        "horizyn/datasets/residue_hdf5.py",
        "horizyn/utils/collate.py",
        "horizyn/hyperbolic_enzyme.py",
        "horizyn/sleec_stage1.py",
    )
    return {relative: digest(ROOT / relative) for relative in paths}


def query_truth(path, reaction_queries, enzyme_queries):
    r2e, e2r = defaultdict(set), defaultdict(set)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t" if str(path).endswith(".tsv") else ",")
        for row in reader:
            r, e = row["reaction_id"], row["protein_id"]
            if reaction_queries is None or r in reaction_queries:
                r2e[r].add(e)
            if enzyme_queries is None or e in enzyme_queries:
                e2r[e].add(r)
    for queries, truth, role in ((reaction_queries, r2e, "reaction"), (enzyme_queries, e2r, "enzyme")):
        if queries is not None and set(queries) - truth.keys():
            raise ValueError(f"Selected {role} queries lack gold associations")
    if not r2e or not e2r:
        raise ValueError("Both retrieval directions need nonempty query gold")
    return r2e, e2r


def evaluate_direction(queries, candidates, query_ids, truth, candidate_lookup, *, batch_size,
                       chunk_size, device, label):
    sums, count = defaultdict(float), 0
    for start in tqdm(range(0, len(query_ids), batch_size), desc=label):
        ids = query_ids[start:start + batch_size]
        positives = []
        for q in ids:
            missing = truth[q] - candidate_lookup.keys()
            if missing:
                raise ValueError(f"{label}: gold positives missing from catalog for {q}: {len(missing)}")
            positives.append([candidate_lookup[p] for p in truth[q]])
        ranks = chunked_positive_ranks(
            queries[start:start + batch_size].to(device), candidates, positives, chunk_size=chunk_size,
        )
        metrics = rank_metrics(ranks, (1, 10, 100, 1000))
        for name, values in metrics.items():
            sums[name] += values.double().sum().item()
        count += len(ids)
    return {"queries": count, "candidates": len(candidates),
            **{name: total / count for name, total in sums.items()}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "config", "pairs", "reactions", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("protein-candidates", "reaction-candidates", "enzyme-query-ids", "reaction-query-ids", "embedding-cache"):
        parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-batch-size", type=int, default=64)
    parser.add_argument("--candidate-chunk-size", type=int, default=32768)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--protocol", choices=("own_source", "reaction_holdout_clustered"), default="own_source")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.batch_size, args.target_batch_size, args.candidate_chunk_size) <= 0:
        raise ValueError("Batch and chunk sizes must be positive")
    config = load_config(args.config)
    sources = {k: file_signature(v) for k, v in vars(args).items()
               if isinstance(v, Path) and k not in {"output", "embedding_cache"}}
    sources["checkpoint_sha256"] = digest(args.checkpoint)
    sources["config_sha256"] = digest(args.config)
    sources["evaluator_sha256"] = digest(__file__)
    sources["ranker_sha256"] = digest(ROOT / "horizyn/benchmarks/chunked_retrieval.py")
    sources["implementation_sha256"] = implementation_signatures()
    for key, value in config.data.items():
        if isinstance(value, str) and ("embed" in key or "vectors_path" in key) and Path(value).is_file():
            sources[key] = file_signature(value)
            if value.endswith((".h5", ".hdf5")):
                import h5py
                with h5py.File(value, "r") as handle:
                    if "vectors" in handle and handle["vectors"].is_virtual:
                        for i, backing in enumerate(handle["vectors"].virtual_sources()):
                            shard = Path(os.fsdecode(backing.file_name))
                            if not shard.is_absolute():
                                shard = Path(value).parent / shard
                            sources[f"{key}_shard_{i}"] = file_signature(shard)
    signature = {
        "schema": 1, "sources": sources, "ties": "score_desc_candidate_row_asc",
        "scoring": {"query_batch_size": args.batch_size,
                    "target_batch_size": args.target_batch_size,
                    "candidate_chunk_size": args.candidate_chunk_size,
                    "score_dtype": "float32"},
    }
    if args.protocol != "own_source":
        signature["protocol"] = args.protocol
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("signature") != signature:
            raise ValueError("Existing evaluation belongs to different inputs; choose a new output path")
        print(f"Using completed evaluation: {args.output}", flush=True)
        return
    module = ProteinPooledLitModule.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    if (module.model.enzyme_prototype_count != 1 or module.embedding_similarity != "cosine"
            or getattr(module.model, "e2r_adapter", None) is not None
            or getattr(module.model, "r2e_adapter", None) is not None):
        raise ValueError("This evaluator requires the agreed shared-vector cosine CIRCE-v2 architecture")
    module.eval().to(args.device)
    residue_dataset = ResidueEmbedDataset(
        config.data.protein_residue_embeds_path, in_memory=False,
        max_tokens=config.data.get("max_protein_tokens", 1022),
        truncation=config.data.get("protein_truncation", "ends_center"),
    )
    protein_ids = read_ids(args.protein_candidates) or list(residue_dataset.keys)
    missing = set(protein_ids) - set(residue_dataset.keys)
    if missing:
        raise ValueError(f"Missing protein features: {len(missing)}")
    residue_dataset = KeySubsetDataset(residue_dataset, protein_ids)
    reaction_features = build_reaction_feature_dataset(
        args.reactions, config, bidirectional=not bool(config.data.get("indexed_pairs_dir")), split_name="train",
    )
    reaction_ids = read_ids(args.reaction_candidates)
    if reaction_ids is None:
        with open(args.reactions, newline="") as handle:
            reaction_ids = [row["reaction_id"] for row in csv.DictReader(handle)]
    available = set(reaction_features.keys)
    feature_keys = []
    for r in reaction_ids:
        key = r if r in available else f"{r}_f"
        if key not in available:
            raise ValueError(f"Missing mandatory reaction features: {r}")
        feature_keys.append(key)
    protein_lookup = {p: i for i, p in enumerate(protein_ids)}
    reaction_lookup = {r: i for i, r in enumerate(reaction_ids)}
    rqueries = read_ids(args.reaction_query_ids)
    equeries = read_ids(args.enzyme_query_ids)
    rtruth, etruth = query_truth(args.pairs, set(rqueries) if rqueries else None, set(equeries) if equeries else None)
    rqueries = rqueries or sorted(rtruth)
    equeries = equeries or sorted(etruth)
    if args.protocol == "reaction_holdout_clustered" and set(reaction_ids) != set(rqueries):
        raise ValueError("Reaction-held-out E2R evaluation must use only the held-out reaction catalog")
    if set(rqueries) - reaction_lookup.keys() or set(equeries) - protein_lookup.keys():
        raise ValueError("Query entities missing from encoded catalogs")
    with torch.inference_mode():
        proteins = None
        if args.embedding_cache and args.embedding_cache.exists():
            payload = torch.load(args.embedding_cache, map_location="cpu", weights_only=True)
            if payload.get("signature") != signature:
                raise ValueError("Embedding cache input signature mismatch; choose a new cache path")
            proteins = payload["embeddings"]
            if proteins.shape != (len(protein_ids), 512) or not torch.isfinite(proteins).all():
                raise ValueError("Invalid cached protein embedding shape or values")
        if proteins is None:
            proteins = encode_targets(module, residue_dataset, args.device, args.target_batch_size, store_on_device=False)
            if args.embedding_cache:
                args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.embedding_cache.with_suffix(f".partial.{os.getpid()}")
                torch.save({"signature": signature, "embeddings": proteins}, temporary)
                os.replace(temporary, args.embedding_cache)
        reactions = encode_reactions(module, reaction_features, feature_keys, args.device, args.batch_size).cpu()
        results = {"signature": signature, "gold_semantics": "observed own-source positives; unknown is not inactivity"}
        if args.protocol == "reaction_holdout_clustered":
            results.update(gold_semantics="source-collapsed reaction-enzyme associations; not all experimentally verified",
                           split_semantics="reaction-held-out; protein overlap is allowed, not an enzyme-cold test",
                           enzyme_to_reaction_protocol="enzymes with test associations ranked against held-out reactions only",
                           reaction_direction_augmentation="forward and reverse chemical directions")
        results["reaction_to_enzyme"] = evaluate_direction(
            reactions[[reaction_lookup[r] for r in rqueries]], proteins, rqueries, rtruth, protein_lookup,
            batch_size=args.batch_size, chunk_size=args.candidate_chunk_size, device=args.device, label="reaction_to_enzyme",
        )
        results["enzyme_to_reaction"] = evaluate_direction(
            proteins[[protein_lookup[e] for e in equeries]], reactions, equeries, etruth, reaction_lookup,
            batch_size=args.batch_size, chunk_size=args.candidate_chunk_size, device=args.device, label="enzyme_to_reaction",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({k: v for k, v in results.items() if k != "signature"}, indent=2))


if __name__ == "__main__":
    main()
