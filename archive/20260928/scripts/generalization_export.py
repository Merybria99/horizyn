#!/usr/bin/env python3
"""Export compact, provenance-checked train/validation features without fitting.

No test pair file or Case1 comparison is read. Raw reaction blocks remain
separate and unnormalized so downstream experiments can fit train-only scalers.
Protein means use the stored ProtT5 residues (already ends-center truncated by
the extractor), with float32 accumulation. Existing F3 embeddings are reused;
the exact FP32 validation cache overrides the compact bf16 training cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def identity(path, hash_content=False):
    path = resolve(path)
    stat = path.stat()
    result = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if hash_content:
        result["sha256"] = digest(path)
    return result


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def strings(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def h5_ids(handle):
    ids = strings(handle["ids"][:])
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {handle.filename}")
    return ids


def read_pairs(path):
    with path.open() as handle:
        reader = csv.DictReader(handle)
        pairs = sorted({(row["reaction_id"], row["protein_id"]) for row in reader})
    if not pairs or any(not q or not p for q, p in pairs):
        raise ValueError(f"Empty or invalid association file: {path}")
    return pairs


def aligned_h5(path, ids, suffix=""):
    with h5py.File(path, "r") as handle:
        keys = h5_ids(handle)
        index = {key: i for i, key in enumerate(keys)}
        resolved = [key if key in index else key + suffix for key in ids]
        missing = [key for key in resolved if key not in index]
        if missing:
            raise ValueError(f"Missing {len(missing)} requested IDs in {path}")
        vectors = np.asarray(handle["vectors"][:], dtype=np.float32)
        output = vectors[[index[key] for key in resolved]]
    if not np.isfinite(output).all():
        raise ValueError(f"Nonfinite vectors: {path}")
    return output


def reaction_block(path, ids, molecule_set=False):
    """Mean participants from the forward reactant bank of self-reactions.

    Official ReactZyme caches duplicate the unordered participant set on both
    sides. Reading one side avoids counting it twice and matches F3's
    side_composition=molecule_set convention.
    """
    with h5py.File(path, "r") as handle:
        index = {key: i for i, key in enumerate(h5_ids(handle))}
        key = "reactant_vectors" if molecule_set else "vectors"
        dim = handle[key].shape[1]
        vectors = np.zeros((len(ids), dim), np.float32)
        mask = np.zeros(len(ids), bool)
        offsets = handle["reactant_offsets"][:] if molecule_set else None
        source = np.asarray(handle[key][:], dtype=np.float32)
        for row, identifier in enumerate(ids):
            physical = index.get(identifier + "_f", index.get(identifier))
            if physical is None:
                continue
            if molecule_set:
                start, stop = offsets[physical:physical + 2]
                if start == stop:
                    continue
                vectors[row] = source[start:stop].mean(0, dtype=np.float32)
            else:
                vectors[row] = source[physical]
            mask[row] = True
    if not np.isfinite(vectors).all():
        raise ValueError(f"Nonfinite reaction features: {path}")
    return vectors, mask


def prepare(args):
    config = yaml.safe_load(args.config.read_text())
    data = config["data"]
    if not data.get("normalize_molecule_sets_as_self_reactions"):
        raise ValueError("This exporter requires the audited participant-set protocol")
    inputs = {key: identity(data[key], hash_content=True) for key in (
        "train_pairs_path", "validation_pairs_path", "train_reactions_path",
        "validation_reactions_path", "validation_retrieval_candidate_ids_path")}
    inputs["config"] = identity(args.config, True)
    inputs["protein_residues"] = identity(args.protein_residue_h5 or data["protein_residue_embeds_path"])
    if args.reuse_mean_cache:
        reuse = args.reuse_mean_cache.resolve()
        previous = json.loads((reuse.parent / "manifest.json").read_text())
        receipt = json.loads((reuse.parent / "complete.json").read_text())
        signature = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
        if previous["inputs"]["protein_residues"] != inputs["protein_residues"]:
            raise ValueError("Mean reuse requires the identical physical residue source")
        if receipt["manifest_signature"] != signature:
            raise ValueError("Mean source completion receipt does not match its manifest")
        with h5py.File(reuse, "r") as source:
            if source.attrs.get("manifest_signature") != signature or not source["complete"][:].all():
                raise ValueError("Mean source is incomplete or belongs to another manifest")
            if source.attrs.get("accumulation_dtype") != "float32":
                raise ValueError("Mean source requires float32 accumulation")
        inputs["reuse_mean_cache"] = identity(reuse)
        inputs["reuse_mean_manifest"] = identity(reuse.parent / "manifest.json", True)
        inputs["reuse_mean_receipt"] = identity(reuse.parent / "complete.json", True)
    for split in ("train", "validation"):
        for modality in ("t5v2", "unimol2", "chiro", "chemistry"):
            ending = "vectors_path" if modality == "chemistry" else "embeds_path"
            key = f"{split}_reaction_{modality}_{ending}"
            inputs[key] = identity(data[key], hash_content=True)
    for cache_name, cache in (("compact", args.f3_cache), ("exact_validation", args.f3_validation_cache)):
        inputs[cache_name + "_manifest"] = identity(cache / "manifest.json", True)
        cached = json.loads((cache / "manifest.json").read_text())
        signature = cached["signature"]["inputs"]
        checkpoint = signature["checkpoint"]
        if identity(checkpoint["path"]) != checkpoint:
            raise ValueError(f"Frozen checkpoint identity changed: {checkpoint['path']}")
        for role, key in (("train_pairs", "train_pairs_path"),
                          ("validation_pairs", "validation_pairs_path")):
            if signature[role]["sha256"] != inputs[key]["sha256"]:
                raise ValueError(f"Frozen cache association source mismatch: {role}")
        if signature["protein_residues"] != identity(data["protein_residue_embeds_path"]):
            raise ValueError("Frozen cache residue source mismatch")
        for key, recorded in signature["reaction_feature_inputs"].items():
            if key in inputs and recorded is not None:
                if any(recorded[field] != inputs[key][field] for field in ("path", "size", "mtime_ns")):
                    raise ValueError(f"Frozen cache reaction feature source mismatch: {key}")
        if cache_name == "exact_validation" and (
            cached["dtype"] != "float32" or cached["signature"]["precision"] != "32"
        ):
            raise ValueError("Exact validation requires FP32 extraction and storage")
        names = ["enzyme_base.h5", "validation_reaction_base.h5"]
        if cache_name == "compact":
            names.append("train_reaction_base.h5")
        for name in names:
            inputs[f"{cache_name}_{name}"] = identity(cache / name, True)
    manifest = dict(schema="generalization_features_v1", inputs=inputs,
                    script_sha256=digest(__file__), test_used=False,
                    notes=["Original F3 training associations; validation never used to fit features.",
                           "Raw blocks are unnormalized, all statistics must be fit on training only.",
                           "F3 training cache used bf16 inference stored float16; validation overridden by FP32.",
                           "Means use all stored ProtT5 residues; extraction already truncated to 1022 tokens."])
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Inputs or export code changed; choose a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(path, manifest)
    (args.output / "export_source.py").write_text(Path(__file__).read_text())
    train = read_pairs(resolve(data["train_pairs_path"]))
    val = read_pairs(resolve(data["validation_pairs_path"]))
    if set(train) & set(val):
        raise ValueError("Train/validation association overlap")
    candidates = sorted({line.strip() for line in resolve(data["validation_retrieval_candidate_ids_path"]).read_text().splitlines() if line.strip()})
    if not {p for _, p in val} <= set(candidates):
        raise ValueError("Validation positives missing from candidate pool")
    proteins = sorted({p for _, p in train} | set(candidates))
    reactions = sorted({q for q, _ in train + val})
    train_q = sorted({q for q, _ in train})
    val_q = sorted({q for q, _ in val})
    catalog = dict(proteins=proteins, reactions=reactions, train_reactions=train_q,
                   validation_reactions=val_q, validation_candidates=candidates,
                   train_proteins=sorted({p for _, p in train}),
                   counts=dict(train_pairs=len(train), validation_pairs=len(val),
                               proteins=len(proteins), reactions=len(reactions),
                               unseen_validation_reactions=len(set(val_q) - set(train_q))))
    atomic_json(args.output / "catalog.json", catalog)
    pi, qi = ({key: i for i, key in enumerate(keys)} for keys in (proteins, reactions))
    atomic_npz(args.output / "pairs.npz",
               train=np.asarray([(qi[q], pi[p]) for q, p in train], dtype=np.int64),
               validation=np.asarray([(qi[q], pi[p]) for q, p in val], dtype=np.int64))
    print(json.dumps(catalog["counts"]), flush=True)
    return data, catalog, manifest


def export_compact(args, data, catalog):
    reactions, proteins = catalog["reactions"], catalog["proteins"]
    ridx = {key: i for i, key in enumerate(reactions)}
    pidx = {key: i for i, key in enumerate(proteins)}
    output = {}
    for modality, molecular in (("t5v2", False), ("unimol2", True), ("chiro", True)):
        # The audited official reaction caches cover both train and validation.
        train_path = resolve(data[f"train_reaction_{modality}_embeds_path"])
        val_path = resolve(data[f"validation_reaction_{modality}_embeds_path"])
        vectors, mask = reaction_block(train_path, reactions, molecular)
        if val_path != train_path:
            val, val_mask = reaction_block(val_path, catalog["validation_reactions"], molecular)
            rows = [ridx[key] for key in catalog["validation_reactions"]]
            vectors[rows], mask[rows] = val, val_mask
        output[modality], output[modality + "_mask"] = vectors, mask
    chemistry = None
    chemistry_mask = np.zeros(len(reactions), bool)
    for split, keys in (("train", catalog["train_reactions"]), ("validation", catalog["validation_reactions"])):
        with np.load(resolve(data[f"{split}_reaction_chemistry_vectors_path"]), allow_pickle=True) as source:
            si = {key: i for i, key in enumerate(strings(source["ids"]))}
            source_vectors = source["vectors"]
            source_mask = source["mask"] if "mask" in source else np.ones(len(si), bool)
            if chemistry is None:
                chemistry = np.zeros((len(reactions), source_vectors.shape[1]), np.float32)
            for key in keys:
                if key in si:
                    i, j = ridx[key], si[key]
                    value = source_vectors[j]
                    present = bool(source_mask[j])
                    if chemistry_mask[i] and present and not np.allclose(chemistry[i], value):
                        raise ValueError(f"Train/validation chemistry disagrees: {key}")
                    chemistry[i], chemistry_mask[i] = value, present
    if not np.isfinite(chemistry).all():
        raise ValueError("Nonfinite chemistry")
    output["chemistry"], output["chemistry_mask"] = chemistry, chemistry_mask
    atomic_npz(args.output / "reaction_features.npz", **output)
    enzymes = aligned_h5(args.f3_cache / "enzyme_base.h5", proteins)
    queries = np.zeros((len(reactions), enzymes.shape[1]), np.float32)
    for split, keys in (("train", catalog["train_reactions"]), ("validation", catalog["validation_reactions"])):
        cache = args.f3_cache if split == "train" else args.f3_validation_cache
        vectors = aligned_h5(cache / f"{split}_reaction_base.h5", keys, "_f")
        queries[[ridx[key] for key in keys]] = vectors
    val_enzymes = aligned_h5(args.f3_validation_cache / "enzyme_base.h5", catalog["validation_candidates"])
    enzymes[[pidx[key] for key in catalog["validation_candidates"]]] = val_enzymes
    # Separate train-query array avoids replacing shared training reactions by
    # the FP32 validation export when fitting on the compact baseline geometry.
    train_queries = aligned_h5(args.f3_cache / "train_reaction_base.h5", catalog["train_reactions"], "_f")
    atomic_npz(args.output / "f3_features.npz", proteins=enzymes, reactions=queries,
               train_reactions=train_queries)
    print("Exported compact reaction and frozen F3 features", flush=True)


def export_means(args, catalog, manifest):
    path = args.output / "protein_mean.h5"
    wanted = catalog["proteins"]
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    source_path = manifest["inputs"]["protein_residues"]["path"]
    with h5py.File(source_path, "r") as source, h5py.File(path, "a") as out:
        ids = h5_ids(source)
        physical = {key: i for i, key in enumerate(ids)}
        absent = set(wanted) - physical.keys()
        if absent:
            raise ValueError(f"Residue cache missing {len(absent)} proteins")
        offsets = source["offsets"][:]
        if offsets.shape != (len(ids) + 1,) or np.any(np.diff(offsets) <= 0):
            raise ValueError("Malformed or empty residue spans")
        if "ids" not in out:
            out.create_dataset("ids", data=np.asarray(wanted, dtype=object), dtype=h5py.string_dtype("utf-8"))
            out.create_dataset("vectors", shape=(len(wanted), source["vectors"].shape[1]), dtype=np.float32)
            out.create_dataset("complete", data=np.zeros(len(wanted), bool))
            out.attrs["manifest_signature"] = signature
            out.attrs["accumulation_dtype"] = "float32"
        elif out.attrs.get("manifest_signature") != signature or h5_ids(out) != wanted:
            raise ValueError("Mean cache resume signature/IDs do not match")
        complete = out["complete"][:]
        if args.reuse_mean_cache:
            # Means contain no fitted statistics or labels. Sharing an exact
            # per-protein transform across splits is therefore inductive.
            with h5py.File(args.reuse_mean_cache, "r") as reused:
                old_ids = {key: i for i, key in enumerate(h5_ids(reused))}
                rows = np.asarray([i for i, key in enumerate(wanted) if not complete[i] and key in old_ids], dtype=np.int64)
                if len(rows):
                    old_vectors = reused["vectors"][:]
                    values = old_vectors[[old_ids[wanted[i]] for i in rows]]
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite reusable means")
                    # h5py constructs point selections quadratically for very
                    # long index lists. Bound them while retaining sparse I/O.
                    for offset in range(0, len(rows), 512):
                        out["vectors"][rows[offset:offset + 512]] = values[offset:offset + 512]
                    out.flush()
                    complete[rows] = True
                    out["complete"][:] = complete
                    out.attrs["reused_proteins"] = int(out.attrs.get("reused_proteins", 0)) + len(rows)
                    out.flush()
                    print(f"Reused {len(rows)} fit-free protein means from identical residue source", flush=True)
        order = sorted((physical[key], row) for row, key in enumerate(wanted) if not complete[row])
        cursor, finished = 0, int(complete.sum())
        started = time.monotonic()
        while cursor < len(order):
            start = int(offsets[order[cursor][0]])
            end = cursor + 1
            while end < len(order) and int(offsets[order[end][0] + 1]) - start <= args.block_residues:
                end += 1
            stop = int(offsets[order[end - 1][0] + 1])
            block = source["vectors"][start:stop]
            rows, values = [], []
            for physical_row, output_row in order[cursor:end]:
                lo, hi = offsets[physical_row:physical_row + 2] - start
                mean = block[lo:hi].mean(0, dtype=np.float32)
                if not np.isfinite(mean).all():
                    raise ValueError(f"Invalid mean for {wanted[output_row]}")
                rows.append(output_row)
                values.append(mean)
            permutation = np.argsort(rows)
            rows = np.asarray(rows)[permutation]
            out["vectors"][rows] = np.asarray(values)[permutation]
            out.flush()
            out["complete"][rows] = True
            out.flush()
            finished += len(rows)
            cursor = end
            print(f"protein means {finished}/{len(wanted)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if not out["complete"][:].all():
            raise ValueError("Incomplete protein mean cache")
    if getattr(args, "write_completion_receipt", True):
        atomic_json(args.output / "complete.json", dict(manifest_signature=signature,
                    proteins=len(wanted), test_used=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/train.yaml")
    parser.add_argument("--protein-residue-h5", type=Path)
    parser.add_argument("--reuse-mean-cache", type=Path, help="Completed protein_mean.h5 from the identical residue source")
    parser.add_argument("--f3-cache", type=Path, default=ROOT / "runs/biological_residual_reaction_smi/cache/reactzyme")
    parser.add_argument("--f3-validation-cache", type=Path, default=ROOT / "runs/biological_residual_reaction_smi/cache/reactzyme_validation_eval")
    parser.add_argument("--block-residues", type=int, default=65536)
    parser.add_argument("--compact-only", action="store_true", help="Export small reaction/F3 caches and defer the residue scan")
    args = parser.parse_args()
    if args.block_residues < 1022:
        parser.error("--block-residues must be at least 1022")
    for key in ("config", "output", "f3_cache", "f3_validation_cache"):
        setattr(args, key, resolve(getattr(args, key)))
    data, catalog, manifest = prepare(args)
    export_compact(args, data, catalog)
    if not args.compact_only:
        export_means(args, catalog, manifest)


if __name__ == "__main__":
    main()
