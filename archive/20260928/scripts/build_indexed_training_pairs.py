#!/usr/bin/env python3
"""Build mmap/CSR train indices without enumerating negative pairs.

The training CSV is read twice into compact NumPy coordinates, not retained as
millions of dictionaries. NPZ members are unpacked to a temporary directory and
validated in chunks. Only train entities enter the index; optional exclusion
CSVs can prohibit edges but never contribute positive support or annotations.
"""

from __future__ import annotations

import argparse
from array import array
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_annotation_negative_pools import (
    _v2_metadata, _validate_v2_arrays, canonical_protein_id,
    ec_annotation_matches_complete, reaction_signature,
)
from horizyn.datasets.indexed_pairs import SEMANTICS, write_indexed_pairs

CHUNK = 32768


def rows(path, required):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path} requires columns {sorted(required)}")
        for row in reader:
            if None in row or any(row.get(key) is None or not row[key].strip() for key in required):
                raise ValueError(f"Malformed required fields in {path}:{reader.line_num}")
            yield row


def chunks(iterator):
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) == CHUNK:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class ProteinLookup:
    def __init__(self, ids):
        self.ids = np.empty(len(ids), dtype=np.asarray(ids).dtype)
        for start in range(0, len(ids), CHUNK):
            stop = min(len(ids), start + CHUNK)
            self.ids[start:stop] = [canonical_protein_id(value) for value in ids[start:stop]]
        self.order = np.argsort(self.ids)
        self.sorted_ids = self.ids[self.order]
        if np.any(self.sorted_ids[1:] == self.sorted_ids[:-1]):
            raise ValueError("Canonical protein aliases collide")

    def many(self, ids):
        values = np.asarray([canonical_protein_id(value) for value in ids])
        pos = np.searchsorted(self.sorted_ids, values)
        valid = pos < len(self.sorted_ids)
        valid[valid] &= self.sorted_ids[pos[valid]] == values[valid]
        result = np.full(len(values), -1, dtype=np.int64)
        result[valid] = self.order[pos[valid]]
        return result


def load_graph(path, scratch):
    query_names = set()
    size = width = 0
    for row in rows(path, ("reaction_id", "protein_id")):
        query_names.add(row["reaction_id"].strip())
        width = max(width, len(row["protein_id"].strip()))
        size += 1
        if size % 1_000_000 == 0:
            print(f"Counting training CSV: {size:,} rows", flush=True)
    if not size:
        raise ValueError("Training pair CSV is empty")
    qids = np.asarray(sorted(query_names))
    query_lookup = {qid: i for i, qid in enumerate(qids)}
    raw_pids = np.lib.format.open_memmap(scratch / "pair_protein_ids.npy", mode="w+", dtype=f"U{width}", shape=(size,))
    edges = np.empty((size, 2), dtype=np.uint32)
    offset = 0
    for chunk in chunks(rows(path, ("reaction_id", "protein_id"))):
        stop = offset + len(chunk)
        raw_pids[offset:stop] = [row["protein_id"].strip() for row in chunk]
        edges[offset:stop, 0] = [query_lookup[row["reaction_id"].strip()] for row in chunk]
        offset = stop
    pids, inverse = np.unique(raw_pids, return_inverse=True)
    edges[:, 1] = inverse
    return qids, pids, np.unique(edges, axis=0)


def load_ec(path, lookup, pairs, num_queries, prefix_depth):
    label_names, label_lookup = [], {}
    records = array("I")
    complete_labels = set()
    for chunk in chunks(rows(path, ("protein_id", "ec_number", "known_depth"))):
        indices = lookup.many([row["protein_id"] for row in chunk])
        for p, row in zip(indices, chunk):
            if p < 0:
                continue
            label = row["ec_number"].strip()
            if label not in label_lookup:
                label_lookup[label] = len(label_names)
                label_names.append(label)
            code = label_lookup[label]
            records.extend((int(p), code))
            if row["known_depth"].strip() == "4" and len(label.split(".")) == 4 and all(part.isdigit() for part in label.split(".")):
                complete_labels.add(code)
    records = np.unique(np.frombuffer(records, dtype=np.uint32).reshape(-1, 2), axis=0)
    complete_ids = np.asarray(sorted(complete_labels), dtype=np.int64)
    if not len(complete_ids):
        raise ValueError("No complete EC evidence on training proteins")
    complete_map = np.full(len(label_names), -1, dtype=np.int32)
    complete_map[complete_ids] = np.arange(len(complete_ids))
    complete_rows = records[complete_map[records[:, 1]] >= 0].copy()
    complete_rows[:, 1] = complete_map[complete_rows[:, 1]]
    pcount = len(lookup.ids)
    protein_ec = np.full(pcount, -1, dtype=np.int32)
    ec_degree = np.bincount(complete_rows[:, 0], minlength=pcount)
    protein_ec[complete_rows[:, 0]] = complete_rows[:, 1]
    protein_ec[ec_degree != 1] = -1
    @lru_cache(maxsize=65536)
    def compatible(label_code, ec_code):
        return ec_annotation_matches_complete(label_names[label_code], label_names[complete_ids[ec_code]])
    conflicting = np.zeros(pcount, dtype=bool)
    for p, label in records:
        ec = int(protein_ec[p])
        if ec >= 0 and not compatible(int(label), ec):
            conflicting[p] = True
    protein_ec[conflicting] = -1
    prefixes = [".".join(label_names[code].split(".")[:prefix_depth]) for code in complete_ids]
    _, ec_prefix = np.unique(prefixes, return_inverse=True)
    # ALL complete labels remain in positive exclusions, including conflicts.
    pptr = np.concatenate(([0], np.cumsum(ec_degree))).astype(np.int64)
    qptr = np.concatenate(([0], np.cumsum(np.bincount(pairs[:, 0], minlength=num_queries)))).astype(np.int64)
    qecptr, qecs = [0], []
    for q in range(num_queries):
        proteins = pairs[qptr[q]:qptr[q + 1], 1]
        counts = ec_degree[proteins]
        total = int(counts.sum())
        if total:
            starts = np.repeat(pptr[proteins], counts)
            shifts = np.repeat(np.cumsum(counts) - counts, counts)
            ecs = np.unique(complete_rows[starts + np.arange(total) - shifts, 1])
        else:
            ecs = np.empty(0, dtype=np.uint32)
        qecs.append(ecs)
        qecptr.append(qecptr[-1] + len(ecs))
    return protein_ec, ec_prefix, np.asarray(qecptr, dtype=np.int64), np.concatenate(qecs)


def load_eligibility(path, lookup):
    result = np.zeros((len(lookup.ids), 2), dtype=bool)
    seen = np.zeros(len(lookup.ids), dtype=bool)
    columns = ("ec_negative_candidate_eligible", "biological_negative_candidate_eligible")
    for chunk in chunks(rows(path, ("protein_id", *columns))):
        indices = lookup.many([row["protein_id"] for row in chunk])
        for p, row in zip(indices, chunk):
            if any(row[col] not in {"0", "1"} for col in columns):
                raise ValueError("Eligibility flags must be exactly 0 or 1")
            if p < 0:
                continue
            flags = tuple(row[col] == "1" for col in columns)
            if seen[p] and tuple(result[p]) != flags:
                raise ValueError("Conflicting duplicate eligibility flags")
            seen[p], result[p] = True, flags
    return result


def load_biofp_bits(path, lookup, scratch):
    with np.load(path, allow_pickle=False) as payload:
        if _v2_metadata(payload, Path(path)) != "train":
            raise ValueError("Indexed training requires explicit train-only v2 BioFP annotations")
        names = {"ids"} | {f"{family}_{field}" for family in ("mechanism", "cofactor", "native_cofactor", "reaction_cofactor") for field in ("targets", "mask", "confidence")}
        names |= {key for key in payload.files if key.endswith("_denominator") and key.split("_denominator")[0] in ("cofactor", "native_cofactor", "reaction_cofactor")}
        if not names.issubset(payload.files):
            raise ValueError("Missing required v2 target/mask/confidence arrays")
    # Explicit member names, no extractall/path traversal; scratch is temporary.
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for name in sorted(names):
            target = scratch / f"biofp_{name}.npy"
            with archive.open(f"{name}.npy") as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest, length=1024 * 1024)
            arrays[name] = np.load(target, mmap_mode="r", allow_pickle=False)
    ids = arrays["ids"]
    if ids.ndim != 1 or ids.dtype.kind != "U":
        raise ValueError("BioFP IDs must be Unicode vector")
    for name, value in arrays.items():
        if name == "ids":
            continue
        width = 8 if name.startswith("mechanism_") else 32
        expected = (len(ids),) if name.endswith("_denominator") else (len(ids), width)
        if value.shape != expected:
            raise ValueError(f"BioFP {name} has shape {value.shape}, expected {expected}")
    # Chunked matching does not construct a 6M-entry profile dictionary.
    result = np.zeros((len(lookup.ids), 3), dtype=np.uint32)
    found = np.zeros(len(lookup.ids), dtype=bool)
    for start in range(0, len(ids), CHUNK):
        stop = min(len(ids), start + CHUNK)
        chunk = {name: value[start:stop] for name, value in arrays.items()}
        _validate_v2_arrays(chunk, Path(path))
        mapped = lookup.many(ids[start:stop])
        valid = mapped >= 0
        target_rows = mapped[valid]
        if len(np.unique(target_rows)) != len(target_rows) or np.any(found[target_rows]):
            raise ValueError("Duplicate canonical training BioFP IDs")
        found[target_rows] = True
        for column, family, width in ((0, "mechanism", 8), (1, "native_cofactor", 31), (2, "reaction_cofactor", 31)):
            bits = np.left_shift(np.uint32(1), np.arange(width, dtype=np.uint32))
            result[target_rows, column] = chunk[f"{family}_targets"][valid, :width].astype(np.uint32) @ bits
        if start // 1_000_000 != stop // 1_000_000:
            print(f"Validated BioFP annotations: {stop:,}/{len(ids):,}", flush=True)
    if not found.all():
        raise ValueError(f"Train BioFP table omits {int((~found).sum())} training proteins")
    return result


def source_record(path):
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def build(*, train_pairs, train_reactions, ec_labels, biofp_targets, candidate_eligibility,
          output_dir, exclude_pairs=(), ec_prefix_depth=2, min_biofp_similarity=0.5):
    if ec_prefix_depth not in {1, 2, 3}:
        raise ValueError("ec_prefix_depth must be 1, 2 or 3")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Refusing to overwrite an existing index")
    sources = [Path(value) for value in (train_pairs, train_reactions, ec_labels, biofp_targets, candidate_eligibility, *exclude_pairs)]
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in sources]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="indexed-pairs-", dir=output_dir.parent) as temp:
        scratch = Path(temp)
        qids, pids, pairs = load_graph(train_pairs, scratch)
        lookup = ProteinLookup(pids)
        print(f"Graph: {len(pairs):,} positive pairs, {len(qids):,} reactions, {len(pids):,} train proteins", flush=True)
        protein_ec, ec_prefix, qecptr, qec = load_ec(ec_labels, lookup, pairs, len(qids), ec_prefix_depth)
        eligibility = load_eligibility(candidate_eligibility, lookup)
        bits = load_biofp_bits(biofp_targets, lookup, scratch)
        eligibility[:, 0] &= protein_ec >= 0
        eligibility[:, 1] &= eligibility[:, 0] & (bits[:, 0] != 0)
        print(f"Eligible train candidates: random={int(eligibility[:, 0].sum()):,}, biological={int(eligibility[:, 1].sum()):,}", flush=True)
        signatures = {}
        query_set = set(qids)
        for row in rows(train_reactions, ("reaction_id", "reaction_smiles")):
            qid = row["reaction_id"].strip()
            if qid in query_set:
                signature = reaction_signature(row["reaction_smiles"], qid)
                if signatures.setdefault(qid, signature) != signature:
                    raise ValueError("Reaction has conflicting signatures")
        if len(signatures) != len(qids):
            raise ValueError("Training reaction table omits graph reactions")
        _, query_signature = np.unique([signatures[q] for q in qids], return_inverse=True)
        qlookup = {str(q): i for i, q in enumerate(qids)}
        exclusions = array("I")
        for path in exclude_pairs:
            for chunk in chunks(rows(path, ("reaction_id", "protein_id"))):
                ps = lookup.many([row["protein_id"] for row in chunk])
                for row, p in zip(chunk, ps):
                    q = qlookup.get(row["reaction_id"].strip())
                    if q is not None and p >= 0:
                        exclusions.extend((q, int(p)))
        provenance = dict(pair_scope="train", annotation_semantics=SEMANTICS,
                          ec_prefix_depth=ec_prefix_depth,
                          source_files=[source_record(p) for p in sources],
                          exclusion_policy="Exclusion inputs prohibit edges only; no positive/profile transfer")
        after = [(p.stat().st_size, p.stat().st_mtime_ns) for p in sources]
        if before != after:
            raise ValueError("Input changed while building indexed pairs")
        return write_indexed_pairs(
            output_dir, query_ids=qids, protein_ids=pids, pairs=pairs,
            protein_ec=protein_ec, ec_prefix=ec_prefix, query_ec_indptr=qecptr, query_ec=qec,
            mechanism_bits=bits[:, 0], native_cofactor_bits=bits[:, 1], reaction_cofactor_bits=bits[:, 2],
            ec_eligible=eligibility[:, 0], biological_eligible=eligibility[:, 1],
            query_signature=query_signature, provenance=provenance,
            excluded_pairs=np.frombuffer(exclusions, dtype=np.uint32).reshape(-1, 2),
            min_biofp_similarity=min_biofp_similarity,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-pairs", "train-reactions", "ec-labels", "biofp-targets", "candidate-eligibility", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--exclude-pairs", type=Path, action="append", default=[])
    parser.add_argument("--ec-prefix-depth", type=int, default=2)
    parser.add_argument("--min-biofp-similarity", type=float, default=0.5)
    args = parser.parse_args()
    manifest = build(**vars(args))
    print(json.dumps({key: manifest[key] for key in ("num_pairs", "num_queries", "num_proteins")}, indent=2))


if __name__ == "__main__":
    main()
