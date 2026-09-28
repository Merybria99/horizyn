#!/usr/bin/env python3
"""Reuse cached native labels and reaggregate only TRAIN reaction-associated evidence.

Never relabel an unsplit NPZ as train-only. Native evidence is copied from the
existing cache; weak evidence is reconstructed from the already annotated
reaction lookup and the representative's OWN training associations. No native
UniProt table, archive, cluster membership table or model is rescanned.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_horizyn1_circe_v2_labels_v2 import SCHEMA, REACTION_SCHEMA, ANNOTATION_SEMANTICS
from scripts.horizyn1_circe_v2_pipeline import atomic_json, csv_rows, signature
from horizyn.capability.biological_targets import MECHANISM_LABELS
from horizyn.capability.cofactor_vocabulary_v2 import COFACTOR_LABELS, COFACTOR_VOCABULARY_VERSION, UNKNOWN_INDEX


def hashed_signature(path):
    before = signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if signature(path) != before:
        raise ValueError(f"Input changed while hashing: {path}")
    return {**before, "sha256": digest.hexdigest()}


def project(source, prepared, output):
    source, prepared, output = map(lambda p: Path(p).resolve(), (source, prepared, output))
    cache = source / "annotations/circe_v2_cofactor_v2"
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Label outputs must be separate from source annotations")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite existing labels")
    split_path = prepared / "reaction_split.json"
    split = json.loads(split_path.read_text())
    train_reactions = {r for r, s in split.items() if s == "train"}
    if not train_reactions or set(split.values()) != {"train", "validation", "test"}:
        raise ValueError("Explicit reaction-held-out train/validation/test assignment required")
    cached = json.loads((cache / "label_manifest.json").read_text())
    if cached.get("schema_version") != SCHEMA or cached.get("pair_scope") != "unsplit_inventory":
        raise ValueError("Expected the completed versioned unsplit label inventory")
    families = {"mechanism": list(MECHANISM_LABELS), "cofactor": list(COFACTOR_LABELS)}
    if cached.get("families") != families or cached.get("cofactor_unknown_index") != UNKNOWN_INDEX:
        raise ValueError("Cached annotation vocabulary differs from the CIRCE-v2 contract")
    cached_paths = {name: cache / file for name, file in (
        ("biofp", "enzyme_biofp_targets.npz"), ("ec", "enzyme_ec_labels.csv"),
        ("eligibility", "candidate_eligibility.csv"), ("vocab", "enzyme_biofp_vocab.json"))}
    ctime_changes = []
    for name, path in cached_paths.items():
        recorded = cached.get("output_signatures", {}).get(name)
        info = path.stat()
        if (not isinstance(recorded, list) or len(recorded) != 5
                or [info.st_ino, info.st_size, info.st_mtime_ns] != recorded[1:4]
                or path.with_name(path.name + ".partial").exists()):
            raise ValueError(f"Cached label inventory changed or is incomplete: {path}")
        if info.st_ctime_ns != recorded[4]:
            ctime_changes.append(str(path))
    if ctime_changes:
        print("Cache ctime changed since export; binding current native-label bytes with fresh SHA256. "
              "This is not proof of historical byte identity or a refresh of the old audit.", flush=True)
    inputs = [*cached_paths.values(), cache / "label_manifest.json", split_path,
              prepared / "preparation_manifest.json", prepared / "train_own_raw_associations.csv",
              source / "annotations/cofactor_v2/reaction_annotations.json"]
    before = [signature(p) for p in inputs]
    sources = [hashed_signature(p) for p in inputs]
    lookup = json.loads(inputs[-1].read_text())
    if (lookup.get("schema_version") != REACTION_SCHEMA or lookup.get("families") != families
            or lookup.get("label_semantics") != "positive_only_unknown_not_negative"
            or lookup.get("cofactor_vocabulary_version") != COFACTOR_VOCABULARY_VERSION):
        raise ValueError("Cached reaction descriptor schema/vocabulary mismatch")
    features = {}
    for rid in train_reactions:
        row = lookup["reactions"].get(rid)
        if row is None:
            continue  # No available evidence remains unknown, never a negative label.
        if row.get("evidence_type") != "reaction_associated_descriptor" or row.get("chemistry_match") != "exact":
            raise ValueError(f"Wrong cached reaction evidence role: {rid}")
        m, c = row["mechanism"], row["cofactor"]
        if (len(set(m)) != len(m) or len(set(c)) != len(c) or "unknown" in c
                or set(m) - set(MECHANISM_LABELS) or set(c) - set(COFACTOR_LABELS)):
            raise ValueError(f"Invalid known reaction descriptor: {rid}")
        features[rid] = [MECHANISM_LABELS.index(x) for x in m], [COFACTOR_LABELS.index(x) for x in c]
    with np.load(cached_paths["biofp"], allow_pickle=False) as payload:
        from scripts.build_annotation_negative_pools import _v2_metadata
        if _v2_metadata(payload, cached_paths["biofp"]) != "unsplit_inventory":
            raise ValueError("Cached NPZ scope differs from its manifest")
        if list(payload["cofactor_labels"]) != list(COFACTOR_LABELS):
            raise ValueError("Cached NPZ label order differs")
        ids = payload["ids"]
        if ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) != cached["representative_count"]:
            raise ValueError("Invalid cached representative IDs")
        order = np.argsort(ids)
        sorted_ids = ids[order]
        if np.any(sorted_ids[1:] == sorted_ids[:-1]):
            raise ValueError("Duplicate cached representative IDs")
        native = payload["native_cofactor_targets"]
        if native.shape != (len(ids), len(COFACTOR_LABELS)) or not np.isin(native, (0, 1)).all():
            raise ValueError("Invalid native cofactor targets")
        native = native.astype(bool)
        if not np.array_equal(native[:, UNKNOWN_INDEX], ~native[:, :UNKNOWN_INDEX].any(axis=1)):
            raise ValueError("Invalid native unknown status")
        native_status = {key: payload[key] for key in ("native_cofactor_has_annotation", "native_cofactor_has_unmapped")}
        if any(value.shape != (len(ids),) or value.dtype != np.dtype(bool) for value in native_status.values()):
            raise ValueError("Invalid native annotation-status arrays")

    def positions(values):
        # Do not cast to the cache's fixed string width: that could truncate an
        # unknown longer accession into an existing ID and silently relabel it.
        values = np.asarray(values)
        loc = np.searchsorted(sorted_ids, values)
        if np.any(loc >= len(ids)) or np.any(sorted_ids[np.minimum(loc, len(ids) - 1)] != values):
            raise ValueError("Annotation projection references an unknown representative")
        return order[loc]

    mechanism = np.zeros((len(ids), len(MECHANISM_LABELS)), dtype=bool)
    reaction = np.zeros_like(native)
    batch, own_rows = [], 0
    def flush():
        if not batch:
            return
        mapped = positions([p for _, p in batch])
        for (rid, _), position in zip(batch, mapped):
            descriptors = features.get(rid)
            if descriptors is not None:
                mechanism[position, descriptors[0]] = True
                reaction[position, descriptors[1]] = True
        batch.clear()
    for own_rows, row in enumerate(csv_rows(prepared / "train_own_raw_associations.csv"), 1):
        rid = row["reaction_id"]
        if rid not in train_reactions:
            raise ValueError(f"Held-out reaction leaked into weak label associations: {rid}")
        batch.append((rid, row["protein_id"]))
        if len(batch) == 32768:
            flush()
        if own_rows % 1000000 == 0:
            print(f"Reaggregated cached training evidence: {own_rows:,} own associations", flush=True)
    flush()
    expected = json.loads((prepared / "preparation_manifest.json").read_text())["training_own_annotation_associations"]
    if own_rows != expected:
        raise ValueError("Training association count differs from preparation manifest")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached_paths["ec"], output / "enzyme_ec_labels.csv")
    with (output / "candidate_eligibility.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["protein_id", "ec_negative_candidate_eligible",
                  "biological_negative_candidate_eligible", "reason", "member_count", "members_with_ec"])
        writer.writeheader()
        batch = []
        eligibility_seen = np.zeros(len(ids), dtype=bool)
        def write_eligibility():
            if not batch:
                return
            mapped = positions([r["protein_id"] for r in batch])
            if len(np.unique(mapped)) != len(mapped) or eligibility_seen[mapped].any():
                raise ValueError("Duplicate cached native eligibility IDs")
            eligibility_seen[mapped] = True
            for row, position in zip(batch, mapped):
                if row["ec_negative_candidate_eligible"] not in {"0", "1"}:
                    raise ValueError("Invalid cached native EC eligibility flag")
                row["biological_negative_candidate_eligible"] = str(int(
                    row["ec_negative_candidate_eligible"] == "1" and mechanism[position].any()))
                writer.writerow(row)
            batch.clear()
        for row in csv_rows(cached_paths["eligibility"]):
            batch.append(row)
            if len(batch) == 32768:
                write_eligibility()
        write_eligibility()
        if not eligibility_seen.all():
            raise ValueError("Cached eligibility omits representative IDs")
    union = native | reaction
    for target in (reaction, union):
        target[:, UNKNOWN_INDEX] = ~target[:, :UNKNOWN_INDEX].any(axis=1)
    arrays = {"ids": ids, "cofactor_labels": np.asarray(COFACTOR_LABELS),
              "cofactor_unknown_index": np.asarray(UNKNOWN_INDEX), "pair_scope": np.asarray("train"),
              "cofactor_vocabulary_version": np.asarray(COFACTOR_VOCABULARY_VERSION),
              "annotation_semantics": np.asarray(ANNOTATION_SEMANTICS), **native_status}
    for family, target in (("mechanism", mechanism), ("native_cofactor", native),
                           ("reaction_cofactor", reaction), ("cofactor", union)):
        arrays[f"{family}_targets"] = target.astype(np.float32)
        arrays[f"{family}_mask"] = target
        confidence = target.astype(np.float32) * (1 if family == "native_cofactor" else .4)
        if family == "cofactor":
            confidence = np.where(native, 1., np.where(reaction, .4, 0.)).astype(np.float32)
        if family != "mechanism":
            confidence[:, UNKNOWN_INDEX] = 0
            arrays[f"{family}_denominator"] = target[:, :UNKNOWN_INDEX].any(axis=1).astype(np.float32)
        arrays[f"{family}_confidence"] = confidence
    with (output / "enzyme_biofp_targets.npz").open("wb") as out:
        np.savez_compressed(out, **arrays)
        out.flush()
        os.fsync(out.fileno())
    metadata = {"schema_version": SCHEMA, "families": families, "pair_scope": "train",
                "training_split_declared": True, "annotation_semantics": ANNOTATION_SEMANTICS,
                "cofactor_vocabulary_version": COFACTOR_VOCABULARY_VERSION,
                "cofactor_unknown_index": UNKNOWN_INDEX,
                "non_biological_classes": {"cofactor": ["unknown"]},
                "representative_count": len(ids), "training_own_association_rows": own_rows,
                "source_roles": {"native": "existing direct native labels reused without inference",
                    "reaction": "cached descriptors reaggregated using own TRAIN associations only"},
                "row_coverage": {name: int(arr.any(axis=1).sum()) for name, arr in (
                    ("mechanism", mechanism), ("native_cofactor", native[:, :UNKNOWN_INDEX]),
                    ("reaction_cofactor", reaction[:, :UNKNOWN_INDEX]), ("cofactor", union[:, :UNKNOWN_INDEX]))},
                "current_input_sha256": sources, "historical_cache_ctime_changes": ctime_changes,
                "provenance_limit": "Fresh binding of current cached native evidence; no claim of historical byte equality when ctime changed.",
                "unknown_policy": "unknown is missing annotation status with zero confidence, never a biological negative"}
    atomic_json(output / "enzyme_biofp_vocab.json", metadata)
    if before != [signature(p) for p in inputs]:
        raise ValueError("Label source changed during projection; no completion manifest")
    metadata["output_sha256"] = [hashed_signature(output / name) for name in (
        "enzyme_ec_labels.csv", "candidate_eligibility.csv", "enzyme_biofp_targets.npz", "enzyme_biofp_vocab.json")]
    atomic_json(output / "label_manifest.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prepared", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(project(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
