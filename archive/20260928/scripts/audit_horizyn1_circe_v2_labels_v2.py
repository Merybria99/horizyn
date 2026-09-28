#!/usr/bin/env python3
"""Independently verify CIRCE-v2 labels against native rows and own associations.

The audit streams sequence, membership, native annotation, pair and CSV files.
Only representative digests, identifiers, EC sets and compact Boolean profiles
remain in memory. No UniProt archive rescan or sequence-model inference occurs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from horizyn.capability.biological_targets import MECHANISM_LABELS, _labels, mechanism_groups
from horizyn.capability.cofactor_vocabulary_v2 import (
    COFACTOR_LABELS, COFACTOR_VOCABULARY_VERSION, UNKNOWN_INDEX, UNKNOWN_LABEL,
    cofactor_name_groups,
)


FAMILIES = {"mechanism": list(MECHANISM_LABELS), "cofactor": list(COFACTOR_LABELS)}
LABEL_SCHEMA = "horizyn1_circe_v2_labels_v2"
PROGRESS_INTERVAL = 1_000_000
LIMITATIONS = [
    "Passing proves consistency of exported labels, not experimental annotation truth.",
    "Mechanism and reaction-associated cofactors remain weak positive descriptors.",
    "Unobserved labels are unknown; positive-only masks do not establish absence.",
    "Archive authenticity and native annotations rely on the frozen extraction provenance; the archive is not rescanned.",
    "A declared training scope is preserved and checked for consistency, not independently proven leakage-free.",
    "Candidate eligibility is a conservative EC conflict guard, not proof that a sampled pair is biologically negative.",
]


def progress(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", file=sys.stderr, flush=True)


def signature(path: Path) -> list[int]:
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def signature_dict(path: Path) -> dict[str, int]:
    values = signature(path)
    return dict(zip(("device", "inode", "size", "mtime_ns", "ctime_ns"), values))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(encoding="utf-8", newline="")


def labels(value: str) -> set[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError("Expected a JSON list of label strings")
    if len(set(parsed)) != len(parsed) or any(not item or item != item.strip() for item in parsed):
        raise ValueError("Duplicate, blank, or non-normalized annotation labels")
    return set(parsed)


def ec_depth(ec: str) -> int:
    parts = ec.split(".")
    if len(parts) != 4:
        return 0
    depth = 0
    for part in parts:
        if not part.isdigit():
            break
        depth += 1
    return depth


def ec_leaf(values: set[str]) -> str | None:
    complete = {value for value in values if ec_depth(value) == 4}
    if len(complete) != 1:
        return None
    leaf = next(iter(complete))
    components = leaf.split(".")
    for value in values:
        parts = value.split(".")
        if len(parts) != 4 or any(part not in {"-", wanted} for part, wanted in zip(parts, components)):
            return None
    return leaf


class Incomplete(RuntimeError):
    pass


class Audit:
    def __init__(self, run_root: Path, label_dir: Path | None = None):
        self.root = run_root.resolve()
        self.label_dir = (label_dir or self.root / "annotations/circe_v2_cofactor_v2").resolve()
        self.paths = {
            "representatives": self.root / "clustered/proteins.fasta",
            "clusters": self.root / "clustered/clusters.tsv",
            "raw_fasta": self.root / "raw/raw_proteins.fasta",
            "raw_pairs": self.root / "raw/raw_pairs.tsv",
            "raw_reactions": self.root / "raw/raw_reactions.tsv",
            "raw_manifest": self.root / "raw/raw_manifest.json",
            "clustered_pairs": self.root / "clustered/pairs.tsv",
            "graph_audit": self.root / "logs/clustered_integrity_audit.json",
            "cluster_manifest": self.root / "clustered/clustered_manifest.json",
            "ec": self.label_dir / "enzyme_ec_labels.csv",
            "biofp": self.label_dir / "enzyme_biofp_targets.npz",
            "eligibility": self.label_dir / "candidate_eligibility.csv",
            "summary": self.label_dir / "cluster_annotation_summary.tsv.gz",
            "vocab": self.label_dir / "enzyme_biofp_vocab.json",
            "label_manifest": self.label_dir / "label_manifest.json",
        }
        self.checks: dict[str, dict[str, Any]] = {}
        self.counts: dict[str, Any] = {}
        self.snapshots: dict[str, list[int]] = {}
        self.incomplete = False

    def check(self, name: str, condition: bool, message: str = "") -> None:
        item = self.checks.setdefault(name, {"status": "passed", "error_count": 0, "examples": []})
        if not condition:
            item["status"] = "failed"
            item["error_count"] += 1
            if len(item["examples"]) < 5:
                item["examples"].append(message)

    def require(self, path: Path) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise Incomplete(f"Missing or empty completed artifact: {path}")
        if path.with_name(path.name + ".partial").exists():
            raise Incomplete(f"Partial counterpart exists for {path}")
        self.snapshots.setdefault(str(path.resolve()), signature(path))

    def json(self, path: Path) -> dict:
        self.require(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        return value

    def rows(self, path: Path, required: set[str], delimiter: str = "\t") -> Iterator[dict[str, str]]:
        with open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path} lacks columns {sorted(required)}")
            for number, row in enumerate(reader, 1):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed table row {number}: {path}")
                if number % PROGRESS_INTERVAL == 0:
                    progress(f"Auditing {path.name}: {number:,} rows")
                yield row

    def resolve_source(self, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def setup(self) -> tuple[dict, dict, dict, dict]:
        self.require(Path(__file__))
        manifest = self.json(self.paths["label_manifest"])
        if manifest.get("schema_version") != LABEL_SCHEMA:
            raise ValueError("Unsupported or incomplete label manifest schema")
        sources = manifest.get("sources", {})
        for key, field in (("native", "native"), ("reaction_lookup", "reaction_annotations"), ("associations", "association_pairs")):
            if not isinstance(sources.get(field), str):
                raise ValueError(f"Label manifest lacks source path {field}")
            self.paths[key] = self.resolve_source(sources[field])
        self.paths["native_manifest"] = self.resolve_source(sources["native_manifest"]) if sources.get("native_manifest") else self.root / "annotations/native_uniprot_manifest.json"
        self.paths["reaction_manifest"] = self.resolve_source(sources["reaction_manifest"]) if sources.get("reaction_manifest") else self.root / "annotations/cofactor_v2/reaction_annotations_manifest.json"
        for path in self.paths.values():
            self.require(path)
        graph = self.json(self.paths["graph_audit"])
        if graph.get("status") != "passed" or graph.get("stage") != "clustered":
            raise Incomplete("The complete clustered graph integrity audit has not passed")
        graph_keys = (("clustered_proteins", "representatives"), ("clusters", "clusters"),
                      ("raw_proteins", "raw_fasta"), ("raw_reactions", "raw_reactions"),
                      ("raw_pairs", "raw_pairs"), ("raw_manifest", "raw_manifest"),
                      ("clustered_pairs", "clustered_pairs"), ("clustered_manifest", "cluster_manifest"))
        audited_signatures = graph.get("artifact_signatures")
        if not isinstance(audited_signatures, dict):
            raise Incomplete("Graph audit lacks artifact signatures; rerun the full clustered graph audit")
        for name, key in graph_keys:
            self.check("graph_gate", graph.get("paths", {}).get(name) == str(self.paths[key]), f"Graph audit does not refer to the current {key}")
            if audited_signatures.get(str(self.paths[key].resolve())) != signature(self.paths[key]):
                self.incomplete = True
                self.check("graph_gate", False, f"Graph artifact changed after its passing audit: {key}")
        native = self.json(self.paths["native_manifest"])
        if native.get("status") != "complete" or native.get("schema_version") != "horizyn1_native_uniprot_annotations_v1":
            raise Incomplete("Native extraction completion is not established")
        native_output = native.get("output", {})
        self.check("native_completion", native_output.get("path") == str(self.paths["native"].resolve()), "Native output path differs")
        self.check("native_completion", native_output.get("signature") == signature_dict(self.paths["native"]), "Native output signature differs")
        self.check("native_completion", native_output.get("sha256") == sha256(self.paths["native"]), "Native output checksum differs")
        target_path = self.resolve_source(native.get("request", {}).get("target_fasta", ""))
        self.check("native_completion", target_path == self.paths["raw_fasta"], "Native extraction must target the complete raw FASTA")
        self.check("native_completion", native.get("input_signatures", {}).get("target_fasta") == signature_dict(self.paths["raw_fasta"]), "Native target FASTA changed")
        archive_name = native.get("request", {}).get("input_path")
        if not isinstance(archive_name, str):
            raise ValueError("Native completion lacks archive provenance")
        self.paths["archive"] = self.resolve_source(archive_name)
        self.require(self.paths["archive"])
        self.check("native_completion", native.get("input_signatures", {}).get("archive") == signature_dict(self.paths["archive"]), "Native archive signature changed")

        reaction_manifest = self.json(self.paths["reaction_manifest"])
        if reaction_manifest.get("status") != "complete":
            raise Incomplete("Reaction annotation completion is not established")
        lookup_meta = reaction_manifest.get("output", {})
        self.check("reaction_completion", lookup_meta.get("path") == str(self.paths["reaction_lookup"].resolve()), "Reaction lookup output path differs")
        self.check("reaction_completion", lookup_meta.get("sha256") == sha256(self.paths["reaction_lookup"]), "Reaction lookup checksum differs")
        for name, item in reaction_manifest.get("inputs", {}).items():
            path = self.resolve_source(item["path"])
            self.paths[f"reaction_input_{name}"] = path
            self.require(path)
            self.check("reaction_completion", item.get("sha256") == sha256(path), f"Reaction input checksum differs: {name}")
        for name, path_string in manifest.get("input_signatures", {}).items():
            path = Path(name)
            self.require(path)
            self.check("label_provenance", path_string == signature(path), f"Label input changed: {path}")
        expected_inputs = [self.paths[key] for key in ("representatives", "clusters", "native", "reaction_lookup", "associations")]
        self.check("label_provenance", all(str(path.resolve()) in manifest.get("input_signatures", {}) for path in expected_inputs), "Label manifest does not bind all required source inputs")
        output_map = {"ec": "ec", "biofp": "biofp", "eligibility": "eligibility", "clusters": "summary", "vocab": "vocab"}
        for output_key, key in output_map.items():
            self.check("label_provenance", manifest.get("output_signatures", {}).get(output_key) == signature(self.paths[key]), f"Label output signature differs: {key}")
            self.check("label_provenance", self.resolve_source(manifest.get("outputs", {}).get(output_key, "")) == self.paths[key], f"Label output path differs: {key}")
        exporter = PROJECT_ROOT / "scripts/build_horizyn1_circe_v2_labels_v2.py"
        self.require(exporter)
        self.check("label_provenance", manifest.get("implementation_sha256") == sha256(exporter), "Label exporter implementation changed")
        vocabulary_source = PROJECT_ROOT / "horizyn/capability/cofactor_vocabulary_v2.py"
        reaction_builder = PROJECT_ROOT / "scripts/build_horizyn1_reaction_annotations_v2.py"
        for source in (vocabulary_source, reaction_builder):
            self.check("label_provenance", str(source.resolve()) in manifest.get("input_signatures", {}), f"Missing versioned code provenance: {source.name}")
        self.check("reaction_completion", self.paths.get("reaction_input_cofactor_vocabulary") == vocabulary_source, "Reaction vocabulary provenance differs")
        self.check("reaction_completion", reaction_manifest.get("schema_version") == "horizyn1_circe_v2_reaction_annotations_v2", "Reaction schema differs")
        vocab = self.json(self.paths["vocab"])
        self.check("vocabulary", manifest.get("families") == FAMILIES and vocab.get("families") == FAMILIES, "CIRCE-v2 label vocabulary or order differs")
        for key, expected in {
            "cofactor_vocabulary_version": COFACTOR_VOCABULARY_VERSION,
            "cofactor_unknown_index": UNKNOWN_INDEX,
            "annotation_semantics": "positive_only_role_aware_v2",
            "positive_only": True,
            "non_biological_classes": {"cofactor": [UNKNOWN_LABEL]},
        }.items():
            self.check("unknown_semantics", manifest.get(key) == expected, f"Invalid annotation-status contract: {key}")
        self.check("unknown_semantics", set(manifest.get("source_roles", {})) == {"native_cofactor", "reaction_cofactor", "cofactor", "unknown"}, "Missing source-role documentation")
        scope = manifest.get("pair_scope")
        self.check("pair_scope", scope in {"train", "unsplit_inventory"}, "Unknown pair scope")
        self.check("pair_scope", manifest.get("training_split_declared") is (scope == "train"), "Training scope flag differs")
        self.check("pair_scope", vocab.get("pair_scope") == scope and vocab.get("training_split_declared") is (scope == "train"), "Vocab scope disagrees with label manifest")
        self.check("pair_scope", "positive_only" in manifest.get("mask_semantics", ""), "Positive-only missingness policy is not recorded")
        for key, value in vocab.items():
            self.check("vocab_manifest", manifest.get(key) == value, f"Metadata differs between vocab and label manifest: {key}")
        return manifest, native, graph, vocab

    def read_representatives(self) -> tuple[list[str], dict[str, int], list[bytes]]:
        ids: list[str] = []
        index: dict[str, int] = {}
        digests: list[bytes] = []
        current = None
        digest = hashlib.sha256()
        residues = total_residues = 0

        def finish() -> None:
            nonlocal total_residues
            if current is None:
                return
            if residues == 0:
                raise ValueError(f"Empty representative sequence: {current}")
            index[current] = len(ids)
            ids.append(current)
            digests.append(digest.digest())
            total_residues += residues

        with open_text(self.paths["representatives"]) as handle:
            for line in handle:
                if line.startswith(">"):
                    finish()
                    parts = line[1:].split()
                    if not parts or parts[0] in index:
                        raise ValueError("Empty or duplicate representative FASTA ID")
                    current = parts[0]
                    digest, residues = hashlib.sha256(), 0
                elif line.strip():
                    sequence = "".join(line.split())
                    if current is None or not sequence.isascii() or not sequence.isalpha():
                        raise ValueError("Malformed representative sequence")
                    digest.update(sequence.upper().encode("ascii"))
                    residues += len(sequence)
            finish()
        if not ids:
            raise ValueError("Empty representative FASTA")
        self.counts.update(representatives=len(ids), representative_residues=total_residues)
        return ids, index, digests

    def reaction_features(self) -> tuple[dict[str, str], dict[str, tuple[list[int], list[int]]]]:
        """Rejoin small cached chemistry independently of the lookup builder."""
        lookup = self.json(self.paths["reaction_lookup"])
        self.check("reaction_lookup", lookup.get("families") == FAMILIES, "Reaction lookup vocabulary/order differs")
        self.check("reaction_lookup", lookup.get("label_semantics") == "positive_only_unknown_not_negative", "Reaction lookup missingness semantics differ")
        self.check("reaction_lookup", lookup.get("schema_version") == "horizyn1_circe_v2_reaction_annotations_v2" and lookup.get("cofactor_vocabulary_version") == COFACTOR_VOCABULARY_VERSION, "Reaction schema/version differs")
        raw_chemistry: dict[str, str] = {}
        for row in self.rows(self.paths["raw_reactions"], {"reaction_id", "reaction_smiles"}):
            if row["reaction_id"] in raw_chemistry or not row["reaction_smiles"]:
                raise ValueError("Duplicate or empty raw reaction chemistry")
            raw_chemistry[row["reaction_id"]] = row["reaction_smiles"]
        bridge_path = self.paths.get("reaction_input_cached_reactions")
        cache_path = self.paths.get("reaction_input_cached_features")
        if bridge_path is None or cache_path is None:
            raise ValueError("Reaction annotation completion lacks cached chemistry provenance")
        if self.paths.get("reaction_input_raw_reactions") != self.paths["raw_reactions"]:
            raise ValueError("Reaction lookup was not built against this raw reaction set")
        cached_bridge: dict[str, str] = {}
        candidates: dict[str, set[str]] = defaultdict(set)
        for row in self.rows(bridge_path, {"reaction_id", "reaction_smiles", "source_reaction_ids"}, ","):
            cached_id = row["reaction_id"]
            chemistry = row["reaction_smiles"].strip()
            if cached_id in cached_bridge and cached_bridge[cached_id] != chemistry:
                raise ValueError("Conflicting cached reaction bridge chemistry")
            cached_bridge[cached_id] = chemistry
            for value in row["source_reaction_ids"].split("|"):
                match = re.fullmatch(r"(?:Rh_|RHEA:)(\d+)", value.strip())
                if match:
                    candidates[f"Rh_{match.group(1)}"].add(cached_id)
        cache = pd.read_parquet(cache_path)
        if "reaction_id" not in cache.columns:
            raise ValueError("Cached chemistry has no reaction IDs")
        cache_by_id: dict[str, list[tuple[str, tuple[str, ...], tuple[str, ...]]]] = defaultdict(list)
        for row in cache.to_dict("records"):
            chemistry = ""
            for column in ("raw_reaction_smiles", "reaction_smiles", "canonical_reaction_smiles"):
                value = row.get(column)
                if isinstance(value, str) and value.strip():
                    chemistry = value.strip()
                    break
            mechanism = mechanism_groups(row)
            cofactor = cofactor_name_groups(_labels(row.get("core_cofactor_labels")) | _labels(row.get("cofactor_labels")))
            cache_by_id[str(row["reaction_id"])].append((chemistry, tuple(label for label in MECHANISM_LABELS if label in mechanism), tuple(label for label in COFACTOR_LABELS if label in cofactor)))
        expected = {}
        for reaction_id, chemistry in raw_chemistry.items():
            candidate_ids = {"rxn_" + reaction_id[3:]} if reaction_id.startswith("Em_") else candidates.get(reaction_id, set()) if reaction_id.startswith("Rh_") else set()
            matches = [(cached_id, mechanism, cofactor)
                       for cached_id in candidate_ids if cached_bridge.get(cached_id) == chemistry
                       for cached_chemistry, mechanism, cofactor in cache_by_id.get(cached_id, [])
                       if cached_chemistry == chemistry]
            descriptors = {(mechanism, cofactor) for _, mechanism, cofactor in matches}
            if len(descriptors) == 1:
                mechanism, cofactor = next(iter(descriptors))
                if mechanism or cofactor:
                    expected[reaction_id] = (mechanism, cofactor, sorted({cached_id for cached_id, _, _ in matches}))
        self.check("reaction_lookup", set(lookup.get("reactions", {})) == set(expected), "Reaction lookup omits positive exact matches or includes mismatched/ambiguous/unknown chemistry")
        for reaction_id, row in lookup.get("reactions", {}).items():
            if reaction_id not in expected:
                self.check("reaction_chemistry", False, f"No unambiguous exact cached chemistry: {reaction_id}")
                continue
            mechanism, cofactor, source_ids = expected[reaction_id]
            self.check("reaction_chemistry", row.get("chemistry_match") == "exact" and row.get("source_reaction_id") == source_ids[0] and row.get("source_reaction_ids") == source_ids, f"Reaction source mapping differs: {reaction_id}")
            self.check("reaction_lookup", row.get("evidence_type") == "reaction_associated_descriptor", f"Reaction evidence type differs: {reaction_id}")
            self.check("reaction_lookup", row.get("mechanism") == list(mechanism) and row.get("cofactor") == list(cofactor), f"Cached descriptors differ: {reaction_id}")
        self.counts["reaction_descriptors"] = len(expected)
        return raw_chemistry, {
            reaction_id: ([FAMILIES["mechanism"].index(value) for value in mechanism], [FAMILIES["cofactor"].index(value) for value in cofactor])
            for reaction_id, (mechanism, cofactor, _) in expected.items()
        }

    def verify(self, manifest: dict, native_manifest: dict, graph: dict) -> None:
        ids, index, digests = self.read_representatives()
        n = len(ids)
        members: dict[str, int] = {}
        member_counts = np.zeros(n, dtype=np.uint32)
        member_ec_counts = np.zeros(n, dtype=np.uint32)
        with open_text(self.paths["clusters"]) as handle:
            for number, line in enumerate(handle, 1):
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 2 or fields[0] not in index or not fields[1] or fields[1] in members:
                    raise ValueError(f"Invalid or duplicate cluster membership at row {number}")
                position = index[fields[0]]
                members[fields[1]] = 2 * position  # Low bit marks a native row seen.
                member_counts[position] += 1
        self.check("cluster_membership", all(members.get(value) == 2 * position for value, position in index.items()), "Representative is not its own member")
        self.counts["raw_members"] = len(members)
        self.check("graph_gate", graph.get("observed_counts", {}).get("clustered_proteins") == n, "Graph audit representative count differs")
        self.check("graph_gate", graph.get("observed_counts", {}).get("clustered_members") == len(members), "Graph audit member count differs")
        cluster_manifest = self.json(self.paths["cluster_manifest"])
        self.check("graph_gate", cluster_manifest.get("clustered_proteins") == n and cluster_manifest.get("clustered_members") == len(members), "Cluster completion counts differ")

        own_ec: dict[int, set[str]] = {}
        member_ec: dict[int, set[str]] = defaultdict(set)
        native_cofactor = np.zeros((n, len(COFACTOR_LABELS)), dtype=bool)
        mechanism = np.zeros((n, len(MECHANISM_LABELS)), dtype=bool)
        reaction_cofactor = np.zeros((n, len(COFACTOR_LABELS)), dtype=bool)
        native_counts: Counter[str] = Counter()
        representative_counts: Counter[str] = Counter()
        native_has_annotation = np.zeros(n, dtype=bool)
        native_has_unmapped = np.zeros(n, dtype=bool)
        annotation_counts = Counter()
        unmapped_all, unmapped_reps = Counter(), Counter()
        name_cache = {}
        native_rows = 0
        for row in self.rows(self.paths["native"], {"protein_id", "annotation_status", "ec_numbers", "cofactor_names", "sequence_sha256"}):
            native_rows += 1
            protein_id = row["protein_id"]
            encoded = members.get(protein_id)
            if encoded is None or encoded & 1:
                raise ValueError(f"Unknown or duplicate native ID: {protein_id}")
            position = encoded // 2
            members[protein_id] = encoded | 1
            status = row["annotation_status"]
            if status not in {"matched", "matched_unannotated", "sequence_mismatch", "unresolved", "ambiguous_accession"}:
                raise ValueError(f"Unknown native status: {status}")
            ecs = {sys.intern(value) for value in labels(row["ec_numbers"])}
            # Cache by source string, but independently recompute mapping from raw names.
            cached = name_cache.get(row["cofactor_names"])
            if cached is None:
                cofactors = labels(row["cofactor_names"])
                groups = cofactor_name_groups(cofactors)
                unmapped = {name for name in cofactors if not cofactor_name_groups([name])}
                cached = name_cache[row["cofactor_names"]] = (cofactors, groups, unmapped)
            cofactors, groups, unmapped = cached
            annotation_counts["all_members_with_annotation"] += bool(cofactors)
            annotation_counts["all_members_with_unmapped"] += bool(unmapped)
            unmapped_all.update(unmapped)
            self.check("native_missingness", status == "matched" or not (ecs or cofactors), f"Unknown/unannotated row carries labels: {protein_id}")
            self.check("native_missingness", status != "matched" or bool(ecs or cofactors), f"Matched row has no native labels: {protein_id}")
            native_counts[status] += 1
            if ecs:
                member_ec[position].update(ecs)
                member_ec_counts[position] += 1
            if protein_id in index:
                representative_counts[status] += 1
                self.check("representative_native_sequence", row["sequence_sha256"] == digests[position].hex(), f"Native target hash differs for {protein_id}")
                if ecs:
                    own_ec[position] = ecs
                native_has_annotation[position] = bool(cofactors)
                native_has_unmapped[position] = bool(unmapped)
                annotation_counts["representatives_with_annotation"] += bool(cofactors)
                annotation_counts["representatives_with_unmapped"] += bool(unmapped)
                unmapped_reps.update(unmapped)
                for family in groups:
                    native_cofactor[position, FAMILIES["cofactor"].index(family)] = True
        self.check("native_coverage", native_rows == len(members) and all(value & 1 for value in members.values()), "Native rows do not cover every raw member exactly once")
        self.check("native_coverage", sum(representative_counts.values()) == n, "A representative lacks an explicit native row")
        self.check("native_completion", native_manifest.get("counts", {}).get("output_rows") == native_rows, "Native manifest row count differs")
        self.check("native_completion", native_manifest.get("annotation_status_counts") == dict(native_counts), "Native manifest status counts differ")
        self.counts["native_rows"] = native_rows

        expected_ec = sum(map(len, own_ec.values()))
        observed_ec = 0
        ec_seen: set[tuple[int, str]] = set()
        for row in self.rows(self.paths["ec"], {"protein_id", "ec_number", "known_depth"}, ","):
            observed_ec += 1
            protein_id, ec = row["protein_id"], row["ec_number"]
            position = index.get(protein_id)
            key = (position, ec)
            self.check("direct_ec", position is not None and ec in own_ec.get(position, set()), f"Non-native or member-inherited direct EC: {protein_id}/{ec}")
            self.check("direct_ec", key not in ec_seen, f"Duplicate direct EC: {protein_id}/{ec}")
            self.check("direct_ec", row["known_depth"] == str(ec_depth(ec)), f"Incorrect EC depth: {protein_id}/{ec}")
            ec_seen.add(key)
        self.check("direct_ec", observed_ec == expected_ec, "Direct EC export omitted or added rows")
        self.counts["direct_ec_rows"] = observed_ec
        del ec_seen, digests

        raw_chemistry, features = self.reaction_features()
        pairs = own_pairs = own_annotated_pairs = 0
        pair_path = self.paths["associations"]
        delimiter = "," if pair_path.name.endswith((".csv", ".csv.gz")) else "\t"
        for row in self.rows(pair_path, {"reaction_id", "protein_id"}, delimiter):
            pairs += 1
            protein_id, reaction_id = row["protein_id"], row["reaction_id"]
            if protein_id not in members or reaction_id not in raw_chemistry:
                raise ValueError(f"Unknown source association: {protein_id}/{reaction_id}")
            position = index.get(protein_id)
            if position is None:
                continue
            own_pairs += 1
            if reaction_id in features:
                own_annotated_pairs += 1
                mechanisms, cofactors = features[reaction_id]
                mechanism[position, mechanisms] = True
                reaction_cofactor[position, cofactors] = True
        self.counts.update(association_rows=pairs, representative_own_association_rows=own_pairs,
                           representative_own_annotated_association_rows=own_annotated_pairs)

        cofactor = native_cofactor | reaction_cofactor
        for array in (native_cofactor, reaction_cofactor, cofactor):
            array[:, UNKNOWN_INDEX] = ~array[:, :UNKNOWN_INDEX].any(axis=1)
        expected_arrays = {"mechanism": mechanism, "native_cofactor": native_cofactor,
                           "reaction_cofactor": reaction_cofactor, "cofactor": cofactor}
        with np.load(self.paths["biofp"], allow_pickle=False) as data:
            for key, expected in {
                "cofactor_labels": np.asarray(COFACTOR_LABELS),
                "cofactor_unknown_index": np.asarray(UNKNOWN_INDEX, dtype=np.int64),
                "cofactor_vocabulary_version": np.asarray(COFACTOR_VOCABULARY_VERSION),
                "annotation_semantics": np.asarray("positive_only_role_aware_v2"),
                "pair_scope": np.asarray(manifest["pair_scope"]),
            }.items():
                actual = data[key]
                self.check("npz_metadata", actual.shape == expected.shape and actual.dtype.kind == expected.dtype.kind and np.array_equal(actual, expected), f"Invalid embedded contract: {key}")
            for key, expected in (
                ("native_cofactor_has_annotation", native_has_annotation),
                ("native_cofactor_has_unmapped", native_has_unmapped),
            ):
                actual = data[key]
                self.check("native_annotation_status", actual.dtype == np.bool_ and np.array_equal(actual, expected), f"Incorrect native status flags: {key}")
            array_ids = data["ids"]
            self.check("npz_ids", array_ids.ndim == 1 and array_ids.dtype.kind in {"U", "S"}, "NPZ IDs are not a one-dimensional string array")
            self.check("npz_ids", array_ids.shape == (n,) and np.array_equal(array_ids, np.asarray(ids)), "NPZ IDs/order differ from representative FASTA")
            for family, expected in expected_arrays.items():
                target, mask = data[f"{family}_targets"], data[f"{family}_mask"]
                correct_shape = target.shape == expected.shape and mask.shape == expected.shape
                self.check("array_shapes", correct_shape, f"Wrong dimensions for {family}")
                self.check("array_dtypes", target.dtype == np.float32 and mask.dtype == np.bool_, f"Wrong dtypes for {family}")
                self.check("array_values", np.isfinite(target).all() and ((target >= 0) & (target <= 1)).all(), f"Non-finite or out-of-range {family} target")
                if correct_shape:
                    if family != "mechanism":
                        self.check("unknown_exclusivity", np.array_equal(target[:, UNKNOWN_INDEX] > 0, ~np.any(target[:, :UNKNOWN_INDEX] > 0, axis=1)), f"Known evidence and unknown must be exclusive: {family}")
                    self.check("positive_only_masks", np.array_equal(mask, target > 0) and np.all(target[~mask.astype(bool)] == 0), f"Unknown/negative mask policy violated: {family}")
                    self.check("source_profile_union", np.array_equal(target, expected.astype(np.float32)) and np.array_equal(mask, expected), f"{family} differs from native/own-association positive union")
                if family != "mechanism":
                    denominator = data[f"{family}_denominator"]
                    wanted = expected[:, :UNKNOWN_INDEX].any(axis=1).astype(np.float32)
                    self.check("known_only_training_eligibility", denominator.dtype == np.float32 and np.array_equal(denominator, wanted), f"Unknown-only rows must not enter biological pretraining: {family}")
            for family in expected_arrays:
                confidence = data[f"{family}_confidence"]
                if family == "cofactor":
                    expected = np.where(native_cofactor, 1.0, np.where(reaction_cofactor, 0.4, 0.0)).astype(np.float32)
                else:
                    weight = 1.0 if family == "native_cofactor" else 0.4
                    expected = expected_arrays[family].astype(np.float32) * np.float32(weight)
                if family != "mechanism":
                    expected[:, UNKNOWN_INDEX] = 0
                self.check("confidence", confidence.dtype == np.float32 and confidence.shape == expected.shape and np.isfinite(confidence).all() and np.array_equal(confidence, expected), f"Incorrect native/weak confidence: {family}")

        eligibility_counts: Counter[str] = Counter()
        for key, required, delimiter in (
            ("summary", {"protein_id", "member_count", "members_with_ec", "member_ec_evidence", "own_ec_labels"}, "\t"),
            ("eligibility", {"protein_id", "member_count", "members_with_ec", "ec_negative_candidate_eligible", "biological_negative_candidate_eligible", "reason"}, ","),
        ):
            seen = np.zeros(n, dtype=bool)
            for row in self.rows(self.paths[key], required, delimiter):
                position = index.get(row["protein_id"])
                if position is None or seen[position]:
                    raise ValueError(f"Unknown/duplicate {key} ID: {row['protein_id']}")
                seen[position] = True
                own, observed = own_ec.get(position, set()), member_ec.get(position, set())
                self.check("cluster_ec_summary", row["member_count"] == str(member_counts[position]) and row["members_with_ec"] == str(member_ec_counts[position]), f"Incorrect member counts: {row['protein_id']}")
                if key == "summary":
                    self.check("cluster_ec_summary", labels(row["member_ec_evidence"]) == observed and labels(row["own_ec_labels"]) == own, f"Incorrect own/member EC evidence: {row['protein_id']}")
                else:
                    own_leaf = ec_leaf(own)
                    eligible = own_leaf is not None and ec_leaf(observed) == own_leaf
                    biological = eligible and bool(mechanism[position].any())
                    reason = "eligible" if eligible else "missing_or_ambiguous_direct_ec" if own_leaf is None else "cluster_member_ec_conflict"
                    self.check("candidate_eligibility", row["ec_negative_candidate_eligible"] == str(int(eligible)) and row["biological_negative_candidate_eligible"] == str(int(biological)) and row["reason"] == reason, f"Eligibility ignores missingness or member conflict: {row['protein_id']}")
                    eligibility_counts[reason] += 1
                    eligibility_counts["ec_eligible"] += int(eligible)
                    eligibility_counts["biological_eligible"] += int(biological)
            self.check("row_coverage", seen.all(), f"Missing representative {key} row")
        expected_metadata = {
            "representative_count": n, "raw_member_count": len(members),
            "association_rows": pairs, "representative_own_association_rows": own_pairs,
            "representative_own_annotated_association_rows": own_annotated_pairs,
            "native_annotation_status": dict(native_counts), "representative_annotation_status": dict(representative_counts),
            "negative_candidate_eligibility": dict(eligibility_counts),
            "row_coverage": {**{family: int((array if family == "mechanism" else array[:, :UNKNOWN_INDEX]).any(axis=1).sum()) for family, array in expected_arrays.items()}, "direct_ec": len(own_ec)},
            "unknown_counts": {family: int(expected_arrays[family][:, UNKNOWN_INDEX].sum()) for family in ("native_cofactor", "reaction_cofactor", "cofactor")},
            "native_cofactor_annotation_counts": dict(annotation_counts),
            "unmapped_native_cofactor_name_counts": {"all_members": dict(unmapped_all), "representatives": dict(unmapped_reps)},
        }
        for key, value in expected_metadata.items():
            self.check("manifest_counts", manifest.get(key) == value, f"Label manifest misreports {key}")
        self.counts["row_coverage"] = expected_metadata["row_coverage"]
        self.counts["pair_scope"] = manifest["pair_scope"]
        self.counts["unknown_counts"] = expected_metadata["unknown_counts"]
        self.counts["native_cofactor_annotation_counts"] = expected_metadata["native_cofactor_annotation_counts"]

    def run(self) -> dict[str, Any]:
        started = datetime.now(timezone.utc).isoformat()
        try:
            manifest, native, graph, _ = self.setup()
            self.verify(manifest, native, graph)
        except Incomplete as error:
            self.incomplete = True
            self.check("completed_inputs", False, str(error))
        except (ValueError, KeyError, TypeError, IndexError, OSError, EOFError, zipfile.BadZipFile, zlib.error) as error:
            self.check("artifact_reading", False, f"{type(error).__name__}: {error}")
        for name, before in self.snapshots.items():
            try:
                if signature(Path(name)) != before:
                    self.incomplete = True
                    self.check("input_stability", False, f"Artifact changed: {name}")
                if Path(name + ".partial").exists():
                    self.incomplete = True
                    self.check("input_stability", False, f"Partial counterpart appeared: {name}")
            except OSError:
                self.incomplete = True
                self.check("input_stability", False, f"Artifact disappeared: {name}")
        self.checks.setdefault("input_stability", {"status": "passed", "error_count": 0, "examples": []})
        errors = [{"check": name, **value} for name, value in self.checks.items() if value["status"] != "passed"]
        return {
            "schema_version": "horizyn1_circe_v2_label_audit_v2",
            "status": "incomplete" if self.incomplete else "failed" if errors else "passed",
            "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
            "observed_counts": self.counts, "checks": self.checks, "errors": errors,
            "paths": {key: str(path) for key, path in self.paths.items()},
            "artifact_signatures": self.snapshots, "limitations": LIMITATIONS,
        }


def write_report(path: Path, report: dict[str, Any], protected: list[Path]) -> None:
    if path.is_symlink() or path.is_dir():
        raise ValueError("Audit report must be a regular non-symlink file")
    for source in protected:
        if path.resolve() == source.resolve() or (path.exists() and source.exists() and path.samefile(source)):
            raise ValueError(f"Audit report aliases an input: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = Audit(args.run_root, args.label_dir)
    report = audit.run()
    write_report(args.output, report, [*audit.paths.values(), *(Path(path) for path in audit.snapshots)])
    print(json.dumps({"status": report["status"], "observed_counts": report["observed_counts"]}, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
