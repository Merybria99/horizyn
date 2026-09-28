#!/usr/bin/env python3
"""Export versioned CIRCE-v2 labels with role-aware cofactor annotation status.

Known classes are sparse positive evidence. The cofactor unknown slot records
missing supported annotation and has zero biological supervision confidence.
Native source names remain in the immutable source table bound by the manifest.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizyn.capability.biological_targets import (  # noqa: E402
    MECHANISM_LABELS,
)
from horizyn.capability.cofactor_vocabulary_v2 import (  # noqa: E402
    COFACTOR_LABELS, COFACTOR_VOCABULARY_VERSION, UNKNOWN_INDEX,
    cofactor_name_groups,
)


SCHEMA = "horizyn1_circe_v2_labels_v2"
REACTION_SCHEMA = "horizyn1_circe_v2_reaction_annotations_v2"
ANNOTATION_SEMANTICS = "positive_only_role_aware_v2"


def progress(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", file=sys.stderr, flush=True)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def signature(path: Path) -> list[int]:
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def reject_aliases(inputs: list[Path], outputs: list[Path]) -> None:
    for index, output in enumerate(outputs):
        for other in inputs + outputs[:index]:
            if output.resolve() == other.resolve() or (
                output.exists() and other.exists() and os.path.samefile(output, other)
            ):
                raise ValueError(f"Output aliases another artifact: {output} -> {other}")


@contextmanager
def atomic_text(path: Path):
    temporary = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        handle = gzip.open(temporary, "wt", encoding="utf-8", newline="")
    else:
        handle = temporary.open("w", encoding="utf-8", newline="")
    try:
        yield handle
    finally:
        handle.close()
    # NFS close/writeback can otherwise change metadata after the manifest appears.
    with temporary.open("rb") as complete:
        os.fsync(complete.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: dict) -> None:
    with atomic_text(path) as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def json_labels(value: str) -> set[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("Annotation labels must be JSON lists of strings")
    return {item.strip() for item in parsed if item.strip()}


def known_depth(ec: str) -> int:
    parts = ec.split(".")
    if len(parts) != 4:
        return 0
    depth = 0
    for part in parts:
        if not part.isdigit():
            break
        depth += 1
    return depth


def consistent_complete_ec(labels: set[str]) -> str | None:
    complete = {value for value in labels if known_depth(value) == 4}
    if len(complete) != 1:
        return None
    leaf = next(iter(complete))
    target = leaf.split(".")
    for label in labels:
        parts = label.split(".")
        if len(parts) != 4 or any(p != "-" and p != t for p, t in zip(parts, target)):
            return None
    return leaf


def fasta_ids(path: Path) -> list[str]:
    result = []
    seen = set()
    with open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                protein_id = line[1:].strip().split()[0]
                if protein_id in seen:
                    raise ValueError(f"Duplicate representative ID: {protein_id}")
                seen.add(protein_id)
                result.append(protein_id)
    if not result:
        raise ValueError("Representative FASTA is empty")
    return result


def build_labels(
    *,
    representative_fasta: Path,
    cluster_map: Path,
    native_annotations: Path,
    reaction_annotations: Path,
    association_pairs: Path,
    output_dir: Path,
    pair_scope: str,
    force: bool = False,
    native_manifest: Path | None = None,
    reaction_manifest: Path | None = None,
) -> dict:
    if pair_scope not in {"unsplit_inventory", "train"}:
        raise ValueError("pair_scope must be unsplit_inventory or train")
    helper = Path(__file__).resolve().parents[1] / "horizyn/capability/biological_targets.py"
    names_helper = helper.with_name("cofactor_vocabulary_v2.py")
    reaction_builder = Path(__file__).with_name("build_horizyn1_reaction_annotations_v2.py")
    inputs = [representative_fasta, cluster_map, native_annotations, reaction_annotations,
              association_pairs, helper, names_helper, reaction_builder]
    inputs.extend(path for path in (native_manifest, reaction_manifest) if path is not None)
    paths = {
        "ec": output_dir / "enzyme_ec_labels.csv",
        "biofp": output_dir / "enzyme_biofp_targets.npz",
        "eligibility": output_dir / "candidate_eligibility.csv",
        "clusters": output_dir / "cluster_annotation_summary.tsv.gz",
        "vocab": output_dir / "enzyme_biofp_vocab.json",
        "manifest": output_dir / "label_manifest.json",
    }
    staging_paths = [path.with_name(path.name + ".partial") for path in paths.values()]
    reject_aliases([*inputs, Path(__file__)], list(paths.values()) + staging_paths)
    before = {str(path.resolve()): signature(path) for path in inputs}
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if paths["manifest"].exists():
        previous = json.loads(paths["manifest"].read_text())
        if previous.get("schema_version") != SCHEMA:
            raise ValueError("Existing label manifest has another schema; use a new versioned output directory")
    if paths["manifest"].exists() and not force:
        if (
            previous.get("input_signatures") == before
            and previous.get("implementation_sha256") == implementation
            and previous.get("pair_scope") == pair_scope
            and all(
                path.is_file() and signature(path) == previous.get("output_signatures", {}).get(name)
                and not path.with_name(path.name + ".partial").exists()
                for name, path in paths.items() if name != "manifest"
            )
        ):
            progress("Skipping completed CIRCE-v2 label export")
            return previous
        raise ValueError("Existing label manifest does not match inputs/outputs; inspect before --force")
    if not force and any(path.exists() for path in list(paths.values()) + staging_paths):
        raise ValueError("Incomplete label outputs already exist; inspect before --force")
    if force and paths["manifest"].exists():
        # A forced rebuild must not advertise stale completion while replacing outputs.
        paths["manifest"].unlink()

    progress("Reading representative IDs and exact cluster membership")
    ids = fasta_ids(representative_fasta)
    index = {value: position for position, value in enumerate(ids)}
    count = len(ids)
    members: dict[str, int] = {}
    member_counts = np.zeros(count, dtype=np.uint32)
    with open_text(cluster_map) as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 or fields[0] not in index:
                raise ValueError(f"Invalid cluster row {number}")
            rep, member = fields
            if member in members:
                raise ValueError(f"Duplicate cluster member: {member}")
            position = index[rep]
            members[member] = position
            member_counts[position] += 1
            if number % 1_000_000 == 0:
                progress(f"Read {number:,} cluster memberships")
    if any(members.get(value) != position for value, position in index.items()):
        raise ValueError("Every representative must be its own cluster member")

    mechanism = np.zeros((count, len(MECHANISM_LABELS)), dtype=bool)
    native_cofactor = np.zeros((count, len(COFACTOR_LABELS)), dtype=bool)
    reaction_cofactor = np.zeros_like(native_cofactor)
    native_has_annotation = np.zeros(count, dtype=bool)
    native_has_unmapped = np.zeros(count, dtype=bool)
    annotation_counts = Counter()
    unmapped_all_members: Counter[str] = Counter()
    unmapped_representatives: Counter[str] = Counter()
    # Millions of source rows repeat a small set of explicit cofactor names.
    name_cache: dict[str, frozenset[str]] = {}
    mechanism_index = {value: i for i, value in enumerate(MECHANISM_LABELS)}
    cofactor_index = {value: i for i, value in enumerate(COFACTOR_LABELS)}
    own_ec: dict[int, set[str]] = {}
    member_ec: dict[int, set[str]] = defaultdict(set)
    member_ec_count = np.zeros(count, dtype=np.uint32)
    native_status = Counter()
    representative_status = Counter()
    seen_members: set[str] = set()
    own_native_seen = np.zeros(count, dtype=bool)
    progress("Reading direct sequence annotations; member labels stay separate")
    with atomic_text(paths["ec"]) as out, open_text(native_annotations) as handle:
        writer = csv.writer(out)
        writer.writerow(["protein_id", "ec_number", "known_depth"])
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"protein_id", "annotation_status", "ec_numbers", "cofactor_names"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Native annotation table lacks {sorted(required)}")
        for number, row in enumerate(reader, 1):
            protein_id = row["protein_id"]
            if protein_id not in members or protein_id in seen_members:
                raise ValueError(f"Unknown or duplicate native annotation ID: {protein_id}")
            seen_members.add(protein_id)
            position = members[protein_id]
            status = row["annotation_status"]
            if status not in {"matched", "matched_unannotated", "sequence_mismatch", "unresolved", "ambiguous_accession"}:
                raise ValueError(f"Unknown native annotation status: {status}")
            ec = {sys.intern(value) for value in json_labels(row["ec_numbers"])}
            cofactor_names = json_labels(row["cofactor_names"])
            if status not in {"matched", "matched_unannotated"} and (ec or cofactor_names):
                raise ValueError(f"Unmatched sequence has native labels: {protein_id}")
            if status == "matched_unannotated" and (ec or cofactor_names):
                raise ValueError(f"Unannotated status carries native labels: {protein_id}")
            mapped_cofactors: set[str] = set()
            unmapped_names: set[str] = set()
            for name in cofactor_names:
                groups = name_cache.get(name)
                if groups is None:
                    groups = frozenset(cofactor_name_groups((name,)))
                    name_cache[name] = groups
                mapped_cofactors.update(groups)
                if not groups:
                    unmapped_names.add(name)
            annotation_counts["all_members_with_annotation"] += bool(cofactor_names)
            annotation_counts["all_members_with_unmapped"] += bool(unmapped_names)
            unmapped_all_members.update(unmapped_names)
            native_status[status] += 1
            if ec:
                member_ec[position].update(ec)
                member_ec_count[position] += 1
            if protein_id in index:
                own_native_seen[position] = True
                representative_status[status] += 1
                if ec:
                    own_ec[position] = ec
                for value in sorted(ec):
                    depth = known_depth(value)
                    # Preserve incomplete/provisional evidence, not synthetic full EC leaves.
                    writer.writerow([protein_id, value, depth])
                native_has_annotation[position] = bool(cofactor_names)
                native_has_unmapped[position] = bool(unmapped_names)
                unmapped_representatives.update(unmapped_names)
                for value in mapped_cofactors:
                    native_cofactor[position, cofactor_index[value]] = True
            if number % 1_000_000 == 0:
                progress(f"Read {number:,} native annotation rows")
    if len(seen_members) != len(members) or not own_native_seen.all():
        raise ValueError("Native annotation table must include every raw member, including unknowns")
    del seen_members

    lookup = json.loads(reaction_annotations.read_text())
    expected_families = {"mechanism": list(MECHANISM_LABELS), "cofactor": list(COFACTOR_LABELS)}
    if not isinstance(lookup, dict):
        raise ValueError("Reaction annotations require a versioned lookup object")
    if lookup.get("schema_version") != REACTION_SCHEMA:
        raise ValueError("Reaction annotation schema does not match the versioned CIRCE-v2 export")
    if (lookup.get("cofactor_vocabulary_version") != COFACTOR_VOCABULARY_VERSION
            or lookup.get("cofactor_unknown_label") != "unknown"):
        raise ValueError("Reaction annotation cofactor vocabulary identity does not match")
    if lookup.get("label_semantics") != "positive_only_unknown_not_negative":
        raise ValueError("Reaction annotation must contain positive-only descriptors")
    if lookup.get("families") != expected_families:
        raise ValueError("Reaction annotation vocabulary/order does not match CIRCE-v2")
    features = {}
    if not isinstance(lookup.get("reactions"), dict):
        raise ValueError("Reaction annotations require a reaction lookup object")
    for reaction_id, row in lookup["reactions"].items():
        if not isinstance(row, dict):
            raise ValueError(f"Invalid reaction descriptor: {reaction_id}")
        for family, allowed in (("mechanism", mechanism_index), ("cofactor", cofactor_index)):
            values = row.get(family)
            if (not isinstance(values, list) or not all(isinstance(value, str) for value in values)
                    or len(values) != len(set(values)) or any(value not in allowed for value in values)
                    or (family == "cofactor" and "unknown" in values)):
                raise ValueError(f"Reaction annotation has invalid known {family} labels: {reaction_id}")
        if row.get("evidence_type") != "reaction_associated_descriptor":
            raise ValueError(f"Reaction annotation has invalid evidence role: {reaction_id}")
        if row.get("chemistry_match") != "exact":
            raise ValueError(f"Reaction annotation is not an exact chemistry match: {reaction_id}")
        features[reaction_id] = (
            [mechanism_index[value] for value in row["mechanism"]],
            [cofactor_index[value] for value in row["cofactor"]],
        )
    progress("Building weak profiles from each representative's OWN source associations")
    pair_rows = 0
    own_pair_rows = 0
    own_annotated_pair_rows = 0
    with open_text(association_pairs) as handle:
        delimiter = "," if association_pairs.name.endswith((".csv", ".csv.gz")) else "\t"
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not {"reaction_id", "protein_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Association pairs require reaction_id and protein_id")
        for pair_rows, row in enumerate(reader, 1):
            protein_id = row["protein_id"]
            if protein_id not in members:
                raise ValueError(f"Association references an unknown raw protein: {protein_id}")
            if protein_id in index:
                own_pair_rows += 1
                labels = features.get(row["reaction_id"])
                if labels is not None:
                    own_annotated_pair_rows += 1
                    position = index[protein_id]
                    mechanism[position, labels[0]] = True
                    reaction_cofactor[position, labels[1]] = True
            if pair_rows % 1_000_000 == 0:
                progress(f"Read {pair_rows:,} source association rows")

    eligibility_counts = Counter()
    with atomic_text(paths["eligibility"]) as eligibility, atomic_text(paths["clusters"]) as summary:
        eligible_writer = csv.writer(eligibility)
        eligible_writer.writerow([
            "protein_id", "ec_negative_candidate_eligible", "biological_negative_candidate_eligible",
            "reason", "member_count", "members_with_ec",
        ])
        summary_writer = csv.writer(summary, delimiter="\t")
        summary_writer.writerow(["protein_id", "member_count", "members_with_ec", "member_ec_evidence", "own_ec_labels"])
        for position, protein_id in enumerate(ids):
            own = own_ec.get(position, set())
            observed = member_ec.get(position, set())
            leaf = consistent_complete_ec(own)
            cluster_leaf = consistent_complete_ec(observed)
            eligible = leaf is not None and cluster_leaf == leaf
            biological = eligible and bool(mechanism[position].any())
            reason = "eligible" if eligible else (
                "missing_or_ambiguous_direct_ec" if leaf is None else "cluster_member_ec_conflict"
            )
            eligibility_counts[reason] += 1
            eligibility_counts["ec_eligible"] += int(eligible)
            eligibility_counts["biological_eligible"] += int(biological)
            eligible_writer.writerow([protein_id, int(eligible), int(biological), reason,
                                      int(member_counts[position]), int(member_ec_count[position])])
            summary_writer.writerow([protein_id, int(member_counts[position]), int(member_ec_count[position]),
                                     json.dumps(sorted(observed)), json.dumps(sorted(own))])

    # Unknown is source-specific annotation status. Union it only after known
    # evidence is combined, so one empty source never contaminates the union.
    cofactor = native_cofactor | reaction_cofactor
    for source in (native_cofactor, reaction_cofactor, cofactor):
        source[:, UNKNOWN_INDEX] = ~source[:, :UNKNOWN_INDEX].any(axis=1)
    cofactor_confidence = np.where(
        native_cofactor, 1.0, np.where(reaction_cofactor, 0.4, 0.0)
    ).astype(np.float32)
    cofactor_confidence[:, UNKNOWN_INDEX] = 0.0
    # Every discoverable source target family also carries explicit confidence;
    # generic dataset loaders otherwise default to its mask, supervising unknown.
    native_cofactor_confidence = native_cofactor.astype(np.float32)
    reaction_cofactor_confidence = reaction_cofactor.astype(np.float32) * 0.4
    native_cofactor_confidence[:, UNKNOWN_INDEX] = 0.0
    reaction_cofactor_confidence[:, UNKNOWN_INDEX] = 0.0
    arrays = {
        "ids": np.asarray(ids, dtype=str),
        "cofactor_labels": np.asarray(COFACTOR_LABELS, dtype=str),
        "cofactor_unknown_index": np.asarray(UNKNOWN_INDEX, dtype=np.int64),
        "cofactor_vocabulary_version": np.asarray(COFACTOR_VOCABULARY_VERSION),
        "annotation_semantics": np.asarray(ANNOTATION_SEMANTICS),
        "pair_scope": np.asarray(pair_scope),
        "native_cofactor_has_annotation": native_has_annotation,
        "native_cofactor_has_unmapped": native_has_unmapped,
        "mechanism_targets": mechanism.astype(np.float32), "mechanism_mask": mechanism,
        "mechanism_confidence": mechanism.astype(np.float32) * 0.4,
        "cofactor_targets": cofactor.astype(np.float32), "cofactor_mask": cofactor,
        "cofactor_confidence": cofactor_confidence,
        "native_cofactor_targets": native_cofactor.astype(np.float32), "native_cofactor_mask": native_cofactor,
        "native_cofactor_confidence": native_cofactor_confidence,
        "reaction_cofactor_targets": reaction_cofactor.astype(np.float32), "reaction_cofactor_mask": reaction_cofactor,
        "reaction_cofactor_confidence": reaction_cofactor_confidence,
    }
    # Standalone pretraining selects active rows using mask AND denominator
    # before reading confidence. Unknown-only rows must not become eligible.
    for family, source in (("cofactor", cofactor), ("native_cofactor", native_cofactor),
                           ("reaction_cofactor", reaction_cofactor)):
        arrays[f"{family}_denominator"] = source[:, :UNKNOWN_INDEX].any(axis=1).astype(np.float32)
    progress("Writing CIRCE-v2 arrays with known positives and explicit cofactor unknown status")
    temporary = paths["biofp"].with_name(paths["biofp"].name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, paths["biofp"])
    metadata = {
        "schema_version": SCHEMA, "families": expected_families, "pair_scope": pair_scope,
        "training_split_declared": pair_scope == "train",
        "cofactor_vocabulary_version": COFACTOR_VOCABULARY_VERSION,
        "cofactor_unknown_index": UNKNOWN_INDEX,
        "annotation_semantics": ANNOTATION_SEMANTICS,
        "positive_only": True,
        "non_biological_classes": {"cofactor": ["unknown"]},
        "mask_semantics": "positive_only known classes; unknown target/mask marks no supported annotation and carries zero biological supervision confidence",
        "source_roles": {
            "native_cofactor": "Direct representative sequence annotation; original cofactor names are retained in the bound native source table.",
            "reaction_cofactor": "Weak descriptors from the representative own supplied source associations; no cluster-member transfer.",
            "cofactor": "Union of known native and reaction-associated positives, with native confidence taking precedence.",
            "unknown": "Annotation status only, not a biological cofactor class, absence label, or negative supervision; set within each source iff no known class is present, and for union iff neither source has known evidence.",
        },
        "unknown_counts": {
            "native_cofactor": int(native_cofactor[:, UNKNOWN_INDEX].sum()),
            "reaction_cofactor": int(reaction_cofactor[:, UNKNOWN_INDEX].sum()),
            "cofactor": int(cofactor[:, UNKNOWN_INDEX].sum()),
        },
        "native_cofactor_annotation_counts": {
            "representatives_with_annotation": int(native_has_annotation.sum()),
            "representatives_with_unmapped": int(native_has_unmapped.sum()),
            "all_members_with_annotation": annotation_counts["all_members_with_annotation"],
            "all_members_with_unmapped": annotation_counts["all_members_with_unmapped"],
        },
        "unmapped_native_cofactor_name_counts": {
            "all_members": dict(sorted(unmapped_all_members.items())),
            "representatives": dict(sorted(unmapped_representatives.items())),
        },
        "confidence_semantics": "known positive provenance weights (native 1.0, reaction-associated 0.4); unknown confidence is zero and is not biological supervision; weights are not calibrated probabilities or experimental validation",
        "denominator_semantics": "Row-wise known-evidence indicator for each cofactor source; zero excludes unknown-only rows from biological pretraining eligibility.",
        "representative_count": count, "raw_member_count": len(members),
        "association_rows": pair_rows, "representative_own_association_rows": own_pair_rows,
        "representative_own_annotated_association_rows": own_annotated_pair_rows,
        "native_annotation_status": dict(native_status),
        "representative_annotation_status": dict(representative_status),
        "negative_candidate_eligibility": dict(eligibility_counts),
        "row_coverage": {"mechanism": int(mechanism.any(axis=1).sum()),
                         "cofactor": int(cofactor[:, :UNKNOWN_INDEX].any(axis=1).sum()),
                         "native_cofactor": int(native_cofactor[:, :UNKNOWN_INDEX].any(axis=1).sum()),
                         "reaction_cofactor": int(reaction_cofactor[:, :UNKNOWN_INDEX].any(axis=1).sum()),
                         "direct_ec": len(own_ec)},
        "row_coverage_semantics": "Known positive classes only; cofactor unknown status does not count as coverage.",
        "sources": {"native": str(native_annotations), "reaction_annotations": str(reaction_annotations),
                    "association_pairs": str(association_pairs),
                    "native_manifest": None if native_manifest is None else str(native_manifest),
                    "reaction_manifest": None if reaction_manifest is None else str(reaction_manifest)},
        "limitations": [
            "Mechanism features are weak reaction-associated descriptors, not measured catalytic mechanisms.",
            "Native cofactor labels and weak reaction-associated cofactor evidence are also exported separately.",
            "The cofactor unknown slot is annotation status with zero confidence and must be excluded from biological supervision and biological similarity.",
            "Specific metal classes preserve element identity; original source names retain oxidation-state and evidence detail.",
            "Cluster-member EC evidence excludes ambiguous negative candidates; it is never assigned as representative gold labels.",
            "No EC, cofactor, or mechanism is inferred solely from 80% sequence identity.",
            "Unsplit inventory profiles must be regenerated from training-only source associations before held-out evaluation.",
            "Use candidate_eligibility.csv when building CIRCE-v2 negative pools; query-specific exclusions still apply.",
            "Positive-only masks are intentionally conservative and are not calibrated absence probabilities.",
        ],
    }
    write_json(paths["vocab"], metadata)
    after = {str(path.resolve()): signature(path) for path in inputs}
    if after != before or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != implementation:
        raise ValueError("An input changed during label export; no completion manifest written")
    metadata.update({"input_signatures": before, "implementation_sha256": implementation,
                     "outputs": {name: str(path) for name, path in paths.items()},
                     "output_signatures": {name: signature(path) for name, path in paths.items() if name != "manifest"}})
    write_json(paths["manifest"], metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representative-fasta", required=True, type=Path)
    parser.add_argument("--cluster-map", required=True, type=Path)
    parser.add_argument("--native-annotations", required=True, type=Path)
    parser.add_argument("--reaction-annotations", required=True, type=Path)
    parser.add_argument("--native-manifest", type=Path)
    parser.add_argument("--reaction-manifest", type=Path)
    parser.add_argument("--association-pairs", required=True, type=Path)
    parser.add_argument("--pair-scope", required=True, choices=["unsplit_inventory", "train"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_labels(**vars(args))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
