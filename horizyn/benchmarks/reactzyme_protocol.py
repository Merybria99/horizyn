"""ReactZyme protocol materialization and feature-coverage audits."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py


REACTZYME_SPLITS = ("time", "enzyme_smi", "reaction_smi")
PAIR_REQUIRED_FIELDS = frozenset({"pr_id", "reaction_id", "protein_id"})
REACTION_MODALITIES = ("reactiont5v2", "unimol2", "chiro")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV while preserving its field and row order."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, str]],
) -> None:
    """Write ordered CSV records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_protein_ids(rows: Iterable[dict[str, str]]) -> list[str]:
    """Return protein IDs in first-occurrence order."""

    seen: set[str] = set()
    protein_ids: list[str] = []
    for row in rows:
        protein_id = row["protein_id"]
        if protein_id not in seen:
            seen.add(protein_id)
            protein_ids.append(protein_id)
    return protein_ids


def write_ids(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def split_train_rows(
    rows: Sequence[dict[str, str]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Reproduce ReactZyme's row-level train/validation split deterministically."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    train_count = int((1.0 - validation_fraction) * len(rows))
    shuffled_indices = list(range(len(rows)))
    random.Random(seed).shuffle(shuffled_indices)
    train_indices = set(shuffled_indices[:train_count])
    train_rows = [row for index, row in enumerate(rows) if index in train_indices]
    validation_rows = [row for index, row in enumerate(rows) if index not in train_indices]
    return train_rows, validation_rows


def split_train_rows_by_reaction_smiles(
    rows: Sequence[dict[str, str]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Hold out complete exact-SMILES groups near a target pair-row fraction."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not rows:
        raise ValueError("Cannot split an empty ReactZyme training table")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        smiles = str(row.get("reaction_smiles", "")).strip()
        if not smiles:
            raise ValueError(f"Training row {index} has no reaction_smiles")
        groups[smiles].append(index)
    if len(groups) < 2:
        raise ValueError("Reaction-disjoint validation requires at least two SMILES groups")

    ordered_groups = sorted(
        groups,
        key=lambda smiles: hashlib.sha256(f"{seed}:{smiles}".encode("utf-8")).hexdigest(),
    )
    target_rows = validation_fraction * len(rows)
    selected_smiles: set[str] = set()
    selected_count = 0
    for smiles in ordered_groups:
        group_size = len(groups[smiles])
        if selected_smiles and selected_count >= target_rows:
            break
        if selected_smiles and selected_count + group_size > target_rows:
            without_group = abs(target_rows - selected_count)
            with_group = abs(target_rows - (selected_count + group_size))
            if without_group <= with_group:
                break
        selected_smiles.add(smiles)
        selected_count += group_size

    if not selected_smiles or len(selected_smiles) == len(groups):
        raise ValueError("Reaction-disjoint split produced an empty train or validation set")
    validation_indices = {
        index for smiles in selected_smiles for index in groups[smiles]
    }
    train_rows = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation_rows = [row for index, row in enumerate(rows) if index in validation_indices]
    return train_rows, validation_rows


def reaction_rows_for_pairs(
    reaction_rows: Sequence[dict[str, str]],
    pair_rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    """Select reaction rows required by a pair subset, preserving source order."""

    selected_ids = {row["reaction_id"] for row in pair_rows}
    selected = [row for row in reaction_rows if row["reaction_id"] in selected_ids]
    missing = selected_ids - {row["reaction_id"] for row in selected}
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Missing {len(missing)} reaction rows; examples: {examples}")
    return selected


def _pair_key(row: dict[str, str]) -> tuple[str, str]:
    return row["reaction_id"], row["protein_id"]


def build_protocol(
    protocol: str,
    *,
    source_root: Path,
    out_root: Path,
    validation_fraction: float,
    seed: int,
    split_method: str = "pair_random",
) -> dict[str, Any]:
    """Materialize one ReactZyme train/validation/test protocol."""

    source_dir = source_root / protocol
    out_dir = out_root / protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_fields, source_train_pairs = read_csv_rows(source_dir / "train_pairs.csv")
    reaction_fields, source_train_rxns = read_csv_rows(source_dir / "train_rxns.csv")
    missing_fields = PAIR_REQUIRED_FIELDS - set(pair_fields)
    if missing_fields:
        raise ValueError(f"{protocol} train pairs missing fields: {sorted(missing_fields)}")

    if split_method == "pair_random":
        train_pairs, validation_pairs = split_train_rows(
            source_train_pairs,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        split_description = (
            "pair-level shuffled indices; floor((1-validation_fraction) * N) train"
        )
    elif split_method == "reaction_smiles_disjoint":
        train_pairs, validation_pairs = split_train_rows_by_reaction_smiles(
            source_train_pairs,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        split_description = (
            "exact reaction-SMILES-group holdout near validation_fraction pair rows"
        )
    else:
        raise ValueError(
            "split_method must be one of: pair_random, reaction_smiles_disjoint"
        )
    train_rxns = reaction_rows_for_pairs(source_train_rxns, train_pairs)
    validation_rxns = reaction_rows_for_pairs(source_train_rxns, validation_pairs)

    write_csv_rows(out_dir / "train_pairs.csv", pair_fields, train_pairs)
    write_csv_rows(out_dir / "validation_pairs.csv", pair_fields, validation_pairs)
    write_csv_rows(out_dir / "train_rxns.csv", reaction_fields, train_rxns)
    write_csv_rows(out_dir / "validation_rxns.csv", reaction_fields, validation_rxns)

    # The released test data is copied byte-for-byte and never rewritten.
    for name in ("test_pairs.csv", "test_rxns.csv"):
        shutil.copy2(source_dir / name, out_dir / name)

    _, test_pairs = read_csv_rows(out_dir / "test_pairs.csv")
    candidate_counts = {}
    for subset, rows in (
        ("train", train_pairs),
        ("validation", validation_pairs),
        ("test", test_pairs),
    ):
        protein_ids = unique_protein_ids(rows)
        write_ids(out_dir / f"{subset}_candidate_ids.txt", protein_ids)
        candidate_counts[subset] = len(protein_ids)

    train_keys = {_pair_key(row) for row in train_pairs}
    validation_keys = {_pair_key(row) for row in validation_pairs}
    source_keys = {_pair_key(row) for row in source_train_pairs}
    reconstructed_keys = train_keys | validation_keys
    if train_keys & validation_keys:
        raise ValueError(f"{protocol} has exact pair overlap between train and validation")
    if reconstructed_keys != source_keys or len(train_pairs) + len(validation_pairs) != len(
        source_train_pairs
    ):
        raise ValueError(f"{protocol} train/validation rows do not reconstruct source train pairs")

    train_reaction_ids = {row["reaction_id"] for row in train_pairs}
    validation_reaction_ids = {row["reaction_id"] for row in validation_pairs}
    train_reaction_smiles = {row["reaction_smiles"] for row in train_pairs}
    validation_reaction_smiles = {row["reaction_smiles"] for row in validation_pairs}
    reaction_id_overlap = train_reaction_ids & validation_reaction_ids
    reaction_smiles_overlap = train_reaction_smiles & validation_reaction_smiles
    if split_method == "reaction_smiles_disjoint" and (
        reaction_id_overlap or reaction_smiles_overlap
    ):
        raise ValueError(
            f"{protocol} reaction-disjoint split leaked reaction IDs or SMILES"
        )

    test_hashes = {}
    for name in ("test_pairs.csv", "test_rxns.csv"):
        source_hash = sha256(source_dir / name)
        output_hash = sha256(out_dir / name)
        if source_hash != output_hash:
            raise ValueError(f"Released test file changed while copying: {protocol}/{name}")
        test_hashes[name] = source_hash

    manifest: dict[str, Any] = {
        "protocol": protocol,
        "source_dir": str(source_dir.resolve()),
        "split_method": split_method,
        "split_description": split_description,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "reaction_direction_mode": "forward_only",
        "counts": {
            "source_train_pairs": len(source_train_pairs),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "test_pairs": len(test_pairs),
            "train_reactions": len(train_rxns),
            "validation_reactions": len(validation_rxns),
            "train_candidates": candidate_counts["train"],
            "validation_candidates": candidate_counts["validation"],
            "test_candidates": candidate_counts["test"],
        },
        "audits": {
            "train_validation_exact_pair_overlap": len(train_keys & validation_keys),
            "source_train_pair_set_reconstructed": reconstructed_keys == source_keys,
            "test_files_byte_identical_to_release": True,
            "test_sha256": test_hashes,
            "train_validation_reaction_id_overlap": len(reaction_id_overlap),
            "train_validation_reaction_smiles_overlap": len(reaction_smiles_overlap),
            "source_train_pair_rows_reconstructed": (
                len(train_pairs) + len(validation_pairs) == len(source_train_pairs)
            ),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def materialize_paper_protocols(
    *,
    source_root: Path,
    out_root: Path,
    validation_fraction: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
    protocols: Sequence[str] = REACTZYME_SPLITS,
) -> dict[str, Any]:
    """Build all requested ReactZyme paper protocol directories."""

    source_root = source_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    manifests = {
        protocol: build_protocol(
            protocol,
            source_root=source_root,
            out_root=out_root,
            validation_fraction=validation_fraction,
            seed=seed,
            split_method="pair_random",
        )
        for protocol in protocols
    }
    manifest = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "protocols": manifests,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    train_percent = 100.0 * (1.0 - validation_fraction)
    validation_percent = 100.0 * validation_fraction
    (out_root / "README.md").write_text(
        "# ReactZyme paper protocol\n\n"
        f"Each official ReactZyme training split is divided into {train_percent:g}% training "
        f"and {validation_percent:g}% validation at the pair-row level with seed {seed}. "
        "Released test CSVs are copied byte-for-byte and are used only for final evaluation. "
        "Candidate ID files contain the unique proteins in their corresponding subset.\n",
        encoding="utf-8",
    )
    return manifest


def materialize_unseen_reaction_protocols(
    *,
    source_root: Path,
    out_root: Path,
    validation_fraction: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
    protocols: Sequence[str] = REACTZYME_SPLITS,
) -> dict[str, Any]:
    """Build paper-compatible splits with unseen-reaction development validation."""

    source_root = source_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    manifests = {}
    for protocol in protocols:
        method = "reaction_smiles_disjoint" if protocol == "reaction_smi" else "pair_random"
        manifests[protocol] = build_protocol(
            protocol,
            source_root=source_root,
            out_root=out_root,
            validation_fraction=validation_fraction,
            seed=seed,
            split_method=method,
        )
    manifest = {
        "schema_version": "reactzyme_unseen_reaction_protocol_v1",
        "source_root": str(source_root),
        "out_root": str(out_root),
        "protocols": manifests,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_root / "README.md").write_text(
        "# ReactZyme unseen-reaction development protocol\n\n"
        "The time and enzyme-similarity protocols retain deterministic pair-level "
        "validation. The reaction-similarity protocol holds out complete exact "
        "reaction-SMILES groups. Official test CSVs are copied byte-for-byte.\n",
        encoding="utf-8",
    )
    return manifest


def _reaction_feature_ids(path: Path) -> set[str]:
    with h5py.File(path, "r") as handle:
        return {
            value.decode() if isinstance(value, bytes) else str(value) for value in handle["ids"][:]
        }


def collect_feature_coverage(
    *,
    protocol_root: Path,
    feature_root: Path,
    protocols: Sequence[str] = REACTZYME_SPLITS,
) -> dict[str, Any]:
    """Audit canonical paper queries against configured modality stores."""

    coverage: dict[str, Any] = {}
    for protocol in protocols:
        protocol_coverage: dict[str, Any] = {}
        for subset, feature_subset in (
            ("train", "train"),
            ("validation", "train"),
            ("test", "test"),
        ):
            _, reaction_rows = read_csv_rows(protocol_root / protocol / f"{subset}_rxns.csv")
            required = {f"{row['reaction_id']}_f" for row in reaction_rows}
            subset_coverage: dict[str, Any] = {}
            for modality in REACTION_MODALITIES:
                available = _reaction_feature_ids(
                    feature_root / protocol / feature_subset / f"{modality}.h5"
                )
                missing = sorted(required - available)
                policy = "required" if modality == "reactiont5v2" else "zero_with_mask"
                subset_coverage[modality] = {
                    "required": len(required),
                    "available": len(required) - len(missing),
                    "missing": len(missing),
                    "missing_examples": missing[:10],
                    "policy": policy,
                }
                if policy == "required" and missing:
                    raise ValueError(
                        f"{protocol}/{subset} is missing {len(missing)} required "
                        "ReactionT5v2 feature records"
                    )
            protocol_coverage[subset] = subset_coverage
        coverage[protocol] = protocol_coverage
    return coverage


__all__ = [
    "REACTZYME_SPLITS",
    "build_protocol",
    "collect_feature_coverage",
    "materialize_paper_protocols",
    "read_csv_rows",
    "split_train_rows",
]
