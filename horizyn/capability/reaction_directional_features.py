"""Leakage-safe, reaction-only Rhea matching and directional feature assembly."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem


CENTER_VOCAB_CAPACITY = 256


def canonical_molecule(smiles: str) -> str:
    value = str(smiles).strip()
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return value
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def participant_multiset(reaction_smiles: str) -> tuple[str, ...]:
    text = str(reaction_smiles).strip()
    if ">>" not in text:
        components = [value for value in text.split(".") if value]
    else:
        left, right = text.split(">>", maxsplit=1)
        left_values = [canonical_molecule(value) for value in left.split(".") if value]
        right_values = [canonical_molecule(value) for value in right.split(".") if value]
        components = left_values if Counter(left_values) == Counter(right_values) else left_values + right_values
        return tuple(sorted(components))
    return tuple(sorted(canonical_molecule(value) for value in components))


def _load_rhea_index(path: str | Path) -> tuple[dict[tuple[str, ...], set[str]], dict[frozenset[str], set[str]], dict[str, str]]:
    frame = pd.read_csv(path, sep="\t")
    required = {"Rhea ID", "substrate", "product"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    exact: dict[tuple[str, ...], set[str]] = defaultdict(set)
    set_only: dict[frozenset[str], set[str]] = defaultdict(set)
    smiles_by_id: dict[str, str] = {}
    for row in frame.to_dict("records"):
        rhea_id = str(row["Rhea ID"]).strip()
        directional = f"{str(row['substrate']).strip()}>>{str(row['product']).strip()}"
        key = participant_multiset(directional)
        exact[key].add(rhea_id)
        set_only[frozenset(key)].add(rhea_id)
        smiles_by_id[rhea_id] = directional
    return dict(exact), dict(set_only), smiles_by_id


def build_reaction_only_rhea_map(
    *,
    reactions_path: str | Path,
    rhea_molecules_path: str | Path,
    output_mapping_path: str | Path,
    output_directional_reactions_path: str | Path,
    output_report_path: str | Path,
) -> dict[str, Any]:
    """Resolve Rhea direction using reaction participants only, never protein data."""

    reactions = pd.read_csv(reactions_path, dtype={"reaction_id": str})
    required = {"reaction_id", "reaction_smiles"}
    missing = required - set(reactions.columns)
    if missing:
        raise ValueError(f"{reactions_path} missing columns: {sorted(missing)}")
    exact, set_only, smiles_by_id = _load_rhea_index(rhea_molecules_path)
    mapping_rows: list[dict[str, Any]] = []
    directional_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in reactions.to_dict("records"):
        reaction_id = str(row["reaction_id"])
        key = participant_multiset(row["reaction_smiles"])
        candidates = exact.get(key, set())
        if len(candidates) == 1:
            status = "exact_unique"
            confidence = 1.0
        elif len(candidates) > 1:
            status = "exact_ambiguous"
            confidence = 0.0
        else:
            candidates = set_only.get(frozenset(key), set())
            if len(candidates) == 1:
                status = "set_unique"
                confidence = 0.75
            elif len(candidates) > 1:
                status = "set_ambiguous"
                confidence = 0.0
            else:
                status = "unmatched"
                confidence = 0.0
        resolved = len(candidates) == 1 and status in {"exact_unique", "set_unique"}
        rhea_id = next(iter(candidates)) if resolved else ""
        mapping_rows.append(
            {
                "reaction_id": reaction_id,
                "rhea_id": rhea_id,
                "match_status": status,
                "match_confidence": confidence,
                "candidate_count": len(candidates),
            }
        )
        counts[status] += 1
        if resolved:
            directional_rows.append(
                {
                    "reaction_id": reaction_id,
                    "reaction_smiles": smiles_by_id[rhea_id],
                    "rhea_id": rhea_id,
                    "match_status": status,
                    "match_confidence": confidence,
                }
            )
    mapping_path = Path(output_mapping_path)
    directional_path = Path(output_directional_reactions_path)
    report_path = Path(output_report_path)
    for path in (mapping_path, directional_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)
    pd.DataFrame(
        directional_rows,
        columns=[
            "reaction_id",
            "reaction_smiles",
            "rhea_id",
            "match_status",
            "match_confidence",
        ],
    ).to_csv(directional_path, index=False)
    resolved_count = len(directional_rows)
    report = {
        "schema_version": "reactzyme_reaction_only_rhea_map_v1",
        "reactions_path": str(reactions_path),
        "rhea_molecules_path": str(rhea_molecules_path),
        "num_reactions": int(len(reactions)),
        "num_resolved": resolved_count,
        "coverage": resolved_count / max(len(reactions), 1),
        "status_counts": dict(sorted(counts.items())),
        "leakage_policy": "reaction participants only; no pair, protein, sequence, or EC input",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _decode_ids(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _fixed_h5(path: str | Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        ids = _decode_ids(handle["ids"][:])
        vectors = np.asarray(handle["vectors"][:], dtype=np.float32)
    return dict(zip(ids, vectors, strict=True))


def _ragged_h5(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        ids = _decode_ids(handle["ids"][:])
        reactants = np.asarray(handle["reactant_vectors"][:], dtype=np.float32)
        reactant_offsets = np.asarray(handle["reactant_offsets"][:], dtype=np.int64)
        products = np.asarray(handle["product_vectors"][:], dtype=np.float32)
        product_offsets = np.asarray(handle["product_offsets"][:], dtype=np.int64)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, reaction_id in enumerate(ids):
        reactant = reactants[reactant_offsets[index] : reactant_offsets[index + 1]]
        product = products[product_offsets[index] : product_offsets[index + 1]]
        out[reaction_id] = (reactant, product)
    return out


def _h5_embedding_dim(path: str | Path, dataset: str) -> int:
    with h5py.File(path, "r") as handle:
        if "embedding_dim" in handle.attrs:
            return int(handle.attrs["embedding_dim"])
        values = handle[dataset]
        if values.ndim != 2 or values.shape[1] <= 0:
            raise ValueError(f"Cannot determine embedding dimension from {path}:{dataset}")
        return int(values.shape[1])


def _symmetric_side_vector(sides: tuple[np.ndarray, np.ndarray], dim: int) -> np.ndarray:
    reactants, products = sides
    reactant = reactants.mean(axis=0) if len(reactants) else np.zeros(dim, dtype=np.float32)
    product = products.mean(axis=0) if len(products) else np.zeros(dim, dtype=np.float32)
    return np.concatenate([0.5 * (reactant + product), np.abs(product - reactant)]).astype(np.float32)


def _as_labels(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
    return []


def symmetric_center_label(label: str) -> str:
    for prefix in ("bond_formed_", "bond_broken_"):
        if label.startswith(prefix):
            return "bond_changed_" + label[len(prefix) :]
    return label


def _center_labels(path: str | Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    frame = pd.read_parquet(path)
    out: dict[str, list[str]] = {}
    for row in frame.to_dict("records"):
        values = _as_labels(row.get("reaction_center_raw_labels")) + _as_labels(
            row.get("reaction_center_coarse_labels")
        )
        out[str(row["reaction_id"])] = sorted(
            {symmetric_center_label(value) for value in values}
        )
    return out


def _build_one_directional_bundle(
    *,
    reactions_path: str | Path,
    mapping_path: str | Path,
    reaction_t5_h5: str | Path,
    unimol2_h5: str | Path,
    chiro_h5: str | Path,
    center_features_path: str | Path | None,
    center_vocab: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    reactions = pd.read_csv(reactions_path, dtype={"reaction_id": str})
    mapping = pd.read_csv(mapping_path, dtype={"reaction_id": str}).set_index("reaction_id")
    reaction_t5 = _fixed_h5(reaction_t5_h5)
    unimol2 = _ragged_h5(unimol2_h5)
    chiro = _ragged_h5(chiro_h5)
    centers = _center_labels(center_features_path)
    center_index = {label: index for index, label in enumerate(center_vocab)}
    t5_dim = _h5_embedding_dim(reaction_t5_h5, "vectors")
    unimol_dim = _h5_embedding_dim(unimol2_h5, "reactant_vectors")
    chiro_dim = _h5_embedding_dim(chiro_h5, "reactant_vectors")
    base_dim = t5_dim + 2 * unimol_dim + 2 * chiro_dim + 5
    vectors = np.zeros(
        (len(reactions), base_dim + CENTER_VOCAB_CAPACITY + 1),
        dtype=np.float32,
    )
    masks = np.zeros(len(reactions), dtype=bool)
    counts: Counter[str] = Counter()
    for row_index, row in enumerate(reactions.to_dict("records")):
        reaction_id = str(row["reaction_id"])
        if reaction_id not in mapping.index:
            counts["missing_mapping_row"] += 1
            continue
        map_row = mapping.loc[reaction_id]
        if str(map_row["match_status"]) not in {"exact_unique", "set_unique"}:
            counts["unresolved"] += 1
            continue
        forward_id = f"{reaction_id}_f"
        reverse_id = f"{reaction_id}_r"
        has_t5 = forward_id in reaction_t5 and reverse_id in reaction_t5
        has_unimol = forward_id in unimol2
        has_chiro = forward_id in chiro
        if not has_t5 or not has_unimol:
            counts["missing_required_embedding"] += 1
            continue
        t5 = 0.5 * (reaction_t5[forward_id] + reaction_t5[reverse_id])
        unimol = _symmetric_side_vector(unimol2[forward_id], unimol_dim)
        chiro_vector = (
            _symmetric_side_vector(chiro[forward_id], chiro_dim)
            if has_chiro
            else np.zeros(2 * chiro_dim, dtype=np.float32)
        )
        confidence = float(map_row["match_confidence"])
        status = str(map_row["match_status"])
        base = np.concatenate(
            [
                t5,
                unimol,
                chiro_vector,
                np.asarray(
                    [has_t5, has_unimol, has_chiro, status == "exact_unique", confidence],
                    dtype=np.float32,
                ),
            ]
        )
        vectors[row_index, :base_dim] = base
        labels = centers.get(reaction_id, [])
        for label in labels:
            if label in center_index:
                vectors[row_index, base_dim + center_index[label]] = 1.0
        vectors[row_index, -1] = float(bool(labels))
        masks[row_index] = True
        counts["resolved"] += 1
        counts["with_chiro"] += int(has_chiro)
        counts["with_center"] += int(bool(labels))
    ids = reactions["reaction_id"].astype(str).to_numpy(dtype=object)
    return ids, vectors, masks, dict(counts)


def build_directional_vector_splits(
    *,
    split_inputs: dict[str, dict[str, str | Path | None]],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Build F5 and F6 vectors, fitting the center vocabulary on train only."""

    train_centers = _center_labels(split_inputs["train"].get("center_features"))
    center_vocab = sorted({label for values in train_centers.values() for label in values})
    if len(center_vocab) > CENTER_VOCAB_CAPACITY:
        raise ValueError(
            f"Training reaction-center vocabulary has {len(center_vocab)} labels, "
            f"exceeding capacity {CENTER_VOCAB_CAPACITY}"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": "reactzyme_directional_vectors_v1",
        "center_vocab": center_vocab,
        "center_vocab_capacity": CENTER_VOCAB_CAPACITY,
        "splits": {},
    }
    for split, inputs in split_inputs.items():
        ids, combined, mask, counts = _build_one_directional_bundle(
            reactions_path=inputs["reactions"],
            mapping_path=inputs["mapping"],
            reaction_t5_h5=inputs["reaction_t5"],
            unimol2_h5=inputs["unimol2"],
            chiro_h5=inputs["chiro"],
            center_features_path=inputs.get("center_features"),
            center_vocab=center_vocab,
        )
        f5_dim = combined.shape[1] - CENTER_VOCAB_CAPACITY - 1
        np.savez_compressed(
            out / f"{split}_reaction_directional_f5.npz",
            ids=ids,
            vectors=combined[:, :f5_dim],
            mask=mask,
        )
        np.savez_compressed(
            out / f"{split}_reaction_directional_f6.npz",
            ids=ids,
            vectors=combined,
            mask=mask,
        )
        report["splits"][split] = {
            "num_reactions": int(len(ids)),
            "num_resolved": int(mask.sum()),
            "coverage": float(mask.mean()) if len(mask) else 0.0,
            "f5_dim": int(f5_dim),
            "f6_dim": int(combined.shape[1]),
            "counts": counts,
        }
    (out / "schema.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "build_directional_vector_splits",
    "build_reaction_only_rhea_map",
    "CENTER_VOCAB_CAPACITY",
    "canonical_molecule",
    "participant_multiset",
    "symmetric_center_label",
]
