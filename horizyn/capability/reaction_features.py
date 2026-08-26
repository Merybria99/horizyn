"""Build deterministic reaction capability features."""

from __future__ import annotations

import hashlib
import importlib.resources
import re
import sys
import types
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from horizyn.capability.annotation_dictionaries import annotation_dictionaries_payload
from horizyn.capability.cofactors import (
    cofactor_labels_for_substrate_filter,
    extract_cofactor_labels,
    load_cofactor_aliases,
    split_cofactor_label_tiers,
)
from horizyn.capability.io import ensure_dir, write_json
from horizyn.capability.reaction_center import (
    coarse_reaction_center_labels,
    extract_reaction_center_raw_labels,
)
from horizyn.capability.reaction_types import reaction_type_labels
from horizyn.capability.substrate_classes import (
    classify_reaction_substrates_products,
    substrate_product_transition_labels,
)


def strip_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def canonicalize_mol_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    mol = strip_atom_maps(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_reaction_smiles(rxn: str) -> str | None:
    parts = str(rxn).split(">")
    if len(parts) == 2:
        left, right = parts
        middle = ""
    elif len(parts) == 3:
        left, middle, right = parts
    else:
        return None

    reactant_side = ".".join([x for x in [left, middle] if x])
    product_side = right

    def canon_side(side: str) -> str | None:
        mols = []
        for smi in side.split("."):
            if not smi:
                continue
            canon = canonicalize_mol_smiles(smi)
            if canon is None:
                return None
            mols.append(canon)
        return ".".join(sorted(mols))

    lhs = canon_side(reactant_side)
    rhs = canon_side(product_side)
    if lhs is None or rhs is None:
        return None
    return f"{lhs}>>{rhs}"


def generated_reaction_id(canonical_reaction_smiles: str) -> str:
    digest = hashlib.sha1(canonical_reaction_smiles.encode()).hexdigest()[:16]
    return f"RXN_{digest}"


def compute_drfp_active_bits(
    reaction_smiles: str,
    n_bits: int = 2048,
) -> tuple[np.ndarray, list[int]]:
    from drfp import DrfpEncoder

    fp = DrfpEncoder.encode([reaction_smiles], n_folded_length=n_bits)[0]
    arr = np.asarray(fp, dtype=np.uint8)
    active = np.flatnonzero(arr).astype(int).tolist()
    return arr, active


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(item) for item in value if str(item)]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.replace(";", ",").replace("|", ",").split(",") if part.strip()]


def _normalized_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column.lower())


def reaction_identifier_aliases(value: Any) -> set[str]:
    aliases: set[str] = set()
    if value is None or pd.isna(value):
        return aliases
    text = str(value).strip()
    if not text:
        return aliases
    aliases.add(text)
    for match in re.finditer(r"(?:RHEA:|Rh_)(\d+)", text, flags=re.IGNORECASE):
        number = match.group(1)
        aliases.update({number, f"RHEA:{number}", f"Rh_{number}"})
    if text.isdigit():
        aliases.update({f"RHEA:{text}", f"Rh_{text}"})
    return aliases


def reaction_aliases_from_row(row: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for column in ("reaction_id", "source_reaction_ids", "source_entries"):
        value = row.get(column)
        if value is None:
            continue
        for item in _listify(value):
            aliases.update(reaction_identifier_aliases(item))
        aliases.update(reaction_identifier_aliases(value))
    return aliases


def load_ec_map(rhea2ec_path: str | Path | None, reaction_smiles_df: pd.DataFrame) -> dict[str, list[str]]:
    ec_map: dict[str, set[str]] = defaultdict(set)
    if "ec_number" in reaction_smiles_df.columns:
        for row in reaction_smiles_df.to_dict("records"):
            aliases = reaction_aliases_from_row(row)
            for ec in _listify(row.get("ec_number")):
                for alias in aliases:
                    ec_map[alias].add(ec)
    if rhea2ec_path is None:
        return {rid: sorted(values) for rid, values in ec_map.items()}
    path = Path(rhea2ec_path)
    if not path.exists():
        raise FileNotFoundError(f"rhea2ec file not found: {path}")
    df = pd.read_csv(path, sep=None, engine="python")
    lowered = {_normalized_column_name(column): column for column in df.columns}
    rid_col = (
        lowered.get("reactionid")
        or lowered.get("rheaid")
        or lowered.get("rhea")
        or lowered.get("masterid")
    )
    ec_col = lowered.get("ecnumber") or lowered.get("ec") or lowered.get("ecid")
    if rid_col is None or ec_col is None:
        raise ValueError(f"Could not find Rhea/reaction id and EC columns in {path}")
    for row in df.to_dict("records"):
        aliases = reaction_identifier_aliases(row.get(rid_col))
        for ec in _listify(row.get(ec_col)):
            for alias in aliases:
                ec_map[alias].add(ec)
    return {rid: sorted(values) for rid, values in ec_map.items()}


def load_participant_texts(*paths: str | Path | None) -> dict[str, list[str]]:
    texts: dict[str, list[str]] = defaultdict(list)
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        df = pd.read_csv(path, sep=None, engine="python")
        lowered = {_normalized_column_name(column): column for column in df.columns}
        rid_col = lowered.get("reactionid") or lowered.get("rheaid") or lowered.get("rhea")
        if rid_col is None:
            continue
        text_cols = [
            column
            for column in df.columns
            if column != rid_col
            and any(
                token in _normalized_column_name(column)
                for token in (
                    "name",
                    "chebi",
                    "smiles",
                    "substrate",
                    "product",
                    "participant",
                    "compound",
                    "molecule",
                )
            )
        ]
        for row in df.to_dict("records"):
            aliases = reaction_identifier_aliases(row.get(rid_col))
            for column in text_cols:
                value = row.get(column)
                if value is not None and not pd.isna(value):
                    for alias in aliases:
                        texts[alias].append(str(value))
    return dict(texts)


def load_atom_mapped_smiles(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Atom-mapped reaction SMILES file not found: {path}")
    df = pd.read_csv(path)
    lowered = {_normalized_column_name(column): column for column in df.columns}
    rid_col = lowered.get("reactionid")
    mapped_col = (
        lowered.get("atommappedreactionsmiles")
        or lowered.get("mappedreactionsmiles")
        or lowered.get("reactionsmiles")
    )
    if rid_col is None or mapped_col is None:
        raise ValueError(f"{path} must contain reaction_id and mapped reaction SMILES columns")
    out: dict[str, str] = {}
    for row in df.to_dict("records"):
        mapped = str(row.get(mapped_col, "")).strip()
        if not mapped:
            continue
        for alias in reaction_identifier_aliases(row.get(rid_col)):
            out[alias] = mapped
    return out


def _build_rxnmapper(use_rxnmapper: bool):
    if not use_rxnmapper:
        return None
    try:
        import pkg_resources  # noqa: F401
    except Exception:
        compatibility = types.ModuleType("pkg_resources")
        compatibility.resource_filename = lambda package, resource: str(
            importlib.resources.files(package).joinpath(resource)
        )
        sys.modules["pkg_resources"] = compatibility
    try:
        from rxnmapper import RXNMapper
    except Exception as exc:
        raise RuntimeError(
            "--use-rxnmapper true requires the rxnmapper package to be installed"
        ) from exc
    return RXNMapper()


def _run_rxnmapper(reaction_smiles: str, mapper) -> tuple[str | None, float | None]:
    if mapper is None:
        return None, None
    result = mapper.get_attention_guided_atom_maps([reaction_smiles])[0]
    mapped = result.get("mapped_rxn")
    confidence = result.get("confidence")
    return str(mapped) if mapped else None, float(confidence) if confidence is not None else None


def _run_rxnmapper_batch(reactions: list[str], mapper) -> list[tuple[str | None, float | None]]:
    if mapper is None or not reactions:
        return [(None, None)] * len(reactions)
    try:
        results = mapper.get_attention_guided_atom_maps(reactions)
    except Exception:
        mapped_one_by_one: list[tuple[str | None, float | None]] = []
        for reaction in reactions:
            try:
                mapped_one_by_one.append(_run_rxnmapper(reaction, mapper))
            except Exception:
                mapped_one_by_one.append((None, None))
        return mapped_one_by_one
    mapped: list[tuple[str | None, float | None]] = []
    for result in results:
        value = result.get("mapped_rxn") if isinstance(result, dict) else None
        confidence = result.get("confidence") if isinstance(result, dict) else None
        mapped.append((str(value) if value else None, float(confidence) if confidence is not None else None))
    return mapped


def _collect_label_vocabs(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    keys = {
        "ec_numbers": "ec_labels",
        "reaction_center_raw_labels": "reaction_center_raw_labels",
        "reaction_center_coarse_labels": "reaction_center_labels",
        "cofactor_labels": "cofactor_labels",
        "core_cofactor_labels": "core_cofactor_labels",
        "metal_ion_labels": "metal_ion_labels",
        "auxiliary_participant_labels": "auxiliary_participant_labels",
        "substrate_class_labels": "substrate_class_labels",
        "product_class_labels": "product_class_labels",
        "substrate_product_transition_labels": "substrate_product_transition_labels",
        "reaction_type_labels": "reaction_type_labels",
    }
    vocabs: dict[str, list[str]] = {}
    for row_key, vocab_key in keys.items():
        values = sorted({label for row in rows for label in _label_values(row.get(row_key))})
        vocabs[vocab_key] = values
    return vocabs


def _label_values(value: Any) -> list[str]:
    if value is None or isinstance(value, str):
        return []
    if isinstance(value, Iterable):
        return [str(label) for label in value if str(label)]
    return []


def _quality_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flag_counts = Counter(flag for row in rows for flag in _label_values(row.get("quality_flags")))
    def count_with(key: str) -> int:
        return sum(1 for row in rows if _label_values(row.get(key)))

    return {
        "num_reactions_total": len(rows),
        "num_reactions_with_valid_smiles": sum(
            1 for row in rows if row["extraction_status"]["smiles"] == "ok"
        ),
        "num_reactions_with_ec": count_with("ec_numbers"),
        "num_reactions_with_drfp": sum(
            1 for row in rows if row["extraction_status"]["drfp"] == "ok"
        ),
        "num_reactions_with_atom_mapping": sum(
            1 for row in rows if row["extraction_status"]["reaction_center"] == "ok"
        ),
        "num_reactions_with_reaction_center_labels": count_with("reaction_center_coarse_labels"),
        "num_reactions_with_cofactor_labels": count_with("cofactor_labels"),
        "num_reactions_with_substrate_product_labels": sum(
            1
            for row in rows
            if _label_values(row.get("substrate_class_labels"))
            or _label_values(row.get("product_class_labels"))
        ),
        "num_reactions_with_reaction_type_labels": count_with("reaction_type_labels"),
        "num_enzymes_total_train": 0,
        "num_enzymes_with_capability_labels": 0,
        "top_cofactor_labels": dict(Counter(label for row in rows for label in _label_values(row["cofactor_labels"])).most_common(25)),
        "top_core_cofactor_labels": dict(Counter(label for row in rows for label in _label_values(row.get("core_cofactor_labels"))).most_common(25)),
        "top_metal_ion_labels": dict(Counter(label for row in rows for label in _label_values(row.get("metal_ion_labels"))).most_common(25)),
        "top_auxiliary_participant_labels": dict(Counter(label for row in rows for label in _label_values(row.get("auxiliary_participant_labels"))).most_common(25)),
        "top_reaction_type_labels": dict(Counter(label for row in rows for label in _label_values(row["reaction_type_labels"])).most_common(25)),
        "top_substrate_class_labels": dict(Counter(label for row in rows for label in _label_values(row["substrate_class_labels"])).most_common(25)),
        "top_product_class_labels": dict(Counter(label for row in rows for label in _label_values(row["product_class_labels"])).most_common(25)),
        "top_substrate_product_transition_labels": dict(Counter(label for row in rows for label in _label_values(row.get("substrate_product_transition_labels"))).most_common(25)),
        "top_reaction_center_labels": dict(Counter(label for row in rows for label in _label_values(row["reaction_center_coarse_labels"])).most_common(25)),
        "missingness_by_ec_level1": {},
        "quality_flag_counts": dict(flag_counts),
        "known_failure_modes": [],
    }


def build_reaction_features(
    *,
    reaction_smiles_path: str | Path,
    out_dir: str | Path,
    train_pairs_path: str | Path | None = None,
    rhea2ec_path: str | Path | None = None,
    chebi_names_path: str | Path | None = None,
    rhea_chebi_smiles_path: str | Path | None = None,
    cofactor_dictionary_path: str | Path | None = None,
    atom_mapped_reaction_smiles_path: str | Path | None = None,
    drfp_bits: int = 2048,
    use_rxnmapper: bool = False,
    rxnmapper_batch_size: int = 64,
) -> pd.DataFrame:
    del train_pairs_path
    out = ensure_dir(out_dir)
    reactions = pd.read_csv(reaction_smiles_path)
    if "reaction_smiles" not in reactions.columns:
        raise ValueError("reaction_smiles CSV must contain a reaction_smiles column")
    if "reaction_id" not in reactions.columns:
        reactions["reaction_id"] = ""
    ec_map = load_ec_map(rhea2ec_path, reactions)
    participant_texts = load_participant_texts(chebi_names_path, rhea_chebi_smiles_path)
    cofactor_aliases = load_cofactor_aliases(cofactor_dictionary_path)
    mapped_by_id = load_atom_mapped_smiles(atom_mapped_reaction_smiles_path)
    rxnmapper = _build_rxnmapper(use_rxnmapper)

    prepared_rows: list[dict[str, Any]] = []
    seen_reaction_ids: set[str] = set()
    for row_index, row in enumerate(reactions.to_dict("records")):
        raw_smiles = None if pd.isna(row.get("reaction_smiles")) else str(row["reaction_smiles"])
        canonical = canonicalize_reaction_smiles(raw_smiles or "")
        reaction_id = str(row.get("reaction_id", "")).strip()
        if not reaction_id and canonical:
            reaction_id = generated_reaction_id(canonical)
        if not reaction_id:
            reaction_id = f"RXN_ROW_{row_index}"
        if reaction_id in seen_reaction_ids:
            raise ValueError(f"Duplicate reaction_id in reaction SMILES table: {reaction_id}")
        seen_reaction_ids.add(reaction_id)
        aliases = reaction_aliases_from_row({**row, "reaction_id": reaction_id})
        prepared_rows.append(
            {
                "row_index": row_index,
                "row": row,
                "raw_smiles": raw_smiles,
                "canonical": canonical,
                "reaction_id": reaction_id,
                "aliases": aliases,
            }
        )

    rxnmapper_mapped_by_id: dict[str, tuple[str, float | None]] = {}
    if rxnmapper is not None:
        to_map = [
            (row["reaction_id"], row["canonical"])
            for row in prepared_rows
            if row["canonical"]
            and not any(alias in mapped_by_id for alias in row["aliases"])
        ]
        batch_size = max(1, int(rxnmapper_batch_size))
        print(
            f"RXNMapper mapping {len(to_map)} reactions with batch_size={batch_size}",
            flush=True,
        )
        for start in range(0, len(to_map), batch_size):
            batch = to_map[start : start + batch_size]
            mapped_batch = _run_rxnmapper_batch([canonical for _, canonical in batch], rxnmapper)
            for (reaction_id, _), (mapped, confidence) in zip(batch, mapped_batch, strict=True):
                if mapped:
                    rxnmapper_mapped_by_id[reaction_id] = (mapped, confidence)
            done = min(start + batch_size, len(to_map))
            if done == len(to_map) or done % max(batch_size * 10, 1) == 0:
                print(f"RXNMapper mapped {done}/{len(to_map)} reactions", flush=True)

    rows: list[dict[str, Any]] = []
    drfp_matrix: list[np.ndarray] = []
    reaction_id_to_index: dict[str, int] = {}

    for prepared in prepared_rows:
        row = prepared["row"]
        raw_smiles = prepared["raw_smiles"]
        canonical = prepared["canonical"]
        reaction_id = prepared["reaction_id"]
        aliases = prepared["aliases"]

        status = {
            "smiles": "ok" if canonical else "invalid",
            "drfp": "skipped",
            "reaction_center": "skipped",
            "cofactor": "unknown",
            "substrate_product_class": "unknown",
            "reaction_type": "unknown",
        }
        quality_flags: list[str] = []
        if canonical is None:
            quality_flags.append("invalid_reaction_smiles")

        drfp_arr = np.zeros(drfp_bits, dtype=np.uint8)
        drfp_active: list[int] = []
        if canonical:
            try:
                drfp_arr, drfp_active = compute_drfp_active_bits(canonical, drfp_bits)
                status["drfp"] = "ok"
            except Exception:
                status["drfp"] = "failed"
                quality_flags.append("drfp_failed")

        raw_center: list[str] = []
        coarse_center: list[str] = []
        reaction_center_mapping_confidence: float | None = None
        if canonical:
            mapped = next((mapped_by_id[alias] for alias in aliases if alias in mapped_by_id), None)
            if mapped is None:
                mapped_pair = rxnmapper_mapped_by_id.get(reaction_id)
                if mapped_pair is not None:
                    mapped, reaction_center_mapping_confidence = mapped_pair
            if mapped:
                try:
                    raw_center = extract_reaction_center_raw_labels(mapped)
                    coarse_center = coarse_reaction_center_labels(raw_center)
                    status["reaction_center"] = "ok"
                except Exception:
                    status["reaction_center"] = "failed"
                    quality_flags.append("no_reaction_center")
            else:
                status["reaction_center"] = "no_atom_map"
                quality_flags.append("no_reaction_center")

        ec_numbers = sorted({ec for alias in aliases for ec in ec_map.get(alias, [])})
        cofactor_texts = [text for alias in aliases for text in participant_texts.get(alias, [])]
        if raw_smiles:
            cofactor_texts.append(raw_smiles)
        cofactors, cofactor_flags, cofactor_status = extract_cofactor_labels(
            cofactor_texts,
            cofactor_aliases=cofactor_aliases,
        )
        cofactor_tiers = split_cofactor_label_tiers(cofactors)
        quality_flags.extend(cofactor_flags)
        status["cofactor"] = cofactor_status

        substrate_unfiltered: list[str] = []
        product_unfiltered: list[str] = []
        substrate_labels: list[str] = []
        product_labels: list[str] = []
        transition_labels: list[str] = []
        if canonical:
            substrate_unfiltered, product_unfiltered, _, _ = (
                classify_reaction_substrates_products(canonical)
            )
            labels_to_filter = cofactor_labels_for_substrate_filter(cofactors)
            substrate_labels, product_labels, class_flags, class_status = (
                classify_reaction_substrates_products(
                    canonical,
                    cofactor_aliases=cofactor_aliases,
                    skip_cofactor_labels=labels_to_filter,
                )
            )
            quality_flags.extend(class_flags)
            status["substrate_product_class"] = class_status
            transition_labels = substrate_product_transition_labels(substrate_labels, product_labels)

        types, type_flags, type_status = reaction_type_labels(
            coarse_center,
            cofactors,
            substrate_labels,
            product_labels,
            ec_numbers,
        )
        quality_flags.extend(type_flags)
        status["reaction_type"] = type_status

        reaction_id_to_index[reaction_id] = len(rows)
        drfp_matrix.append(drfp_arr)
        rows.append(
            {
                "reaction_id": reaction_id,
                "canonical_reaction_smiles": canonical,
                "raw_reaction_smiles": raw_smiles,
                "ec_numbers": ec_numbers,
                "drfp_active_bits": drfp_active,
                "drfp_num_bits": int(drfp_bits),
                "reaction_center_raw_labels": raw_center,
                "reaction_center_coarse_labels": coarse_center,
                "reaction_center_mapping_confidence": reaction_center_mapping_confidence,
                "cofactor_labels": cofactors,
                "core_cofactor_labels": cofactor_tiers["core_cofactor_labels"],
                "metal_ion_labels": cofactor_tiers["metal_ion_labels"],
                "auxiliary_participant_labels": cofactor_tiers["auxiliary_participant_labels"],
                "substrate_class_labels_unfiltered": substrate_unfiltered,
                "product_class_labels_unfiltered": product_unfiltered,
                "substrate_class_labels": substrate_labels,
                "product_class_labels": product_labels,
                "substrate_product_transition_labels": transition_labels,
                "reaction_type_labels": types,
                "extraction_status": status,
                "quality_flags": sorted(set(quality_flags)),
            }
        )
        if len(rows) % 5000 == 0 or len(rows) == len(prepared_rows):
            print(f"Built reaction features for {len(rows)}/{len(prepared_rows)} reactions", flush=True)

    features = pd.DataFrame(rows)
    features.to_parquet(out / "reaction_features.parquet", index=False)
    np.savez_compressed(
        out / "reaction_drfp.npz",
        vectors=np.stack(drfp_matrix),
        ids=np.array(list(reaction_id_to_index)),
    )
    write_json(out / "reaction_id_to_index.json", reaction_id_to_index)
    write_json(out / "label_vocabs.json", _collect_label_vocabs(rows))
    write_json(out / "annotation_dictionaries.json", annotation_dictionaries_payload())
    write_json(out / "annotation_quality_report.json", _quality_report(rows))
    sample = features.sample(n=min(50, len(features)), random_state=13) if len(features) else features
    sample[
        [
            "reaction_id",
            "canonical_reaction_smiles",
            "ec_numbers",
            "cofactor_labels",
            "core_cofactor_labels",
            "metal_ion_labels",
            "auxiliary_participant_labels",
            "reaction_center_coarse_labels",
            "substrate_class_labels",
            "product_class_labels",
            "substrate_product_transition_labels",
            "reaction_type_labels",
            "quality_flags",
        ]
    ].to_csv(out / "reaction_feature_examples.csv", index=False)
    return features
