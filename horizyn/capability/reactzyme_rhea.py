"""Recover directional Rhea reactions for ReactZyme train rows.

ReactZyme split files store reaction inputs as unordered molecule sets.  For
chemistry-supervised capability pretraining we only use rows that can be backed
by an explicit Rhea substrate/product definition.
"""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


REACTZYME_TRAIN_SPLITS = ("time", "enzyme_smi", "reaction_smi")


def clean_sequence(sequence: str) -> str:
    return str(sequence).replace(" ", "").replace("\n", "").replace("\r", "").upper()


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def protein_uid(sequence: str) -> str:
    return f"uprot_{sha1_text(clean_sequence(sequence))[:16]}"


def reaction_uid(reaction_smiles: str) -> str:
    return f"rxn_{sha1_text(str(reaction_smiles).strip())[:16]}"


def parse_semicolon_list(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip() and item.strip().lower() != "nan"
    ]


def directional_rhea_smiles(substrate: str, product: str) -> str:
    substrate = str(substrate).strip()
    product = str(product).strip()
    if not substrate or not product:
        raise ValueError("Rhea substrate and product must both be non-empty")
    return f"{substrate}>>{product}"


@lru_cache(maxsize=100_000)
def canonical_molecule(smiles: str) -> str:
    """Canonicalize one molecule while retaining isotope and stereo information."""

    from rdkit import Chem

    value = str(smiles).strip()
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return value
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=30_000)
def molecule_multiset(smiles: str) -> Counter[str]:
    """Return the canonical dot-component multiset for an unordered reaction row."""

    return Counter(
        canonical_molecule(component)
        for component in str(smiles).split(".")
        if component.strip()
    )


def row_specific_rhea_matches(
    source_molecule_set: str,
    candidate_rhea_ids: list[str],
    rhea_reactions: dict[str, dict[str, str]],
) -> list[tuple[str, float, str]]:
    """Match sequence-associated Rhea IDs to one ReactZyme molecule-set row."""

    source = molecule_multiset(source_molecule_set)
    exact: list[tuple[str, float, str]] = []
    set_only: list[tuple[str, float, str]] = []
    for rhea_id in candidate_rhea_ids:
        reaction = rhea_reactions.get(rhea_id)
        if reaction is None:
            continue
        candidate = molecule_multiset(
            f"{reaction['substrate']}.{reaction['product']}"
        )
        if candidate == source:
            exact.append((rhea_id, 1.0, "exact_molecule_multiset"))
        elif set(candidate) == set(source):
            set_only.append((rhea_id, 0.75, "molecule_set_without_stoichiometry"))
    return exact if exact else set_only


def load_sequence_rhea_index(cleaned_uniprot_rhea_path: str | Path) -> dict[str, dict[str, set[str]]]:
    """Map exact protein sequence to Rhea IDs, UniProt entries, and EC numbers."""

    df = pd.read_csv(cleaned_uniprot_rhea_path, sep="\t")
    required = {"Entry", "EC number", "Rhea ID", "Sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{cleaned_uniprot_rhea_path} missing columns: {sorted(missing)}")

    by_sequence: dict[str, dict[str, set[str]]] = {}
    for row in df.to_dict("records"):
        sequence = clean_sequence(row["Sequence"])
        if not sequence:
            continue
        record = by_sequence.setdefault(
            sequence,
            {"entries": set(), "ec_numbers": set(), "rhea_ids": set()},
        )
        entry = str(row.get("Entry", "")).strip()
        if entry:
            record["entries"].add(entry)
        record["ec_numbers"].update(parse_semicolon_list(row.get("EC number")))
        record["rhea_ids"].update(parse_semicolon_list(row.get("Rhea ID")))
    return by_sequence


def load_rhea_reaction_map(rhea_molecules_path: str | Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(rhea_molecules_path, sep="\t")
    required = {"Rhea ID", "substrate", "product"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{rhea_molecules_path} missing columns: {sorted(missing)}")

    out: dict[str, dict[str, str]] = {}
    for row in df.to_dict("records"):
        rhea_id = str(row["Rhea ID"]).strip()
        substrate = str(row.get("substrate", "")).strip()
        product = str(row.get("product", "")).strip()
        if not rhea_id or not substrate or not product:
            continue
        reaction_smiles = directional_rhea_smiles(substrate, product)
        out[rhea_id] = {
            "rhea_id": rhea_id,
            "substrate": substrate,
            "product": product,
            "reaction_smiles": reaction_smiles,
            "reaction_id": reaction_uid(reaction_smiles),
        }
    return out


def _join(values: set[str] | list[str]) -> str:
    return "|".join(sorted(str(value) for value in values if str(value)))


def reconstruct_reactzyme_train_rhea(
    *,
    reactzyme_eval_root: str | Path,
    cleaned_uniprot_rhea_path: str | Path,
    rhea_molecules_path: str | Path,
    splits: tuple[str, ...] = REACTZYME_TRAIN_SPLITS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return collapsed pair/reaction tables plus member-level audit rows."""

    sequence_index = load_sequence_rhea_index(cleaned_uniprot_rhea_path)
    rhea_reactions = load_rhea_reaction_map(rhea_molecules_path)
    eval_root = Path(reactzyme_eval_root)

    members: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "reactzyme_eval_root": str(eval_root),
        "cleaned_uniprot_rhea_path": str(cleaned_uniprot_rhea_path),
        "rhea_molecules_path": str(rhea_molecules_path),
        "splits": {},
        "num_sequence_index_entries": len(sequence_index),
        "num_rhea_reactions": len(rhea_reactions),
    }

    for split in splits:
        path = eval_root / split / "train_pairs.csv"
        df = pd.read_csv(path)
        required = {"reaction_id", "protein_id", "reaction_smiles", "protein_sequence"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        split_stats = {
            "input_rows": int(len(df)),
            "deduplicated_source_pairs": 0,
            "rows_with_sequence_rhea": 0,
            "rows_with_single_rhea": 0,
            "rows_with_multiple_rhea": 0,
            "rows_without_sequence_match": 0,
            "rows_without_rhea_molecule_record": 0,
            "rows_without_molecule_match": 0,
            "rows_with_exact_molecule_match": 0,
            "rows_with_set_only_match": 0,
            "expanded_directional_members": 0,
        }
        seen_source_pairs: set[tuple[str, str]] = set()
        source_split = f"{split}_train"

        for row_index, row in enumerate(df.to_dict("records")):
            source_pair = (str(row["reaction_id"]), str(row["protein_id"]))
            if source_pair in seen_source_pairs:
                continue
            seen_source_pairs.add(source_pair)
            split_stats["deduplicated_source_pairs"] += 1

            sequence = clean_sequence(row["protein_sequence"])
            sequence_record = sequence_index.get(sequence)
            if sequence_record is None:
                split_stats["rows_without_sequence_match"] += 1
                continue

            sequence_rhea_ids = sorted(
                rhea_id
                for rhea_id in sequence_record["rhea_ids"]
                if rhea_id in rhea_reactions
            )
            if not sequence_rhea_ids:
                split_stats["rows_without_rhea_molecule_record"] += 1
                continue

            matched_rhea = row_specific_rhea_matches(
                str(row["reaction_smiles"]),
                sequence_rhea_ids,
                rhea_reactions,
            )
            if not matched_rhea:
                split_stats["rows_without_molecule_match"] += 1
                continue

            split_stats["rows_with_sequence_rhea"] += 1
            if matched_rhea[0][2] == "exact_molecule_multiset":
                split_stats["rows_with_exact_molecule_match"] += 1
            else:
                split_stats["rows_with_set_only_match"] += 1
            if len(matched_rhea) == 1:
                split_stats["rows_with_single_rhea"] += 1
                row_reconstruction_status = "sequence_rhea_single"
            else:
                split_stats["rows_with_multiple_rhea"] += 1
                row_reconstruction_status = "sequence_rhea_multi_expanded"

            for rhea_id, match_confidence, molecule_match_status in matched_rhea:
                reaction = rhea_reactions[rhea_id]
                members.append(
                    {
                        "reaction_id": reaction["reaction_id"],
                        "reaction_smiles": reaction["reaction_smiles"],
                        "protein_id": protein_uid(sequence),
                        "protein_uid": protein_uid(sequence),
                        "source_datasets": "reactzyme",
                        "source_splits": source_split,
                        "source_name": f"reactzyme_{source_split}",
                        "source_reaction_id": str(row["reaction_id"]),
                        "source_protein_id": str(row["protein_id"]),
                        "source_pair_index": row_index,
                        "source_molecule_set_smiles": str(row["reaction_smiles"]),
                        "rhea_id": rhea_id,
                        "molecule_match_status": molecule_match_status,
                        "match_confidence": match_confidence,
                        "rhea_entries": _join(sequence_record["entries"]),
                        "ec_numbers": _join(sequence_record["ec_numbers"]),
                        "reconstruction_status": row_reconstruction_status,
                    }
                )
                split_stats["expanded_directional_members"] += 1

        report["splits"][split] = split_stats

    pair_groups: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    reaction_groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for member in members:
        pair_groups.setdefault((member["reaction_id"], member["protein_id"]), []).append(member)
        reaction_groups.setdefault(member["reaction_id"], []).append(member)

    pair_rows: list[dict[str, Any]] = []
    for idx, ((reaction_id, protein_id), group) in enumerate(pair_groups.items()):
        source_entries = {
            (
                f"{member['source_name']}:{member['source_reaction_id']}:"
                f"{member['source_protein_id']}:{member['source_pair_index']}:"
                f"{member['rhea_id']}"
            )
            for member in group
        }
        pair_rows.append(
            {
                "pr_id": idx,
                "reaction_id": reaction_id,
                "protein_id": protein_id,
                "member_count": len(group),
                "protein_uid": protein_id,
                "source_datasets": "reactzyme",
                "source_splits": _join({member["source_splits"] for member in group}),
                "source_entries": _join(source_entries),
            }
        )

    reaction_rows: list[dict[str, Any]] = []
    for reaction_id, group in reaction_groups.items():
        reaction_smiles = group[0]["reaction_smiles"]
        rhea_ids = {member["rhea_id"] for member in group}
        source_entries = {member["source_name"] for member in group}
        reaction_rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_smiles": reaction_smiles,
                "source_entries": _join(source_entries | {"reactzyme_rhea_reconstructed"}),
                "source_reaction_ids": _join(rhea_ids),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    reaction_df = pd.DataFrame(reaction_rows)
    member_df = pd.DataFrame(members)

    report["expanded_directional_members"] = int(len(member_df))
    report["collapsed_directional_pairs"] = int(len(pair_df))
    report["collapsed_directional_reactions"] = int(len(reaction_df))
    report["collapsed_directional_proteins"] = (
        int(pair_df["protein_id"].nunique()) if len(pair_df) else 0
    )
    report["status_counts"] = (
        member_df["reconstruction_status"].value_counts().sort_index().to_dict()
        if len(member_df)
        else {}
    )
    return pair_df, reaction_df, member_df, report


def combine_source_collapse_variant(
    *,
    base_pairs: pd.DataFrame,
    base_reactions: pd.DataFrame,
    reconstructed_pairs: pd.DataFrame,
    reconstructed_reactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace ReactZyme molecule-set rows with reconstructed directional rows."""

    if "source_datasets" not in base_pairs.columns:
        raise ValueError("base_pairs must contain source_datasets")
    if "source_entries" not in base_reactions.columns:
        raise ValueError("base_reactions must contain source_entries")

    non_reactzyme_pairs = base_pairs[
        ~base_pairs["source_datasets"].fillna("").astype(str).str.contains("reactzyme")
    ].copy()
    needed_base_reactions = set(non_reactzyme_pairs["reaction_id"].astype(str))
    non_reactzyme_reactions = base_reactions[
        base_reactions["reaction_id"].astype(str).isin(needed_base_reactions)
    ].copy()

    pair_fields = [
        "reaction_id",
        "protein_id",
        "member_count",
        "protein_uid",
        "source_datasets",
        "source_splits",
        "source_entries",
    ]
    all_pairs = pd.concat(
        [non_reactzyme_pairs[pair_fields], reconstructed_pairs[pair_fields]],
        ignore_index=True,
    )

    combined_pair_rows: list[dict[str, Any]] = []
    for idx, ((reaction_id, protein_id), group) in enumerate(
        all_pairs.groupby(["reaction_id", "protein_id"], sort=False)
    ):
        combined_pair_rows.append(
            {
                "pr_id": idx,
                "reaction_id": reaction_id,
                "protein_id": protein_id,
                "member_count": int(pd.to_numeric(group["member_count"], errors="coerce").fillna(1).sum()),
                "protein_uid": str(group["protein_uid"].iloc[0]),
                "source_datasets": _join(set("|".join(group["source_datasets"].astype(str)).split("|"))),
                "source_splits": _join(set("|".join(group["source_splits"].astype(str)).split("|"))),
                "source_entries": _join(set("|".join(group["source_entries"].astype(str)).split("|"))),
            }
        )
    combined_pairs = pd.DataFrame(combined_pair_rows)

    reaction_fields = ["reaction_id", "reaction_smiles", "source_entries", "source_reaction_ids"]
    all_reactions = pd.concat(
        [non_reactzyme_reactions[reaction_fields], reconstructed_reactions[reaction_fields]],
        ignore_index=True,
    )

    combined_reaction_rows: list[dict[str, Any]] = []
    for reaction_id, group in all_reactions.groupby("reaction_id", sort=False):
        combined_reaction_rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_smiles": str(group["reaction_smiles"].iloc[0]),
                "source_entries": _join(set("|".join(group["source_entries"].astype(str)).split("|"))),
                "source_reaction_ids": _join(
                    set("|".join(group["source_reaction_ids"].astype(str)).split("|"))
                ),
            }
        )
    combined_reactions = pd.DataFrame(combined_reaction_rows)
    return combined_pairs, combined_reactions
