#!/usr/bin/env python3
"""Build a biologically oriented, pair-level audit of the ReactZyme splits.

The released ReactZyme inputs are UniProt-specific *molecular sets*, not
directional atom-mapped reactions.  This audit therefore keeps molecular-set
properties separate from directional Rhea metadata and preserves the original
generic/wildcard provenance that ReactZyme normalizes by replacing ``*`` with
``C``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Lipinski

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_PYTHON_DEPS = ROOT / ".deps/python"
if LOCAL_PYTHON_DEPS.exists() and str(LOCAL_PYTHON_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_PYTHON_DEPS))

try:
    import Levenshtein
except ImportError:  # Optional; MMseqs results remain useful without it.
    Levenshtein = None

from horizyn.capability.cofactors import split_cofactor_label_tiers
from horizyn.capability.enzyme_cofactor_annotations import (
    extract_cofactor_labels_from_molecule_set,
    extract_cofactor_labels_from_uniprotkb_comment,
)


PROTOCOLS = ("time", "enzyme_smi", "reaction_smi")
SUBSETS = ("train", "validation", "test")
EC_CLASSES = {
    "1": "oxidoreductases",
    "2": "transferases",
    "3": "hydrolases",
    "4": "lyases",
    "5": "isomerases",
    "6": "ligases",
    "7": "translocases",
}
METAL_ATOMIC_NUMBERS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
}


def stable_id(prefix: str, value: str, length: int = 16) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def split_values(value: Any, separator: str = ";") -> list[str]:
    if value is None or pd.isna(value):
        return []
    return sorted(
        {
            item.strip()
            for item in str(value).split(separator)
            if item.strip() and item.strip().lower() != "nan"
        }
    )


def count_joined(value: Any) -> int:
    text = str(value or "")
    return len([item for item in text.split("|") if item])


def percentile_summary(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "mean": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "p05": float(numeric.quantile(0.05)),
        "median": float(numeric.median()),
        "p95": float(numeric.quantile(0.95)),
        "max": float(numeric.max()),
    }


def ec_prefixes(ec_values: str, depth: int) -> set[str]:
    prefixes: set[str] = set()
    for ec in str(ec_values or "").split("|"):
        parts = ec.split(".")
        if len(parts) < depth or any(part in {"", "-"} for part in parts[:depth]):
            continue
        prefixes.add(".".join(parts[:depth]))
    return prefixes


def summarize_ec_novelty(ec_values: str, train_prefixes: dict[int, set[str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for depth in range(1, 5):
        values = ec_prefixes(ec_values, depth)
        seen = values & train_prefixes[depth]
        result[f"ec{depth}_count"] = len(values)
        result[f"ec{depth}_seen_train_count"] = len(seen)
        result[f"ec{depth}_all_seen_train"] = bool(values) and values <= train_prefixes[depth]
        result[f"ec{depth}_any_seen_train"] = bool(seen)
    return result


def aggregate_source_rows(source: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = [
        *keys,
        "accession",
        "ec_numbers",
        "rhea_ids",
        "directional_rhea_ids",
        "enzyme_cofactor_comment_present",
        "enzyme_cofactor_labels",
        "enzyme_core_cofactor_labels",
        "creation_date",
        "source_cleaned_supported",
    ]
    states: dict[tuple[Any, ...], dict[str, Any]] = {}
    for values in source[columns].itertuples(index=False, name=None):
        key = tuple(values[: len(keys)])
        (
            accession,
            ec_numbers,
            rhea_ids,
            directional_rhea_ids,
            enzyme_cofactor_comment_present,
            enzyme_cofactor_labels,
            enzyme_core_cofactor_labels,
            creation_date,
            cleaned_supported,
        ) = values[len(keys) :]
        state = states.setdefault(
            key,
            {
                "accessions": set(),
                "ec_numbers": set(),
                "rhea_ids": set(),
                "directional_rhea_ids": set(),
                "enzyme_cofactor_labels": set(),
                "enzyme_core_cofactor_labels": set(),
                "enzyme_cofactor_comment_present": False,
                "creation_dates": [],
                "source_cleaned_supported": False,
            },
        )
        state["accessions"].add(str(accession))
        state["ec_numbers"].update(item for item in str(ec_numbers).split("|") if item)
        state["rhea_ids"].update(item for item in str(rhea_ids).split("|") if item)
        state["directional_rhea_ids"].update(
            item for item in str(directional_rhea_ids).split("|") if item
        )
        state["enzyme_cofactor_labels"].update(
            item for item in str(enzyme_cofactor_labels).split("|") if item
        )
        state["enzyme_core_cofactor_labels"].update(
            item for item in str(enzyme_core_cofactor_labels).split("|") if item
        )
        state["enzyme_cofactor_comment_present"] = bool(
            state["enzyme_cofactor_comment_present"] or enzyme_cofactor_comment_present
        )
        if not pd.isna(creation_date):
            state["creation_dates"].append(creation_date)
        state["source_cleaned_supported"] = bool(
            state["source_cleaned_supported"] or cleaned_supported
        )

    records: list[dict[str, Any]] = []
    for key, state in states.items():
        ec_numbers = "|".join(sorted(state["ec_numbers"]))
        rhea_ids = "|".join(sorted(state["rhea_ids"]))
        directional_rhea_ids = "|".join(sorted(state["directional_rhea_ids"]))
        enzyme_cofactor_labels = "|".join(sorted(state["enzyme_cofactor_labels"]))
        enzyme_core_cofactor_labels = "|".join(
            sorted(state["enzyme_core_cofactor_labels"])
        )
        dates = state["creation_dates"]
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "source_accessions": "|".join(sorted(state["accessions"])),
                "source_accession_count": len(state["accessions"]),
                "ec_numbers": ec_numbers,
                "ec_number_count": count_joined(ec_numbers),
                "ec_top_classes": "|".join(sorted(ec_prefixes(ec_numbers, 1))),
                "rhea_ids": rhea_ids,
                "rhea_id_count": count_joined(rhea_ids),
                "directional_rhea_ids": directional_rhea_ids,
                "directional_rhea_id_count": count_joined(directional_rhea_ids),
                "enzyme_cofactor_comment_present": state[
                    "enzyme_cofactor_comment_present"
                ],
                "enzyme_cofactor_labels": enzyme_cofactor_labels,
                "enzyme_core_cofactor_labels": enzyme_core_cofactor_labels,
                "has_enzyme_cofactor_label": bool(enzyme_cofactor_labels),
                "has_enzyme_core_cofactor_label": bool(enzyme_core_cofactor_labels),
                "creation_date_min": min(dates) if dates else pd.NaT,
                "creation_date_max": max(dates) if dates else pd.NaT,
                "source_cleaned_supported": state["source_cleaned_supported"],
            }
        )
    return pd.DataFrame.from_records(records)


def load_source_metadata(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(args.uniprot_rhea, sep="\t", dtype=str)
    cleaned = pd.read_csv(args.cleaned_uniprot_rhea, sep="\t", dtype=str)
    molecules = pd.read_csv(args.uniprot_molecules, sep="\t", dtype=str)
    rhea = pd.read_csv(args.rhea_molecules, sep="\t", dtype=str)

    full = full.rename(
        columns={
            "Entry": "accession",
            "EC number": "ec_numbers",
            "Rhea ID": "rhea_ids",
            "Date of creation": "creation_date_raw",
            "Sequence": "protein_sequence",
        }
    )
    full["protein_id"] = full["protein_sequence"].map(lambda value: stable_id("prot", value))
    full["creation_date"] = pd.to_datetime(full["creation_date_raw"], format="%Y%m%d", errors="coerce")
    full["ec_numbers"] = full["ec_numbers"].map(
        lambda value: "|".join(split_values(value))
    )
    full["rhea_ids"] = full["rhea_ids"].map(
        lambda value: "|".join(split_values(value))
    )
    cleaned_accessions = set(cleaned["Entry"].astype(str))
    full["source_cleaned_supported"] = full["accession"].isin(cleaned_accessions)

    molecules = molecules.rename(
        columns={"uniprot_id": "accession", "molecules": "raw_molecule_set_smiles"}
    )
    molecules["reaction_smiles"] = molecules["raw_molecule_set_smiles"].str.replace(
        "*", "C", regex=False
    )
    molecules["reaction_id"] = molecules["reaction_smiles"].map(lambda value: stable_id("rxn", value))
    molecules["source_wildcard_atom_count"] = molecules["raw_molecule_set_smiles"].str.count(
        r"\*"
    )

    source = full.merge(molecules, on="accession", how="inner", validate="one_to_one")
    rhea_ids_available = set(rhea["Rhea ID"].astype(str))
    source["directional_rhea_ids"] = source["rhea_ids"].map(
        lambda value: "|".join(
            rhea_id
            for rhea_id in str(value).split("|")
            if rhea_id and rhea_id in rhea_ids_available
        )
    )
    source["directional_rhea_id_count"] = source["directional_rhea_ids"].map(count_joined)
    source["enzyme_cofactor_comment_present"] = False
    source["enzyme_cofactor_labels"] = ""
    source["enzyme_core_cofactor_labels"] = ""
    if args.uniprotkb_cofactor_cache is not None:
        cache = pd.read_csv(args.uniprotkb_cofactor_cache, sep="\t", dtype=str)
        if not {"Entry", "Cofactor"}.issubset(cache.columns):
            raise ValueError(
                f"{args.uniprotkb_cofactor_cache} must contain Entry and Cofactor columns"
            )
        cache = cache.drop_duplicates("Entry", keep="last")
        comment_present: dict[str, bool] = {}
        cofactor_labels: dict[str, str] = {}
        core_labels: dict[str, str] = {}
        for row in cache.itertuples(index=False):
            accession = str(row.Entry)
            comment = "" if pd.isna(row.Cofactor) else str(row.Cofactor).strip()
            labels, _flags = extract_cofactor_labels_from_uniprotkb_comment(comment)
            tiers = split_cofactor_label_tiers(labels)
            comment_present[accession] = bool(comment)
            cofactor_labels[accession] = "|".join(labels)
            core_labels[accession] = "|".join(tiers["core_cofactor_labels"])
        source["enzyme_cofactor_comment_present"] = (
            source["accession"].map(comment_present).fillna(False).astype(bool)
        )
        source["enzyme_cofactor_labels"] = (
            source["accession"].map(cofactor_labels).fillna("")
        )
        source["enzyme_core_cofactor_labels"] = (
            source["accession"].map(core_labels).fillna("")
        )
    return source, molecules, rhea


def molecule_set_features(reaction_smiles: str) -> tuple[dict[str, Any], DataStructs.ExplicitBitVect]:
    tokens = [token for token in str(reaction_smiles).split(".") if token]
    unique_tokens = set(tokens)
    fingerprint = DataStructs.ExplicitBitVect(2048)
    valid = 0
    atoms = 0
    heavy_atoms = 0
    max_heavy_atoms = 0
    rings = 0
    chiral_centers = 0
    formal_charge_abs = 0
    metal_atoms = 0
    phosphorus_atoms = 0
    sulfur_atoms = 0
    parse_failures = 0
    for token in tokens:
        molecule = Chem.MolFromSmiles(token)
        if molecule is None:
            parse_failures += 1
            continue
        valid += 1
        atom_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
        atoms += len(atom_numbers)
        token_heavy_atoms = int(molecule.GetNumHeavyAtoms())
        heavy_atoms += token_heavy_atoms
        max_heavy_atoms = max(max_heavy_atoms, token_heavy_atoms)
        rings += int(Lipinski.RingCount(molecule))
        chiral_centers += len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True))
        formal_charge_abs += sum(abs(atom.GetFormalCharge()) for atom in molecule.GetAtoms())
        metal_atoms += sum(number in METAL_ATOMIC_NUMBERS for number in atom_numbers)
        phosphorus_atoms += atom_numbers.count(15)
        sulfur_atoms += atom_numbers.count(16)
        token_fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        fingerprint |= token_fp

    labels, _flags = extract_cofactor_labels_from_molecule_set(reaction_smiles)
    tiers = split_cofactor_label_tiers(labels)
    return (
        {
            "molecule_count": len(tokens),
            "unique_molecule_count": len(unique_tokens),
            "valid_molecule_count": valid,
            "molecule_parse_failure_count": parse_failures,
            "molecule_parse_complete": valid == len(tokens),
            "total_atom_count": atoms,
            "total_heavy_atom_count": heavy_atoms,
            "max_component_heavy_atom_count": max_heavy_atoms,
            "total_ring_count": rings,
            "stereocenter_count": chiral_centers,
            "absolute_formal_charge": formal_charge_abs,
            "metal_atom_count": metal_atoms,
            "phosphorus_atom_count": phosphorus_atoms,
            "sulfur_atom_count": sulfur_atoms,
            "has_stereochemistry": chiral_centers > 0 or "@" in reaction_smiles,
            "has_metal": metal_atoms > 0,
            "has_phosphorus": phosphorus_atoms > 0,
            "has_sulfur": sulfur_atoms > 0,
            "cofactor_labels": "|".join(labels),
            "core_cofactor_labels": "|".join(tiers["core_cofactor_labels"]),
            "metal_ion_labels": "|".join(tiers["metal_ion_labels"]),
            "auxiliary_participant_labels": "|".join(
                tiers["auxiliary_participant_labels"]
            ),
            "has_any_cofactor_label": bool(labels),
            "has_core_cofactor_label": bool(tiers["core_cofactor_labels"]),
        },
        fingerprint,
    )


def build_reaction_annotations(
    molecules: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, DataStructs.ExplicitBitVect]]:
    records: list[dict[str, Any]] = []
    fingerprints: dict[str, DataStructs.ExplicitBitVect] = {}
    for reaction_id, group in molecules.groupby("reaction_id", sort=False):
        reaction_smiles = str(group["reaction_smiles"].iloc[0])
        features, fingerprint = molecule_set_features(reaction_smiles)
        raw_values = sorted(set(group["raw_molecule_set_smiles"].astype(str)))
        records.append(
            {
                "reaction_id": reaction_id,
                "reaction_smiles": reaction_smiles,
                "raw_molecule_set_smiles": raw_values[0],
                "source_raw_string_count": len(raw_values),
                "source_accession_count": int(group["accession"].nunique()),
                "source_wildcard_atom_count": int(group["source_wildcard_atom_count"].max()),
                "source_had_generic_wildcard": bool(
                    (group["source_wildcard_atom_count"] > 0).any()
                ),
                "is_directional_reaction": ">>" in reaction_smiles,
                **features,
            }
        )
        fingerprints[str(reaction_id)] = fingerprint
    return pd.DataFrame.from_records(records), fingerprints


def add_reaction_source_annotations(
    reactions: pd.DataFrame, source: pd.DataFrame
) -> pd.DataFrame:
    source_annotations = aggregate_source_rows(source, ["reaction_id"])
    protein_counts = source.groupby("reaction_id")["protein_id"].nunique()
    source_annotations["associated_protein_count"] = source_annotations["reaction_id"].map(
        protein_counts
    )
    source_annotations["associated_ec_top_class_count"] = source_annotations[
        "ec_top_classes"
    ].map(count_joined)
    keep = {
        "reaction_id": "reaction_id",
        "source_accessions": "associated_source_accessions",
        "ec_numbers": "associated_ec_numbers",
        "ec_number_count": "associated_ec_number_count",
        "ec_top_classes": "associated_ec_top_classes",
        "associated_ec_top_class_count": "associated_ec_top_class_count",
        "rhea_ids": "associated_rhea_ids",
        "rhea_id_count": "associated_rhea_id_count",
        "associated_protein_count": "associated_protein_count",
    }
    source_annotations = source_annotations[list(keep)].rename(columns=keep)
    return reactions.merge(
        source_annotations, on="reaction_id", how="left", validate="one_to_one"
    )


def build_enzyme_annotations(source: pd.DataFrame) -> pd.DataFrame:
    annotations = aggregate_source_rows(source, ["protein_id"])
    lengths = source.groupby("protein_id", sort=False)["protein_sequence"].first().str.len()
    annotations["sequence_length"] = annotations["protein_id"].map(lengths)
    return annotations


def build_pair_source_annotations(source: pd.DataFrame) -> pd.DataFrame:
    return aggregate_source_rows(source, ["protein_id", "reaction_id"])


def add_nearest_reaction_similarity(
    protocol_frames: dict[str, dict[str, pd.DataFrame]],
    fingerprints: dict[str, DataStructs.ExplicitBitVect],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for protocol, frames in protocol_frames.items():
        train_ids = sorted(set(frames["train"]["reaction_id"]))
        train_fps = [fingerprints[reaction_id] for reaction_id in train_ids]
        for reaction_id in sorted(set(frames["test"]["reaction_id"])):
            if reaction_id in set(train_ids):
                nearest_id = reaction_id
                similarity = 1.0
            else:
                similarities = DataStructs.BulkTanimotoSimilarity(
                    fingerprints[reaction_id], train_fps
                )
                best_index = int(np.argmax(similarities))
                nearest_id = train_ids[best_index]
                similarity = float(similarities[best_index])
            records.append(
                {
                    "protocol": protocol,
                    "reaction_id": reaction_id,
                    "nearest_train_reaction_id": nearest_id,
                    "nearest_train_molecular_set_morgan_tanimoto": similarity,
                }
            )
    return pd.DataFrame.from_records(records)


def export_fasta(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = frame[["protein_id", "protein_sequence"]].drop_duplicates("protein_id")
    with path.open("w", encoding="utf-8") as handle:
        for row in unique.itertuples(index=False):
            handle.write(f">{row.protein_id}\n{row.protein_sequence}\n")


def run_mmseqs(
    binary: Path,
    output_dir: Path,
    protocol_frames: dict[str, dict[str, pd.DataFrame]],
    threads: int,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for protocol, frames in protocol_frames.items():
        mmseqs_dir = output_dir / "mmseqs" / protocol
        train_fasta = mmseqs_dir / "train.fasta"
        test_fasta = mmseqs_dir / "test.fasta"
        result_path = mmseqs_dir / "test_to_train.tsv"
        export_fasta(train_fasta, frames["train"])
        export_fasta(test_fasta, frames["test"])
        command = [
            str(binary),
            "easy-search",
            str(test_fasta),
            str(train_fasta),
            str(result_path),
            str(mmseqs_dir / "tmp"),
            "--format-output",
            "query,target,fident,alnlen,qcov,tcov,evalue,bits",
            "--max-seqs",
            "1",
            "--threads",
            str(threads),
            "-s",
            "7.5",
        ]
        subprocess.run(command, check=True)
        columns = [
            "protein_id",
            "nearest_train_protein_id",
            "local_sequence_identity",
            "alignment_length",
            "query_coverage",
            "target_coverage",
            "evalue",
            "bits",
        ]
        result = pd.read_csv(result_path, sep="\t", names=columns)
        if Levenshtein is not None:
            query_sequences = dict(
                frames["test"][["protein_id", "protein_sequence"]]
                .drop_duplicates("protein_id")
                .itertuples(index=False, name=None)
            )
            target_sequences = dict(
                frames["train"][["protein_id", "protein_sequence"]]
                .drop_duplicates("protein_id")
                .itertuples(index=False, name=None)
            )
            distances: list[int] = []
            normalized_differences: list[float] = []
            for row in result.itertuples(index=False):
                query = query_sequences[row.protein_id]
                target = target_sequences[row.nearest_train_protein_id]
                distance = int(Levenshtein.distance(query, target))
                distances.append(distance)
                normalized_differences.append(distance / max(len(query), len(target)))
            result["mmseqs_neighbor_global_levenshtein_distance"] = distances
            result[
                "mmseqs_neighbor_normalized_global_levenshtein_difference"
            ] = normalized_differences
        result.insert(0, "protocol", protocol)
        records.append(result)
    return pd.concat(records, ignore_index=True)


def load_protocol_frames(
    paper_root: Path,
    pair_source: pd.DataFrame,
    enzymes: pd.DataFrame,
    reactions: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    output: dict[str, dict[str, pd.DataFrame]] = {}
    enzyme_columns = ["protein_id", "sequence_length"]
    reaction_columns = [
        "reaction_id",
        "molecule_count",
        "source_had_generic_wildcard",
        "has_any_cofactor_label",
        "has_core_cofactor_label",
        "cofactor_labels",
        "core_cofactor_labels",
    ]
    for protocol in PROTOCOLS:
        frames: dict[str, pd.DataFrame] = {}
        for subset in SUBSETS:
            frame = pd.read_csv(paper_root / protocol / f"{subset}_pairs.csv")
            original_count = len(frame)
            frame = frame.merge(
                pair_source,
                on=["protein_id", "reaction_id"],
                how="left",
                validate="many_to_one",
            )
            if frame["source_accessions"].isna().any():
                missing = int(frame["source_accessions"].isna().sum())
                raise ValueError(f"{protocol}/{subset}: {missing} pairs did not map to source metadata")
            frame = frame.merge(enzymes[enzyme_columns], on="protein_id", how="left", validate="many_to_one")
            frame = frame.merge(reactions[reaction_columns], on="reaction_id", how="left", validate="many_to_one")
            if len(frame) != original_count:
                raise ValueError(f"{protocol}/{subset}: pair count changed during annotation")
            protein_degree = frame.groupby("protein_id")["reaction_id"].nunique()
            reaction_degree = frame.groupby("reaction_id")["protein_id"].nunique()
            frame["protein_reaction_degree_subset"] = frame["protein_id"].map(protein_degree)
            frame["reaction_protein_degree_subset"] = frame["reaction_id"].map(reaction_degree)
            frames[subset] = frame

        train_pairs = set(
            frames["train"][["reaction_id", "protein_id"]].itertuples(index=False, name=None)
        )
        train_proteins = set(frames["train"]["protein_id"])
        train_reactions = set(frames["train"]["reaction_id"])
        train_prefixes = {
            depth: set().union(
                *(ec_prefixes(value, depth) for value in frames["train"]["ec_numbers"])
            )
            for depth in range(1, 5)
        }
        for subset, frame in frames.items():
            frame["exact_pair_seen_train"] = [
                pair in train_pairs
                for pair in frame[["reaction_id", "protein_id"]].itertuples(index=False, name=None)
            ]
            frame["protein_seen_train"] = frame["protein_id"].isin(train_proteins)
            frame["reaction_seen_train"] = frame["reaction_id"].isin(train_reactions)
            novelty = pd.DataFrame.from_records(
                [summarize_ec_novelty(value, train_prefixes) for value in frame["ec_numbers"]],
                index=frame.index,
            )
            frame[novelty.columns] = novelty
            frame.insert(0, "subset", subset)
            frame.insert(0, "protocol", protocol)
        output[protocol] = frames
    return output


def distribution_rows(
    frames: dict[str, dict[str, pd.DataFrame]], column: str, kind: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol, subsets in frames.items():
        for subset, frame in subsets.items():
            counts: Counter[str] = Counter()
            for value in frame[column]:
                counts.update(item for item in str(value or "").split("|") if item)
            for label, count in counts.most_common():
                rows.append(
                    {
                        "protocol": protocol,
                        "subset": subset,
                        "kind": kind,
                        "label": label,
                        "description": EC_CLASSES.get(label, "") if kind == "ec_top_class" else "",
                        "pair_presence_count": int(count),
                        "pair_presence_fraction": float(count / len(frame)),
                    }
                )
    return rows


def summarize_frames(
    frames: dict[str, dict[str, pd.DataFrame]],
    nearest_reactions: pd.DataFrame,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"protocols": {}, "cross_protocol": {}}
    for protocol, subsets in frames.items():
        protocol_summary: dict[str, Any] = {}
        for subset, frame in subsets.items():
            dates = pd.to_datetime(frame["creation_date_min"], errors="coerce")
            protocol_summary[subset] = {
                "pairs": int(len(frame)),
                "proteins": int(frame["protein_id"].nunique()),
                "reactions": int(frame["reaction_id"].nunique()),
                "accessions": int(
                    len(set().union(*(set(str(v).split("|")) for v in frame["source_accessions"])))
                ),
                "protein_seen_train_fraction": float(frame["protein_seen_train"].mean()),
                "proteins_seen_train": int(
                    frame.loc[frame["protein_seen_train"], "protein_id"].nunique()
                ),
                "reaction_seen_train_fraction": float(frame["reaction_seen_train"].mean()),
                "reactions_seen_train": int(
                    frame.loc[frame["reaction_seen_train"], "reaction_id"].nunique()
                ),
                "exact_pair_seen_train_fraction": float(frame["exact_pair_seen_train"].mean()),
                "ec_annotated_fraction": float((frame["ec_number_count"] > 0).mean()),
                "complete_ec4_fraction": float((frame["ec4_count"] > 0).mean()),
                "all_ec4_seen_train_fraction": float(frame["ec4_all_seen_train"].mean()),
                "directional_rhea_supported_fraction": float(
                    (frame["directional_rhea_id_count"] > 0).mean()
                ),
                "generic_wildcard_provenance_fraction": float(
                    frame["source_had_generic_wildcard"].mean()
                ),
                "cofactor_context_fraction": float(frame["has_any_cofactor_label"].mean()),
                "core_cofactor_context_fraction": float(frame["has_core_cofactor_label"].mean()),
                "enzyme_cofactor_comment_fraction": float(
                    frame["enzyme_cofactor_comment_present"].mean()
                ),
                "enzyme_cofactor_label_fraction": float(
                    frame["has_enzyme_cofactor_label"].mean()
                ),
                "enzyme_core_cofactor_label_fraction": float(
                    frame["has_enzyme_core_cofactor_label"].mean()
                ),
                "creation_year_min": int(dates.dt.year.min()) if dates.notna().any() else None,
                "creation_year_median": float(dates.dt.year.median()) if dates.notna().any() else None,
                "creation_year_max": int(dates.dt.year.max()) if dates.notna().any() else None,
                "sequence_length": percentile_summary(frame.drop_duplicates("protein_id")["sequence_length"]),
                "rhea_ids_per_pair": percentile_summary(frame["rhea_id_count"]),
                "molecules_per_reaction": percentile_summary(
                    frame.drop_duplicates("reaction_id")["molecule_count"]
                ),
                "proteins_per_reaction": percentile_summary(
                    frame.drop_duplicates("reaction_id")["reaction_protein_degree_subset"]
                ),
                "reactions_per_protein": percentile_summary(
                    frame.drop_duplicates("protein_id")["protein_reaction_degree_subset"]
                ),
            }
        test_similarity = nearest_reactions[nearest_reactions["protocol"] == protocol]
        protocol_summary["test"]["nearest_train_reaction_morgan_tanimoto"] = percentile_summary(
            test_similarity["nearest_train_molecular_set_morgan_tanimoto"]
        )
        summary["protocols"][protocol] = protocol_summary

    universes = {
        protocol: set().union(
            *(
                set(frame[["reaction_id", "protein_id"]].itertuples(index=False, name=None))
                for frame in subsets.values()
            )
        )
        for protocol, subsets in frames.items()
    }
    summary["cross_protocol"]["same_pair_universe"] = all(
        universe == universes[PROTOCOLS[0]] for universe in universes.values()
    )
    summary["cross_protocol"]["pair_universe_size"] = len(universes[PROTOCOLS[0]])
    test_sets = {
        protocol: set(
            subsets["test"][["reaction_id", "protein_id"]].itertuples(index=False, name=None)
        )
        for protocol, subsets in frames.items()
    }
    overlaps: dict[str, int] = {}
    for index, left in enumerate(PROTOCOLS):
        for right in PROTOCOLS[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(test_sets[left] & test_sets[right])
    overlaps["all_three"] = len(set.intersection(*test_sets.values()))
    summary["cross_protocol"]["test_pair_intersections"] = overlaps

    time_source_train = pd.concat(
        [frames["time"]["train"], frames["time"]["validation"]], ignore_index=True
    )
    time_test = frames["time"]["test"].copy()
    cutoff = pd.Timestamp("2010-01-01")
    for frame in (time_source_train, time_test):
        frame["date_min"] = pd.to_datetime(frame["creation_date_min"], errors="coerce")
        frame["date_max"] = pd.to_datetime(frame["creation_date_max"], errors="coerce")
    unequivocally_pre = time_source_train["date_max"] < cutoff
    unequivocally_post = time_source_train["date_min"] >= cutoff
    ambiguous = ~(unequivocally_pre | unequivocally_post)
    pre_reactions = set(time_source_train.loc[unequivocally_pre, "reaction_id"])
    late_train = time_source_train.loc[unequivocally_post]
    summary["protocols"]["time"]["release_date_audit"] = {
        "audit_boundary": "2010-01-01",
        "paper_text_boundary": "2010-12-31",
        "official_train_rows_unambiguously_before_boundary": int(unequivocally_pre.sum()),
        "official_train_rows_unambiguously_on_or_after_boundary": int(
            unequivocally_post.sum()
        ),
        "official_train_rows_with_accession_aliases_spanning_boundary": int(ambiguous.sum()),
        "test_rows_unambiguously_before_boundary": int((time_test["date_max"] < cutoff).sum()),
        "test_rows_unambiguously_on_or_after_boundary": int(
            (time_test["date_min"] >= cutoff).sum()
        ),
        "test_rows_with_accession_aliases_spanning_boundary": int(
            (
                ~(
                    (time_test["date_max"] < cutoff)
                    | (time_test["date_min"] >= cutoff)
                )
            ).sum()
        ),
        "late_train_rows": int(len(late_train)),
        "late_train_unique_reactions": int(late_train["reaction_id"].nunique()),
        "late_train_rows_whose_reaction_has_no_unambiguous_pre_boundary_train_row": int(
            (~late_train["reaction_id"].isin(pre_reactions)).sum()
        ),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path("data/revised_protocols/reactzyme_paper"),
    )
    parser.add_argument(
        "--uniprot-rhea",
        type=Path,
        default=Path("data/paper/reactzyme/raw/uniprot_rhea.tsv"),
    )
    parser.add_argument(
        "--cleaned-uniprot-rhea",
        type=Path,
        default=Path("data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv"),
    )
    parser.add_argument(
        "--uniprot-molecules",
        type=Path,
        default=Path("data/paper/reactzyme/raw/uniprot_molecules.tsv"),
    )
    parser.add_argument(
        "--rhea-molecules",
        type=Path,
        default=Path("data/paper/reactzyme/raw/rhea_molecules.tsv"),
    )
    parser.add_argument(
        "--uniprotkb-cofactor-cache",
        type=Path,
        default=None,
        help="Optional UniProtKB TSV cache with Entry and Cofactor columns",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/reactzyme_split_biology"))
    parser.add_argument("--mmseqs", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")

    source, molecules, _rhea = load_source_metadata(args)
    enzymes = build_enzyme_annotations(source)
    reactions, fingerprints = build_reaction_annotations(molecules)
    reactions = add_reaction_source_annotations(reactions, source)
    pair_source = build_pair_source_annotations(source)
    protocol_frames = load_protocol_frames(args.paper_root, pair_source, enzymes, reactions)
    nearest_reactions = add_nearest_reaction_similarity(protocol_frames, fingerprints)

    all_pairs = pd.concat(
        [frame for subsets in protocol_frames.values() for frame in subsets.values()],
        ignore_index=True,
    )
    pair_export_columns = [
        column
        for column in all_pairs.columns
        if column not in {"protein_sequence", "reaction_smiles"}
    ]
    all_pairs[pair_export_columns].to_parquet(
        args.output_dir / "pair_annotations.parquet", index=False, compression="zstd"
    )
    enzymes.to_parquet(args.output_dir / "enzyme_annotations.parquet", index=False, compression="zstd")
    reactions.drop(columns=["reaction_smiles", "raw_molecule_set_smiles"]).to_parquet(
        args.output_dir / "reaction_annotations.parquet", index=False, compression="zstd"
    )
    nearest_reactions.to_csv(args.output_dir / "reaction_nearest_train.csv", index=False)

    distributions = []
    distributions.extend(distribution_rows(protocol_frames, "ec_top_classes", "ec_top_class"))
    distributions.extend(distribution_rows(protocol_frames, "ec_numbers", "ec_number"))
    distributions.extend(distribution_rows(protocol_frames, "cofactor_labels", "cofactor"))
    distributions.extend(
        distribution_rows(protocol_frames, "core_cofactor_labels", "core_cofactor")
    )
    distributions.extend(
        distribution_rows(protocol_frames, "enzyme_cofactor_labels", "enzyme_cofactor")
    )
    distributions.extend(
        distribution_rows(
            protocol_frames, "enzyme_core_cofactor_labels", "enzyme_core_cofactor"
        )
    )
    pd.DataFrame.from_records(distributions).to_csv(
        args.output_dir / "label_distributions.csv", index=False
    )

    summary = summarize_frames(protocol_frames, nearest_reactions)
    summary["source"] = {
        "full_uniprot_rhea_rows": int(pd.read_csv(args.uniprot_rhea, sep="\t", usecols=["Entry"]).shape[0]),
        "cleaned_uniprot_rhea_rows": int(pd.read_csv(args.cleaned_uniprot_rhea, sep="\t", usecols=["Entry"]).shape[0]),
        "uniprot_molecule_rows": int(len(molecules)),
        "source_pairs": int(len(pair_source)),
        "unique_proteins": int(len(enzymes)),
        "unique_reactions": int(len(reactions)),
        "pair_source_mapping_fraction": 1.0,
        "wildcard_normalization": "literal '*' replaced with 'C' before reaction hashing",
        "directional_reaction_fraction": float(reactions["is_directional_reaction"].mean()),
        "molecule_parse_complete_fraction": float(reactions["molecule_parse_complete"].mean()),
        "molecular_sets_with_multiple_ec_top_classes_fraction": float(
            (reactions["associated_ec_top_class_count"] > 1).mean()
        ),
        "associated_ec_numbers_per_molecular_set": percentile_summary(
            reactions["associated_ec_number_count"]
        ),
        "associated_proteins_per_molecular_set": percentile_summary(
            reactions["associated_protein_count"]
        ),
        "uniprotkb_cofactor_cache": (
            str(args.uniprotkb_cofactor_cache) if args.uniprotkb_cofactor_cache else None
        ),
    }

    if args.mmseqs is not None:
        sequence_nearest = run_mmseqs(
            args.mmseqs.resolve(), args.output_dir, protocol_frames, args.threads
        )
        sequence_nearest.to_csv(args.output_dir / "protein_nearest_train_mmseqs.csv", index=False)
        summary["sequence_similarity"] = {}
        for protocol in PROTOCOLS:
            protocol_nearest = sequence_nearest[
                sequence_nearest["protocol"] == protocol
            ]
            query_count = protocol_frames[protocol]["test"]["protein_id"].nunique()
            summary["sequence_similarity"][protocol] = {
                "test_proteins": int(query_count),
                "proteins_with_significant_hit": int(
                    protocol_nearest["protein_id"].nunique()
                ),
                "significant_hit_fraction": float(
                    protocol_nearest["protein_id"].nunique() / query_count
                ),
                "best_local_sequence_identity": percentile_summary(
                    protocol_nearest["local_sequence_identity"]
                ),
                "query_coverage": percentile_summary(protocol_nearest["query_coverage"]),
                "fraction_of_all_test_proteins_with_identity_at_least_0_9": float(
                    (protocol_nearest["local_sequence_identity"] >= 0.9).sum()
                    / query_count
                ),
                "fraction_of_all_test_proteins_with_identity_at_least_0_5": float(
                    (protocol_nearest["local_sequence_identity"] >= 0.5).sum()
                    / query_count
                ),
            }
            difference_column = (
                "mmseqs_neighbor_normalized_global_levenshtein_difference"
            )
            if difference_column in protocol_nearest.columns:
                summary["sequence_similarity"][protocol][
                    "mmseqs_neighbor_normalized_global_levenshtein_difference"
                ] = percentile_summary(protocol_nearest[difference_column])
                summary["sequence_similarity"][protocol][
                    "fraction_of_all_test_proteins_proven_below_0_60_global_difference"
                ] = float((protocol_nearest[difference_column] < 0.60).sum() / query_count)
        summary["sequence_similarity_note"] = (
            "MMseqs2 best local alignment identity; this is not the global Levenshtein "
            "criterion used to construct enzyme_smi."
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
