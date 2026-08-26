#!/usr/bin/env python3
"""Arrange Case1 structures by model, target reaction, split, and Excel overlap."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ALL_SPLITS_DIR = SCRIPT_DIR / "all_splits"
TREE_ROOT = SCRIPT_DIR / "structures/F3_set_chemistry/D-fructose_to_D-tagatose"
SPLITS = ("time", "enzyme_smi", "reaction_smi")
STRUCTURE_MANIFESTS = (
    SCRIPT_DIR / "structures/top25_structure_manifest.csv",
    ALL_SPLITS_DIR / "structures_consensus/top25_structure_manifest.csv",
    SCRIPT_DIR / "structures/not_in_ranked_candidates/top29_structure_manifest.csv",
    TREE_ROOT / "manifest.csv",
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _structure_index() -> dict[str, dict[str, str]]:
    candidates: dict[str, list[dict[str, str]]] = {}
    for manifest in STRUCTURE_MANIFESTS:
        if not manifest.is_file():
            continue
        for row in _load_csv(manifest):
            source = Path(row.get("unique_structure_path") or row.get("structure_path") or "")
            if source.is_file():
                normalized = {
                    **row,
                    "unique_structure_path": str(source),
                }
                candidates.setdefault(row["sequence_sha256"], []).append(normalized)

    index: dict[str, dict[str, str]] = {}
    for sha256, rows in candidates.items():
        rows.sort(
            key=lambda row: (
                row["structure_source"] != "AlphaFoldDB",
                row["unique_structure_path"],
            )
        )
        index[sha256] = rows[0]
    return index


def _link_or_copy(source: Path, destination: Path) -> str:
    if destination.is_file() and os.path.samefile(source, destination):
        return "hardlink"
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def organize() -> Path:
    campaign = json.loads((ALL_SPLITS_DIR / "manifest.json").read_text(encoding="utf-8"))
    structures = _structure_index()
    TREE_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    for split in SPLITS:
        result_path = Path(campaign["result_paths"][split])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        comparison_path = (
            ALL_SPLITS_DIR / f"ranked_candidates/{split}/top25_vs_ranked_candidates.csv"
        )
        comparison_by_id = {row["entry_id"]: row for row in _load_csv(comparison_path)}
        split_rows: list[dict[str, Any]] = []

        for ranking in result["rankings"][:25]:
            protein_id = str(ranking["protein_id"])
            comparison = comparison_by_id[protein_id]
            overlap = comparison["in_ranked_candidates"].lower() == "true"
            category = "overlapping" if overlap else "non-overlapping"
            category_dir = TREE_ROOT / split / category
            category_dir.mkdir(parents=True, exist_ok=True)

            sha256 = str(ranking["sha256"])
            structure = structures.get(sha256)
            if structure is None:
                raise ValueError(
                    f"No validated structure is available for {split} rank "
                    f"{ranking['rank']} {protein_id} ({sha256})"
                )
            source_path = Path(structure["unique_structure_path"]).resolve()
            destination = category_dir / f"rank{int(ranking['rank']):02d}_{protein_id}.pdb"
            storage = _link_or_copy(source_path, destination)
            row = {
                "split": split,
                "rank": int(ranking["rank"]),
                "protein_id": protein_id,
                "name": ranking.get("name", ""),
                "cosine_similarity": float(ranking["cosine_similarity"]),
                "sequence_sha256": sha256,
                "category": category,
                "match_basis": comparison["match_basis"],
                "ranked_candidate_positions": comparison["ranked_candidate_positions"],
                "structure_source": structure["structure_source"],
                "source_accession": structure["source_accession"],
                "mean_plddt": structure["mean_plddt"],
                "storage": storage,
                "structure_path": str(destination),
                "source_structure_path": str(source_path),
                "checkpoint": result["model"]["checkpoint"],
            }
            split_rows.append(row)
            all_rows.append(row)

        _write_csv(TREE_ROOT / split / "manifest.csv", split_rows)
        overlap_count = sum(row["category"] == "overlapping" for row in split_rows)
        if overlap_count + sum(row["category"] == "non-overlapping" for row in split_rows) != 25:
            raise RuntimeError(f"{split} structure partition does not contain 25 rows")

    _write_csv(TREE_ROOT / "manifest.csv", all_rows)
    counts = {
        split: {
            category: sum(row["split"] == split and row["category"] == category for row in all_rows)
            for category in ("overlapping", "non-overlapping")
        }
        for split in SPLITS
    }
    tree_manifest = {
        "schema_version": "case1_structure_tree_v1",
        "model": "F3_set_chemistry",
        "target_reaction": "D-fructose_to_D-tagatose",
        "splits": list(SPLITS),
        "selection": "top 25 per split",
        "overlap_rule": (
            "overlapping when a Homolog has an evidence-ID or exact-sequence match "
            "in the resolved Ranked Candidates sheet"
        ),
        "counts": counts,
        "total_structure_rows": len(all_rows),
        "tree_root": str(TREE_ROOT),
    }
    (TREE_ROOT / "manifest.json").write_text(
        json.dumps(tree_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    report = "\n".join(
        [
            "# F3 D-fructose to D-tagatose Structures",
            "",
            "Each split contains its top-25 structures partitioned against the workbook's "
            "resolved `Ranked Candidates` sheet.",
            "",
            "`overlapping` means an evidence-ID or exact-sequence match. "
            "`non-overlapping` means neither match is present.",
            "",
            "| Split | Overlapping | Non-overlapping | Total |",
            "|---|---:|---:|---:|",
            *[
                f"| {split} | {counts[split]['overlapping']} | "
                f"{counts[split]['non-overlapping']} | 25 |"
                for split in SPLITS
            ],
            "",
            "PDB files are hard-linked to validated source structures when supported by the "
            "filesystem, with ordinary copies as fallback. Each split manifest records the "
            "checkpoint, model score, match basis, structure source, confidence, and original "
            "structure path.",
            "",
        ]
    )
    (TREE_ROOT / "README.md").write_text(report, encoding="utf-8")
    return TREE_ROOT


def main() -> None:
    print(organize())


if __name__ == "__main__":
    main()
