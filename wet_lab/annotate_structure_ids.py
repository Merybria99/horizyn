#!/usr/bin/env python3
"""Add durable identifiers and source metadata to predicted PDB outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


ANNOTATION_PREFIX = "REMARK 900 HORIZYN_"
PREDICTION_METHOD = "facebook/esmfold_v1"
EXPERIMENTAL_PDB_ID = "NA (predicted model; no PDB accession assigned)"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_ranking_metadata(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    ranking_results = Path(manifest["ranking_results"])
    ranking_csv = ranking_results.with_name("top_10.csv")
    _, rows = read_csv(ranking_csv)
    return {row["protein_id"]: row for row in rows}


def annotation_lines(
    structure_id: str,
    ranking_model: str,
    rank: int,
    protein_id: str,
    metadata: dict[str, str],
) -> list[str]:
    values = (
        ("MODEL_ID", structure_id),
        ("RANKING_MODEL", ranking_model),
        ("RANK", str(rank)),
        ("REFSEQ_ACCESSION", protein_id),
        ("PROTEIN_DESCRIPTION", metadata.get("description", "NA")),
        ("SOURCE_ORGANISM", metadata.get("organism", "NA")),
        ("PREDICTION_METHOD", PREDICTION_METHOD),
        ("EXPERIMENTAL_PDB_ID", EXPERIMENTAL_PDB_ID),
    )
    return [f"{ANNOTATION_PREFIX}{key} {value}\n" for key, value in values]


def annotate_pdb(path: Path, lines: list[str]) -> None:
    original = path.read_text().splitlines(keepends=True)
    original = [line for line in original if not line.startswith(ANNOTATION_PREFIX)]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(lines + original))
    os.replace(temporary, path)


def annotate_group(root: Path) -> None:
    manifest_path = root / "manifest.json"
    status_path = root / "status.csv"
    manifest = json.loads(manifest_path.read_text())
    metadata_by_id = load_ranking_metadata(manifest)
    old_fields, statuses = read_csv(status_path)

    ranking_model = root.name.removesuffix("_top10").upper()
    structure_dir = root / "structures" / "esmfold"
    id_rows: list[dict[str, Any]] = []

    for status in statuses:
        rank = int(status["rank"])
        protein_id = status["protein_id"]
        metadata = metadata_by_id.get(protein_id, {})
        structure_id = f"{ranking_model}_RANK{rank:02d}_{protein_id}_ESMFOLD"
        new_path = structure_dir / f"{structure_id}.pdb"

        recorded_path = Path(status["structure_path"])
        if not recorded_path.exists():
            candidates = list(structure_dir.glob(f"*{protein_id}*.pdb"))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected one PDB for {protein_id} in {structure_dir}, found {len(candidates)}"
                )
            recorded_path = candidates[0]
        if recorded_path != new_path:
            recorded_path.rename(new_path)

        annotate_pdb(
            new_path,
            annotation_lines(structure_id, ranking_model, rank, protein_id, metadata),
        )

        extra = {
            "structure_id": structure_id,
            "structure_filename": new_path.name,
            "refseq_accession": protein_id,
            "description": metadata.get("description", ""),
            "organism": metadata.get("organism", ""),
            "prediction_method": PREDICTION_METHOD,
            "experimental_pdb_id": EXPERIMENTAL_PDB_ID,
        }
        status.update(extra)
        status["structure_path"] = str(new_path.resolve())
        id_rows.append(
            {
                "ranking_model": ranking_model,
                "rank": rank,
                **extra,
                "structure_path": str(new_path.resolve()),
            }
        )

    extra_fields = [
        "structure_id",
        "structure_filename",
        "refseq_accession",
        "description",
        "organism",
        "prediction_method",
        "experimental_pdb_id",
    ]
    fields = old_fields[:]
    insert_at = fields.index("protein_id") + 1
    for field in reversed(extra_fields):
        if field not in fields:
            fields.insert(insert_at, field)
    write_csv(status_path, fields, statuses)

    identifier_fields = [
        "ranking_model",
        "rank",
        "structure_id",
        "structure_filename",
        "refseq_accession",
        "description",
        "organism",
        "prediction_method",
        "experimental_pdb_id",
        "structure_path",
    ]
    write_csv(root / "structure_ids.csv", identifier_fields, id_rows)

    status_by_key = {(int(row["rank"]), row["protein_id"]): row for row in statuses}
    for section in ("candidates", "statuses"):
        for item in manifest.get(section, []):
            key = (int(item["rank"]), item["protein_id"])
            enriched = status_by_key[key]
            for field in extra_fields:
                item[field] = enriched[field]
            if section == "statuses":
                item["structure_path"] = enriched["structure_path"]
    manifest["structure_identifiers"] = {
        "scheme": "{RANKING_MODEL}_RANK{rank:02d}_{RefSeq_accession}_ESMFOLD",
        "prediction_method": PREDICTION_METHOD,
        "experimental_pdb_id_note": EXPERIMENTAL_PDB_ID,
        "identifier_log": str((root / "structure_ids.csv").resolve()),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_manifest, manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory containing *_top10 folders")
    args = parser.parse_args()
    for group in sorted(args.root.glob("*_top10")):
        annotate_group(group.resolve())
        print(f"Annotated {group}")


if __name__ == "__main__":
    main()
