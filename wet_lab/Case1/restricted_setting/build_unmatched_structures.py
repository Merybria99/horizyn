#!/usr/bin/env python3
"""Build structures for all-split top-25 candidates absent from Ranked Candidates."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tarfile
from pathlib import Path
from typing import Any

from build_top25_structures import (
    DEFAULT_FASTA,
    DEFAULT_SEQUENCE_MANIFEST,
    build_structures,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_TABLE = SCRIPT_DIR.parent / "sequence_pool/final_entry_sequences.csv"
ALL_SPLITS_MANIFEST = SCRIPT_DIR / "all_splits/manifest.json"
CONSENSUS_RESULTS = SCRIPT_DIR / "all_splits/consensus_results.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "structures/not_in_ranked_candidates"
EVIDENCE_PATTERN = re.compile(r"\b(?:H|P)\d{1,3}\b", flags=re.IGNORECASE)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ranked_candidate_indexes() -> tuple[set[str], set[str]]:
    resolved = [
        row
        for row in _load_csv(SOURCE_TABLE)
        if row.get("sheet") == "Ranked Candidates" and row.get("sequence")
    ]
    evidence_ids = {
        identifier.upper()
        for row in resolved
        for identifier in EVIDENCE_PATTERN.findall(row.get("evidence_row", ""))
    }
    sequence_shas = {row["sha256"] for row in resolved}
    return evidence_ids, sequence_shas


def prepare_selection(output_dir: Path) -> tuple[Path, int]:
    campaign = json.loads(ALL_SPLITS_MANIFEST.read_text(encoding="utf-8"))
    consensus = json.loads(CONSENSUS_RESULTS.read_text(encoding="utf-8"))
    consensus_by_id = {str(row["protein_id"]): row for row in consensus["rankings"]}
    evidence_ids, ranked_shas = _ranked_candidate_indexes()

    union: dict[str, dict[str, Any]] = {}
    for split, result_path in campaign["result_paths"].items():
        result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        for row in result["rankings"][:25]:
            protein_id = str(row["protein_id"])
            candidate = union.setdefault(
                protein_id,
                {
                    "protein_id": protein_id,
                    "name": row.get("name", ""),
                    "sha256": row.get("sha256", ""),
                    "time_rank": "",
                    "enzyme_smi_rank": "",
                    "reaction_smi_rank": "",
                },
            )
            candidate[f"{split}_rank"] = int(row["rank"])

    unmatched = [
        candidate
        for candidate in union.values()
        if candidate["protein_id"].upper() not in evidence_ids
        and candidate["sha256"] not in ranked_shas
    ]
    unmatched.sort(key=lambda row: int(consensus_by_id[row["protein_id"]]["rank"]))
    if len(union) != 41 or len(unmatched) != 29:
        raise ValueError(
            f"Expected 41 union candidates and 29 unmatched rows, found "
            f"{len(union)} and {len(unmatched)}"
        )
    if len({row["sha256"] for row in unmatched}) != 26:
        raise ValueError("Expected 26 unique unmatched sequences")

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, Any]] = []
    filtered_rankings: list[dict[str, Any]] = []
    for candidate in unmatched:
        consensus_row = dict(consensus_by_id[candidate["protein_id"]])
        selection_rows.append(
            {
                "consensus_rank": consensus_row["rank"],
                **candidate,
                "selected_by_splits": ";".join(
                    split
                    for split in ("time", "enzyme_smi", "reaction_smi")
                    if candidate[f"{split}_rank"] != ""
                ),
                "exclusion_check": "no evidence-ID match and no exact-sequence match",
            }
        )
        filtered_rankings.append(consensus_row)

    selection_csv = output_dir / "selection.csv"
    _write_csv(selection_csv, selection_rows)
    selection_result = {
        "schema_version": "case1_restricted_unmatched_selection_v1",
        "reaction": consensus["reaction"],
        "model": {
            "name": "Union of F3 split top 25, filtered against Ranked Candidates",
            "checkpoint": "time + enzyme_smi + reaction_smi F3 checkpoints",
            "config": "build_unmatched_structures.py",
        },
        "candidate_pool": {"candidate_count": len(filtered_rankings)},
        "selection": {
            "source": "union of top 25 from each split-specific F3 ranking",
            "excluded": "Ranked Candidates evidence IDs and exact resolved sequences",
            "ordering": "three-split reciprocal-rank-fusion consensus rank",
        },
        "rankings": filtered_rankings,
    }
    results_path = output_dir / "selection_results.json"
    results_path.write_text(json.dumps(selection_result, indent=2) + "\n", encoding="utf-8")
    return results_path, len(filtered_rankings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    results_path, count = prepare_selection(output_dir)
    manifest_path = build_structures(
        results_path=results_path,
        fasta_path=DEFAULT_FASTA,
        sequence_manifest_path=DEFAULT_SEQUENCE_MANIFEST,
        output_dir=output_dir,
        gpu=args.gpu,
        force=args.force,
        top_k=count,
        reuse_manifests=[
            SCRIPT_DIR / "structures/top25_structure_manifest.csv",
            SCRIPT_DIR / "all_splits/structures_consensus/top25_structure_manifest.csv",
        ],
    )
    structure_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    structure_csv = Path(structure_manifest["manifest_csv"])
    report_path = output_dir / "README.md"
    generated_report = report_path.read_text(encoding="utf-8")
    heading = "# Structures Absent From Ranked Candidates"
    generated_report = generated_report.replace(
        f"# Restricted Setting Top-{count} Structures",
        heading,
        1,
    )
    selection_note = (
        "\nThis set is the union of the top 25 from the `time`, `enzyme_smi`, and "
        "`reaction_smi` F3 rankings after excluding every candidate with either an "
        "evidence-ID match or an exact-sequence match in the workbook's `Ranked "
        "Candidates` sheet. Table ranks are three-checkpoint consensus ranks and may "
        "therefore contain gaps. See `selection.csv` for all three source ranks.\n"
    )
    report_path.write_text(
        generated_report.replace(heading, heading + selection_note, 1),
        encoding="utf-8",
    )

    generic_archive = output_dir / f"case1_restricted_top{count}_structures.tar.gz"
    archive_path = output_dir / "case1_not_in_ranked_candidates_structures.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in (
            report_path,
            manifest_path,
            structure_csv,
            output_dir / "selection.csv",
            results_path,
            output_dir / "unique",
            output_dir / "by_rank",
        ):
            archive.add(path, arcname=path.relative_to(output_dir))
    generic_archive.unlink(missing_ok=True)
    print(f"Unmatched structure archive: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
