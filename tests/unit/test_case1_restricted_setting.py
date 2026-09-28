from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


RESTRICTED_DIR = Path(__file__).resolve().parents[2] / "wet_lab/Case1/restricted_setting"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RESTRICTED_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load_module("case1_prepare_candidate_pool", "prepare_candidate_pool.py")
compare = _load_module("case1_compare_ranked_candidates", "compare_ranked_candidates.py")


def _sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "sheet",
        "excel_row",
        "entry_id",
        "name",
        "status",
        "sequence_source",
        "parent_identifier",
        "applied_changes",
        "length",
        "sha256",
        "sequence",
        "reason",
        "evidence_row",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _row(sheet: str, entry_id: str, sequence: str, **updates: str) -> dict[str, str]:
    row = {
        "sheet": sheet,
        "excel_row": "2",
        "entry_id": entry_id,
        "name": entry_id,
        "status": "resolved_reference",
        "sequence_source": "test",
        "parent_identifier": "",
        "applied_changes": "",
        "length": str(len(sequence)),
        "sha256": _sha(sequence),
        "sequence": sequence,
        "reason": "",
        "evidence_row": "",
    }
    row.update(updates)
    return row


def test_build_candidate_pool_preserves_duplicate_rows(tmp_path: Path) -> None:
    source = tmp_path / "entries.csv"
    _write_rows(
        source,
        [
            _row("Homologs", "H001", "ACDE"),
            _row("Homologs", "H002", "ACDE"),
            _row("Homologs", "H003", "FGHI", status="unresolved"),
            _row("Ranked Candidates", "RANK_001", "ACDE"),
        ],
    )

    manifest = prepare.build_candidate_pool(source, tmp_path / "pool", expected_count=2)

    assert manifest["candidate_rows"] == 2
    assert manifest["unique_sequences"] == 1
    assert (tmp_path / "pool/candidate_ids_prott5_order.txt").read_text().splitlines() == [
        "H001",
        "H002",
    ]
    metadata = list(csv.DictReader((tmp_path / "pool/proteins.csv").open()))
    assert metadata[0]["sequence_duplicate_count"] == "2"
    assert metadata[1]["sequence_duplicate_ids"] == "H001;H002"


def test_compare_distinguishes_evidence_and_sequence_matches(tmp_path: Path) -> None:
    source = tmp_path / "entries.csv"
    _write_rows(
        source,
        [
            _row("Homologs", "H001", "ACDE"),
            _row("Homologs", "H002", "FGHI"),
            _row(
                "Ranked Candidates",
                "RANK_001",
                "FGHI",
                evidence_row="H001",
            ),
        ],
    )
    results = {
        "reaction": {"id": "query"},
        "model": {"name": "CIRCE-v2 test model"},
        "candidate_pool": {"candidate_count": 2},
        "rankings": [
            {
                "rank": 1,
                "protein_id": "H001",
                "name": "one",
                "cosine_similarity": 0.9,
                "sha256": _sha("ACDE"),
            },
            {
                "rank": 2,
                "protein_id": "H002",
                "name": "two",
                "cosine_similarity": 0.8,
                "sha256": _sha("FGHI"),
            },
        ],
    }
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(results), encoding="utf-8")

    summary, rows = compare.compare_rankings(result_path, source, tmp_path / "comparison", top_k=2)

    assert rows[0]["match_basis"] == "evidence_id"
    assert rows[1]["match_basis"] == "exact_sequence"
    assert summary["top_k_rows_matching_ranked_candidates"] == 2
    assert summary["eligible_ranked_unique_sequences_recovered"] == 1
    report = (tmp_path / "comparison/report.md").read_text()
    assert "CIRCE-v2 test model" in report
    assert "| Model rank |" in report
    assert "F3" not in report
