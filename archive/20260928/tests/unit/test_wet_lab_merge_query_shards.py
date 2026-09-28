import csv
import json

import pytest

from wet_lab.merge_query_shards import merge_sharded_results


def _write_shard(tmp_path, index, rows):
    shard = tmp_path / f"shard_{index}"
    shard.mkdir()
    csv_path = shard / "top_2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "protein_id", "cosine_similarity", "sequence"],
        )
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "reaction": {"id": "r1", "smiles": "C>>C", "warnings": []},
        "model": {"name": "model", "checkpoint": "model.ckpt", "config": "model.yaml"},
        "scoring": {
            "type": "cosine_similarity",
            "calibrated_probability": False,
            "top_k": [1, 2],
        },
        "candidate_pool": {
            "requested_candidate_count": 2,
            "candidate_count": 2,
            "missing_candidate_id_count": 0,
            "zero_length_candidate_count": 0,
        },
        "rankings": [
            {key: value for key, value in row.items() if key != "sequence"} for row in rows
        ],
        "artifacts": {"top_k_csv": {"2": str(csv_path)}},
    }
    result_path = shard / "results.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


def test_merge_sharded_results_recovers_exact_global_top_k(tmp_path):
    first = _write_shard(
        tmp_path,
        0,
        [
            {"rank": 1, "protein_id": "P1", "cosine_similarity": 0.9, "sequence": "AAAA"},
            {"rank": 2, "protein_id": "P3", "cosine_similarity": 0.6, "sequence": "CCCC"},
        ],
    )
    second = _write_shard(
        tmp_path,
        1,
        [
            {"rank": 1, "protein_id": "P2", "cosine_similarity": 0.8, "sequence": "GGGG"},
            {"rank": 2, "protein_id": "P4", "cosine_similarity": 0.5, "sequence": "TTTT"},
        ],
    )

    result_path = merge_sharded_results([first, second], tmp_path / "merged")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert [row["protein_id"] for row in result["rankings"]] == ["P1", "P2"]
    assert [row["rank"] for row in result["rankings"]] == [1, 2]
    assert [row["cosine_similarity"] for row in result["rankings"]] == [0.9, 0.8]
    assert result["candidate_pool"]["candidate_count"] == 4
    assert (tmp_path / "merged/top_2.fasta").read_text(encoding="utf-8").startswith(">P1 rank=1")


def test_merge_sharded_results_rejects_non_finite_scores(tmp_path):
    first = _write_shard(
        tmp_path,
        0,
        [
            {"rank": 1, "protein_id": "P1", "cosine_similarity": "nan", "sequence": "AAAA"},
            {"rank": 2, "protein_id": "P2", "cosine_similarity": 0.5, "sequence": "CCCC"},
        ],
    )
    second = _write_shard(
        tmp_path,
        1,
        [
            {"rank": 1, "protein_id": "P3", "cosine_similarity": 0.4, "sequence": "GGGG"},
            {"rank": 2, "protein_id": "P4", "cosine_similarity": 0.3, "sequence": "TTTT"},
        ],
    )

    with pytest.raises(ValueError, match="not finite"):
        merge_sharded_results([first, second], tmp_path / "merged")
