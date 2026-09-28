import json

from wet_lab.query_campaign import consolidate_results


def _result(name, rows):
    return {
        "model": {"name": name, "checkpoint": f"{name}.ckpt"},
        "rankings": [
            {
                "rank": rank,
                "protein_id": protein_id,
                "cosine_similarity": score,
            }
            for rank, protein_id, score in rows
        ],
    }


def test_consolidate_results_uses_reciprocal_rank_fusion(tmp_path):
    paths = []
    for index, result in enumerate(
        [
            _result("m1", [(1, "P1", 0.9), (2, "P2", 0.8)]),
            _result("m2", [(1, "P2", 0.7), (2, "P1", 0.6)]),
        ]
    ):
        path = tmp_path / f"model_{index}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(path)

    result_path = consolidate_results(
        reaction={"id": "r1", "smiles": "C>>C"},
        result_paths=paths,
        output_dir=tmp_path / "campaign",
        rrf_constant=60.0,
        max_rank=100,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    consensus = result["consensus"]["rankings"]
    assert {row["protein_id"] for row in consensus[:2]} == {"P1", "P2"}
    assert all(row["model_support"] == 2 for row in consensus[:2])
    assert (tmp_path / "campaign/summary.md").is_file()
    assert (tmp_path / "campaign/per_model_rankings.csv").is_file()
