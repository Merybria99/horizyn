import json

import torch

from wet_lab.fold import _mean_plddt, _normalize_plddt_scale, load_ranked_candidates, run_folding


def _write_campaign(path):
    path.write_text(
        json.dumps(
            {
                "reaction": {"id": "rxn", "smiles": "C>>C"},
                "consensus": {
                    "rankings": [
                        {
                            "consensus_rank": 1,
                            "protein_id": "P1",
                            "rrf_score": 0.2,
                        },
                        {
                            "consensus_rank": 2,
                            "protein_id": "P2",
                            "rrf_score": 0.1,
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_ranked_candidates_attaches_sequences(tmp_path):
    results = tmp_path / "results.json"
    fasta = tmp_path / "candidates.fasta"
    _write_campaign(results)
    fasta.write_text(">P1\nACDE\n>P2\nGGUT\n", encoding="utf-8")

    reaction, candidates = load_ranked_candidates(results, fasta, top_k=2)

    assert reaction["id"] == "rxn"
    assert [candidate["protein_id"] for candidate in candidates] == ["P1", "P2"]
    assert [candidate["sequence_length"] for candidate in candidates] == [4, 4]
    assert candidates[1]["selenocysteine_positions"] == [3]


def test_load_ranked_candidates_supports_explicit_ids(tmp_path):
    results = tmp_path / "results.json"
    fasta = tmp_path / "candidates.fasta"
    _write_campaign(results)
    fasta.write_text(">P1\nACDE\n>P2\nGGUT\n", encoding="utf-8")

    _, candidates = load_ranked_candidates(
        results,
        fasta,
        top_k=10,
        protein_ids=["P2"],
    )

    assert [candidate["protein_id"] for candidate in candidates] == ["P2"]
    assert candidates[0]["rank"] == 2


def test_prepare_only_writes_exact_af3_jobs(tmp_path, monkeypatch):
    results = tmp_path / "results.json"
    fasta = tmp_path / "candidates.fasta"
    output = tmp_path / "folding"
    config = tmp_path / "fold.yaml"
    _write_campaign(results)
    fasta.write_text(">P1\nACDE\n>P2\nGGUT\n", encoding="utf-8")
    config.write_text(
        "\n".join(
            [
                "ranking:",
                f"  results: {results}",
                f"  candidate_fasta: {fasta}",
                "  top_k: 2",
                "backend:",
                "  name: alphafold3",
                "  prepare_only: true",
                "  model_seeds: [7]",
                "output:",
                f"  directory: {output}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    manifest_path = run_folding(config)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [row["status"] for row in manifest["statuses"]] == ["prepared", "prepared"]
    first_job_path = output / "inputs/alphafold3/01_P1.json"
    first_job = json.loads(first_job_path.read_text(encoding="utf-8"))
    assert first_job["modelSeeds"] == [7]
    assert first_job["sequences"][0]["protein"]["sequence"] == "ACDE"
    assert first_job["dialect"] == "alphafold3"
    second_job = json.loads((output / "inputs/alphafold3/02_P2.json").read_text(encoding="utf-8"))
    assert second_job["sequences"][0]["protein"]["sequence"] == "GGCT"
    assert second_job["sequences"][0]["protein"]["modifications"] == [
        {"ptmType": "CCD_SEC", "ptmPosition": 3}
    ]


def test_normalize_plddt_scale_uses_standard_zero_to_100_range():
    output = {
        "plddt": torch.tensor([[[0.75, 0.80]]]),
        "atom37_atom_exists": torch.ones(1, 1, 2),
    }

    scale = _normalize_plddt_scale(output)

    assert scale == 100.0
    assert _mean_plddt(output) == 77.5
