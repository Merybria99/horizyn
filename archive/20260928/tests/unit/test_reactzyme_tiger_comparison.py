import argparse
import csv
import json

import pytest

from scripts.reactzyme_tiger_comparison import aggregate, build_jobs


def _build_fixture(tmp_path):
    promotion = tmp_path / "promotion.json"
    promotion.write_text(
        json.dumps({"stage": "recipe_confirm", "winner": "R4"}),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.tsv"
    fields = ["variant", "seed", "split", "run_id", "checkpoint_sidecar", "test_config"]
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for seed in (7, 42, 137):
            for split in ("time", "enzyme_smi", "reaction_smi"):
                run_dir = tmp_path / f"seed{seed}" / split
                run_dir.mkdir(parents=True)
                checkpoint = run_dir / "model.ckpt"
                checkpoint.write_text(f"{seed}-{split}", encoding="utf-8")
                sidecar = run_dir / "selected_checkpoint.json"
                sidecar.write_text(json.dumps({"checkpoint": str(checkpoint)}), encoding="utf-8")
                config = run_dir / "test.yaml"
                config.write_text("model: {}\n", encoding="utf-8")
                writer.writerow(
                    {
                        "variant": "R4",
                        "seed": seed,
                        "split": split,
                        "run_id": f"R4_seed{seed}_{split}",
                        "checkpoint_sidecar": sidecar,
                        "test_config": config,
                    }
                )
    return promotion, plan


def test_aggregate_requires_and_reports_one_validation_selected_recipe(tmp_path):
    promotion, plan = _build_fixture(tmp_path)
    output_root = tmp_path / "evaluations"
    winner, jobs = build_jobs(promotion, plan, output_root)
    assert winner == "R4"
    assert len(jobs) == 9

    for job in jobs:
        seed_offset = {"7": 0.0, "42": 0.1, "137": 0.2}[job["seed"]]
        payload = {
            "evaluation_protocol": "paper_test_candidates",
            "ground_truth_pairs": "test_pairs_only",
            "direction": "both",
            "checkpoint": job["checkpoint"],
            "config": job["config"],
            "num_enzyme_candidates": 100,
            "num_reaction_candidates": 10,
        }
        for direction in ("enzyme_to_reaction", "reaction_to_enzyme"):
            payload[f"{direction}/top_1"] = 0.4 + seed_offset
            payload[f"{direction}/reactzyme_mrr"] = 0.3 + seed_offset
            payload[f"{direction}/first_positive_mrr"] = 0.5 + seed_offset
        output = tmp_path / job["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload), encoding="utf-8")

    output_json = tmp_path / "report.json"
    output_markdown = tmp_path / "report.md"
    aggregate(
        argparse.Namespace(
            promotion=promotion,
            plan=plan,
            output_root=output_root,
            output_json=output_json,
            output_tsv=tmp_path / "report.tsv",
            output_markdown=output_markdown,
        )
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["selection"]["variant"] == "R4"
    assert report["selection"]["test_set_used_for_selection"] is False
    assert report["ours"]["time"]["enzyme_to_reaction"]["reactzyme_mrr"]["mean"] == pytest.approx(0.4)
    assert report["ours"]["time"]["enzyme_to_reaction"]["reactzyme_mrr"]["std"] == pytest.approx(0.1)
    assert "First-positive MRR is retained only as a ranking diagnostic" in output_markdown.read_text(encoding="utf-8")


def test_build_jobs_rejects_non_confirmation_promotion(tmp_path):
    promotion, plan = _build_fixture(tmp_path)
    promotion.write_text(json.dumps({"stage": "recipe", "winner": "R4"}), encoding="utf-8")

    with pytest.raises(ValueError, match="three-seed recipe_confirm"):
        build_jobs(promotion, plan, tmp_path / "out")
