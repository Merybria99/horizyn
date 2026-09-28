import csv
import json
from pathlib import Path

import pytest

from scripts.report_reactzyme_reaction_feature_ablation import generate_report


SPLITS = ("time", "enzyme_smi", "reaction_smi")
DIRECTIONS = ("enzyme_to_reaction", "reaction_to_enzyme")


def _write_fixture(tmp_path: Path) -> Path:
    plan = tmp_path / "train_plan.tsv"
    fields = [
        "wave",
        "variant",
        "label",
        "split",
        "run_id",
        "test_config",
        "eval_json",
        "description",
    ]
    for split_index, split in enumerate(SPLITS):
        directional_dir = tmp_path / "data" / split / "reaction_directional"
        directional_dir.mkdir(parents=True)
        test_reactions = 10 + split_index
        schema = {
            "schema_version": "reactzyme_directional_vectors_v1",
            "splits": {
                "train": {
                    "num_reactions": 20,
                    "num_resolved": 4,
                    "coverage": 0.2,
                    "counts": {"with_center": 3},
                },
                "validation": {
                    "num_reactions": 10,
                    "num_resolved": 2,
                    "coverage": 0.2,
                    "counts": {"with_center": 2},
                },
                "test": {
                    "num_reactions": test_reactions,
                    "num_resolved": 2,
                    "coverage": 2 / test_reactions,
                    "counts": {"with_center": 1},
                },
            },
        }
        (directional_dir / "schema.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )

    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(7):
            label = f"F{index}"
            variant = f"F{index}_variant"
            for split_index, split in enumerate(SPLITS):
                run_id = f"{split}_{variant}"
                run_dir = tmp_path / split / variant
                run_dir.mkdir(parents=True)
                checkpoint = run_dir / "model.ckpt"
                checkpoint.write_bytes(f"{label}-{split}".encode())
                config = run_dir / "test.yaml"
                config.write_text("test: true\n", encoding="utf-8")
                eval_path = run_dir / "test_both.json"
                sidecar = Path(f"{eval_path}.checkpoint.json")
                sidecar.write_text(
                    json.dumps({"run_id": run_id, "checkpoint": str(checkpoint)}),
                    encoding="utf-8",
                )
                offset = index * 0.01 + split_index * 0.001
                payload = {
                    "evaluation_protocol": "paper_test_candidates",
                    "ground_truth_pairs": "test_pairs_only",
                    "direction": "both",
                    "reaction_query_expansion": "canonical_forward_only",
                    "checkpoint": str(checkpoint),
                    "config": str(config),
                    "num_enzyme_candidates": 100 + split_index,
                    "num_reaction_candidates": 10 + split_index,
                    "enzyme_to_reaction/num_queries": 100 + split_index,
                    "reaction_to_enzyme/num_queries": 10 + split_index,
                }
                for direction in DIRECTIONS:
                    payload[f"{direction}/top_1"] = 0.4 + offset
                    payload[f"{direction}/reactzyme_mrr"] = 0.3 + offset
                    payload[f"{direction}/first_positive_mrr"] = 0.5 + offset
                payload["balanced_reactzyme_mrr"] = 0.3 + offset
                eval_path.write_text(json.dumps(payload), encoding="utf-8")
                writer.writerow(
                    {
                        "wave": index,
                        "variant": variant,
                        "label": label,
                        "split": split,
                        "run_id": run_id,
                        "test_config": config,
                        "eval_json": eval_path,
                        "description": f"Description {label}",
                    }
                )
    return plan


def _generate(tmp_path: Path, plan: Path):
    return generate_report(
        plan_path=plan,
        output_json=tmp_path / "report.json",
        output_csv=tmp_path / "report.csv",
        output_markdown=tmp_path / "report.md",
    )


def test_report_requires_and_validates_full_f0_f6_grid(tmp_path):
    plan = _write_fixture(tmp_path)

    report = _generate(tmp_path, plan)

    assert report["status"] == {
        "complete": True,
        "expected_evaluations": 21,
        "validated_evaluations": 21,
    }
    assert report["variants"]["F0"]["cells"]["time"]["enzyme_to_reaction"][
        "reactzyme_mrr"
    ] == pytest.approx(0.3)
    assert report["variants"]["F0"]["cells"]["time"]["enzyme_to_reaction"][
        "first_positive_mrr"
    ] == pytest.approx(0.5)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Table 1 format" in markdown
    assert "TIGER (ESM2Text)" in markdown
    assert "https://arxiv.org/pdf/2605.24489" in markdown
    assert "F6" in markdown
    assert "single-seed" in markdown
    assert "## Directional feature coverage" in markdown
    assert "Checkpoint selection: validation arithmetic mean" in markdown
    assert report["directional_feature_coverage"]["reaction_smi"]["test"][
        "num_resolved"
    ] == 2
    assert len(list(csv.DictReader((tmp_path / "report.csv").open()))) == 54


def test_report_rejects_missing_evaluation(tmp_path):
    plan = _write_fixture(tmp_path)
    missing = tmp_path / "reaction_smi" / "F6_variant" / "test_both.json"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="Missing required artifact"):
        _generate(tmp_path, plan)


def test_report_rejects_first_positive_mrr_as_paper_mrr(tmp_path):
    plan = _write_fixture(tmp_path)
    eval_path = tmp_path / "time" / "F0_variant" / "test_both.json"
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    del payload["reaction_to_enzyme/reactzyme_mrr"]
    eval_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="reaction_to_enzyme/reactzyme_mrr"):
        _generate(tmp_path, plan)
