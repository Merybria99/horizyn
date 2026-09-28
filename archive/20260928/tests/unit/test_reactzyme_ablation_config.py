import csv
import json
from pathlib import Path

import pytest
import yaml

from horizyn.benchmarks.reactzyme_ablations import load_matrix_specs, load_variants
from horizyn.benchmarks.reactzyme_runtime import (
    build_train_ec_subset,
    freeze_selection,
    list_plan_variants,
    read_checkpoint_sidecar,
    selection_rows,
    write_checkpoint_sidecar,
    write_summary,
)


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs/benchmarks/reactzyme_paper"


def write_csv(path, fieldnames, rows, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_active_matrices_are_declarative_and_validated():
    splits, latent = load_matrix_specs(CONFIG_ROOT, "latent")
    _, representation = load_matrix_specs(CONFIG_ROOT, "representation")
    _, biological = load_matrix_specs(CONFIG_ROOT, "biological")
    _, reaction_features = load_matrix_specs(CONFIG_ROOT, "reaction_features")

    assert list(splits) == ["time", "enzyme_smi", "reaction_smi"]
    assert [variant["label"] for variant in latent] == [f"A{i}" for i in range(8)]
    assert [variant["label"] for variant in representation] == [
        "A8",
        "A9",
        "A10",
        "A11",
        "A12",
        "A15",
    ]
    assert [variant["label"] for variant in biological] == [f"B{i}" for i in range(5)]
    assert [variant["label"] for variant in reaction_features] == [f"F{i}" for i in range(7)]
    assert biological[3]["supervised_families"] == ["mechanism", "cofactor"]
    assert biological[4]["ec_pretrain"] is True


def test_matrix_loader_rejects_invalid_block_dimensions(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "variants": [
                    {
                        "id": "bad",
                        "label": "A0",
                        "description": "bad layout",
                        "template": "sleec",
                        "mode": "raw_mean_sleec_blockwise",
                        "loss_name": "FullBatchMLNCELoss",
                        "positive_pair_source": "observed_pairs",
                        "hard_negative": False,
                        "structure": False,
                        "capability": False,
                        "block_dims": {"core": 1},
                        "block_weights": {"core": 1.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sum to 512"):
        load_variants(path)


def test_runtime_ec_subset_and_checkpoint_sidecar(tmp_path):
    train_pairs = tmp_path / "pairs.csv"
    ec_source = tmp_path / "ec.csv"
    output_csv = tmp_path / "out/ec.csv"
    output_report = tmp_path / "out/report.json"
    write_csv(
        train_pairs,
        ["protein_id"],
        [{"protein_id": "prot_A"}, {"protein_id": "prot_B"}],
    )
    write_csv(
        ec_source,
        ["protein_id", "ec_number"],
        [
            {"protein_id": "source_A", "ec_number": "1.1.1.1;2.2.2.2"},
            {"protein_id": "source_C", "ec_number": "3.3.3.3"},
        ],
    )

    report = build_train_ec_subset(
        train_pairs_path=train_pairs,
        ec_source_path=ec_source,
        output_csv=output_csv,
        output_report=output_report,
    )
    assert report["train_proteins"] == 2
    assert report["train_proteins_with_ec"] == 1
    assert "prot_A,1.1.1.1;2.2.2.2" in output_csv.read_text()

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    sidecar = tmp_path / "checkpoint.json"
    write_checkpoint_sidecar(sidecar, run_id="time_A0", checkpoint=str(checkpoint))
    assert read_checkpoint_sidecar(sidecar) == str(checkpoint)


def test_runtime_lists_variants_and_writes_summary(tmp_path):
    plan = tmp_path / "train_plan.tsv"
    plan_rows = []
    for index, split in enumerate(("time", "enzyme_smi", "reaction_smi")):
        eval_path = tmp_path / f"{split}.json"
        eval_path.write_text(
            json.dumps(
                {
                    "reaction_to_enzyme/mrr": 0.1 + index * 0.1,
                    "reaction_to_enzyme/reactzyme_mrr": 0.05 + index * 0.05,
                    "reaction_to_enzyme/first_positive_mrr": 0.1 + index * 0.1,
                    "enzyme_to_reaction/mrr": 0.2 + index * 0.1,
                    "enzyme_to_reaction/reactzyme_mrr": 0.1 + index * 0.05,
                    "enzyme_to_reaction/first_positive_mrr": 0.2 + index * 0.1,
                    "balanced_mrr": 0.15 + index * 0.1,
                    "balanced_reactzyme_mrr": 0.075 + index * 0.05,
                }
            ),
            encoding="utf-8",
        )
        plan_rows.append(
            {
                "variant": "A0_mean_observed",
                "label": "A0",
                "split": split,
                "run_id": f"{split}_A0",
                "eval_json": str(eval_path),
            }
        )
    write_csv(
        plan,
        ["variant", "label", "split", "run_id", "eval_json"],
        plan_rows,
        delimiter="\t",
    )

    summary_csv = tmp_path / "summary.csv"
    summary_md = tmp_path / "summary.md"
    assert list_plan_variants(plan) == ["A0_mean_observed"]
    write_summary(plan_path=plan, output_csv=summary_csv, output_markdown=summary_md)

    assert len(summary_csv.read_text().splitlines()) == 4
    summary_rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    assert float(summary_rows[0]["r2e_mrr"]) == pytest.approx(0.05)
    assert float(summary_rows[0]["r2e_first_positive_mrr"]) == pytest.approx(0.1)
    assert float(summary_rows[0]["balanced_mrr"]) == pytest.approx(0.075)
    summary_text = summary_md.read_text()
    assert "R2E Paper MRR" in summary_text
    assert "First-positive MRR is retained" in summary_text
    # The aggregate primary score is the macro mean of all-positive symmetric MRR.
    assert "| A0_mean_observed | 0.1250 | 0.1000 | 0.1250 | 0.0500 |" in summary_text


def test_runtime_freezes_and_verifies_checkpoint_hashes(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"frozen checkpoint")
    eval_json = tmp_path / "test.json"
    sidecar = Path(str(eval_json) + ".checkpoint.json")
    write_checkpoint_sidecar(sidecar, run_id="time_B0", checkpoint=str(checkpoint))
    plan = tmp_path / "train_plan.tsv"
    write_csv(
        plan,
        [
            "variant", "label", "split", "run_id", "test_config", "eval_json",
            "master_port",
        ],
        [
            {
                "variant": "B0_unsupervised_control",
                "label": "B0",
                "split": "time",
                "run_id": "time_B0",
                "test_config": "test.yaml",
                "eval_json": str(eval_json),
                "master_port": "25000",
            }
        ],
        delimiter="\t",
    )
    frozen = tmp_path / "frozen.json"

    payload = freeze_selection(plan_path=plan, output_path=frozen, variants=None)

    assert payload["schema_version"] == "reactzyme_frozen_selection_v1"
    assert selection_rows(frozen)[0]["checkpoint"] == str(checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after selection"):
        selection_rows(frozen)
