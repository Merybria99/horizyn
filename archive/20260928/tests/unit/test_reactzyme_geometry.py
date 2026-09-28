import csv
import json
from pathlib import Path

import yaml

from horizyn.benchmarks.reactzyme_geometry import (
    PLAN_FIELDS,
    generate_stage,
    load_spec,
    promote,
)
from horizyn.benchmarks.reactzyme_runtime import write_checkpoint_sidecar


ROOT = Path(__file__).resolve().parents[2]


def test_geometry_spec_declares_every_stage_and_valid_block_layout():
    spec = load_spec()

    assert list(spec["stages"]) == ["recipe", "reaction", "enzyme", "hard_negative"]
    assert [item["id"] for item in spec["stages"]["recipe"]["variants"]] == [
        f"R{index}" for index in range(7)
    ]
    assert [item["id"] for item in spec["stages"]["reaction"]["variants"]] == [
        f"A{index}" for index in range(8)
    ]
    assert [item["id"] for item in spec["stages"]["enzyme"]["variants"]] == [
        f"E{index}" for index in range(6)
    ]


def test_recipe_generation_applies_balanced_anchor_recipe_without_labels(tmp_path):
    plan = generate_stage(
        run_root=tmp_path / "run",
        protocol_root=tmp_path / "protocol",
        feature_root=tmp_path / "features",
        stage="recipe",
        seeds=[42],
        selected={"R4"},
        parent_promotion=None,
        max_epochs=3,
        train_batch_size=32,
        retrieval_batch_size=16,
        validation_retrieval_batch_size=16,
        num_workers=0,
        python_bin=Path("/usr/bin/python3"),
        setup_python_bin=Path("/usr/bin/python3"),
        wandb_project="test",
        wandb_entity="test",
        wandb_mode="disabled",
        base_master_port=24000,
    )

    with plan.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 3
    config = yaml.safe_load(Path(rows[2]["config"]).read_text(encoding="utf-8"))
    assert config["training"]["loss"]["name"] == "BidirectionalAnchorBalancedSupConLoss"
    assert config["training"]["loss"]["positive_pair_source"] == "all_known_in_batch"
    assert config["data"]["train_sampler"]["name"] == "reaction_degree_balanced"
    assert "protein_biofp_targets_path" not in config["data"]


def test_promotion_applies_balanced_gates_then_reaction_smi_e2r(tmp_path):
    plan = tmp_path / "recipe.tsv"
    rows = []
    metrics = {
        "R0": {
            "time": (0.70, 0.70),
            "enzyme_smi": (0.80, 0.80),
            "reaction_smi": (0.50, 0.40),
        },
        "R5": {
            "time": (0.70, 0.70),
            "enzyme_smi": (0.80, 0.80),
            "reaction_smi": (0.51, 0.45),
        },
    }
    for variant, split_metrics in metrics.items():
        for split, (r2e, e2r) in split_metrics.items():
            checkpoint = tmp_path / variant / split / "model.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{variant}-{split}".encode())
            sidecar = checkpoint.parent / "selected.json"
            write_checkpoint_sidecar(
                sidecar,
                run_id=f"recipe_{variant}_seed42_{split}",
                checkpoint=str(checkpoint),
            )
            eval_path = checkpoint.parent / "validation.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "reaction_to_enzyme/mrr": r2e,
                        "enzyme_to_reaction/mrr": e2r,
                        "balanced_mrr": (r2e + e2r) / 2,
                    }
                ),
                encoding="utf-8",
            )
            rows.append(
                {
                    **{field: "-" for field in PLAN_FIELDS},
                    "stage": "recipe",
                    "variant": variant,
                    "label": variant,
                    "seed": "42",
                    "split": split,
                    "run_id": f"recipe_{variant}_seed42_{split}",
                    "config": f"{variant}-{split}-train.yaml",
                    "test_config": f"{variant}-{split}-test.yaml",
                    "checkpoint_sidecar": str(sidecar),
                    "validation_eval_json": str(eval_path),
                }
            )
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    result = promote(plan_path=plan, output_path=tmp_path / "promotion.json", top_k=1)

    assert result["winner"] == "R5"
    assert len(result["entries"]) == 3
