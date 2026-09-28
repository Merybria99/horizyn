import json

from horizyn.benchmarks.reactzyme_e2r_pareto import (
    ROOT,
    generate_campaign,
    load_yaml,
)
from horizyn.benchmarks.reactzyme_e2r_report import (
    E2R_KEY,
    R2E_KEY,
    build_report,
)
from horizyn.config import load_config


def test_campaign_configs_preserve_f3_and_isolate_directional_inputs(tmp_path):
    run_root = tmp_path / "campaign"
    manifest = generate_campaign(
        run_root=run_root,
        source_run_root=ROOT / "runs/reactzyme_reaction_features_v1",
        project="test-e2r",
        entity="test",
        mode="disabled",
        seeds=(42,),
    )

    q0 = load_yaml(
        run_root / "configs/seed42/Q0_f3/reaction_smi/train.yaml"
    )
    q1 = load_config(
        run_root / "configs/seed42/Q1_e2r_adapter/reaction_smi/train.yaml"
    )
    q2 = load_config(
        run_root / "configs/seed42/Q2_e2r_hardneg/reaction_smi/train.yaml"
    )
    q3 = load_config(
        run_root / "configs/seed42/Q3_dense_transform/reaction_smi/train.yaml"
    )
    q4 = load_config(
        run_root
        / "configs/seed42/Q4_factorized_modalities/reaction_smi/train.yaml"
    )
    q5 = load_config(
        run_root / "configs/seed42/Q5_combined/reaction_smi/train.yaml"
    )
    validation = load_yaml(
        run_root
        / "configs/seed42/Q3_dense_transform/reaction_smi/validation.yaml"
    )

    assert "e2r_adapter" not in q0["model"]
    assert q1.training.training_stage == "e2r_adapter"
    assert q1.training.loss.lambda_r2e == 0.0
    assert q1.training.loss.lambda_e2r == 1.0
    assert q2.data.hard_negative_direction == "enzyme_to_reaction"
    assert q2.training.loss.lambda_e2r_hard_neg > 0
    assert q2.training.use_distributed_sampler is False
    assert q3.data.reaction_load_directional is True
    assert q3.data.reaction_use_directional is False
    assert q3.model.e2r_adapter.use_directional_inputs is True
    assert q4.model.e2r_adapter.use_factorized_inputs is True
    assert q5.training.use_distributed_sampler is False
    assert (
        validation["data"]["test_pairs_path"]
        == q3.data.validation_pairs_path
    )
    assert manifest["selection"]["primary"].startswith(
        "validation reaction_smi"
    )


def _write_result(path, r2e, e2r):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({R2E_KEY: r2e, E2R_KEY: e2r}),
        encoding="utf-8",
    )


def test_report_promotes_on_validation_without_using_test(tmp_path):
    run_root = tmp_path / "campaign"
    for subset in ("validation", "test"):
        for split in ("time", "enzyme_smi", "reaction_smi"):
            _write_result(
                run_root / f"eval/seed42/{split}/Q0_f3/{subset}.json",
                0.5,
                0.5,
            )
            _write_result(
                run_root
                / f"eval/seed42/{split}/Q1_e2r_adapter/{subset}.json",
                0.5,
                0.6 if split == "reaction_smi" else 0.5,
            )
            _write_result(
                run_root
                / f"eval/seed42/{split}/Q2_e2r_hardneg/{subset}.json",
                0.5,
                0.55 if split == "reaction_smi" else 0.5,
            )
    _write_result(
        run_root / "eval/seed42/reaction_smi/Q1_e2r_adapter/test.json",
        0.1,
        0.1,
    )

    report = build_report(run_root, seed=42)

    assert report["promoted_variants"] == [
        "Q1_e2r_adapter",
        "Q2_e2r_hardneg",
    ]
    assert report["test_used_for_selection"] is False
