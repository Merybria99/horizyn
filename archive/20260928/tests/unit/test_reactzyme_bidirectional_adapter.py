from horizyn.benchmarks.reactzyme_bidirectional_adapter import (
    ROOT,
    VARIANT,
    generate_campaign,
)
from horizyn.config import load_config


def test_q6_campaign_reproduces_q5_while_training_both_adapters(tmp_path):
    run_root = tmp_path / "campaign"
    manifest = generate_campaign(
        run_root=run_root,
        source_run_root=ROOT / "runs/reactzyme_e2r_pareto_v1",
        project="test-bidirectional",
        entity="test",
        mode="disabled",
    )

    config = load_config(run_root / f"configs/seed42/{VARIANT}/reaction_smi/train.yaml")

    assert len(manifest["runs"]) == 3
    assert config.model.e2r_adapter.enabled is True
    assert config.model.r2e_adapter.enabled is True
    assert config.model.r2e_adapter.use_factorized_inputs is True
    assert config.training.training_stage == "bidirectional_adapters"
    assert config.training.loss.lambda_r2e == 1.0
    assert config.training.loss.lambda_e2r == 1.0
    assert config.training.loss.e2r_identity_weight == 0.05
    assert config.training.loss.r2e_identity_weight == 0.05
    assert config.training.loss.lambda_e2r_hard_neg == 0.25
    assert config.training.loss.e2r_hard_neg_top_k == 64
    assert config.training.loss.e2r_hard_neg_margin == 0.05
    assert config.training.use_distributed_sampler is False
    assert config.logging.checkpoint_monitor == "val/balanced_mrr"
    assert config.data.hard_negative_direction == "enzyme_to_reaction"
    assert "Q0_f3/selected.ckpt" in config.training.init_from_checkpoint
    assert config.ablation.parent == "Q0_f3"
