from pathlib import Path

import yaml

from horizyn.benchmarks.reactzyme_f3_shortcut_campaign import (
    DEFAULT_BASE_ROOT,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_PROTOCOL_ROOT,
    PRIOR_WEIGHTS,
    VARIANTS,
    _select_checkpoint,
    configure_run,
    generate_campaign,
)


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _configured(tmp_path: Path, variant: str, split: str = "reaction_smi"):
    base = _load(DEFAULT_BASE_ROOT / split / "train.yaml")
    return base, configure_run(
        base,
        run_root=tmp_path,
        protocol_root=DEFAULT_PROTOCOL_ROOT,
        feature_root=DEFAULT_FEATURE_ROOT,
        variant=variant,
        split=split,
        seed=42,
        evaluation_subset="validation",
    )


def test_s0_preserves_f3_architecture_and_loss(tmp_path):
    base, config = _configured(tmp_path, "S0")

    assert config["model"] == base["model"]
    assert config["training"]["loss"] == base["training"]["loss"]
    assert config["logging"]["checkpoint_monitor"] == (
        "val/enzyme_to_reaction/reactzyme_mrr"
    )
    assert "reaction_reactiont5v2" not in "\n".join(config["data"])
    assert config["data"]["train_reaction_t5v2_embeds_path"].endswith(
        "reaction_smi/train/reactiont5v2.h5"
    )


def test_shortcut_variants_add_only_declared_controls(tmp_path):
    configs = {name: _configured(tmp_path, name)[1] for name in VARIANTS}

    s1 = configs["S1"]["model"]["reaction_multimodal_attention"]
    assert s1["fusion"] == "attention"
    assert s1["modality_l2_normalize"] is True

    for name in ("S2", "S3", "S4", "S5"):
        attention = configs[name]["model"]["reaction_multimodal_attention"]
        assert attention["fusion"] == "prior_bounded_attention"
        assert attention["attention_prior_weights"] == PRIOR_WEIGHTS
        assert attention["attention_adaptation_strength"] == 0.4

    assert configs["S3"]["model"]["reaction_multimodal_attention"][
        "chemistry_dropout"
    ] == 0.25
    assert configs["S4"]["training"]["reaction_chemistry_consistency"] == {
        "weight": 0.03
    }
    assert configs["S5"]["model"]["reaction_multimodal_attention"][
        "chemistry_dropout"
    ] == 0.25
    assert configs["S5"]["training"]["reaction_chemistry_consistency"] == {
        "weight": 0.03
    }


def test_reaction_checkpoint_selection_enforces_s0_r2e_guard():
    rows = [
        {"checkpoint": "bad.ckpt", "e2r": 0.70, "r2e": 0.37},
        {"checkpoint": "eligible.ckpt", "e2r": 0.65, "r2e": 0.395},
        {"checkpoint": "lower.ckpt", "e2r": 0.60, "r2e": 0.40},
    ]

    selected = _select_checkpoint(
        rows,
        split="reaction_smi",
        reaction_r2e_floor=0.39,
    )

    assert selected["checkpoint"] == "eligible.ckpt"


def test_generate_campaign_writes_six_variants_three_splits_three_seeds(tmp_path):
    manifest = generate_campaign(tmp_path)

    assert set(manifest["variants"]) == set(VARIANTS)
    rows = (tmp_path / "campaign.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 6 * 3 * 3
    validation = _load(
        tmp_path / "configs/S5/reaction_smi/seed42/validation.yaml"
    )
    assert validation["data"]["test_pairs_path"].endswith(
        "reaction_smi/validation_pairs.csv"
    )
    assert validation["data"]["reaction_t5v2_embeds_path"].endswith(
        "reaction_smi/train/reactiont5v2.h5"
    )
