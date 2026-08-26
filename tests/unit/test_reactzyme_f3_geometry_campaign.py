from pathlib import Path

import yaml

from horizyn.benchmarks.reactzyme_f3_geometry_campaign import (
    DEFAULT_BASE_ROOT,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_PROTOCOL_ROOT,
    FACTOR_DIMS,
    FACTOR_WEIGHTS,
    VARIANTS,
    configure_run,
    generate_campaign,
)
from horizyn.config import DotDict, validate_config


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


def test_m0_preserves_f3_model_and_loss(tmp_path):
    base, config = _configured(tmp_path, "M0")

    assert config["model"] == base["model"]
    assert config["training"]["loss"] == base["training"]["loss"]
    assert config["data"]["train_batch_size"] == 2048
    assert config["training"]["devices"] == 1
    assert config["ablation"]["cluster_validation_used"] is False


def test_geometry_variants_are_exactly_scoped(tmp_path):
    configs = {name: _configured(tmp_path, name)[1] for name in VARIANTS}
    assert configs["M1"]["model"]["reaction_multimodal_attention"][
        "modality_l2_normalize"
    ] is True

    for name in ("M2", "M3"):
        attention = configs[name]["model"]["reaction_multimodal_attention"]
        assert attention["fusion"] == "factorized_concat"
        assert attention["output_projection"] == "residual_mlp"
        assert attention["factorized_dims"] == FACTOR_DIMS
        assert attention["factorized_weights"] == FACTOR_WEIGHTS
        assert attention["residual_gate_init"] == 0.1
        assert configs[name]["training"]["reaction_residual_identity"] == {
            "weight": 0.02
        }
        validate_config(DotDict(configs[name]))

    assert configs["M3"]["training"]["reaction_chemistry_consistency"] == {
        "weight": 0.03
    }


def test_generate_campaign_writes_four_variants_three_splits_three_seeds(tmp_path):
    manifest = generate_campaign(tmp_path)

    assert set(manifest["variants"]) == set(VARIANTS)
    rows = (tmp_path / "campaign.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 4 * 3 * 3
    assert manifest["execution"]["effective_global_batch_size"] == 2048
