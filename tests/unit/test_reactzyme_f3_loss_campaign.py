from pathlib import Path

import yaml

from horizyn.benchmarks.reactzyme_f3_loss_campaign import (
    DEFAULT_BASE_ROOT,
    LOSS_VARIANTS,
    SPLITS,
    configure_f3_run,
    generate_campaign,
    loss_config,
)


def _load(path):
    return yaml.safe_load(Path(path).read_text())


def test_all_loss_variants_use_all_known_pairs_and_fixed_global_beta():
    for variant in LOSS_VARIANTS:
        config = loss_config(variant, "time")
        assert config["positive_pair_source"] == "all_known_in_batch"
        assert config["beta"] == 10.0
        assert config["learn_beta"] is False


def test_reaction_similarity_uses_requested_e2r_weight():
    assert loss_config("L3", "reaction_smi")["lambda_e2r"] == 0.55
    assert loss_config("L3", "time")["lambda_e2r"] == 0.5
    assert loss_config("L2", "reaction_smi")["lambda_e2r"] == 0.5


def test_configure_preserves_f3_model_and_sets_correct_monitor(tmp_path):
    base = _load(DEFAULT_BASE_ROOT / "time/train.yaml")
    generated = configure_f3_run(
        base,
        run_root=tmp_path,
        variant="L4",
        split="time",
        seed=42,
    )
    assert generated["model"] == base["model"]
    assert generated["logging"]["checkpoint_monitor"] == "val/mean_bidirectional_reactzyme_mrr"
    assert (
        generated["training"]["early_stopping"]["monitor"] == "val/mean_bidirectional_reactzyme_mrr"
    )
    assert generated["training"]["devices"] == 4
    assert "isomeric" in generated["data"]["train_reactions_path"]


def test_generator_produces_every_variant_split_seed_launcher_config(tmp_path):
    manifest = generate_campaign(tmp_path, seeds=(42,))
    assert manifest["num_generated_runs"] == len(LOSS_VARIANTS) * len(SPLITS)
    for variant in LOSS_VARIANTS:
        for split in SPLITS:
            assert (tmp_path / "configs" / variant / split / "seed42/train.yaml").is_file()
            assert (tmp_path / "configs" / variant / split / "seed42/test.yaml").is_file()


def test_launcher_precomputes_one_resumable_chiro_cache():
    launcher = Path("scripts/run_reactzyme_f3_loss_campaign.sh").read_text()

    assert 'CHIRO_MOLECULE_CACHE="$RUN_ROOT/data/features/chiro_molecules.sqlite3"' in launcher
    assert '--molecule-cache "$CHIRO_MOLECULE_CACHE" --cache-only' in launcher
    assert '--max-pending-tasks "$CHIRO_MAX_PENDING_TASKS"' in launcher
    assert "CUDA_VISIBLE_DEVICES=3" in launcher
    assert '--molecule-cache "$CHIRO_MOLECULE_CACHE" --cache-read-only' in launcher
