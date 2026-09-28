import csv
from pathlib import Path

import yaml

from horizyn.benchmarks.reactzyme_cluster_evaluation import (
    audit_checkpoint_training_overlap,
    build_f3_cluster_test_config,
    build_f3_cluster_training_config,
    build_f3_panel_config,
    discover_epoch_checkpoints,
    harmonic_mrr,
    select_checkpoint,
)


def test_checkpoint_discovery_reports_requested_gaps(tmp_path):
    for epoch in (10, 12, 30):
        (tmp_path / f"protein-pooling-epoch={epoch}.ckpt").touch()
    (tmp_path / "last.ckpt").touch()

    checkpoints, missing = discover_epoch_checkpoints(
        tmp_path,
        minimum_epoch=10,
        maximum_epoch=12,
    )

    assert list(checkpoints) == [10, 12]
    assert missing == [11]


def test_selection_maximizes_all_positive_e2r_under_r2e_constraint():
    rows = [
        {
            "epoch": 26,
            "e2r_all_positive_mrr": 0.50,
            "r2e_all_positive_mrr": 0.39,
            "harmonic_all_positive_mrr": harmonic_mrr(0.50, 0.39),
        },
        {
            "epoch": 28,
            "e2r_all_positive_mrr": 0.48,
            "r2e_all_positive_mrr": 0.40,
            "harmonic_all_positive_mrr": harmonic_mrr(0.48, 0.40),
        },
        {
            "epoch": 29,
            "e2r_all_positive_mrr": 0.55,
            "r2e_all_positive_mrr": 0.37,
            "harmonic_all_positive_mrr": harmonic_mrr(0.55, 0.37),
        },
    ]

    selection = select_checkpoint(rows, baseline_epoch=28, r2e_tolerance=0.015)

    assert selection["r2e_floor"] == 0.385
    assert selection["selected"]["epoch"] == 26
    assert next(row for row in selection["rows"] if row["epoch"] == 29)[
        "r2e_eligible"
    ] is False


def test_f3_panel_config_uses_validation_pool_and_train_modalities(tmp_path):
    panel = tmp_path / "similarity_0p85"
    panel.mkdir()
    for name in (
        "validation_pairs.csv",
        "validation_rxns.csv",
        "validation_candidate_ids.txt",
    ):
        (panel / name).touch()
    features = tmp_path / "features"
    features.mkdir()
    for name in ("reactiont5v2.h5", "unimol2.h5", "chiro.h5"):
        (features / name).touch()
    chemistry = tmp_path / "chemistry.npz"
    chemistry.touch()
    protein_h5 = tmp_path / "proteins.h5"
    protein_h5.touch()
    base = {
        "data": {"protein_residue_embeds_path": str(protein_h5)},
        "training": {},
        "ablation": {},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    output = tmp_path / "evaluation.yaml"

    config = build_f3_panel_config(
        base_config_path=base_path,
        panel_dir=panel,
        chemistry_path=chemistry,
        official_train_feature_dir=features,
        output_path=output,
    )

    assert config["data"]["test_pairs_path"] == str(
        (panel / "validation_pairs.csv").resolve()
    )
    assert config["data"]["validation_retrieval_candidate_ids_path"] == str(
        (panel / "validation_candidate_ids.txt").resolve()
    )
    assert config["data"]["reaction_t5v2_embeds_path"] == str(
        (features / "reactiont5v2.h5").resolve()
    )
    assert config["data"]["reaction_chemistry_vectors_path"] == str(
        chemistry.resolve()
    )
    assert config["ablation"]["evaluation_protocol"] == (
        "configured_forward_candidates"
    )
    assert output.exists()


def test_overlap_audit_blocks_contaminated_retrospective_selection(tmp_path):
    fields = ["pr_id", "reaction_id", "protein_id"]
    training = tmp_path / "training.csv"
    panel = tmp_path / "panel.csv"
    for path, rows in (
        (training, [["0", "r0", "p0"]]),
        (panel, [["1", "r0", "p0"], ["2", "r1", "p1"]]),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)

    audit = audit_checkpoint_training_overlap(
        panel_pairs_path=panel,
        checkpoint_training_pairs_path=training,
    )

    assert audit["overlapping_reactions"] == 1
    assert audit["overlapping_pair_keys"] == 1
    assert audit["retrospective_selection_valid"] is False


def test_cluster_training_config_retains_all_epochs_for_offline_guard(tmp_path):
    panel = tmp_path / "similarity_0p85"
    chemistry = panel / "features/reaction_set"
    features = tmp_path / "features"
    panel.mkdir()
    chemistry.mkdir(parents=True)
    features.mkdir()
    protein_h5 = tmp_path / "proteins.h5"
    protein_h5.touch()
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "data": {"protein_residue_embeds_path": str(protein_h5)},
                "logging": {"wandb": {"tags": []}},
                "training": {"early_stopping": {"enabled": True}},
                "ablation": {},
            }
        ),
        encoding="utf-8",
    )

    config = build_f3_cluster_training_config(
        base_config_path=base_path,
        panel_dir=panel,
        chemistry_dir=chemistry,
        official_train_feature_dir=features,
        run_root=tmp_path / "run",
        output_path=tmp_path / "train.yaml",
    )

    assert config["logging"]["checkpoint_monitor"] == (
        "val/enzyme_to_reaction/reactzyme_mrr"
    )
    assert config["logging"]["save_top_k"] == -1
    assert config["training"]["early_stopping"]["enabled"] is False
    assert config["ablation"]["checkpoint_selection"]["secondary"] == (
        "harmonic_all_positive_mrr"
    )


def test_cluster_test_config_uses_released_test_and_clean_chemistry(tmp_path):
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    for name in ("test_pairs.csv", "test_rxns.csv", "test_candidate_ids.txt"):
        (protocol / name).touch()
    features = tmp_path / "features"
    features.mkdir()
    for name in ("reactiont5v2.h5", "unimol2.h5", "chiro.h5"):
        (features / name).touch()
    chemistry = tmp_path / "test_chemistry.npz"
    chemistry.touch()
    proteins = tmp_path / "proteins.h5"
    proteins.touch()
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "data": {"protein_residue_embeds_path": str(proteins)},
                "training": {},
                "ablation": {},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "test.yaml"

    config = build_f3_cluster_test_config(
        base_config_path=base_path,
        test_protocol_dir=protocol,
        chemistry_path=chemistry,
        official_test_feature_dir=features,
        output_path=output,
    )

    assert config["data"]["test_pairs_path"] == str(
        (protocol / "test_pairs.csv").resolve()
    )
    assert config["data"]["reaction_t5v2_embeds_path"] == str(
        (features / "reactiont5v2.h5").resolve()
    )
    assert config["data"]["reaction_chemistry_vectors_path"] == str(
        chemistry.resolve()
    )
    assert config["training"]["validation_retrieval_candidate_ids_path"] == str(
        (protocol / "test_candidate_ids.txt").resolve()
    )
    assert config["ablation"]["evaluation_protocol"] == "paper_test_candidates"
    assert output.exists()
