#!/usr/bin/env python3
"""Create the audited four-GPU F3 run on the corrected ReactZyme split."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT
    / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi"
)
PROTOCOL_ROOT = ROOT / "data/revised_protocols/reactzyme_paper_v2/reaction_smi"
FEATURE_ROOT = ROOT / "data/revised_protocols/reactzyme_official/features/reaction_smi"
DEFAULT_RUN_ROOT = ROOT / "runs/reactzyme_f3_reaction_smi_paper_v2_4gpu"
RUN_ID = "F3_reaction_smi_paper_v2_seed42_4gpu"
WANDB_PROJECT = "horizyn-reactzyme-level1-f3-v2"
MONITOR = "val/enzyme_to_reaction/reactzyme_mrr"
MODALITIES = ("reaction_t5v2", "reaction_unimol2", "reaction_chiro")
ARTIFACTS = {
    "reaction_t5v2": "reactiont5v2.h5",
    "reaction_unimol2": "unimol2.h5",
    "reaction_chiro": "chiro.h5",
}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_audited_f3(config: dict[str, Any]) -> None:
    model = config["model"]
    attention = model["reaction_multimodal_attention"]
    expected = {
        "query_encoder_dims": [512, 4096, 4096, 512],
        "target_encoder_dims": [512, 512],
        "embedding_dim": 512,
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"Canonical F3 {key} changed: {model.get(key)!r}")
    if attention.get("modality_encoder_widths") != [4096, 4096]:
        raise ValueError("Canonical F3 modality tower widths changed")
    if attention.get("fusion") != "attention" or attention.get("output_projection") != "mlp":
        raise ValueError("Canonical F3 fusion changed")
    if config["training"]["loss"].get("positive_pair_source") != "observed_pairs":
        raise ValueError("Canonical F3 must use observed pair positives")
    if int(config["data"]["train_batch_size"]) * int(config["training"]["devices"]) != 2048:
        raise ValueError("Canonical F3 global batch must be 2048")


def configure_logging(config: dict[str, Any], run_root: Path, subset: str) -> None:
    logging = config["logging"]
    logging.update(
        {
            "log_dir": str(run_root / "logs" / RUN_ID / subset),
            "checkpoint_dir": str(run_root / "checkpoints" / RUN_ID),
            "checkpoint_monitor": MONITOR,
            "checkpoint_mode": "max",
            "checkpoint_on_validation_end": True,
            "save_top_k": 3,
        }
    )
    logging.setdefault("wandb", {}).update(
        {
            "enabled": subset == "train",
            "project": WANDB_PROJECT,
            "entity": "omnai",
            "run_name": "reactzyme-level1-f3-reaction-smi-paper-v2-seed42",
            "mode": "online",
            "tags": [
                "reactzyme",
                "level1",
                "reaction-smi",
                "paper-v2",
                "F3",
                "seed-42",
                "four-gpu",
                "global-batch-2048",
                "observed-pairs",
            ],
        }
    )


def configure_common(config: dict[str, Any], run_root: Path, subset: str) -> None:
    config["seed"] = 42
    configure_logging(config, run_root, subset)
    config["training"]["early_stopping"].update(
        {"enabled": True, "monitor": MONITOR, "mode": "max"}
    )
    config["ablation"] = {
        "campaign": "reactzyme_f3_reaction_smi_paper_v2_4gpu",
        "run_id": RUN_ID,
        "variant": "F3",
        "split": "reaction_smi",
        "seed": 42,
        "evaluation_subset": subset,
        "description": "Audited F3 on the corrected deterministic ReactZyme split.",
        "source_variant": "F3_set_chemistry",
        "protocol_schema": "reactzyme_paper_protocol_v2",
        "validation_fidelity": "deterministic_positive_only_reactzyme_algorithm_analogue",
        "test_protocol": "paper_test_candidates",
        "audited_global_batch_size": 2048,
        "released_test_final_only": True,
    }


def configure_train(base: dict[str, Any], run_root: Path) -> dict[str, Any]:
    config = copy.deepcopy(base)
    assert_audited_f3(config)
    configure_common(config, run_root, "train")
    data = config["data"]
    data.update(
        {
            "train_pairs_path": str(PROTOCOL_ROOT / "train_pairs.csv"),
            "train_reactions_path": str(PROTOCOL_ROOT / "train_rxns.csv"),
            "validation_pairs_path": str(PROTOCOL_ROOT / "validation_pairs.csv"),
            "validation_reactions_path": str(PROTOCOL_ROOT / "validation_rxns.csv"),
            "validation_retrieval_candidate_ids_path": str(
                PROTOCOL_ROOT / "validation_candidate_ids.txt"
            ),
            "train_batch_size": 512,
            "train_reaction_chemistry_vectors_path": str(
                run_root / "data/reaction_set/train_reaction_set_features.npz"
            ),
            "validation_reaction_chemistry_vectors_path": str(
                run_root / "data/reaction_set/validation_reaction_set_features.npz"
            ),
        }
    )
    for prefix in MODALITIES:
        artifact = ARTIFACTS[prefix]
        data[f"train_{prefix}_embeds_path"] = str(FEATURE_ROOT / "train" / artifact)
        data[f"validation_{prefix}_embeds_path"] = str(
            FEATURE_ROOT / "train" / artifact
        )
    config["training"].update(
        {
            "devices": 4,
            "strategy": "ddp_find_unused_parameters_true",
            "validation_retrieval_candidate_ids_path": str(
                PROTOCOL_ROOT / "validation_candidate_ids.txt"
            ),
        }
    )
    assert_audited_f3(config)
    return config


def configure_eval(
    base: dict[str, Any], run_root: Path, *, subset: str
) -> dict[str, Any]:
    if subset not in {"validation", "test"}:
        raise ValueError(f"Unsupported evaluation subset: {subset}")
    config = copy.deepcopy(base)
    assert_audited_f3(config)
    configure_common(config, run_root, subset)
    feature_subset = "train" if subset == "validation" else "test"
    data = config["data"]
    data.update(
        {
            "train_pairs_path": str(PROTOCOL_ROOT / "train_pairs.csv"),
            "train_reactions_path": str(PROTOCOL_ROOT / "train_rxns.csv"),
            "test_pairs_path": str(PROTOCOL_ROOT / f"{subset}_pairs.csv"),
            "test_reactions_path": str(PROTOCOL_ROOT / f"{subset}_rxns.csv"),
            "validation_retrieval_candidate_ids_path": str(
                PROTOCOL_ROOT / f"{subset}_candidate_ids.txt"
            ),
            "train_reaction_chemistry_vectors_path": str(
                run_root / "data/reaction_set/train_reaction_set_features.npz"
            ),
            "validation_reaction_chemistry_vectors_path": str(
                run_root / f"data/reaction_set/{subset}_reaction_set_features.npz"
            ),
            "reaction_chemistry_vectors_path": str(
                run_root / f"data/reaction_set/{subset}_reaction_set_features.npz"
            ),
        }
    )
    for prefix in MODALITIES:
        artifact = ARTIFACTS[prefix]
        data[f"train_{prefix}_embeds_path"] = str(FEATURE_ROOT / "train" / artifact)
        data[f"validation_{prefix}_embeds_path"] = str(
            FEATURE_ROOT / feature_subset / artifact
        )
        data[f"{prefix}_embeds_path"] = str(FEATURE_ROOT / feature_subset / artifact)
    config["training"].update(
        {
            "devices": 1,
            "strategy": "auto",
            "validation_retrieval_candidate_ids_path": str(
                PROTOCOL_ROOT / f"{subset}_candidate_ids.txt"
            ),
        }
    )
    return config


def main() -> None:
    run_root = DEFAULT_RUN_ROOT.resolve()
    train_source = SOURCE_ROOT / "train.yaml"
    test_source = SOURCE_ROOT / "test.yaml"
    train = configure_train(load_yaml(train_source), run_root)
    test_base = load_yaml(test_source)
    validation = configure_eval(test_base, run_root, subset="validation")
    test = configure_eval(test_base, run_root, subset="test")

    config_dir = run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    configs = {"train": train, "validation": validation, "test": test}
    config_paths: dict[str, str] = {}
    for name, payload in configs.items():
        path = config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        config_paths[name] = str(path)

    protocol_manifest = PROTOCOL_ROOT.parent / "manifest.json"
    manifest = {
        "schema_version": "reactzyme_f3_reaction_smi_paper_v2_4gpu_v1",
        "run_id": RUN_ID,
        "source_train_config": str(train_source),
        "source_train_config_sha256": sha256(train_source),
        "source_test_config": str(test_source),
        "source_test_config_sha256": sha256(test_source),
        "protocol_manifest": str(protocol_manifest),
        "protocol_manifest_sha256": sha256(protocol_manifest),
        "released_test_sha256": {
            name: sha256(PROTOCOL_ROOT / name)
            for name in ("test_pairs.csv", "test_rxns.csv")
        },
        "configs": config_paths,
        "devices": [0, 1, 2, 3],
        "batch_size_per_gpu": 512,
        "global_batch_size": 2048,
        "checkpoint_monitor": MONITOR,
        "test_protocol": "paper_test_candidates",
        "architecture": {
            "query_encoder_dims": [512, 4096, 4096, 512],
            "modality_encoder_widths": [4096, 4096],
            "target_encoder_dims": [512, 512],
            "embedding_dim": 512,
            "positive_pair_source": "observed_pairs",
        },
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
