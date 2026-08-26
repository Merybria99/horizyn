"""Generate the single-variant F3 small-reaction-tower capacity experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CONFIG = (
    ROOT
    / "runs/reactzyme_f3_geometry_v1/configs/M0/reaction_smi/seed42/train.yaml"
)
DEFAULT_RUN_ROOT = ROOT / "runs/reactzyme_f3_capacity_v1"
VARIANT = "F3S_small_reaction_tower"
QUERY_ENCODER_DIMS = [512, 1024, 512]
MODALITY_ENCODER_WIDTHS = [1024, 512]
EXPECTED_REACTION_ENCODER_PARAMETERS = 6_937_093
WANDB_PROJECT = "horizyn-reactzyme-f3-capacity-v1"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def configure_small_f3(
    base: dict[str, Any],
    *,
    run_root: Path,
) -> dict[str, Any]:
    """Return exact F3 with only the reaction projection hidden widths reduced."""
    config = copy.deepcopy(base)
    model = config["model"]
    attention = model["reaction_multimodal_attention"]

    if model.get("query_encoder_dims") != [512, 4096, 4096, 512]:
        raise ValueError("F3S must be generated from the audited F3 query encoder")
    if attention.get("modality_encoder_widths") != [4096, 4096]:
        raise ValueError("F3S must be generated from the audited F3 modality encoders")
    if attention.get("fusion") != "attention" or attention.get("output_projection") != "mlp":
        raise ValueError("F3S requires the original F3 attention-plus-MLP fusion")
    if model.get("embedding_dim") != 512:
        raise ValueError("F3S preserves the original 512-dimensional retrieval space")

    model["query_encoder_dims"] = list(QUERY_ENCODER_DIMS)
    attention["modality_encoder_widths"] = list(MODALITY_ENCODER_WIDTHS)

    run_id = f"{VARIANT}_reaction_smi_seed42"
    logging = config["logging"]
    logging["log_dir"] = str(run_root / "logs" / run_id / "train")
    logging["checkpoint_dir"] = str(run_root / "checkpoints" / run_id)
    logging.setdefault("wandb", {}).update(
        {
            "enabled": True,
            "project": WANDB_PROJECT,
            "run_name": f"reactzyme-f3-capacity-{run_id}",
            "tags": [
                "reactzyme-f3-capacity",
                "F3S",
                "small-reaction-tower",
                "reaction_smi",
                "seed-42",
                "single-change",
            ],
        }
    )
    config["ablation"] = {
        "campaign": "reactzyme_f3_capacity_v1",
        "run_id": run_id,
        "variant": VARIANT,
        "split": "reaction_smi",
        "seed": 42,
        "evaluation_subset": "validation",
        "description": (
            "Exact F3 with modality projection widths [1024, 512] and output "
            "projection dimensions [512, 1024, 512]."
        ),
        "source_variant": "M0",
        "single_change": "reaction_tower_hidden_capacity",
        "released_test_final_only": True,
        "cluster_validation_used": False,
    }
    return config


def generate(
    *,
    source_config: Path = DEFAULT_SOURCE_CONFIG,
    run_root: Path = DEFAULT_RUN_ROOT,
) -> dict[str, Any]:
    source_config = source_config.resolve()
    run_root = run_root.resolve()
    base = _load_yaml(source_config)
    config = configure_small_f3(base, run_root=run_root)

    config_path = run_root / "configs" / VARIANT / "reaction_smi" / "seed42" / "train.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest = {
        "schema_version": "reactzyme_f3_capacity_v1",
        "variant": VARIANT,
        "source_config": str(source_config),
        "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        "config": str(config_path),
        "query_encoder_dims": QUERY_ENCODER_DIMS,
        "modality_encoder_widths": MODALITY_ENCODER_WIDTHS,
        "embedding_dim": 512,
        "expected_reaction_encoder_parameters": EXPECTED_REACTION_ENCODER_PARAMETERS,
        "selection": "Validation-only comparison against the audited M0/F3 control.",
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    print(json.dumps(generate(source_config=args.source_config, run_root=args.run_root), indent=2))


if __name__ == "__main__":
    main()
