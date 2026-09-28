#!/usr/bin/env python3
"""Instantiate the unchanged ReactZyme F3 architecture/loss for EnzymeCAGE data."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=4)
    args = parser.parse_args()
    root = args.run_root.resolve()
    data = root / "data"
    features = root / "features"
    config = yaml.safe_load(args.template.read_text())
    paths = config["data"]
    paths.update(
        train_pairs_path=str(data / "train_pairs.csv"),
        train_reactions_path=str(data / "normalized/train_rxns.csv"),
        validation_pairs_path=str(data / "validation_pairs.csv"),
        validation_reactions_path=str(data / "normalized/validation_rxns.csv"),
        validation_retrieval_candidate_ids_path=str(data / "validation_candidate_ids.txt"),
        protein_residue_embeds_path=str(features / "proteins_prott5_residue.h5"),
        train_reaction_t5v2_embeds_path=str(features / "reactiont5v2.forward.h5"),
        validation_reaction_t5v2_embeds_path=str(features / "reactiont5v2.forward.h5"),
        train_reaction_unimol2_embeds_path=str(features / "unimol2.forward.h5"),
        validation_reaction_unimol2_embeds_path=str(features / "unimol2.forward.h5"),
        train_reaction_chiro_embeds_path=str(features / "chiro.forward.h5"),
        validation_reaction_chiro_embeds_path=str(features / "chiro.forward.h5"),
        train_reaction_chemistry_vectors_path=str(features / "chemistry/train_reaction_set_features.npz"),
        validation_reaction_chemistry_vectors_path=str(features / "chemistry/validation_reaction_set_features.npz"),
    )
    config["training"].update(devices=args.devices,
                              validation_retrieval_candidate_ids_path=str(data / "validation_candidate_ids.txt"))
    config["logging"].update(log_dir=str(root / "logs/train"),
                             checkpoint_dir=str(root / "checkpoints"))
    config["logging"]["wandb"]["enabled"] = False
    config["ablation"].update(run_id="enzymecage_f3", split="enzymecage_original_valid",
                              data_protocol="enzymecage_original_train_valid_positive_only",
                              evaluation_subset="original_validation",
                              evaluation_protocol="all_validation_positive_proteins",
                              description="F3 architecture and loss on EnzymeCAGE positive pairs")
    target = root / "configs/train.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(config, sort_keys=False)
    if target.exists() and target.read_text() != serialized:
        old = yaml.safe_load(target.read_text())
        allowed = {}
        for split in ("train", "validation"):
            allowed[f"{split}_reactions_path"] = str(data / f"{split}_rxns.csv")
            for key, name in (("t5v2", "reactiont5v2"), ("unimol2", "unimol2"), ("chiro", "chiro")):
                allowed[f"{split}_reaction_{key}_embeds_path"] = str(features / f"{name}.h5")
        for key, previous in allowed.items():
            if old["data"].get(key) == previous:
                old["data"][key] = paths[key]
        if old != config:
            raise SystemExit(f"Existing configuration differs beyond the audited reaction/cache paths: {target}")
        backup = target.with_name("train.pre_forward_ids.yaml")
        if backup.exists():
            raise SystemExit(f"Configuration backup already exists: {backup}")
        shutil.copy2(target, backup)
    target.write_text(serialized)
    print(target)


if __name__ == "__main__":
    main()
