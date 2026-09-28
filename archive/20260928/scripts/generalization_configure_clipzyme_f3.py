#!/usr/bin/env python3
"""Configure the locked, fresh F3 architecture on the official EnzymeMap split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=4)
    args = parser.parse_args()
    catalog = args.catalog.resolve()
    preparation = json.loads((catalog / "preparation.json").read_text())
    if (preparation["schema"] != "clipzyme_fixed_f3_catalog_preparation_v1"
            or [preparation["split_files"][split]["association_rows"]
                for split in ("train", "validation", "test")] != [34427, 7287, 4642]
            or preparation["candidate_ids"] != 261907
            or preparation["candidate_ids_without_sequence"]):
        raise ValueError("Official EnzymeMap catalog is incomplete or changed")
    config = yaml.safe_load(args.template.read_text())
    features = catalog / "features"
    data = config["data"]
    data.update(
        train_pairs_path=str(catalog / "train_pairs.csv"),
        train_reactions_path=str(catalog / "train_rxns.csv"),
        validation_pairs_path=str(catalog / "validation_pairs.csv"),
        validation_reactions_path=str(catalog / "validation_rxns.csv"),
        validation_retrieval_candidate_ids_path=str(catalog / "validation_candidate_ids.txt"),
        protein_residue_embeds_path=str(features / "proteins_prott5_residue.h5"),
        train_reaction_t5v2_embeds_path=str(features / "reactiont5v2.forward.h5"),
        validation_reaction_t5v2_embeds_path=str(features / "reactiont5v2.forward.h5"),
        train_reaction_unimol2_embeds_path=str(features / "unimol2.forward.h5"),
        validation_reaction_unimol2_embeds_path=str(features / "unimol2.forward.h5"),
        train_reaction_chiro_embeds_path=str(features / "chiro.forward.h5"),
        validation_reaction_chiro_embeds_path=str(features / "chiro.forward.h5"),
        train_reaction_chemistry_vectors_path=str(catalog / "chemistry/train_reaction_set_features.npz"),
        validation_reaction_chemistry_vectors_path=str(catalog / "chemistry/validation_reaction_set_features.npz"),
    )
    training = config["training"]
    training.update(devices=args.devices,
                    validation_retrieval_candidate_ids_path=str(catalog / "validation_candidate_ids.txt"))
    config["logging"].update(log_dir=str(catalog / "logs/train"),
                             checkpoint_dir=str(catalog / "checkpoints"))
    config["logging"]["wandb"]["enabled"] = False
    config["ablation"].update(run_id="clipzyme_enzymemap_f3",
                              split="official_rule_split",
                              data_protocol="official_EnzymeMap_train_positives_only",
                              evaluation_subset="official_validation_for_selection",
                              evaluation_protocol="official_261907_screening_test_only",
                              description="Fresh locked F3 towers trained only on official EnzymeMap train associations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(config, sort_keys=False)
    if args.output.exists() and args.output.read_text() != serialized:
        raise ValueError(f"Existing EnzymeMap F3 config differs: {args.output}")
    args.output.write_text(serialized)
    receipt = {"schema": "clipzyme_enzymemap_f3_config_v1",
               "catalog_preparation_sha256": sha256(catalog / "preparation.json"),
               "template_sha256": sha256(args.template),
               "config_sha256": sha256(args.output),
               "train_rows": 34427, "validation_rows": 7287,
               "test_rows_for_evaluation_only": 4642,
               "tower_initialization": "fresh_random",
               "pair_supervision": "official_EnzymeMap_train_only",
               "source_sha256": sha256(Path(__file__))}
    (args.output.parent / "config_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
