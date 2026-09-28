#!/usr/bin/env python3
"""Export another native checkpoint on the fixed Case1 representation controls.

The F3 preparation supplies shared immutable pretrained inputs. No candidate
reference labels, assay labels, rankings, or performance metrics are loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from horizyn.benchmarks.retrieval import BenchmarkTask, build_reaction_inputs, encode_reactions, encode_residue_targets, load_repo_checkpoint
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from scripts.generalization_export import atomic_json, atomic_npz, identity, resolve
from wet_lab.query import resolve_reaction_input_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, output, config_path, checkpoint = map(resolve, (args.feature_root, args.output, args.config, args.checkpoint))
    if (output / "complete.json").exists():
        raise ValueError("Completed controls already exist; retain their provenance")
    output.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "catalog.json"
    proteins = json.loads(catalog_path.read_text())["proteins"]
    if len(proteins) != 123 or len(set(proteins)) != 123:
        raise ValueError("Expected the fixed 123 unique-sequence catalog")
    config = load_config(str(config_path))
    policy = resolve_reaction_input_policy({"feature_generation": {}}, config)
    if policy["policy"] != "participant_self_reaction":
        raise ValueError("These matched input controls require participant self-reaction training")
    matched = root / "f3_epoch29"
    preparation = json.loads((matched / "complete.json").read_text())
    canonical_config = load_config(preparation["config"]["path"])
    feature_keys = [f"train_reaction_{m}_{'vectors_path' if m == 'chemistry' else 'embeds_path'}"
                    for m in ("t5v2", "unimol2", "chiro", "chemistry")]
    for key in feature_keys:
        if resolve(config.data[key]) != resolve(canonical_config.data[key]):
            raise ValueError(f"Checkpoint was trained on a different pretrained feature source: {key}")
    historical = ROOT / "wet_lab/Case1/restricted_setting/runs/tagatose_4_epimerase_e07cc5e4ab31"
    residue_path = ROOT / "wet_lab/Case1/restricted_setting/candidate_pool/proteins_prott5_residue.h5"
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    module, _ = load_repo_checkpoint(checkpoint, config, args.device)
    module.requires_grad_(False)
    if module.model.query_encoder.side_composition != "molecule_set":
        raise ValueError("Native checkpoint does not use participant molecular pooling")
    residue = ResidueEmbedDataset(str(residue_path), in_memory=False, dtype=torch.float32,
        max_tokens=config.data.max_protein_tokens, truncation=config.data.protein_truncation)
    p = encode_residue_targets(module, residue, proteins, args.device, 32, False).float().cpu().numpy()
    query_id = "tagatose_4_epimerase"
    queries = {}
    for name, directory, csv_path in (("historical", historical / "features", historical / "reaction.csv"),
                                      ("matched", matched, matched / "feature_reaction.csv")):
        config.data.reaction_chemistry_vectors_path = str(directory / "reaction_set_features.npz")
        task = BenchmarkTask(name=f"case1_control_{name}", task_type="screening", dataset="wet_lab", task_label=query_id,
            split="query", pairs=csv_path, reactions=csv_path,
            reaction_model_embeds_h5=directory / "reactiont5v2.h5", reaction_unimol2_embeds_h5=directory / "unimol2.h5",
            reaction_chiro_embeds_h5=directory / "chiro.h5", directions=("reaction_to_enzyme",))
        inputs = build_reaction_inputs(task, config)
        queries[name] = encode_reactions(module, inputs, [query_id], args.device, 1).float().cpu().numpy()
    if p.shape != (123, 512) or any(q.shape != (1, 512) for q in queries.values()):
        raise ValueError("Unexpected native encoder dimensions")
    if not all(np.isfinite(value).all() for value in [p, *queries.values()]):
        raise ValueError("Nonfinite native embeddings")
    atomic_npz(output / "features.npz", proteins=p, query_historical=queries["historical"], query_matched=queries["matched"])
    manifest = dict(schema="case1_native_checkpoint_embedding_controls_v1", reference_labels_used=False,
        checkpoint=identity(checkpoint, True), config=identity(config_path, True), catalog=identity(catalog_path, True),
        input_policy=policy, matched_preparation=identity(matched / "complete.json", True), residues=identity(residue_path, True),
        sources={f"{name}/{filename}": identity(directory / filename, True)
                 for name, directory in (("historical", historical / "features"), ("matched", matched))
                 for filename in ("reactiont5v2.h5", "unimol2.h5", "chiro.h5", "reaction_set_features.npz")},
        precision="float32_highest", proteins=123, query_count=1,
        original_reaction_smiles=preparation["original_reaction_smiles"], encoded_reaction_smiles=preparation["encoded_reaction_smiles"],
        output=identity(output / "features.npz", True), script=identity(__file__, True))
    atomic_json(output / "complete.json", manifest)
    (output / "export_source.py").write_text(Path(__file__).read_text())
    print(json.dumps(dict(output=str(output), proteins=123, query_count=1)), flush=True)


if __name__ == "__main__":
    main()
