#!/usr/bin/env python3
"""Export exact F3 Case1 embeddings with historical and matched input controls.

No reference candidate labels or ranking metrics are loaded. The representative
candidate catalog is sequence-deduplicated before this script is called.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch

from horizyn.benchmarks.retrieval import (
    BenchmarkTask, build_reaction_inputs, encode_reactions,
    encode_residue_targets, load_repo_checkpoint,
)
from horizyn.capability.reaction_set_features import materialize_reaction_set_features
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from scripts.generalization_export import atomic_json, atomic_npz, identity
from wet_lab.query import reaction_smiles_for_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.feature_root.resolve()
    out = root / "f3_epoch29"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise ValueError("Export already complete; retain its immutable provenance")
    catalog = json.loads((root / "catalog.json").read_text())
    ids = catalog["proteins"]
    case = ROOT / "wet_lab/Case1/restricted_setting"
    source = case / "runs/tagatose_4_epimerase_e07cc5e4ab31"
    config_path = ROOT / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/train.yaml"
    checkpoint = ROOT / "runs/reactzyme_reaction_features_v1/checkpoints/reaction_smi/F3_set_chemistry/protein-pooling-epoch=29.ckpt"
    config = load_config(str(config_path))
    if not config.data.normalize_molecule_sets_as_self_reactions or config.model.reaction_multimodal_attention.side_composition != "molecule_set":
        raise ValueError("Expected the audited F3 participant-set model")
    with (source / "reaction.csv").open() as handle:
        reaction, = list(csv.DictReader(handle))
    query_id = reaction["reaction_id"]
    matched = reaction_smiles_for_model(reaction["reaction_smiles"], normalize_molecule_sets_as_self_reactions=True)
    reaction_path = out / "feature_reaction.csv"
    with reaction_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reaction_id", "reaction_smiles"])
        writer.writeheader()
        writer.writerow(dict(reaction_id=query_id, reaction_smiles=matched))
    # Molecular features are fixed per molecule. Reuse both participants from
    # the historical directed cache; no new conformer or model randomness.
    for modality in ("unimol2", "chiro"):
        destination = out / f"{modality}.h5"
        with h5py.File(source / f"features/{modality}.h5", "r") as src:
            if list(src["ids"].asstr()[:]) != [query_id]:
                raise ValueError("Unexpected historical molecular cache IDs")
            blocks = [src[f"{side}_vectors"][:] for side in ("reactant", "product")]
            if any(len(block) != 1 for block in blocks):
                raise ValueError("Case1 should have one molecule on each physical side")
            values = np.concatenate(blocks)
            with h5py.File(destination, "w") as dst:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                dst.attrs["reaction_feature_policy"] = "participant_self_reaction_v1"
                dst["ids"] = np.asarray([query_id], dtype=h5py.string_dtype())
                for side in ("reactant", "product"):
                    dst[f"{side}_vectors"] = values
                    dst[f"{side}_offsets"] = [0, len(values)]
    t5 = out / "reactiont5v2.h5"
    model_path = ROOT.parent / "hf_cache/hub/models--sagawa--ReactionT5v2-forward/snapshots/933114058cb2604dc1bf536dbebdfcefbe83d4fc"
    command = [sys.executable, "scripts/extract_reaction_t5v2_embeddings.py",
               "--reactions", str(reaction_path), "--output", str(t5),
               "--model-name", str(model_path), "--batch-size", "1", "--max-length", "512",
               "--pooling", "mean", "--device", args.device, "--dtype", "float16",
               "--no-bidirectional", "--no-allow-pseudo-reactions", "--force"]
    subprocess.run(command, cwd=ROOT, check=True)
    schema = ROOT / "runs/reactzyme_reaction_features_v1/data/reaction_smi/reaction_set/schema.json"
    cofactors = ROOT / "data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv"
    chemistry = out / "reaction_set_features.npz"
    chemistry_report = materialize_reaction_set_features(
        reactions_path=reaction_path, schema_path=schema,
        cofactor_dictionary_path=cofactors, output_path=chemistry)
    with np.load(chemistry, allow_pickle=True) as new, np.load(source / "features/reaction_set_features.npz", allow_pickle=True) as old:
        chemistry_delta = float(np.max(np.abs(new["vectors"] - old["vectors"])))
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    module, _ = load_repo_checkpoint(checkpoint, config, args.device)
    module.requires_grad_(False)
    residue_path = case / "candidate_pool/proteins_prott5_residue.h5"
    residues = ResidueEmbedDataset(str(residue_path), in_memory=False, dtype=torch.float32,
        max_tokens=config.data.max_protein_tokens, truncation=config.data.protein_truncation)
    proteins = encode_residue_targets(module, residues, ids, args.device, 32, False).float().cpu().numpy()
    queries = {}
    for name, directory, csv_path in (("historical", source / "features", source / "reaction.csv"),
                                      ("matched", out, reaction_path)):
        config.data.reaction_chemistry_vectors_path = str(directory / "reaction_set_features.npz")
        task = BenchmarkTask(name=f"case1_f3_{name}", task_type="screening", dataset="wet_lab",
            task_label=query_id, split="query", pairs=csv_path, reactions=csv_path,
            reaction_model_embeds_h5=directory / "reactiont5v2.h5",
            reaction_unimol2_embeds_h5=directory / "unimol2.h5",
            reaction_chiro_embeds_h5=directory / "chiro.h5", directions=("reaction_to_enzyme",))
        inputs = build_reaction_inputs(task, config)
        queries[name] = encode_reactions(module, inputs, [query_id], args.device, 1).float().cpu().numpy()
    if not all(np.isfinite(value).all() for value in [proteins, *queries.values()]):
        raise ValueError("Nonfinite learned embeddings")
    atomic_npz(out / "features.npz", proteins=proteins,
               query_historical=queries["historical"], query_matched=queries["matched"])
    manifest = dict(schema="case1_f3_exact_embedding_controls_v1", reference_labels_used=False,
        checkpoint=identity(checkpoint, True), config=identity(config_path, True),
        catalog=identity(root / "catalog.json", True), residues=identity(residue_path),
        original_reaction_smiles=reaction["reaction_smiles"], encoded_reaction_smiles=matched,
        protein_count=len(ids), precision="float32_highest", extraction_command=command,
        chemistry=dict(schema=identity(schema, True), report=chemistry_report,
                       historical_max_abs_difference=chemistry_delta),
        historical_sources={str(p.name): identity(p, True) for p in (source / "features").iterdir() if p.is_file()},
        output=identity(out / "features.npz", True), script=identity(__file__, True))
    atomic_json(out / "complete.json", manifest)
    print(json.dumps(dict(output=str(out), protein_count=len(ids), chemistry_max_abs_delta=chemistry_delta)), flush=True)


if __name__ == "__main__":
    main()
