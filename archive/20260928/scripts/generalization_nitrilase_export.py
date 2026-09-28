#!/usr/bin/env python3
"""Export the complete nitrilase assay input panel with frozen ReactZyme F3 and matched inputs.

Only the identity catalog, template reaction equations, and pretrained features
are read. Pairwise activity labels and old model scores are never accessed.
Nitrilase-specific invariant: one substrate and one carboxylic-acid vector in
each historical cache; water and ammonia were skipped by those extractors.
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

from horizyn.benchmarks.retrieval import BenchmarkTask, build_reaction_inputs, encode_reactions, encode_residue_targets, load_repo_checkpoint
from horizyn.capability.reaction_set_features import materialize_reaction_set_features
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from scripts.generalization_export import atomic_json, atomic_npz, h5_ids, identity, reaction_block
from wet_lab.query import reaction_smiles_for_model, resolve_reaction_input_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    root = args.feature_root.resolve()
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    proteins, reactions = catalog["proteins"], catalog["reactions"]
    if len(proteins) != 18 or len(reactions) != 38 or len(set(proteins)) != 18 or len(set(reactions)) != 38:
        raise ValueError("The original 38 x 18 panel must remain intact")
    source = ROOT / catalog["source_features"]
    output = root / "f3_epoch29"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        raise ValueError("Completed export exists; preserve its provenance")
    config_path = ROOT / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/train.yaml"
    checkpoint = ROOT / "runs/reactzyme_reaction_features_v1/checkpoints/reaction_smi/F3_set_chemistry/protein-pooling-epoch=29.ckpt"
    config = load_config(str(config_path))
    policy = resolve_reaction_input_policy({"feature_generation": {}}, config)
    if policy["policy"] != "participant_self_reaction":
        raise ValueError("Expected the audited ReactZyme F3 participant-set protocol")
    original_reactions = ROOT / catalog["source_reactions"]
    with original_reactions.open() as handle:
        original = {row["reaction_id"]: row["reaction_smiles"] for row in csv.DictReader(handle)}
    if set(original) != set(reactions):
        raise ValueError("Reaction catalog/feature IDs disagree")
    reaction_path = output / "feature_reactions.csv"
    with reaction_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reaction_id", "reaction_smiles"])
        for key in reactions:
            writer.writerow([key, reaction_smiles_for_model(original[key], normalize_molecule_sets_as_self_reactions=True)])
    for modality in ("unimol2", "chiro"):
        with h5py.File(source / f"{modality}.h5") as src:
            index = {key: i for i, key in enumerate(h5_ids(src))}
            rows, offsets = [], [0]
            for key in reactions:
                i = index[key]
                blocks = []
                for side in ("reactant", "product"):
                    start, stop = src[f"{side}_offsets"][i:i + 2]
                    blocks.append(src[f"{side}_vectors"][start:stop])
                if any(len(block) != 1 for block in blocks):
                    raise ValueError("Expected one substrate and product per original nitrilase reaction")
                values = np.concatenate(blocks)
                if not np.isfinite(values).all():
                    raise ValueError("Invalid pretrained molecular vectors")
                rows.append(values)
                offsets.append(offsets[-1] + len(values))
            with h5py.File(output / f"{modality}.h5", "w") as dst:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                dst.attrs["reaction_input_policy"] = "participant_self_reaction"
                dst["ids"] = np.asarray(reactions, dtype=h5py.string_dtype())
                for side in ("reactant", "product"):
                    dst[f"{side}_offsets"] = np.asarray(offsets, np.int64)
                    dst[f"{side}_vectors"] = np.concatenate(rows)
    model_path = ROOT.parent / "hf_cache/hub/models--sagawa--ReactionT5v2-forward/snapshots/933114058cb2604dc1bf536dbebdfcefbe83d4fc"
    command = [sys.executable, "scripts/extract_reaction_t5v2_embeddings.py", "--reactions", str(reaction_path),
        "--output", str(output / "reactiont5v2.h5"), "--model-name", str(model_path), "--batch-size", "4",
        "--max-length", "512", "--pooling", "mean", "--device", args.device, "--dtype", "float16",
        "--no-bidirectional", "--no-allow-pseudo-reactions", "--force"]
    subprocess.run(command, cwd=ROOT, check=True)
    schema = ROOT / "runs/reactzyme_reaction_features_v1/data/reaction_smi/reaction_set/schema.json"
    cofactors = ROOT / "data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv"
    chemistry_path = output / "reaction_set_features.npz"
    materialize_reaction_set_features(reactions_path=reaction_path, schema_path=schema,
        cofactor_dictionary_path=cofactors, output_path=chemistry_path)
    raw = {}
    for modality, filename, molecular in (("t5v2", "reactiont5v2.h5", False), ("unimol2", "unimol2.h5", True), ("chiro", "chiro.h5", True)):
        raw[modality], raw[modality + "_mask"] = reaction_block(output / filename, reactions, molecular)
        if not raw[modality + "_mask"].all():
            raise ValueError(f"Missing {modality} features; panel cannot shrink")
    with np.load(chemistry_path, allow_pickle=True) as chemistry:
        index = {str(key): i for i, key in enumerate(chemistry["ids"])}
        order = [index[key] for key in reactions]
        raw["chemistry"], raw["chemistry_mask"] = chemistry["vectors"][order], chemistry["mask"][order]
    atomic_npz(output / "reaction_features.npz", **raw)
    torch.set_num_threads(4)
    if str(args.device).startswith("cuda"):
        cap = 9.5 * 1024**3 / torch.cuda.get_device_properties(args.device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(cap, 1.0), device=args.device)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    module, _ = load_repo_checkpoint(checkpoint, config, args.device)
    module.requires_grad_(False)
    residues = ResidueEmbedDataset(str(source / "proteins.h5"), in_memory=False, dtype=torch.float32,
        max_tokens=config.data.max_protein_tokens, truncation=config.data.protein_truncation)
    p = encode_residue_targets(module, residues, proteins, args.device, 4, False).float().cpu().numpy()
    config.data.reaction_chemistry_vectors_path = str(chemistry_path)
    for modality in ("unimol2", "chiro", "chemistry"):
        config.data[f"reaction_allow_missing_{modality}"] = False
    task = BenchmarkTask(name="nitrilase_f3_epoch29", task_type="screening", dataset="nitrilase", task_label="complete_38x18",
        split="external", pairs=reaction_path, reactions=reaction_path,
        reaction_model_embeds_h5=output / "reactiont5v2.h5", reaction_unimol2_embeds_h5=output / "unimol2.h5",
        reaction_chiro_embeds_h5=output / "chiro.h5", directions=("reaction_to_enzyme",))
    inputs = build_reaction_inputs(task, config)
    q = encode_reactions(module, inputs, reactions, args.device, 4).float().cpu().numpy()
    if p.shape != (18, 512) or q.shape != (38, 512) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Learned embeddings are incomplete or invalid")
    atomic_npz(output / "features.npz", proteins=p, reactions=q)
    manifest = dict(schema="nitrilase_reactzyme_f3_frozen_export_v1", activity_labels_used=False,
        catalog=identity(catalog_path, True), checkpoint=identity(checkpoint, True), config=identity(config_path, True),
        input_policy=policy, precision="float32_highest", chemistry_schema=identity(schema, True),
        extraction_command=command, proteins=18, reactions=38,
        sources={**{name: identity(source / name, True) for name in ("proteins.h5", "unimol2.h5", "chiro.h5")}, "reactions.csv": identity(original_reactions, True)},
        molecular_bank_policy="Reuse original substrate and carboxylic-acid vectors; original extractors skipped single-atom O and N. Full canonical participants remain in T5 and chemistry.",
        outputs={name: identity(output / name, True) for name in ("features.npz", "reaction_features.npz", "feature_reactions.csv")},
        script=identity(__file__, True))
    atomic_json(output / "complete.json", manifest)
    (output / "export_source.py").write_text(Path(__file__).read_text())
    print(json.dumps(dict(output=str(output), proteins=18, reactions=38)), flush=True)


if __name__ == "__main__":
    main()
