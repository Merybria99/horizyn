#!/usr/bin/env python3
"""Export fresh EnzymeCAGE-trained F3 P450 scores with physical reactions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from horizyn.benchmarks.retrieval import (BenchmarkTask, build_query_inputs,
    build_reaction_inputs, cosine_scores, encode_residue_targets,
    load_repo_checkpoint)
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_epoch(metrics: Path) -> tuple[int, float]:
    with metrics.open(newline="") as handle:
        scored = [(int(row["epoch"]), float(row["val/mean_bidirectional_mrr"]))
                  for row in csv.DictReader(handle)
                  if row.get("val/mean_bidirectional_mrr")]
    if not scored:
        raise ValueError("No target validation results")
    return max(scored, key=lambda pair: pair[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path,
                        default=ROOT / "runs/enzymecage_p450_reproduction_20260918/f3_features")
    parser.add_argument("--catalog", type=Path,
                        default=ROOT / "runs/generalization_20260919_2251/p450_audit/features/catalog.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.run.resolve()
    metrics = run / "logs/protein_pooling_training/version_0/metrics.csv"
    config_path = run / "configs/train.yaml"
    training_log = run / "pipeline.log"
    if "PROTEIN-POOLING TRAINING COMPLETE" not in training_log.read_text():
        raise ValueError("Fresh target-base training has not completed")
    epoch, value = selected_epoch(metrics)
    checkpoint = run / f"checkpoints/protein-pooling-epoch={epoch:02d}.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best validation checkpoint missing: {checkpoint}")
    launch = json.loads((run / "launch.json").read_text())
    prep = json.loads((run / "preparation.json").read_text())
    if (sha256(config_path) != launch["config_sha256"]
            or prep["model_matches_lock"] is not True
            or sha256(config_path) != prep["prepared_config"]["sha256"]):
        raise ValueError("Fresh training config/architecture provenance changed")
    catalog = json.loads(args.catalog.read_text())
    proteins, reactions = catalog["proteins"], catalog["reactions"]
    if len(proteins) != 490 or len(reactions) != 191:
        raise ValueError("P450 candidate/query pool changed")
    feature = args.feature_root.resolve()
    with (feature / "reaction_mapping.csv").open(newline="") as handle:
        mapping = list(csv.DictReader(handle))
    if [row["reaction_id"] for row in mapping] != reactions:
        raise ValueError("P450 reaction order changed")
    config = load_config(str(config_path))
    if config.data.reaction_direction_mode != "forward_only":
        raise ValueError("Expected physical forward-only target input")
    config.data.reaction_chemistry_vectors_path = str(feature / "chemistry_f3.npz")
    for modality in ("unimol2", "chiro", "chemistry"):
        config.data[f"reaction_allow_missing_{modality}"] = False
    task = BenchmarkTask(name="matched_enzymecage_p450", task_type="retrieval",
        dataset="EnzymeCAGE_P450", task_label="within_family", split="external",
        pairs=feature / "encoding_pairs.csv", reactions=feature / "reactions.csv",
        reaction_model_embeds_h5=feature / "reactiont5.h5",
        reaction_unimol2_embeds_h5=feature / "unimol2.h5",
        reaction_chiro_embeds_h5=feature / "chiro.h5")
    data = build_reaction_inputs(task, config)
    if set(data.keys) != set(reactions):
        raise ValueError("P450 reaction feature coverage changed")
    device = args.device
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    model, kind = load_repo_checkpoint(checkpoint, config, device)
    if kind != "residue":
        raise ValueError("Unexpected model input kind")
    model.eval()
    residue = ResidueEmbedDataset(str(feature / "proteins.h5"),
        max_tokens=config.data.max_protein_tokens,
        truncation=config.data.protein_truncation)
    try:
        if set(residue.keys) != set(proteins):
            raise ValueError("P450 protein feature coverage changed")
        with torch.inference_mode():
            enzyme = encode_residue_targets(model, residue, proteins, device, 16, False)
            inputs = build_query_inputs(data, reactions, device)
            reaction = torch.cat([
                model.model.encode_queries({key: value[i:i + 8]
                                            for key, value in inputs.items()}).cpu()
                for i in range(0, len(reactions), 8)
            ])
            scores = cosine_scores(reaction.float(), enzyme.float().cpu()).numpy()
    finally:
        residue.close()
    if scores.shape != (191, 490) or not np.isfinite(scores).all():
        raise ValueError("Incomplete or nonfinite P450 score matrix")
    args.output.mkdir(parents=True)
    score_path = args.output / "scores.npz"
    np.savez_compressed(score_path, reaction_ids=np.asarray(reactions),
                        protein_ids=np.asarray(proteins), scores=scores.astype(np.float32))
    source_names = ("reaction_mapping.csv", "reactions.csv", "encoding_pairs.csv",
                    "proteins.h5", "reactiont5.h5", "unimol2.h5", "chiro.h5",
                    "chemistry_f3.npz")
    receipt = {
        "schema": "fresh_matched_enzymecage_f3_p450_score_export_v1",
        "run": str(run), "validation_epoch": epoch, "validation_mrr": value,
        "validation_metrics_sha256": sha256(metrics),
        "training_log_sha256": sha256(training_log),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(config_path),
        "catalog_sha256": sha256(args.catalog),
        "physical_input_features": {name: sha256(feature / name) for name in source_names},
        "score_matrix_sha256": sha256(score_path),
        "score_shape": list(scores.shape),
        "test_labels_read": False,
        "training_completion_verified_by_this_script": True,
        "source_code_sha256": sha256(Path(__file__)),
    }
    (args.output / "export.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"epoch": epoch, "validation_mrr": value,
                      "scores": str(score_path)}), flush=True)


if __name__ == "__main__":
    main()
