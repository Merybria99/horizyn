#!/usr/bin/env python3
"""Prepare an isolated cached ReactZyme run; select a measured Arginine batch size."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_reactzyme_tyrosine_recipe import BIO, rooted, safe_label_proteins, sha

SOURCE = ROOT / "runs/reactzyme_reaction_smi_tyrosine_recipe_glutamine_20260911_031941/configs/train_30epochs.yaml"
BATCHES = (400, 800, 1600, 3200)


def filtered_targets(payload, safe, train_ids):
    """Preserve legacy soft targets/confidences, masking unsafe aggregates entirely."""
    import numpy as np

    ids = np.asarray(payload["ids"], dtype=str)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate annotation IDs")
    keep = np.asarray([value in safe and value in train_ids for value in ids])
    result = {"ids": ids[keep]}
    coverage = {}
    for family, width in (("mechanism", 8), ("cofactor", 10)):
        for suffix in ("targets", "mask", "denominator", "confidence"):
            key = f"{family}_{suffix}"
            value = payload[key]
            if value.shape != (len(ids), width) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} shape/values")
            if np.any(value < 0) or (suffix in ("targets", "mask", "confidence") and np.any(value > 1)):
                raise ValueError(f"Invalid {key} range")
            if suffix == "mask" and np.any((value != 0) & (value != 1)):
                raise ValueError(f"Nonbinary {key}")
            result[key] = value[keep]
        valid = (result[f"{family}_mask"].astype(bool)
                 & (result[f"{family}_denominator"] > 0)
                 & (result[f"{family}_confidence"] > 0))
        coverage[family] = dict(proteins=int(valid.any(axis=1).sum()), cells=int(valid.sum()))
        if not valid.any():
            raise ValueError(f"No usable training annotations for {family}")
    return result, coverage


def run_config(source, output, batch):
    config = copy.deepcopy(source)
    data, training, logging = config["data"], config["training"], config["logging"]
    data.update(train_batch_size=batch, typed_negative_positive_fraction=0.5,
                typed_negative_biological_fraction=0.5, num_workers=2,
                prefetch_factor=1, persistent_workers=True, pin_memory=True,
                retrieval_batch_size=512, validation_retrieval_batch_size=512,
                protein_biofp_targets_path=str(output / "biofp_targets.npz"),
                protein_biofp_vocab_path=str(BIO / "enzyme_biofp_vocab.json"))
    training.update(devices=4, max_epochs=30, max_steps=-1, precision="bf16-mixed",
                    fused_adamw=True, contrastive_fp32=True, fp32_sensitive_modules=True,
                    float32_matmul_precision="high", cpu_num_threads=4,
                    ddp_gradient_as_bucket_view=True, accumulate_grad_batches=1,
                    validation_retrieval_batch_size=512, validation_interval_steps=None,
                    check_val_every_n_epoch=1)
    training["early_stopping"] = {"enabled": False}
    for key in ("init_from_checkpoint", "biofp_pretrain_checkpoint", "limit_train_batches"):
        training.pop(key, None)
    training["loss"].update(biofp_aux_weight=0.2,
                            biofp_family_weights={"mechanism": 0.5, "cofactor": 0.5},
                            biofp_confidence_cap=1.0)
    logging.update(log_dir=str(output / "configured_logs"),
                   checkpoint_dir=str(output / "configured_checkpoints"),
                   log_every_n_steps=10, recovery_every_n_train_steps=50)
    logging["wandb"] = {"enabled": False, "mode": "disabled"}
    config["ablation"] = dict(split="reaction_smi", variant="classification_02_50_50_h200",
                               effective_global_batch_size=4 * batch, feature_extraction=False)
    return config


def prepare(output):
    import numpy as np
    import yaml
    from horizyn.config import load_config

    source = yaml.safe_load(SOURCE.read_text())
    if source.get("ablation", {}).get("split") != "reaction_smi":
        raise ValueError("Expected the existing reaction_smi recipe")
    if (source["model"].get("enzyme_input_mode") != "raw_mean_sleec_biological_factorized"
            or source["model"]["biofp"]["family_dims"] != {"mechanism": 8, "cofactor": 10}):
        raise ValueError("Expected the CIRCE-v2 mechanism/cofactor classification heads")
    data = source["data"]
    index = rooted(data["indexed_pairs_dir"])
    manifest = json.loads((index / "manifest.json").read_text())
    for record in manifest["provenance"]["source_files"]:
        if sha(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Index source changed: {record['path']}")
    vocab = json.loads((BIO / "enzyme_biofp_vocab.json").read_text())
    if vocab["schema_version"] != "biofp_minimal_v1":
        raise ValueError("Unexpected BioFP schema")
    for role, path in vocab["sources"].items():
        if sha(rooted(path)) != vocab["source_sha256"][role]:
            raise ValueError(f"Annotation provenance changed: {role}")
    with rooted(data["train_pairs_path"]).open(newline="") as handle:
        edges = {(row["reaction_id"], row["protein_id"]) for row in csv.DictReader(handle)}
    qids = np.load(index / "query_ids.npy", allow_pickle=False)
    pids = np.load(index / "protein_ids.npy", allow_pickle=False)
    pairs = np.load(index / "pairs.npy", mmap_mode="r", allow_pickle=False)
    if edges != {(str(qids[q]), str(pids[p])) for q, p in pairs}:
        raise ValueError("Configured pairs differ from the train-only index")
    safe, unsafe = safe_label_proteins(rooted(vocab["sources"]["train_pairs"]),
                                     {(q.removesuffix("_f"), p) for q, p in edges})
    with np.load(BIO / "enzyme_biofp_soft_targets.npz", allow_pickle=True) as payload:
        targets, coverage = filtered_targets(payload, safe, set(pids))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "biofp_targets.npz", **targets)
    for batch in BATCHES:
        path = output / f"batch_{batch}.yaml"
        path.write_text(yaml.safe_dump(run_config(source, output, batch), sort_keys=False))
        load_config(path)  # Exercise the same configuration validator as training.
    report = dict(source_config=str(SOURCE), source_sha256=sha(SOURCE), train_pairs=len(pairs),
                  train_proteins=len(pids), safe_label_proteins=len(targets["ids"]),
                  excluded_source_aggregates=len(unsafe), coverage=coverage,
                  classification_weight=0.2, positive_fraction=0.5,
                  annotation_policy="Original per-label masks and weak-negative confidence retained; missing/unsafe aggregates masked, not negative classes",
                  cached_backbones=True, retrieval_tower_warm_start=False)
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def select_batch(output, memory_bytes):
    """Use the slowest rank's measured throughput; retain 15% VRAM headroom."""
    candidates = []
    for batch in BATCHES:
        paths = [output / f"pilot_{batch}/timing_rank{rank}.json" for rank in range(4)]
        if not all(path.is_file() for path in paths):
            continue
        records = [json.loads(path.read_text()) for path in paths]
        valid = all(row["complete"] and row["measured_steps"] >= 5
                    and row["world_size"] == 4 and row["rank"] == rank
                    and math.isfinite(row["mean_step_seconds"]) and row["mean_step_seconds"] > 0
                    and row["peak_cuda_allocated_bytes"] <= 0.85 * memory_bytes
                    and all(math.isfinite(row.get("classification", {}).get(key, 0))
                            and row.get("classification", {}).get(key, 0) > 0
                            for key in ("weighted_biofp", "biofp_mechanism_active", "biofp_cofactor_active"))
                    for rank, row in enumerate(records))
        if valid:
            candidates.append((4 * batch / max(row["mean_step_seconds"] for row in records), batch))
    if not candidates:
        raise ValueError("No pilot passed timing/memory checks; no full training launched")
    throughput, batch = max(candidates)
    (output / "train.yaml").write_bytes((output / f"batch_{batch}.yaml").read_bytes())
    report = dict(batch_per_gpu=batch, global_pair_rows=4 * batch,
                  positive_rows=2 * batch, negative_rows=2 * batch,
                  measured_pair_rows_per_second=throughput, candidates=candidates)
    (output / "batch_selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "select"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--memory-bytes", type=int, default=140 * 1024**3)
    args = parser.parse_args()
    output = args.run_root.resolve()
    if output == ROOT / "runs" or not output.is_relative_to(ROOT / "runs"):
        parser.error("Use a dedicated directory under runs/")
    if args.action == "prepare":
        prepare(output)
    else:
        select_batch(output, args.memory_bytes)
