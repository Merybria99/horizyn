#!/usr/bin/env python3
"""
Train the SLEEC stage-1 functional residue classifier.

This is the stage-1 classifier from Section 3 of the SLEEC paper, not the
Horizyn reaction/protein retrieval model. It expects:
  - pre-generated residue embeddings in the repo's ragged HDF5 schema
    (ESM2, ProtT5, ESMC, or another compatible residue encoder);
  - mCSA sequence-indexed residue labels with train/val splits;
  - optional MSA pseudo-labels prepared from Ligns A3Ms.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
import h5py
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from horizyn.config import DotDict, apply_overrides, parse_overrides  # noqa: E402
from horizyn.sleec_stage1 import (  # noqa: E402
    ResidueLabelDataset,
    SLEECStage1Classifier,
    balance_binary_records,
    binary_classification_metrics,
    confidence_aware_stage1_loss,
    load_residue_label_records,
    residue_label_collate,
)
from horizyn.wandb_utils import resolve_wandb_settings, to_plain_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--export-val-predictions", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=None, help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="W&B logging mode",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional W&B tags")
    parser.add_argument("--wandb-log-model", action="store_true", help="Upload best/last checkpoints")
    return parser.parse_known_args()


def load_config(path: Path, overrides: dict[str, Any]) -> DotDict:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    config = DotDict(raw)
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is None or device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def infer_residue_embedding_dim(residue_h5_path: str | Path) -> int:
    with h5py.File(residue_h5_path, "r") as h5_file:
        if "vectors" not in h5_file:
            raise KeyError(f"{residue_h5_path} missing dataset 'vectors'")
        if h5_file["vectors"].ndim != 2:
            raise ValueError(
                f"{residue_h5_path} vectors must have shape [num_residues, dim], "
                f"got {h5_file['vectors'].shape}"
            )
        return int(h5_file["vectors"].shape[1])


def json_safe_hdf5_attr(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def read_residue_embedding_metadata(residue_h5_path: str | Path) -> dict[str, Any]:
    with h5py.File(residue_h5_path, "r") as h5_file:
        metadata: dict[str, Any] = {
            "residue_embedding_dim": int(h5_file["vectors"].shape[1])
            if "vectors" in h5_file
            else None
        }
        for key in (
            "embedding_model_type",
            "embedding_backend",
            "model_name",
            "hidden_layer",
            "residue_dim",
            "max_sequence_length",
        ):
            if key in h5_file.attrs:
                metadata[key] = json_safe_hdf5_attr(h5_file.attrs[key])
        return metadata


def resolve_model_input_dim(config: DotDict, residue_h5_path: str | Path) -> int:
    configured = config.model.get("input_dim", "auto")
    if configured in {None, "", "auto", "Auto", "AUTO"}:
        inferred = infer_residue_embedding_dim(residue_h5_path)
        config.model.input_dim = inferred
        print(f"Inferred model.input_dim={inferred} from {residue_h5_path}", flush=True)
        return inferred
    return int(configured)


def parse_device_ids(value: Any) -> list[int] | None:
    if value in {None, "", "none", "None", "null", "Null"}:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        ids = [item.strip() for item in value.split(",") if item.strip()]
        return [int(item) for item in ids]
    return [int(item) for item in value]


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def load_model_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    target = unwrap_model(model)
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    target.load_state_dict(state_dict)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def maybe_wrap_data_parallel(
    model: SLEECStage1Classifier,
    config: DotDict,
    *,
    device: torch.device,
) -> nn.Module:
    enabled = bool(config.training.get("data_parallel", False))
    if not enabled:
        return model
    if device.type != "cuda":
        raise ValueError("training.data_parallel=true requires a CUDA device")
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count < 2:
        raise ValueError(
            "training.data_parallel=true requires at least 2 visible CUDA devices; "
            f"found {visible_gpu_count}"
        )
    device_ids = parse_device_ids(config.training.get("device_ids", None))
    if device_ids is None:
        device_ids = list(range(visible_gpu_count))
    invalid_ids = [device_id for device_id in device_ids if device_id < 0 or device_id >= visible_gpu_count]
    if invalid_ids:
        raise ValueError(
            f"Invalid CUDA device_ids {invalid_ids}; visible device count is {visible_gpu_count}"
        )
    print(f"Using DataParallel across CUDA devices: {device_ids}")
    return nn.DataParallel(model, device_ids=device_ids)


def validate_supervised_counts(records: list, *, strict: bool = False) -> None:
    train_proteins = {record.protein_id for record in records if record.split == "train"}
    val_proteins = {record.protein_id for record in records if record.split in {"val", "valid"}}
    total_proteins = len(train_proteins | val_proteins)
    positives = sum(1 for record in records if record.label == 1)
    message = (
        "mCSA labels loaded: "
        f"{len(train_proteins)} train proteins, {len(val_proteins)} val proteins, "
        f"{total_proteins} total proteins, {positives} positives"
    )
    print(message)
    expected = len(train_proteins) == 403 and len(val_proteins) == 417 and positives == 3716
    if strict and not expected:
        warning = (
            "Loaded mCSA counts do not exactly match the paper's 403/417 proteins "
            "and 3716 functional residues. Use a paper-exact split/curation CSV if needed."
        )
        raise ValueError(warning)


def resolve_supervised_pos_weight(value: Any, records: list) -> float | None:
    if value in {None, False, "", "none", "None", "null", "Null"}:
        return None
    positives = sum(1 for record in records if int(record.label) == 1)
    negatives = len(records) - positives
    if positives <= 0:
        raise ValueError("Cannot use supervised positive weighting without positive labels")
    if isinstance(value, str):
        if value.lower() != "auto":
            raise ValueError(
                "training.supervised_pos_weight must be null, a number, or 'auto'"
            )
        return negatives / positives
    weight = float(value)
    if weight <= 0:
        raise ValueError("training.supervised_pos_weight must be positive")
    return weight


def resolve_optional_int(value: Any, *, name: str) -> int | None:
    if value in {None, False, "", "none", "None", "null", "Null"}:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive when set")
    return resolved


def resolve_float_list(value: Any, *, name: str) -> list[float]:
    if value in {None, False, "", "none", "None", "null", "Null"}:
        return []
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        parts = [part.strip() for part in value.split(",") if part.strip()]
        resolved = [float(part) for part in parts]
    elif isinstance(value, (list, tuple)):
        resolved = [float(item) for item in value]
    else:
        resolved = [float(value)]
    invalid = [item for item in resolved if item <= 0.0 or item >= 1.0]
    if invalid:
        raise ValueError(f"{name} values must be in (0, 1); got {invalid}")
    return sorted(set(resolved))


def threshold_key(threshold: float) -> str:
    return f"{threshold:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def build_dataloader(
    records: list,
    residue_h5_path: str,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = ResidueLabelDataset(records, residue_h5_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=residue_label_collate,
        drop_last=False,
    )


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = batch["embeddings"].to(device=device, dtype=torch.float32, non_blocking=True)
    labels = batch["labels"].to(device=device, dtype=torch.float32, non_blocking=True)
    return embeddings, labels


def init_wandb_run(
    args: argparse.Namespace,
    config: DotDict,
    *,
    output_dir: Path,
    model: SLEECStage1Classifier,
    train_records: list,
    val_records: list,
    pseudo_records: list,
) -> Any | None:
    settings = resolve_wandb_settings(args, config)
    if not settings["enabled"]:
        print(f"W&B logging disabled ({settings['mode']})")
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested but wandb is not installed. "
            "Install it with: uv pip install wandb"
        ) from exc

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    run = wandb.init(
        project=settings["project"],
        entity=settings["entity"],
        name=settings["run_name"],
        mode=settings["mode"],
        tags=settings["tags"],
        dir=str(output_dir),
        config={
            "config_path": str(args.config),
            "output_dir": str(output_dir),
            "config": to_plain_config(config),
            "wandb": settings,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "train_records": len(train_records),
            "val_records": len(val_records),
            "pseudo_records": len(pseudo_records),
            "train_positives": sum(1 for record in train_records if record.label == 1),
            "val_positives": sum(1 for record in val_records if record.label == 1),
            "pseudo_positives": sum(1 for record in pseudo_records if record.label == 1),
        },
    )
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("val/*", step_metric="step")
    wandb.define_metric("best/*", step_metric="step")
    print(f"W&B logging enabled: project={settings['project']} mode={settings['mode']}")
    return run


@torch.no_grad()
def evaluate(
    model: SLEECStage1Classifier,
    loader: DataLoader,
    device: torch.device,
    *,
    thresholds: list[float] | None = None,
    balanced_max_gap: float = 0.1,
    balanced_min_precision: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        embeddings, labels = batch_to_device(batch, device)
        logits = model(embeddings)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        total_loss += float(loss.item()) * labels.numel()
        total_count += int(labels.numel())
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())
    if not logits_list:
        return {"loss": float("nan"), "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    logits_all = torch.cat(logits_list)
    labels_all = torch.cat(labels_list)
    metrics = binary_classification_metrics(logits_all, labels_all)
    metrics["loss"] = total_loss / max(total_count, 1)
    if thresholds:
        sweep_rows: list[dict[str, float]] = []
        for threshold in thresholds:
            row = binary_classification_metrics(logits_all, labels_all, threshold=threshold)
            row["threshold"] = float(threshold)
            row["predicted"] = float(row["tp"] + row["fp"])
            row["precision_recall_gap"] = abs(float(row["precision"]) - float(row["recall"]))
            sweep_rows.append(row)
        best = max(sweep_rows, key=lambda row: float(row["f1"]))
        balanced_candidates = [
            row
            for row in sweep_rows
            if float(row["precision_recall_gap"]) <= balanced_max_gap
            and float(row["precision"]) >= balanced_min_precision
        ]
        balanced = (
            max(balanced_candidates, key=lambda row: float(row["f1"]))
            if balanced_candidates
            else min(sweep_rows, key=lambda row: (float(row["precision_recall_gap"]), -float(row["f1"])))
        )
        equal_pr = min(
            sweep_rows,
            key=lambda row: (float(row["precision_recall_gap"]), -float(row["f1"])),
        )
        metrics["threshold_sweep"] = sweep_rows
        for prefix, row in (
            ("threshold_best", best),
            ("threshold_balanced", balanced),
            ("threshold_equal_pr", equal_pr),
        ):
            metrics[f"{prefix}_threshold"] = float(row["threshold"])
            metrics[f"{prefix}_f1"] = float(row["f1"])
            metrics[f"{prefix}_precision"] = float(row["precision"])
            metrics[f"{prefix}_recall"] = float(row["recall"])
            metrics[f"{prefix}_gap"] = float(row["precision_recall_gap"])
            metrics[f"{prefix}_predicted"] = float(row["predicted"])
    return metrics


def log_wandb(run: Any | None, payload: dict[str, float], *, step: int) -> None:
    if run is None:
        return
    run.log({"step": step, **payload}, step=step)


def append_threshold_sweep(path: Path, *, step: int, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    import csv

    fieldnames = [
        "step",
        "threshold",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "predicted",
        "precision_recall_gap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({"step": step, **{key: row.get(key, "") for key in fieldnames if key != "step"}})


def maybe_log_model_artifact(
    run: Any | None,
    checkpoint_path: Path,
    *,
    artifact_name: str,
    artifact_type: str = "model",
) -> None:
    if run is None or not checkpoint_path.exists():
        return
    import wandb

    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(str(checkpoint_path))
    run.log_artifact(artifact)


@torch.no_grad()
def export_predictions(
    model: SLEECStage1Classifier,
    loader: DataLoader,
    device: torch.device,
    path: Path,
) -> None:
    import csv

    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_id", "residue_index", "label", "logit", "probability"],
        )
        writer.writeheader()
        for batch in loader:
            embeddings, labels = batch_to_device(batch, device)
            logits = model(embeddings)
            probs = torch.sigmoid(logits)
            residue_indices = batch["residue_index"].detach().cpu().tolist()
            labels_cpu = labels.detach().cpu().tolist()
            for protein_id, residue_index, label, logit, prob in zip(
                batch["protein_id"],
                residue_indices,
                labels_cpu,
                logits.detach().cpu().tolist(),
                probs.detach().cpu().tolist(),
            ):
                writer.writerow(
                    {
                        "protein_id": protein_id,
                        "residue_index": residue_index,
                        "label": int(label),
                        "logit": f"{float(logit):.10g}",
                        "probability": f"{float(prob):.10g}",
                    }
                )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DotDict,
    step: int,
    best_val_f1: float,
    best_metric_name: str = "f1",
    best_metric_value: float | None = None,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "unwrapped_model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(config),
            "step": step,
            "best_val_f1": best_val_f1,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_val_f1 if best_metric_value is None else best_metric_value,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args, unknown = parse_args()
    overrides = parse_overrides(unknown)
    config = load_config(args.config, overrides)
    if args.seed is not None:
        config.seed = args.seed
    output_dir = args.output_dir or Path(config.logging.get("output_dir", "checkpoints/SLEEC/sleec_stage1"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(args.device or config.training.get("device", "auto"))

    residue_h5_path = config.data.residue_embeds_path
    embedding_metadata = read_residue_embedding_metadata(residue_h5_path)
    model_input_dim = resolve_model_input_dim(config, residue_h5_path)
    supervised_records = load_residue_label_records(config.data.mcsa_labels_path)
    validate_supervised_counts(
        supervised_records,
        strict=bool(config.data.get("strict_paper_mcsa_counts", False)),
    )
    train_records = [record for record in supervised_records if record.split == "train"]
    val_records = [
        record for record in supervised_records if record.split in {"val", "valid", "validation"}
    ]
    if not train_records:
        raise ValueError("No supervised mCSA train records found")
    if not val_records:
        raise ValueError("No supervised mCSA validation records found")

    pseudo_records = []
    pseudo_path = config.data.get("pseudo_labels_path", None)
    if pseudo_path:
        pseudo_records = load_residue_label_records(pseudo_path)
        pseudo_balanced = False
        if config.data.get("balance_pseudo_negatives", True):
            pseudo_records = balance_binary_records(
                pseudo_records,
                seed=int(config.get("seed", 42)),
                negative_to_positive_ratio=float(config.data.get("pseudo_negative_ratio", 1.0)),
            )
            pseudo_balanced = True
        print(
            "MSA pseudo labels loaded: "
            f"{len(pseudo_records)} records, "
            f"{sum(1 for record in pseudo_records if record.label == 1)} positives"
            f"{' (balanced in memory)' if pseudo_balanced else ''}"
        )
    else:
        print("No pseudo-label CSV configured; training supervised-only.")

    num_workers = int(config.data.get("num_workers", 0))
    pin_memory = bool(config.data.get("pin_memory", device.type == "cuda"))
    supervised_loader = build_dataloader(
        train_records,
        residue_h5_path,
        batch_size=int(config.training.get("supervised_batch_size", 8192)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_dataloader(
        val_records,
        residue_h5_path,
        batch_size=int(config.training.get("eval_batch_size", 8192)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    pseudo_loader = None
    if pseudo_records:
        pseudo_loader = build_dataloader(
            pseudo_records,
            residue_h5_path,
            batch_size=int(config.training.get("pseudo_batch_size", 8192)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    model = SLEECStage1Classifier(
        input_dim=model_input_dim,
        hidden_dim=int(config.model.get("hidden_dim", 256)),
        variant=str(config.model.get("variant", "paper")),
        dropout=float(config.model.get("dropout", 0.0)),
        layer_norm=bool(config.model.get("layer_norm", False)),
    ).to(device)
    print(
        "Stage-1 model: "
        f"variant={model.variant}, input_dim={model.input_dim}, hidden_dim={model.hidden_dim}, "
        f"dropout={model.dropout}, layer_norm={model.layer_norm}"
    )

    start_step = 0
    best_val_f1 = -1.0
    best_metric_value: float | None = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_model_state(model, checkpoint["model_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_val_f1 = float(checkpoint.get("best_val_f1", best_val_f1))
        if checkpoint.get("best_metric_value") is not None:
            best_metric_value = float(checkpoint["best_metric_value"])
        print(f"Resumed from {args.resume} at step {start_step}")

    model = maybe_wrap_data_parallel(model, config, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.get("learning_rate", 1e-5)),
        weight_decay=float(config.training.get("weight_decay", 0.01)),
    )
    if args.resume is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)

    wandb_run = init_wandb_run(
        args,
        config,
        output_dir=output_dir,
        model=model,
        train_records=train_records,
        val_records=val_records,
        pseudo_records=pseudo_records,
    )
    wandb_settings = resolve_wandb_settings(args, config)

    max_steps = int(config.training.get("max_steps", 10000))
    eval_every = int(config.training.get("eval_every_steps", 500))
    save_every = int(config.training.get("save_every_steps", 1000))
    lambda_pseudo = float(config.training.get("lambda_pseudo", 1.0))
    confidence_threshold = float(config.training.get("confidence_threshold", 0.9))
    supervised_pos_weight = resolve_supervised_pos_weight(
        config.training.get("supervised_pos_weight", None),
        train_records,
    )
    if supervised_pos_weight is not None:
        print(f"Supervised positive class weight: {supervised_pos_weight:.6g}")
    early_stopping_patience = resolve_optional_int(
        config.training.get("early_stopping_patience_evals", None),
        name="training.early_stopping_patience_evals",
    )
    metric_for_best = str(config.training.get("metric_for_best", "f1"))
    lower_is_better = metric_for_best == "loss" or metric_for_best.endswith("_loss")
    if best_metric_value is None:
        best_metric_value = float("inf") if lower_is_better else -1.0
    threshold_sweep = resolve_float_list(
        config.training.get("threshold_sweep", None),
        name="training.threshold_sweep",
    )
    threshold_sweep_balance_max_gap = float(
        config.training.get("threshold_sweep_balance_max_gap", 0.1)
    )
    threshold_sweep_min_precision = float(
        config.training.get("threshold_sweep_min_precision", 0.5)
    )
    if threshold_sweep:
        print(
            "Validation threshold sweep enabled: "
            f"{','.join(f'{threshold:.4g}' for threshold in threshold_sweep)}; "
            f"balanced=max_f1 with precision>= {threshold_sweep_min_precision:.3g} "
            f"and |precision-recall|<= {threshold_sweep_balance_max_gap:.3g}"
        )

    manifest = {
        "created_unix_time": time.time(),
        "config_path": str(args.config),
        "output_dir": str(output_dir),
        "device": str(device),
        "data_parallel": isinstance(model, nn.DataParallel),
        "device_ids": parse_device_ids(config.training.get("device_ids", None)),
        "paper_stage": "SLEEC stage 1",
        "optimizer": "AdamW",
        "model_variant": str(config.model.get("variant", "paper")),
        "residue_embeds_path": str(residue_h5_path),
        "residue_embedding_metadata": embedding_metadata,
        "model_dropout": float(config.model.get("dropout", 0.0)),
        "model_layer_norm": bool(config.model.get("layer_norm", False)),
        "learning_rate": float(config.training.get("learning_rate", 1e-5)),
        "weight_decay": float(config.training.get("weight_decay", 0.01)),
        "max_steps": max_steps,
        "supervised_batch_size": int(config.training.get("supervised_batch_size", 8192)),
        "pseudo_batch_size": int(config.training.get("pseudo_batch_size", 8192)),
        "confidence_threshold": confidence_threshold,
        "lambda_pseudo": lambda_pseudo,
        "supervised_pos_weight": supervised_pos_weight,
        "early_stopping_patience_evals": early_stopping_patience,
        "metric_for_best": metric_for_best,
        "threshold_sweep": threshold_sweep,
        "threshold_sweep_balance_max_gap": threshold_sweep_balance_max_gap,
        "threshold_sweep_min_precision": threshold_sweep_min_precision,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    supervised_iter = cycle(supervised_loader)
    pseudo_iter = cycle(pseudo_loader) if pseudo_loader is not None else None
    evals_without_improvement = 0
    for step in range(start_step + 1, max_steps + 1):
        model.train()
        supervised_batch = next(supervised_iter)
        supervised_embeddings, supervised_labels = batch_to_device(supervised_batch, device)
        supervised_logits = model(supervised_embeddings)

        pseudo_logits = None
        pseudo_labels = None
        if pseudo_iter is not None:
            pseudo_batch = next(pseudo_iter)
            pseudo_embeddings, pseudo_labels = batch_to_device(pseudo_batch, device)
            pseudo_logits = model(pseudo_embeddings)

        loss, components = confidence_aware_stage1_loss(
            supervised_logits,
            supervised_labels,
            pseudo_logits,
            pseudo_labels,
            lambda_pseudo=lambda_pseudo,
            confidence_threshold=confidence_threshold,
            supervised_pos_weight=supervised_pos_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % int(config.logging.get("log_every_steps", 50)) == 0:
            print(
                f"step={step} loss={float(loss.item()):.6f} "
                f"supervised={float(components['supervised_loss']):.6f} "
                f"pseudo={float(components['pseudo_loss']):.6f} "
                f"pseudo_coverage={float(components['pseudo_coverage']):.4f}",
                flush=True,
            )
            log_wandb(
                wandb_run,
                {
                    "train/loss": float(loss.item()),
                    "train/supervised_loss": float(components["supervised_loss"]),
                    "train/pseudo_loss": float(components["pseudo_loss"]),
                    "train/pseudo_coverage": float(components["pseudo_coverage"]),
                    "train/pseudo_selected": float(components["pseudo_selected"]),
                    "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                },
                step=step,
            )

        should_eval = step == max_steps or (eval_every > 0 and step % eval_every == 0)
        should_stop = False
        if should_eval:
            metrics = evaluate(
                model,
                val_loader,
                device,
                thresholds=threshold_sweep,
                balanced_max_gap=threshold_sweep_balance_max_gap,
                balanced_min_precision=threshold_sweep_min_precision,
            )
            if metric_for_best not in metrics:
                raise ValueError(
                    f"metric_for_best='{metric_for_best}' is not available; "
                    f"choose one of {sorted(metrics)}"
                )
            print(
                f"val step={step} loss={metrics['loss']:.6f} "
                f"f1={metrics['f1']:.6f} precision={metrics['precision']:.6f} "
                f"recall={metrics['recall']:.6f}",
                flush=True,
            )
            val_payload = {
                "val/loss": float(metrics["loss"]),
                "val/f1": float(metrics["f1"]),
                "val/precision": float(metrics["precision"]),
                "val/recall": float(metrics["recall"]),
                "val/accuracy": float(metrics["accuracy"]),
                "val/tp": float(metrics["tp"]),
                "val/fp": float(metrics["fp"]),
                "val/fn": float(metrics["fn"]),
                "val/tn": float(metrics["tn"]),
            }
            if threshold_sweep:
                print(
                    "val_thresholds "
                    f"step={step} "
                    f"best_thr={metrics['threshold_best_threshold']:.4f} "
                    f"best_f1={metrics['threshold_best_f1']:.6f} "
                    f"best_p={metrics['threshold_best_precision']:.6f} "
                    f"best_r={metrics['threshold_best_recall']:.6f} "
                    f"balanced_thr={metrics['threshold_balanced_threshold']:.4f} "
                    f"balanced_f1={metrics['threshold_balanced_f1']:.6f} "
                    f"balanced_p={metrics['threshold_balanced_precision']:.6f} "
                    f"balanced_r={metrics['threshold_balanced_recall']:.6f} "
                    f"equal_pr_thr={metrics['threshold_equal_pr_threshold']:.4f} "
                    f"equal_pr_f1={metrics['threshold_equal_pr_f1']:.6f}",
                    flush=True,
                )
                append_threshold_sweep(
                    output_dir / "threshold_sweep.csv",
                    step=step,
                    rows=metrics["threshold_sweep"],
                )
                for prefix in ("threshold_best", "threshold_balanced", "threshold_equal_pr"):
                    val_payload[f"val_sweep/{prefix}_threshold"] = float(
                        metrics[f"{prefix}_threshold"]
                    )
                    val_payload[f"val_sweep/{prefix}_f1"] = float(metrics[f"{prefix}_f1"])
                    val_payload[f"val_sweep/{prefix}_precision"] = float(
                        metrics[f"{prefix}_precision"]
                    )
                    val_payload[f"val_sweep/{prefix}_recall"] = float(
                        metrics[f"{prefix}_recall"]
                    )
                    val_payload[f"val_sweep/{prefix}_gap"] = float(metrics[f"{prefix}_gap"])
                for row in metrics["threshold_sweep"]:
                    key = threshold_key(float(row["threshold"]))
                    val_payload[f"val_threshold/f1_t{key}"] = float(row["f1"])
                    val_payload[f"val_threshold/precision_t{key}"] = float(row["precision"])
                    val_payload[f"val_threshold/recall_t{key}"] = float(row["recall"])
            log_wandb(wandb_run, val_payload, step=step)
            current_metric = float(metrics[metric_for_best])
            improved = (
                current_metric < best_metric_value
                if lower_is_better
                else current_metric > best_metric_value
            )
            if improved:
                best_metric_value = current_metric
                best_val_f1 = float(metrics["f1"])
                evals_without_improvement = 0
                save_checkpoint(
                    output_dir / "best.ckpt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    best_val_f1=best_val_f1,
                    best_metric_name=metric_for_best,
                    best_metric_value=best_metric_value,
                    metrics=metrics,
                )
                log_wandb(
                    wandb_run,
                    {
                        "best/f1": best_val_f1,
                        f"best/{metric_for_best}": best_metric_value,
                    },
                    step=step,
                )
            else:
                evals_without_improvement += 1
            if (
                early_stopping_patience is not None
                and evals_without_improvement >= early_stopping_patience
            ):
                print(
                    "Early stopping: "
                    f"no {metric_for_best} improvement for "
                    f"{evals_without_improvement} eval(s)",
                    flush=True,
                )
                should_stop = True

        if step == max_steps or should_stop or (save_every > 0 and step % save_every == 0):
            metrics = {"best_val_f1": best_val_f1}
            save_checkpoint(
                output_dir / "last.ckpt",
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                best_val_f1=best_val_f1,
                best_metric_name=metric_for_best,
                best_metric_value=best_metric_value,
                metrics=metrics,
            )
        if should_stop:
            break

    if args.export_val_predictions:
        export_predictions(model, val_loader, device, output_dir / "val_predictions.csv")

    if wandb_run is not None:
        if wandb_settings.get("log_model", False):
            artifact_base = (
                wandb_run.name
                or str(config.model.get("variant", "paper"))
                or "sleec-stage1"
            )
            maybe_log_model_artifact(
                wandb_run,
                output_dir / "best.ckpt",
                artifact_name=f"{artifact_base}-best",
            )
            maybe_log_model_artifact(
                wandb_run,
                output_dir / "last.ckpt",
                artifact_name=f"{artifact_base}-last",
            )
        wandb_run.finish()


if __name__ == "__main__":
    main()
