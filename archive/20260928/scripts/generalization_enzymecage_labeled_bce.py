#!/usr/bin/env python3
"""Train fixed F3 residual towers with every EnzymeCAGE labeled row and BCE.

The target-trained F3 base stays frozen. Checkpoint selection uses the
released EnzymeCAGE validation labels' global AUROC, as in its train.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

from horizyn.generalization_residual import FrozenGeometryResidual


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


class LabeledResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.towers = FrozenGeometryResidual(512, 1024, 0.2)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-3.0))

    def forward(self, reaction: torch.Tensor, enzyme: torch.Tensor) -> torch.Tensor:
        q = self.towers.encode_reactions(reaction)
        p = self.towers.encode_enzymes(enzyme)
        cosine = (q * p).sum(-1)
        return self.logit_scale.exp().clamp(max=100) * cosine + self.logit_bias


def validate(model: LabeledResidual, reaction: torch.Tensor, enzyme: torch.Tensor,
             rows: torch.Tensor, batch_size: int) -> dict:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            block = rows[start:start + batch_size]
            output.append(model(reaction[block[:, 0]], enzyme[block[:, 1]]).cpu().numpy())
    scores = np.concatenate(output)
    labels = rows[:, 2].cpu().numpy()
    return {"auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "rows": len(rows), "positives": int(labels.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    source = json.loads((args.features / "receipt.json").read_text())
    if (source["schema"] != "enzymecage_labeled_f3_features_v1"
            or source["external_labels_or_scores_read"] is not False):
        raise ValueError("Training feature provenance changed")
    for name in ("base.npz", "labeled_pairs.npz"):
        if sha256(args.features / name) != source["outputs"][name]:
            raise ValueError(f"Labeled training feature changed: {name}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    with np.load(args.features / "base.npz", allow_pickle=False) as data:
        enzyme = torch.as_tensor(data["enzymes"], device=args.device)
        reaction = torch.as_tensor(data["reactions"], device=args.device)
    with np.load(args.features / "labeled_pairs.npz", allow_pickle=False) as data:
        train = torch.as_tensor(data["train"], device=args.device, dtype=torch.long)
        valid = torch.as_tensor(data["validation"], device=args.device, dtype=torch.long)
    if (len(train) != source["training_rows"] or len(valid) != source["validation_rows"]
            or not set(torch.unique(train[:, 2]).tolist()) <= {0, 1}
            or not set(torch.unique(valid[:, 2]).tolist()) <= {0, 1}):
        raise ValueError("Full labeled split changed")
    model = LabeledResidual().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    args.output.mkdir(parents=True)
    registry = {
        "schema": "enzymecage_labeled_bce_frozen_f3_residual_v1",
        "features_receipt_sha256": sha256(args.features / "receipt.json"),
        "base_features_sha256": source["outputs"]["base.npz"],
        "labeled_pairs_sha256": source["outputs"]["labeled_pairs.npz"],
        "rows_train": len(train), "rows_validation": len(valid),
        "loss": "unweighted BCEWithLogitsLoss on every labeled row, including duplicates and conflicts",
        "checkpoint_selection": "maximum original EnzymeCAGE validation AUROC",
        "base_f3_frozen": True,
        "f3_base_pretrained_on_positive_subset": True,
        "external_scores_or_labels_used": False,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "lr_decay_per_epoch": 0.95,
        "seed": args.seed, "source_sha256": sha256(Path(__file__)),
    }
    (args.output / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    records = []
    best_auc = -math.inf
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train), device=args.device)
        loss_sum = 0.0
        for start in range(0, len(train), args.batch_size):
            rows = train[order[start:start + args.batch_size]]
            logits = model(reaction[rows[:, 0]], enzyme[rows[:, 1]])
            loss = F.binary_cross_entropy_with_logits(logits, rows[:, 2].float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(rows)
        result = validate(model, reaction, enzyme, valid, args.batch_size)
        record = {"epoch": epoch, "train_bce": loss_sum / len(train),
                  "validation": result, "elapsed_seconds": time.monotonic() - started}
        records.append(record)
        if result["auroc"] > best_auc:
            best_auc = result["auroc"]
            torch.save({"state_dict": model.state_dict(), "registry": registry,
                        "selected_validation": record}, args.output / "selected.pt")
        (args.output / "history.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps({"epoch": epoch, "train_bce": record["train_bce"],
                          "validation_auroc": result["auroc"],
                          "validation_ap": result["average_precision"]}), flush=True)
        scheduler.step()
    (args.output / "complete.json").write_text(json.dumps(
        {"selected_epoch": max(records, key=lambda row: row["validation"]["auroc"])["epoch"],
         "selected_validation_auroc": best_auc, "elapsed_seconds": time.monotonic() - started,
         "external_scores_or_labels_used": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
