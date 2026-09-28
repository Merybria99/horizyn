"""
Utilities for SLEEC stage-1 functional residue classifier training.

The SLEEC paper's first stage trains a residue-level binary classifier from
precomputed PLM residue embeddings, supervised mCSA labels, and MSA-derived
pseudo-labels. This module keeps the reusable pieces independent of the CLI
training/preparation scripts so they can be unit-tested directly.
"""

from __future__ import annotations

import csv
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


CANONICAL_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
CANONICAL_AA_SET = set(CANONICAL_AMINO_ACIDS)
GAP_CHARS = {"-", "."}


@dataclass(frozen=True)
class FastaRecord:
    protein_id: str
    sequence: str


@dataclass(frozen=True)
class ResidueLabelRecord:
    protein_id: str
    residue_index: int
    label: int
    split: str = ""
    source: str = ""
    entropy: float | None = None


class SLEECStage1Classifier(nn.Module):
    """MLP used for SLEEC stage-1 residue classification."""

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        *,
        variant: str = "paper",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if variant not in {"paper", "regularized"}:
            raise ValueError("variant must be one of: paper, regularized")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.variant = variant
        self.dropout = float(dropout) if variant == "regularized" else 0.0
        self.layer_norm = bool(layer_norm) if variant == "regularized" else False

        layers: list[nn.Module] = []
        if self.layer_norm:
            layers.append(nn.LayerNorm(self.input_dim))
        layers.extend(
            [
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
            ]
        )
        if self.dropout > 0.0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape [num_residues, input_dim], "
                f"got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {embeddings.shape[-1]}")
        return self.net(embeddings).squeeze(-1)


def confidence_aware_stage1_loss(
    supervised_logits: torch.Tensor,
    supervised_labels: torch.Tensor,
    pseudo_logits: torch.Tensor | None = None,
    pseudo_labels: torch.Tensor | None = None,
    *,
    lambda_pseudo: float = 1.0,
    confidence_threshold: float = 0.9,
    supervised_pos_weight: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the SLEEC stage-1 objective.

    The supervised term is standard BCE-with-logits. The pseudo-label term is
    BCE-with-logits weighted by the paper's confidence indicator:
    ``1[max(p, 1-p) > tau]``.
    """
    supervised_labels = supervised_labels.to(dtype=supervised_logits.dtype)
    pos_weight_tensor = None
    if supervised_pos_weight is not None:
        if isinstance(supervised_pos_weight, torch.Tensor):
            pos_weight_tensor = supervised_pos_weight.to(
                device=supervised_logits.device,
                dtype=supervised_logits.dtype,
            )
        else:
            pos_weight_tensor = supervised_logits.new_tensor(float(supervised_pos_weight))
    supervised_loss = F.binary_cross_entropy_with_logits(
        supervised_logits,
        supervised_labels,
        pos_weight=pos_weight_tensor,
    )

    device = supervised_logits.device
    pseudo_loss = supervised_logits.new_tensor(0.0)
    pseudo_selected = supervised_logits.new_tensor(0.0)
    pseudo_coverage = supervised_logits.new_tensor(0.0)
    if pseudo_logits is not None and pseudo_labels is not None and pseudo_logits.numel() > 0:
        pseudo_labels = pseudo_labels.to(device=pseudo_logits.device, dtype=pseudo_logits.dtype)
        pseudo_bce = F.binary_cross_entropy_with_logits(
            pseudo_logits,
            pseudo_labels,
            reduction="none",
        )
        with torch.no_grad():
            pseudo_probs = torch.sigmoid(pseudo_logits)
            confidence = torch.maximum(pseudo_probs, 1.0 - pseudo_probs)
            weights = (confidence > confidence_threshold).to(dtype=pseudo_logits.dtype)
        pseudo_loss = (weights * pseudo_bce).mean()
        pseudo_selected = weights.sum()
        pseudo_coverage = weights.mean()

    total = supervised_loss + float(lambda_pseudo) * pseudo_loss
    return total, {
        "supervised_loss": supervised_loss.detach(),
        "pseudo_loss": pseudo_loss.detach().to(device),
        "pseudo_selected": pseudo_selected.detach().to(device),
        "pseudo_coverage": pseudo_coverage.detach().to(device),
    }


def clean_a3m_sequence(sequence: str) -> str:
    """Remove A3M lowercase insertion columns and whitespace."""
    return "".join(ch for ch in sequence.strip() if not ch.islower() and not ch.isspace())


def normalize_sequence(sequence: str) -> str:
    sequence = re.sub(r"\s+", "", sequence.upper())
    return re.sub(r"[UZOB]", "X", sequence)


def read_fasta(path: str | Path, *, a3m: bool = False) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    current_id: str | None = None
    chunks: list[str] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequence = "".join(chunks)
                    if a3m:
                        sequence = clean_a3m_sequence(sequence)
                    records.append(FastaRecord(current_id, normalize_sequence(sequence)))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        sequence = "".join(chunks)
        if a3m:
            sequence = clean_a3m_sequence(sequence)
        records.append(FastaRecord(current_id, normalize_sequence(sequence)))

    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def msa_column_entropies(
    aligned_records: Sequence[FastaRecord],
    *,
    query_id: str | None = None,
) -> list[float]:
    """
    Compute per-query-residue entropy from an aligned FASTA/A3M record set.

    Entropies are returned only for columns where the query has a residue. Gaps
    and non-canonical amino acids in homolog rows are ignored in the per-column
    amino-acid distribution.
    """
    if not aligned_records:
        raise ValueError("aligned_records must be non-empty")

    if query_id is None:
        query = aligned_records[0]
    else:
        matches = [record for record in aligned_records if record.protein_id == query_id]
        if not matches:
            raise KeyError(f"query_id not found in alignment: {query_id}")
        query = matches[0]

    alignment_length = len(query.sequence)
    for record in aligned_records:
        if len(record.sequence) != alignment_length:
            raise ValueError(
                "All aligned sequences must have equal length after A3M cleanup; "
                f"{record.protein_id} has {len(record.sequence)}, expected {alignment_length}"
            )

    entropies: list[float] = []
    for col_idx, query_aa in enumerate(query.sequence):
        if query_aa in GAP_CHARS:
            continue
        counts = {aa: 0 for aa in CANONICAL_AMINO_ACIDS}
        for record in aligned_records:
            aa = record.sequence[col_idx]
            if aa in CANONICAL_AA_SET:
                counts[aa] += 1
        total = sum(counts.values())
        if total == 0:
            entropies.append(math.log(len(CANONICAL_AMINO_ACIDS)))
            continue
        entropy = 0.0
        for count in counts.values():
            if count == 0:
                continue
            probability = count / total
            entropy -= probability * math.log(probability)
        entropies.append(entropy)
    return entropies


def low_entropy_labels(
    entropies: Sequence[float],
    *,
    positive_fraction: float = 0.10,
) -> list[int]:
    """Label the lowest-entropy residue positions as functional pseudo-positives."""
    if not (0.0 < positive_fraction <= 1.0):
        raise ValueError("positive_fraction must be in (0, 1]")
    if not entropies:
        return []
    positive_count = max(1, int(math.ceil(len(entropies) * positive_fraction)))
    ranked = sorted(range(len(entropies)), key=lambda idx: (float(entropies[idx]), idx))
    positives = set(ranked[:positive_count])
    return [1 if idx in positives else 0 for idx in range(len(entropies))]


def write_residue_label_records(
    records: Iterable[ResidueLabelRecord],
    path: str | Path,
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_id", "residue_index", "label", "split", "source", "entropy"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "protein_id": record.protein_id,
                    "residue_index": record.residue_index,
                    "label": int(record.label),
                    "split": record.split,
                    "source": record.source,
                    "entropy": "" if record.entropy is None else f"{record.entropy:.10g}",
                }
            )
            count += 1
    return count


def load_residue_label_records(
    path: str | Path,
    *,
    split: str | None = None,
    index_base: int = 0,
) -> list[ResidueLabelRecord]:
    path = Path(path)
    records: list[ResidueLabelRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"protein_id", "residue_index", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            row_split = row.get("split", "")
            if split is not None and row_split != split:
                continue
            entropy_text = row.get("entropy", "")
            records.append(
                ResidueLabelRecord(
                    protein_id=row["protein_id"],
                    residue_index=int(row["residue_index"]) - index_base,
                    label=int(float(row["label"])),
                    split=row_split,
                    source=row.get("source", ""),
                    entropy=float(entropy_text) if entropy_text not in {"", None} else None,
                )
            )
    return records


def balance_binary_records(
    records: Sequence[ResidueLabelRecord],
    *,
    seed: int = 42,
    negative_to_positive_ratio: float = 1.0,
) -> list[ResidueLabelRecord]:
    """Return all positives plus a sampled negative set at the requested ratio."""
    if negative_to_positive_ratio <= 0:
        raise ValueError("negative_to_positive_ratio must be positive")
    positives = [record for record in records if int(record.label) == 1]
    negatives = [record for record in records if int(record.label) == 0]
    if not positives:
        raise ValueError("Cannot balance records without positive examples")
    if not negatives:
        raise ValueError("Cannot balance records without negative examples")

    rng = random.Random(seed)
    negative_count = max(1, int(round(len(positives) * negative_to_positive_ratio)))
    if negative_count <= len(negatives):
        sampled_negatives = rng.sample(negatives, negative_count)
    else:
        sampled_negatives = [rng.choice(negatives) for _ in range(negative_count)]
    balanced = positives + sampled_negatives
    rng.shuffle(balanced)
    return balanced


class RaggedResidueEmbeddingStore:
    """Lazy HDF5 lookup for the repo's ragged residue embedding schema."""

    def __init__(self, file_path: str | Path, *, dtype: torch.dtype = torch.float32):
        file_path_obj = Path(file_path)
        if not file_path_obj.is_file():
            raise FileNotFoundError(f"Residue HDF5 file not found: {file_path}")
        self.file_path = str(file_path_obj)
        self.dtype = dtype
        self.file: h5py.File | None = None
        self._file_pid: int | None = None
        with h5py.File(self.file_path, "r") as h5_file:
            for dataset_name in ("ids", "vectors", "offsets"):
                if dataset_name not in h5_file:
                    raise KeyError(f"{self.file_path} missing dataset '{dataset_name}'")
            vectors_shape = h5_file["vectors"].shape
            if len(vectors_shape) != 2 or h5_file["vectors"].dtype.kind not in {"f", "i", "u"}:
                raise ValueError("Residue HDF5 'vectors' must be a numeric rank-2 dataset")
            if h5_file["offsets"].dtype.kind not in {"i", "u"}:
                raise ValueError("Residue HDF5 'offsets' must have an integer dtype")
            ids_data = h5_file["ids"][:]
            self.ids = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in ids_data
            ]
            if any(not protein_id.strip() for protein_id in self.ids):
                raise ValueError("Residue HDF5 'ids' must not contain empty strings")
            if len(self.ids) != len(set(self.ids)):
                raise ValueError("Residue HDF5 'ids' must be unique")
            self.offsets = torch.as_tensor(h5_file["offsets"][:], dtype=torch.long)
            if self.offsets.ndim != 1 or len(self.offsets) != len(self.ids) + 1:
                raise ValueError("Residue HDF5 'offsets' length must equal len(ids) + 1")
            if int(self.offsets[0].item()) != 0:
                raise ValueError("Residue HDF5 'offsets' must start at 0")
            if int(self.offsets[-1].item()) != vectors_shape[0]:
                raise ValueError("Residue HDF5 'offsets' must end at the vector row count")
            if not torch.all(self.offsets[1:] >= self.offsets[:-1]):
                raise ValueError("Residue HDF5 'offsets' must be monotonically non-decreasing")
            self.id_to_idx = {protein_id: idx for idx, protein_id in enumerate(self.ids)}
            self.embedding_dim = int(vectors_shape[1])

    def _ensure_file(self) -> h5py.File:
        current_pid = os.getpid()
        if self.file is not None and self._file_pid != current_pid:
            self.close()
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
            self._file_pid = current_pid
        return self.file

    def get(self, protein_id: str, residue_index: int) -> torch.Tensor:
        if protein_id not in self.id_to_idx:
            raise KeyError(f"Protein id not found in embeddings: {protein_id}")
        if residue_index < 0:
            raise IndexError("residue_index must be non-negative")
        protein_idx = self.id_to_idx[protein_id]
        start = int(self.offsets[protein_idx].item())
        end = int(self.offsets[protein_idx + 1].item())
        length = end - start
        if residue_index >= length:
            raise IndexError(
                f"Residue index {residue_index} out of range for {protein_id} length {length}"
            )
        h5_file = self._ensure_file()
        vector = torch.from_numpy(h5_file["vectors"][start + residue_index]).to(dtype=self.dtype)
        return vector

    def close(self) -> None:
        file_handle = getattr(self, "file", None)
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                pass
        self.file = None
        self._file_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["file"] = None
        state["_file_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ResidueLabelDataset(Dataset):
    """Map sequence-indexed residue labels onto precomputed residue embeddings."""

    def __init__(
        self,
        records: Sequence[ResidueLabelRecord],
        residue_h5_path: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        if not records:
            raise ValueError("records must be non-empty")
        self.records = list(records)
        self.store = RaggedResidueEmbeddingStore(residue_h5_path, dtype=dtype)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        return {
            "embedding": self.store.get(record.protein_id, record.residue_index),
            "label": torch.tensor(float(record.label), dtype=torch.float32),
            "protein_id": record.protein_id,
            "residue_index": torch.tensor(record.residue_index, dtype=torch.long),
        }


def residue_label_collate(batch: Sequence[dict[str, torch.Tensor | str]]) -> dict[str, object]:
    if not batch:
        return {}
    return {
        "embeddings": torch.stack([item["embedding"] for item in batch]),  # type: ignore[list-item]
        "labels": torch.stack([item["label"] for item in batch]),  # type: ignore[list-item]
        "protein_id": [str(item["protein_id"]) for item in batch],
        "residue_index": torch.stack([item["residue_index"] for item in batch]),  # type: ignore[list-item]
    }


def binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels_bool = labels.to(dtype=torch.bool)
    preds = torch.sigmoid(logits) >= threshold
    tp = int((preds & labels_bool).sum().item())
    fp = int((preds & ~labels_bool).sum().item())
    fn = int((~preds & labels_bool).sum().item())
    tn = int((~preds & ~labels_bool).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }
