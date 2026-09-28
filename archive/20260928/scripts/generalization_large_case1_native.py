"""Frozen native F3/CIRCE projections sharing one already-loaded residue block.

This helper performs no file read of protein inputs, scoring, sampling or fitting.
The caller owns HDF5 reads and ALL-stored-row raw means. Native inputs use the
checkpoint's ends_center truncation to 1022 residues. Fixed batches of 32 are a
numerical part of the scan recipe; native projections are not batch invariant.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from horizyn.datasets.residue_hdf5 import truncate_residue_embeddings
from horizyn.utils import residue_collate_fn

ROOT = Path(__file__).resolve().parents[1]
MODEL_RECORDS = {
    "f3": {
        "config": ("runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/train.yaml", "b0d79f2a0bcc82774e721870ff13a6d1b7ac2e3b4f003c9987f5600e7a60438b"),
        "checkpoint": ("runs/reactzyme_reaction_features_v1/checkpoints/reaction_smi/F3_set_chemistry/protein-pooling-epoch=29.ckpt", "e4f1537d4182698de492d9c4f85013a97bdfb3f882b116b39ed30960c1782068"),
    },
    "circev2": {
        "config": ("configs/reactzyme_reaction_smi_prott5_annotation_negatives.yaml", "b3934f87f2aae05199ab37c1ddf0ab5782f84f3c6337605409d258cc201e1939"),
        "checkpoint": ("runs/circe_v2_prott5_annotation_negatives/checkpoints/protein-pooling-epoch=23.ckpt", "5bc1e20094c2e316831d029f62e2240067ffe2fcbb2df9b72b7798f986472d29"),
    },
}


def artifact(path: Path) -> dict:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def validate_block(residues, offsets, ids, batch_size=32):
    if batch_size != 32:
        raise ValueError("The frozen native projection recipe requires batch_size=32")
    if not isinstance(residues, np.ndarray) or residues.dtype != np.float16 or residues.ndim != 2 or residues.shape[1] != 1024:
        raise ValueError("Residues must be a NumPy FP16 matrix with 1024 columns")
    offsets = np.asarray(offsets)
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) != len(ids) + 1:
        raise ValueError("Offsets must be an integer vector of len(ids)+1")
    if not len(ids) or any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("Protein IDs must be unique nonempty strings")
    if offsets[0] != 0 or offsets[-1] != len(residues) or np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("Local offsets must span the complete block with nonempty ordered spans")
    if not np.isfinite(residues).all():
        raise ValueError("Nonfinite stored residues")
    return offsets.astype(np.int64, copy=False)


class SharedNativeEncoders:
    """Load the two authenticated checkpoints once and reuse them per block."""

    def __init__(self, device="cuda:0"):
        from horizyn.benchmarks.retrieval import load_repo_checkpoint
        from horizyn.config import load_config

        self.device = torch.device(device)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.models = {}
        self.identities = {}
        self.load_seconds = {}
        for name, entries in MODEL_RECORDS.items():
            checked = {}
            for kind, (relative, expected) in entries.items():
                checked[kind] = artifact(ROOT / relative)
                if checked[kind]["sha256"] != expected:
                    raise ValueError(f"Frozen {name} {kind} identity mismatch")
            config = load_config(checked["config"]["path"])
            data = config.data
            if (data.residue_dim, data.max_protein_tokens, data.protein_truncation) != (1024, 1022, "ends_center"):
                raise ValueError(f"Unexpected native {name} residue preprocessing")
            if config.model.sleec_pooling.get("score_embedding_source", "same") != "same":
                raise ValueError(f"The {name} checkpoint needs another residue stream")
            start = time.monotonic()
            module, _ = load_repo_checkpoint(checked["checkpoint"]["path"], config, str(self.device))
            self.models[name] = module.eval().requires_grad_(False)
            self.load_seconds[name] = time.monotonic() - start
            self.identities[name] = checked

    @torch.inference_mode()
    def encode_block(self, residues, local_offsets, ids, device=None, batch_size=32):
        """Return native CPU FP32 arrays in exactly the supplied ID order.

        Result keys are ``f3``, ``circev2`` and ``timing``. Timing reports wall
        time including validation/collation/copies and projection-only device
        time. No normalization is applied here. No raw mean is computed here.
        """
        started = time.monotonic()
        if device is not None and torch.device(device) != self.device:
            raise ValueError("The block device must match the loaded models")
        if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("Native projection requires highest FP32 precision and TF32 disabled")
        local_offsets = validate_block(residues, local_offsets, ids, batch_size)
        finite_seconds = time.monotonic() - started
        arrays = {name: np.empty((len(ids), 512), dtype=np.float32) for name in self.models}
        projection_seconds = {name: 0.0 for name in self.models}
        for start in range(0, len(ids), batch_size):
            stop = min(start + batch_size, len(ids))
            samples = []
            for index in range(start, stop):
                a, b = local_offsets[index:index + 2]
                native = truncate_residue_embeddings(torch.from_numpy(residues[a:b]).float(), max_tokens=1022, strategy="ends_center")
                samples.append({"residue_embeddings": native, "target_id": ids[index]})
            batch = residue_collate_fn(samples)
            shared = batch["residue_embeddings"].to(self.device)
            mask = batch["residue_padding_mask"].to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=False):
                for name, module in self.models.items():
                    if self.device.type == "cuda":
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                    wall = time.monotonic()
                    encoded = module.model.encode_targets(
                        shared, residue_padding_mask=mask,
                        score_residue_embeddings=None, score_residue_padding_mask=None,
                        capability_vectors=None, capability_mask=None,
                        text_vectors=None, text_mask=None,
                        retrieval_direction="reaction_to_enzyme",
                    )
                    if self.device.type == "cuda":
                        end.record()
                        end.synchronize()
                        elapsed = begin.elapsed_time(end) / 1000
                    else:
                        elapsed = time.monotonic() - wall
                    projection_seconds[name] += elapsed
                    if encoded.dtype != torch.float32 or tuple(encoded.shape) != (stop - start, 512) or not torch.isfinite(encoded).all() or (encoded.norm(dim=1) <= 1e-8).any():
                        raise ValueError(f"Invalid native {name} output")
                    arrays[name][start:stop] = encoded.cpu().numpy()
        return {**arrays, "timing": {"wall_seconds": time.monotonic() - started,
                "validation_seconds": finite_seconds, "projection_seconds": projection_seconds,
                "proteins": len(ids), "batch_size": batch_size}}
