"""Pre-encodable retrieval using frozen F3, residual heads, and train anchors.

The dot product of the returned endpoint vectors is exactly a fixed weighted
sum of the residual/base cosine and the semantic-anchor cosine. No cross
encoder, candidate-pool statistic, or query-specific reranker is involved.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .generalization_residual import FrozenGeometryResidual
from .semantic_anchors import (centered_unit, reaction_features,
    nearest_training_proteins, enzyme_anchor_features, reaction_anchor_features)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_artifact(record, parent=Path(".")):
    """Resolve and authenticate an artifact before consuming its contents."""
    path = Path(record["path"])
    if not path.is_absolute():
        path = Path(parent) / path
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Model artifact hash mismatch: {path}")
    return path


def validate_frozen_residual(path, state, frozen, feature_manifest_sha256):
    """Reject unselected checkpoints and residuals fitted to another catalog."""
    allowed = [record for seeds in frozen["models"].values() for record in seeds.values()]
    actual = sha256(path)
    if not any(Path(record["path"]).resolve() == Path(path).resolve()
               and record["sha256"] == actual for record in allowed):
        raise ValueError(f"Residual checkpoint is not in the frozen recipe: {path}")
    registry = state.get("registry", {})
    if registry.get("feature_manifest_sha256") != feature_manifest_sha256:
        raise ValueError("Residual and anchor feature manifests disagree")
    if registry.get("test_used") is not False:
        raise ValueError("Residual training provenance must explicitly exclude test data")


def validate_anchor_recipe(anchors, frozen):
    for key in ("modalities", "enzyme_temperature", "reaction_temperature",
                "protein_neighbors", "reaction_neighbors"):
        if anchors[key] != frozen["anchors"][key]:
            raise ValueError(f"Anchor setting differs from frozen recipe: {key}")


@torch.inference_mode()
def canonical_dot(reactions, enzymes):
    """FP64 dot accumulation, rounded once to the FP32 score contract.

    This stabilizes the near-tied scores of shared reaction/protein families
    across GEMM shapes while retaining a compact deterministic score artifact.
    """
    return (reactions.double() @ enzymes.double().T).float()


class GeneralizationDualEncoder(nn.Module):
    """Independent endpoint transforms on top of a separately frozen F3 model.

    ``base`` arguments are native frozen F3 embeddings (before the evaluation
    cosine normalization). Raw protein inputs are mean frozen residue vectors;
    raw reaction blocks use the SAME input policy and train-fitted chemistry
    schema as the F3 checkpoint. This class never fits from inference batches.
    """

    def __init__(self, anchors, residuals=(), alpha=.5):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("Anchor weight must lie in [0, 1]")
        self.alpha = float(alpha)
        self.residuals = nn.ModuleList(residuals)
        self.modalities = list(anchors["modalities"])
        self.enzyme_temperature = float(anchors["enzyme_temperature"])
        self.reaction_temperature = float(anchors["reaction_temperature"])
        self.protein_neighbors = int(anchors["protein_neighbors"])
        self.reaction_neighbors = int(anchors["reaction_neighbors"])
        for key in ("protein_center", "train_proteins", "train_reactions", "adjacency"):
            self.register_buffer(key, anchors[key])
        for key in self.modalities:
            self.register_buffer("reaction_center_" + key, anchors["reaction_centers"][key])
        self.requires_grad_(False)
        self.eval()

    def _base(self, base, direction):
        if not self.residuals:
            return F.normalize(base, dim=-1)
        encode = "encode_enzymes" if direction == "enzyme" else "encode_reactions"
        values = [getattr(model, encode)(base) for model in self.residuals]
        return torch.cat(values, dim=1) / math.sqrt(len(values))

    @torch.inference_mode()
    def encode_enzymes(self, base, protein_mean, batch_size=2048):
        learned = self._base(base, "enzyme")
        if not self.alpha:
            return learned
        encoded = centered_unit(protein_mean, self.protein_center)
        values, indices = nearest_training_proteins(encoded, self.train_proteins,
                                                    self.protein_neighbors, batch_size)
        semantic = enzyme_anchor_features(values, indices, self.adjacency,
                                          len(self.train_reactions), self.enzyme_temperature)
        return torch.cat((math.sqrt(1 - self.alpha) * learned,
                          math.sqrt(self.alpha) * semantic), dim=1)

    @torch.inference_mode()
    def encode_reactions(self, base, blocks, masks):
        learned = self._base(base, "reaction")
        if not self.alpha:
            return learned
        centers = {key: getattr(self, "reaction_center_" + key) for key in self.modalities}
        encoded = reaction_features(blocks, centers, masks, self.modalities)
        semantic = reaction_anchor_features(encoded, self.train_reactions,
                                            self.reaction_temperature, self.reaction_neighbors)
        return torch.cat((math.sqrt(1 - self.alpha) * learned,
                          math.sqrt(self.alpha) * semantic), dim=1)

    @torch.inference_mode()
    def encode_enzyme_index(self, base, protein_mean, batch_size=2048):
        """Return a dense learned block and a sparse CSR anchor block.

        This avoids materializing N x (number of training reactions) for a
        large enzyme collection. The index is reusable across reaction queries.
        Each temporary anchor block has at most ``batch_size`` rows.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Index batch size must be a positive integer")
        if len(base) != len(protein_mean):
            raise ValueError("Protein means and base embeddings must have identical row counts")
        learned = math.sqrt(1 - self.alpha) * self._base(base, "enzyme")
        pointers = [torch.zeros(1, dtype=torch.int64, device=base.device)]
        columns, weights, total = [], [], 0
        for start in range(0, len(base), batch_size):
            stop = min(start + batch_size, len(base))
            if self.alpha:
                encoded = centered_unit(protein_mean[start:stop], self.protein_center)
                val, idx = nearest_training_proteins(encoded, self.train_proteins,
                                                      self.protein_neighbors, batch_size)
                semantic = enzyme_anchor_features(val, idx, self.adjacency,
                    len(self.train_reactions), self.enzyme_temperature)
                sparse = (math.sqrt(self.alpha) * semantic).to_sparse_csr()
                pointers.append(sparse.crow_indices()[1:] + total)
                columns.append(sparse.col_indices())
                weights.append(sparse.values())
                total += sparse.values().numel()
            else:
                pointers.append(torch.full((stop-start,), total, dtype=torch.int64, device=base.device))
        columns = torch.cat(columns) if columns else torch.empty(0, dtype=torch.int64, device=base.device)
        weights = torch.cat(weights) if weights else torch.empty(0, dtype=base.dtype, device=base.device)
        sparse = torch.sparse_csr_tensor(torch.cat(pointers), columns, weights,
            size=(len(base), len(self.train_reactions) if self.alpha else 0), device=base.device)
        return {"dense": learned, "anchors": sparse}

    @staticmethod
    @torch.inference_mode()
    def score_index(reactions, index):
        """Score a precomputed sparse enzyme index with encoded reactions."""
        width = index["dense"].shape[1]
        expected = width + index["anchors"].shape[1]
        if reactions.shape[1] != expected:
            raise ValueError("Reaction and index dimensions disagree")
        scores = reactions[:, :width].double() @ index["dense"].double().T
        if index["anchors"].shape[1]:
            scores = scores + torch.sparse.mm(index["anchors"].double(), reactions[:, width:].double().T).T
        return scores.float()

    @classmethod
    def from_bundle(cls, path, device="cpu", verify=True):
        path = Path(path)
        specification = json.loads(path.read_text())
        if specification.get("schema") != "generalization_dual_encoder_bundle_v1":
            raise ValueError("Unrecognized dual-encoder bundle schema")
        def artifact(record):
            result = Path(record["path"])
            if not result.is_absolute():
                result = path.parent / result
            if verify and sha256(result) != record["sha256"]:
                raise ValueError(f"Model artifact hash mismatch: {result}")
            return result
        frozen = None
        if verify:
            frozen = json.loads(artifact(specification["frozen_recipe"]).read_text())
            if not frozen.get("frozen_before_held_out_evaluation"):
                raise ValueError("Recipe manifest has not been frozen")
        anchors = torch.load(artifact(specification["anchors"]), map_location=device, weights_only=False)
        if verify:
            if anchors.get("feature_manifest_sha256") != specification["feature_manifest_sha256"]:
                raise ValueError("Anchor and bundle feature manifests disagree")
            validate_anchor_recipe(anchors, frozen)
        residuals = []
        for record in specification["residuals"]:
            checkpoint = artifact(record)
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            if verify:
                validate_frozen_residual(checkpoint, state, frozen, specification["feature_manifest_sha256"])
            model = FrozenGeometryResidual(**state["model_config"]).to(device)
            model.load_state_dict(state["state_dict"], strict=True)
            residuals.append(model)
        return cls(anchors, residuals, specification["anchor_alpha"]).to(device), specification
