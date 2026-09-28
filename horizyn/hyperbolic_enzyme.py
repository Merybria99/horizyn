"""Unimodal Lorentz enzyme pretraining modules."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from horizyn.lorentz import (
    expmap0_lorentz,
    logmap0_lorentz,
    lorentz_constraint_error,
    lorentz_distance,
    lorentz_inner,
    pairwise_lorentz_distance,
)
from horizyn.model import FunctionalResidueScorer


EC_LABEL_PATTERN = re.compile(r"^[1-7](?:\.(?:\d+|-)){0,3}$")


def parse_ec_prefixes(ec_label: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Parse an EC label into known cumulative prefix levels.

    Unknown EC levels are represented by ``None``. Once an unknown level is
    encountered, later levels are ignored because they are not meaningful
    hierarchy prefixes. For example:

    - ``1.1.1.1`` -> (``1``, ``1.1``, ``1.1.1``, ``1.1.1.1``)
    - ``1.1.1.-`` -> (``1``, ``1.1``, ``1.1.1``, ``None``)
    - ``1.1`` -> (``1``, ``1.1``, ``None``, ``None``)
    """
    ec = str(ec_label).strip()
    ec = re.sub(r"^EC[-:\s]*", "", ec, flags=re.IGNORECASE)
    if not EC_LABEL_PATTERN.fullmatch(ec):
        raise ValueError(f"Invalid EC label: {ec_label!r}")

    parts = ec.split(".")
    if len(parts) > 4:
        raise ValueError(f"Invalid EC label with more than four levels: {ec_label!r}")
    parts = parts + ["-"] * (4 - len(parts))

    prefixes: list[str | None] = []
    known_parts: list[str] = []
    unknown_seen = False
    for part in parts:
        if unknown_seen or part == "-":
            unknown_seen = True
            prefixes.append(None)
            continue
        known_parts.append(part)
        prefixes.append(".".join(known_parts))
    if prefixes[0] is None:
        raise ValueError(f"EC label must have a known top-level class: {ec_label!r}")
    return prefixes[0], prefixes[1], prefixes[2], prefixes[3]


def format_ec_prefixes(prefixes: tuple[str | None, str | None, str | None, str | None]) -> str:
    """Format parsed EC prefixes as a normalized four-level EC label."""
    parts: list[str] = []
    for level, prefix in enumerate(prefixes, start=1):
        if prefix is None:
            parts.append("-")
        else:
            parts.append(prefix.split(".")[level - 1])
    return ".".join(parts)


def known_ec_depth(ec_label: str) -> int:
    """Return the number of known leading EC levels."""
    return sum(prefix is not None for prefix in parse_ec_prefixes(ec_label))


def parse_complete_ec(ec_label: str) -> tuple[str, str, str, str]:
    """Parse a complete EC number into cumulative prefix levels."""
    prefixes = parse_ec_prefixes(ec_label)
    if any(prefix is None for prefix in prefixes):
        raise ValueError(f"Invalid complete EC label: {ec_label!r}")
    return prefixes  # type: ignore[return-value]


def compute_ec_shared_depth(ec_labels: list[str]) -> torch.Tensor:
    """Return pairwise EC shared-prefix depths in {0, 1, 2, 3, 4}."""
    prefixes = [parse_ec_prefixes(label) for label in ec_labels]
    batch_size = len(prefixes)
    depths = torch.zeros(batch_size, batch_size, dtype=torch.long)
    for row_idx, row_prefixes in enumerate(prefixes):
        for col_idx, col_prefixes in enumerate(prefixes):
            depth = 0
            for lhs, rhs in zip(row_prefixes, col_prefixes):
                if lhs is None or rhs is None:
                    break
                if lhs != rhs:
                    break
                depth += 1
            depths[row_idx, col_idx] = depth
    return depths


def build_hierarchy_triplets(
    ec_depth_matrix: torch.Tensor,
    max_triplets_per_anchor: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build depth-ordered triplets (anchor, positive, negative).

    Positives are enzymes sharing at least one EC level with the anchor.
    Negatives are selected from the closest lower-depth group first.
    """
    if ec_depth_matrix.ndim != 2 or ec_depth_matrix.shape[0] != ec_depth_matrix.shape[1]:
        raise ValueError("ec_depth_matrix must have shape [B, B]")
    if max_triplets_per_anchor <= 0:
        raise ValueError("max_triplets_per_anchor must be positive")

    device = ec_depth_matrix.device
    batch_size = ec_depth_matrix.shape[0]
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []

    for anchor_idx in range(batch_size):
        row = ec_depth_matrix[anchor_idx]
        other_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        other_mask[anchor_idx] = False
        positive_indices = torch.nonzero((row > 0) & other_mask, as_tuple=False).flatten()
        if positive_indices.numel() == 0:
            continue
        positive_depths = row[positive_indices]
        order = torch.argsort(positive_depths, descending=True)
        triplets_for_anchor = 0
        for pos_idx in positive_indices[order].tolist():
            pos_depth = int(row[pos_idx].item())
            for neg_depth in range(pos_depth - 1, -1, -1):
                candidate_negatives = torch.nonzero(
                    (row == neg_depth) & other_mask,
                    as_tuple=False,
                ).flatten()
                if candidate_negatives.numel() == 0:
                    continue
                neg_idx = int(candidate_negatives[triplets_for_anchor % candidate_negatives.numel()].item())
                anchors.append(anchor_idx)
                positives.append(pos_idx)
                negatives.append(neg_idx)
                triplets_for_anchor += 1
                break
            if triplets_for_anchor >= max_triplets_per_anchor:
                break

    if not anchors:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty
    return (
        torch.tensor(anchors, dtype=torch.long, device=device),
        torch.tensor(positives, dtype=torch.long, device=device),
        torch.tensor(negatives, dtype=torch.long, device=device),
    )


class SLEECGuidedAttentionPool(nn.Module):
    """Trainable residue attention pooling guided by a frozen SLEEC prior."""

    def __init__(
        self,
        input_dim: int = 1280,
        scorer_hidden_dim: int = 256,
        p0: float = 0.34,
        sleec_checkpoint_path: str | None = None,
        freeze_sleec: bool = True,
        attention_bias: bool = True,
        initial_prior_strength: float = 1.0,
        train_prior_strength: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not (0.0 < p0 < 1.0):
            raise ValueError("p0 must be in the open interval (0, 1)")
        if initial_prior_strength <= 0:
            raise ValueError("initial_prior_strength must be positive")
        self.input_dim = int(input_dim)
        self.p0 = float(p0)
        self.eps = float(eps)
        self.attention = nn.Linear(self.input_dim, 1, bias=attention_bias)
        self.sleec_scorer = FunctionalResidueScorer(
            hidden_dim=self.input_dim,
            scorer_hidden_dim=scorer_hidden_dim,
        )
        if sleec_checkpoint_path:
            self.sleec_scorer.load_stage1_checkpoint(sleec_checkpoint_path)
        if freeze_sleec:
            for parameter in self.sleec_scorer.parameters():
                parameter.requires_grad = False
        threshold_tensor = torch.tensor(self.p0, dtype=torch.float32)
        self.register_buffer("threshold_logit", torch.logit(threshold_tensor))
        raw_strength = torch.log(torch.expm1(torch.tensor(float(initial_prior_strength))))
        self.raw_prior_strength = nn.Parameter(raw_strength)
        self.raw_prior_strength.requires_grad = bool(train_prior_strength)

    @property
    def prior_strength(self) -> torch.Tensor:
        return F.softplus(self.raw_prior_strength)

    def _valid_mask(
        self,
        residue_embeddings: torch.Tensor,
        residue_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if residue_embeddings.ndim != 3:
            raise ValueError(
                "residue_embeddings must have shape [B, L, D], "
                f"got {tuple(residue_embeddings.shape)}"
            )
        if residue_embeddings.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {residue_embeddings.shape[-1]}")
        if residue_mask is None:
            return torch.ones(
                residue_embeddings.shape[:2],
                dtype=torch.bool,
                device=residue_embeddings.device,
            )
        if residue_mask.shape != residue_embeddings.shape[:2]:
            raise ValueError(
                "residue_mask must have shape [B, L], "
                f"got mask={tuple(residue_mask.shape)}, residues={tuple(residue_embeddings.shape)}"
            )
        return residue_mask.to(device=residue_embeddings.device, dtype=torch.bool)

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
        sleec_logits: torch.Tensor | None = None,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid_mask = self._valid_mask(residue_embeddings, residue_mask)
        if valid_mask.sum(dim=1).eq(0).any():
            raise ValueError("Each enzyme must contain at least one valid residue")

        if sleec_logits is None:
            sleec_logits, _ = self.sleec_scorer(residue_embeddings, attention_mask=valid_mask)
        else:
            if sleec_logits.shape != residue_embeddings.shape[:2]:
                raise ValueError(
                    "sleec_logits must have shape [B, L], "
                    f"got {tuple(sleec_logits.shape)}"
                )
            sleec_logits = sleec_logits.to(device=residue_embeddings.device, dtype=residue_embeddings.dtype)
            sleec_logits = sleec_logits.masked_fill(~valid_mask, 0.0)

        learned_logits = self.attention(residue_embeddings).squeeze(-1)
        centered_prior = sleec_logits - self.threshold_logit.to(
            device=residue_embeddings.device,
            dtype=residue_embeddings.dtype,
        )
        combined_logits = learned_logits + self.prior_strength.to(
            dtype=residue_embeddings.dtype
        ) * centered_prior
        masked_logits = combined_logits.masked_fill(~valid_mask, float("-inf"))
        attention_weights = torch.softmax(masked_logits, dim=-1).masked_fill(~valid_mask, 0.0)
        pooled = torch.einsum("bl,bld->bd", attention_weights, residue_embeddings)

        if not return_details:
            return pooled

        details = {
            "h_enzyme": pooled,
            "attention_weights": attention_weights,
            "attention_logits": combined_logits.masked_fill(~valid_mask, 0.0),
            "learned_attention_logits": learned_logits.masked_fill(~valid_mask, 0.0),
            "sleec_logits": sleec_logits.masked_fill(~valid_mask, 0.0),
            "prior_strength": self.prior_strength.detach(),
        }
        return pooled, details


class LorentzEnzymeProjector(nn.Module):
    """Project pooled enzyme embeddings to the Lorentz model."""

    def __init__(
        self,
        input_dim: int = 1280,
        hyp_dim: int = 512,
        curvature: float = 0.25,
        tangent_clip: float | None = 5.0,
        tangent_clip_mode: str = "hard",
        input_normalization: str = "none",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hyp_dim <= 0:
            raise ValueError("hyp_dim must be positive")
        if curvature <= 0:
            raise ValueError("curvature must be positive")
        if tangent_clip is not None and tangent_clip <= 0:
            raise ValueError("tangent_clip must be positive when provided")
        if tangent_clip_mode not in {"hard", "soft"}:
            raise ValueError("tangent_clip_mode must be one of: hard, soft")
        if input_normalization not in {"none", "layernorm"}:
            raise ValueError("input_normalization must be one of: none, layernorm")
        self.input_dim = int(input_dim)
        self.hyp_dim = int(hyp_dim)
        self.curvature = float(curvature)
        self.tangent_clip = None if tangent_clip is None else float(tangent_clip)
        self.tangent_clip_mode = tangent_clip_mode
        self.eps = float(eps)
        self.input_norm = (
            nn.LayerNorm(self.input_dim)
            if input_normalization == "layernorm"
            else nn.Identity()
        )
        self.projection = nn.Linear(self.input_dim, self.hyp_dim)

    def _clip_tangent(self, tangent: torch.Tensor) -> torch.Tensor:
        if self.tangent_clip is None:
            return tangent
        norm = tangent.norm(dim=-1, keepdim=True)
        if self.tangent_clip_mode == "soft":
            scale = torch.tanh(norm / self.tangent_clip) * self.tangent_clip
            return tangent / norm.clamp_min(self.eps) * scale
        clipped = tangent / norm.clamp_min(self.eps) * self.tangent_clip
        return torch.where(norm > self.tangent_clip, clipped, tangent)

    def forward(self, enzyme_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if enzyme_emb.ndim != 2 or enzyme_emb.shape[-1] != self.input_dim:
            raise ValueError(
                f"enzyme_emb must have shape [B, {self.input_dim}], got {tuple(enzyme_emb.shape)}"
            )
        normalized = self.input_norm(enzyme_emb)
        tangent = self._clip_tangent(self.projection(normalized))
        z_hyp = expmap0_lorentz(tangent, kappa=self.curvature, eps=self.eps)
        z_tangent = logmap0_lorentz(z_hyp, kappa=self.curvature, eps=self.eps)
        return z_hyp, z_tangent


@dataclass
class RankingLossOutput:
    loss: torch.Tensor
    logs: dict[str, torch.Tensor]
    triplets: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class HierarchyRankingLoss(nn.Module):
    """Margin ranking loss over enzyme-enzyme Lorentz distances."""

    def __init__(
        self,
        base_margin: float = 0.05,
        curvature: float = 0.25,
        max_triplets_per_anchor: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if base_margin < 0:
            raise ValueError("base_margin must be non-negative")
        self.base_margin = float(base_margin)
        self.curvature = float(curvature)
        self.max_triplets_per_anchor = int(max_triplets_per_anchor)
        self.eps = float(eps)

    def forward(
        self,
        z_hyp: torch.Tensor,
        ec_depth_matrix: torch.Tensor,
        triplets: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> RankingLossOutput:
        if triplets is None:
            triplets = build_hierarchy_triplets(
                ec_depth_matrix,
                max_triplets_per_anchor=self.max_triplets_per_anchor,
            )
        anchor_idx, pos_idx, neg_idx = [
            tensor.to(device=z_hyp.device, dtype=torch.long) for tensor in triplets
        ]
        depth_matrix = ec_depth_matrix.to(device=z_hyp.device, dtype=torch.long)
        if anchor_idx.numel() == 0:
            zero = z_hyp.sum() * 0.0
            logs = {
                "loss_rank": zero.detach(),
                "num_triplets": torch.zeros((), dtype=z_hyp.dtype, device=z_hyp.device),
                "mean_positive_distance": zero.detach(),
                "mean_negative_distance": zero.detach(),
                "mean_depth_positive": zero.detach(),
                "mean_depth_negative": zero.detach(),
            }
            return RankingLossOutput(zero, logs, (anchor_idx, pos_idx, neg_idx))

        distances = pairwise_lorentz_distance(
            z_hyp,
            z_hyp,
            kappa=self.curvature,
            eps=self.eps,
        )
        pos_dist = distances[anchor_idx, pos_idx]
        neg_dist = distances[anchor_idx, neg_idx]
        depth_pos = depth_matrix[anchor_idx, pos_idx].to(dtype=z_hyp.dtype)
        depth_neg = depth_matrix[anchor_idx, neg_idx].to(dtype=z_hyp.dtype)
        margin = self.base_margin * (depth_pos - depth_neg).clamp_min(0.0)
        loss = F.relu(margin + pos_dist - neg_dist).mean()
        logs = {
            "loss_rank": loss.detach(),
            "num_triplets": torch.tensor(float(anchor_idx.numel()), dtype=z_hyp.dtype, device=z_hyp.device),
            "mean_positive_distance": pos_dist.mean().detach(),
            "mean_negative_distance": neg_dist.mean().detach(),
            "mean_depth_positive": depth_pos.mean().detach(),
            "mean_depth_negative": depth_neg.mean().detach(),
        }
        return RankingLossOutput(loss, logs, (anchor_idx, pos_idx, neg_idx))


def same_ec4_radius_loss(
    z_hyp: torch.Tensor,
    ec_depth_matrix: torch.Tensor,
    curvature: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Encourage enzymes with identical EC4 labels to have similar radii."""
    depth_matrix = ec_depth_matrix.to(device=z_hyp.device)
    pair_mask = torch.triu(depth_matrix == 4, diagonal=1)
    if not bool(pair_mask.any()):
        return z_hyp.sum() * 0.0
    origin = z_hyp.new_zeros(z_hyp.shape[-1])
    origin[0] = 1.0 / (curvature**0.5)
    radii = lorentz_distance(
        z_hyp,
        origin.unsqueeze(0).expand_as(z_hyp),
        kappa=curvature,
        eps=eps,
    )
    row_idx, col_idx = torch.nonzero(pair_mask, as_tuple=True)
    return (radii[row_idx] - radii[col_idx]).abs().mean()


def radius_target_loss(
    z_hyp: torch.Tensor,
    radius_target: float,
    curvature: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize enzyme points whose Lorentz radius exceeds ``radius_target``."""
    if radius_target <= 0:
        return z_hyp.sum() * 0.0
    origin = z_hyp.new_zeros(z_hyp.shape[-1])
    origin[0] = 1.0 / (curvature**0.5)
    radii = lorentz_distance(
        z_hyp,
        origin.unsqueeze(0).expand_as(z_hyp),
        kappa=curvature,
        eps=eps,
    )
    return F.relu(radii - radius_target).pow(2).mean()


class LorentzEntailmentConeLoss(nn.Module):
    """HyCoCLIP-style Lorentz cone entailment loss."""

    def __init__(
        self,
        curvature: float = 0.25,
        K: float = 0.1,
        eta: float = 1.0,
        cone_loss_power: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.curvature = float(curvature)
        self.K = float(K)
        self.eta = float(eta)
        self.cone_loss_power = float(cone_loss_power)
        self.eps = float(eps)

    def terms(self, child: torch.Tensor, parent: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return angle, aperture, violation, and powered loss terms."""
        kappa = self.curvature
        sqrt_k = kappa**0.5
        parent_spatial_norm = parent[..., 1:].norm(dim=-1).clamp_min(self.eps)
        omega_arg = (2.0 * self.K / (sqrt_k * parent_spatial_norm)).clamp(
            min=self.eps,
            max=1.0 - self.eps,
        )
        omega = torch.asin(omega_arg)

        inner = lorentz_inner(child, parent)
        k_inner = kappa * inner
        sqrt_arg = (k_inner.pow(2) - 1.0).clamp_min(self.eps)
        numerator = child[..., 0] + parent[..., 0] * kappa * inner
        denominator = parent_spatial_norm * torch.sqrt(sqrt_arg)
        phi_arg = (numerator / denominator.clamp_min(self.eps)).clamp(
            min=-1.0 + self.eps,
            max=1.0 - self.eps,
        )
        phi = torch.acos(phi_arg)
        violation = F.relu(phi - self.eta * omega)
        return {
            "angle": phi,
            "aperture": omega,
            "violation": violation,
            "loss": violation.pow(self.cone_loss_power),
        }

    def forward(self, child: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
        return self.terms(child, parent)["loss"]


class NonParametricCentroidConeLoss(nn.Module):
    """Optional centroid-cone regularizer without trainable EC representations."""

    def __init__(
        self,
        curvature: float = 0.25,
        K: float = 0.1,
        eta: float = 1.0,
        cone_loss_power: float = 1.0,
        min_group_size: int = 2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.curvature = float(curvature)
        self.min_group_size = int(min_group_size)
        self.eps = float(eps)
        self.cone_loss = LorentzEntailmentConeLoss(
            curvature=curvature,
            K=K,
            eta=eta,
            cone_loss_power=cone_loss_power,
            eps=eps,
        )

    def _centroids(self, z_hyp: torch.Tensor, ec_labels: list[str]) -> dict[str, torch.Tensor]:
        tangent = logmap0_lorentz(z_hyp, kappa=self.curvature, eps=self.eps)
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, label in enumerate(ec_labels):
            for prefix in parse_ec_prefixes(label):
                if prefix is not None:
                    groups[prefix].append(idx)
        centroids: dict[str, torch.Tensor] = {}
        for prefix, indices in groups.items():
            if len(indices) < self.min_group_size:
                continue
            index_tensor = torch.tensor(indices, dtype=torch.long, device=z_hyp.device)
            mean_tangent = tangent[index_tensor].mean(dim=0, keepdim=True)
            centroids[prefix] = expmap0_lorentz(
                mean_tangent,
                kappa=self.curvature,
                eps=self.eps,
            ).squeeze(0)
        return centroids

    def forward(self, z_hyp: torch.Tensor, ec_labels: list[str]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        centroids = self._centroids(z_hyp, ec_labels)
        losses: list[torch.Tensor] = []
        for enzyme_idx, label in enumerate(ec_labels):
            prefixes = parse_ec_prefixes(label)
            for parent_prefix, child_prefix in zip(prefixes[:-1], prefixes[1:]):
                if parent_prefix is None or child_prefix is None:
                    continue
                if child_prefix in centroids and parent_prefix in centroids:
                    losses.append(self.cone_loss(centroids[child_prefix], centroids[parent_prefix]))
            deepest_prefix = next(
                (prefix for prefix in reversed(prefixes) if prefix is not None),
                None,
            )
            if deepest_prefix is not None and deepest_prefix in centroids:
                losses.append(self.cone_loss(z_hyp[enzyme_idx], centroids[deepest_prefix]))
        if not losses:
            zero = z_hyp.sum() * 0.0
            return zero, {
                "loss_centroid_cone": zero.detach(),
                "num_centroid_cone_terms": torch.zeros((), dtype=z_hyp.dtype, device=z_hyp.device),
            }
        loss_values = torch.stack([loss.reshape(()) for loss in losses])
        loss = loss_values.mean()
        return loss, {
            "loss_centroid_cone": loss.detach(),
            "num_centroid_cone_terms": torch.tensor(float(len(losses)), dtype=z_hyp.dtype, device=z_hyp.device),
        }


class ECPrefixEntailmentLoss(nn.Module):
    """Batch-centroid EC-prefix entailment loss without trainable EC nodes."""

    def __init__(
        self,
        curvature: float = 0.25,
        min_radius: float = 0.1,
        eta: float = 1.0,
        min_group_size: int = 2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min_group_size <= 0:
            raise ValueError("min_group_size must be positive")
        self.curvature = float(curvature)
        self.min_group_size = int(min_group_size)
        self.eps = float(eps)
        self.cone_loss = LorentzEntailmentConeLoss(
            curvature=curvature,
            K=min_radius,
            eta=eta,
            cone_loss_power=1.0,
            eps=eps,
        )

    def _centroids(self, z_hyp: torch.Tensor, ec_labels: list[str]) -> dict[str, torch.Tensor]:
        tangent = logmap0_lorentz(z_hyp, kappa=self.curvature, eps=self.eps)
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, label in enumerate(ec_labels):
            for prefix in parse_ec_prefixes(label):
                if prefix is not None:
                    groups[prefix].append(idx)

        centroids: dict[str, torch.Tensor] = {}
        for prefix, indices in groups.items():
            if len(indices) < self.min_group_size:
                continue
            index_tensor = torch.tensor(indices, dtype=torch.long, device=z_hyp.device)
            centroid_tangent = tangent[index_tensor].mean(dim=0, keepdim=True)
            centroids[prefix] = expmap0_lorentz(
                centroid_tangent,
                kappa=self.curvature,
                eps=self.eps,
            ).squeeze(0)
        return centroids

    def forward(self, z_hyp: torch.Tensor, ec_labels: list[str]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        centroids = self._centroids(z_hyp, ec_labels)
        child_terms: list[torch.Tensor] = []
        parent_terms: list[torch.Tensor] = []

        centroid_edges: set[tuple[str, str]] = set()
        for label in ec_labels:
            prefixes = parse_ec_prefixes(label)
            for parent_prefix, child_prefix in zip(prefixes[:-1], prefixes[1:]):
                if parent_prefix is None or child_prefix is None:
                    continue
                if parent_prefix in centroids and child_prefix in centroids:
                    centroid_edges.add((child_prefix, parent_prefix))

        for child_prefix, parent_prefix in sorted(centroid_edges):
            child_terms.append(centroids[child_prefix])
            parent_terms.append(centroids[parent_prefix])

        for enzyme_idx, label in enumerate(ec_labels):
            deepest_prefix = next(
                (
                    prefix
                    for prefix in reversed(parse_ec_prefixes(label))
                    if prefix is not None
                ),
                None,
            )
            if deepest_prefix is not None and deepest_prefix in centroids:
                child_terms.append(z_hyp[enzyme_idx])
                parent_terms.append(centroids[deepest_prefix])

        if not child_terms:
            zero = z_hyp.sum() * 0.0
            logs = {
                "loss_ec_entailment": zero.detach(),
                "num_ec_entailment_terms": torch.zeros((), dtype=z_hyp.dtype, device=z_hyp.device),
                "ec_entailment_violation_rate": zero.detach(),
                "mean_ec_entailment_angle": zero.detach(),
                "mean_ec_entailment_aperture": zero.detach(),
            }
            return zero, logs

        children = torch.stack(child_terms, dim=0)
        parents = torch.stack(parent_terms, dim=0)
        terms = self.cone_loss.terms(children, parents)
        loss_values = terms["loss"]
        loss = loss_values.mean()
        violation = terms["violation"]
        logs = {
            "loss_ec_entailment": loss.detach(),
            "num_ec_entailment_terms": torch.tensor(float(children.shape[0]), dtype=z_hyp.dtype, device=z_hyp.device),
            "ec_entailment_violation_rate": (violation > 0).to(dtype=z_hyp.dtype).mean().detach(),
            "mean_ec_entailment_angle": terms["angle"].mean().detach(),
            "mean_ec_entailment_aperture": terms["aperture"].mean().detach(),
        }
        return loss, logs


def mean_attention_entropy(attention_weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return -(attention_weights.clamp_min(eps).log() * attention_weights).sum(dim=1).mean()


def attention_mass_above_threshold(
    attention_weights: torch.Tensor,
    sleec_logits: torch.Tensor,
    threshold: float,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    threshold_logit = torch.logit(
        torch.tensor(threshold, dtype=sleec_logits.dtype, device=sleec_logits.device)
    )
    positive = (sleec_logits > threshold_logit) & residue_mask.to(torch.bool)
    return (attention_weights * positive.to(dtype=attention_weights.dtype)).sum(dim=1).mean()


__all__ = [
    "SLEECGuidedAttentionPool",
    "LorentzEnzymeProjector",
    "HierarchyRankingLoss",
    "RankingLossOutput",
    "LorentzEntailmentConeLoss",
    "NonParametricCentroidConeLoss",
    "ECPrefixEntailmentLoss",
    "format_ec_prefixes",
    "known_ec_depth",
    "parse_ec_prefixes",
    "parse_complete_ec",
    "compute_ec_shared_depth",
    "build_hierarchy_triplets",
    "same_ec4_radius_loss",
    "radius_target_loss",
    "mean_attention_entropy",
    "attention_mass_above_threshold",
]
