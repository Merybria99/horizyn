"""Historical architectures retained only for compatibility with shared constructor branches.
No maintained pipeline selects these architectures; use horizyn.pipelines."""

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from horizyn.checkpoint_utils import extract_query_encoder_state_dict
from horizyn.biological_residual import PromiscuityAwareBiologicalResidual

from .common import (BaseModel, MLP, NormalizeLayer, signed_power_transform)
from .reaction import (MoleculeSetAttentionPooling, MoleculeSetMeanPooling)


class ResidualEnzymePrototypeHead(nn.Module):
    """Expand one sequence-derived enzyme embedding into latent functional modes.

    The head is deliberately residual: at initialization every prototype stays
    very close to the parent enzyme embedding, while independent low-rank
    branches can specialize from enzyme--reaction supervision.  Prototype
    priors are also sequence-derived, so no trainable enzyme identifiers are
    required and the head remains usable for unseen proteins.
    """

    def __init__(
        self,
        embedding_dim: int,
        prototype_count: int = 2,
        bottleneck_dim: int = 64,
        residual_gate_init: float = 0.05,
        aggregation_temperature: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if prototype_count <= 0:
            raise ValueError("prototype_count must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if not 0.0 < residual_gate_init < 1.0:
            raise ValueError("residual_gate_init must be in (0, 1)")
        if aggregation_temperature <= 0.0:
            raise ValueError("aggregation_temperature must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.embedding_dim = int(embedding_dim)
        self.prototype_count = int(prototype_count)
        self.bottleneck_dim = int(bottleneck_dim)
        self.aggregation_temperature = float(aggregation_temperature)
        # Keep the K=1 compatibility path parameter-free.  Besides producing
        # exactly the historical cosine score, this lets old checkpoints load
        # strictly: adding the prototype feature does not introduce missing
        # state-dict keys unless K>1 was explicitly configured.
        branch_count = self.prototype_count if self.prototype_count > 1 else 0
        self.residual_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.embedding_dim),
                    nn.Linear(self.embedding_dim, self.bottleneck_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.bottleneck_dim, self.embedding_dim),
                )
                for _ in range(branch_count)
            ]
        )
        # Small independent final projections break slot symmetry without
        # perturbing the warm-started retrieval geometry appreciably.
        for branch in self.residual_branches:
            final = branch[-1]
            if not isinstance(final, nn.Linear):
                raise RuntimeError("Prototype residual branch must end in Linear")
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final.bias)

        if self.prototype_count > 1:
            gate_logit = math.log(residual_gate_init / (1.0 - residual_gate_init))
            self.residual_gate_logits = nn.Parameter(
                torch.full((self.prototype_count,), gate_logit, dtype=torch.float32)
            )
            self.prior_projection: nn.Linear | None = nn.Linear(
                self.embedding_dim,
                self.prototype_count,
            )
            nn.init.zeros_(self.prior_projection.weight)
            nn.init.zeros_(self.prior_projection.bias)
        else:
            self.register_parameter("residual_gate_logits", None)
            self.prior_projection = None

    def forward(
        self,
        enzyme_embeddings: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if enzyme_embeddings.ndim != 2 or enzyme_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                "enzyme_embeddings must have shape "
                f"[batch, {self.embedding_dim}], got {tuple(enzyme_embeddings.shape)}"
            )

        base = F.normalize(enzyme_embeddings, p=2, dim=-1, eps=1e-12)
        if self.prototype_count == 1:
            prototypes = base.unsqueeze(1)
            log_priors = base.new_zeros((base.shape[0], 1))
            gates = base.new_zeros((1,))
        else:
            if self.residual_gate_logits is None or self.prior_projection is None:
                raise RuntimeError("Multi-prototype parameters were not initialized")
            deltas = torch.stack(
                [branch(base) for branch in self.residual_branches],
                dim=1,
            )
            gates = torch.sigmoid(self.residual_gate_logits).to(dtype=base.dtype)
            prototypes = F.normalize(
                base.unsqueeze(1) + gates.view(1, -1, 1) * deltas,
                p=2,
                dim=-1,
                eps=1e-12,
            )
            log_priors = F.log_softmax(self.prior_projection(base), dim=-1)

        if not return_details:
            return prototypes
        return prototypes, {
            "log_priors": log_priors,
            "prior_probabilities": log_priors.exp(),
            "residual_gates": gates,
        }

    def score(
        self,
        reaction_embeddings: torch.Tensor,
        enzyme_embeddings: torch.Tensor,
        *,
        similarity: str = "cosine",
        feature_power: float = 1.0,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a reaction-by-enzyme compatibility matrix."""

        if reaction_embeddings.ndim != 2:
            raise ValueError("reaction_embeddings must be rank-2")
        if reaction_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"reaction embedding dim must be {self.embedding_dim}, "
                f"got {reaction_embeddings.shape[-1]}"
            )
        if similarity not in {"cosine", "dot"}:
            raise ValueError("similarity must be cosine or dot")
        if feature_power != 1.0 and similarity != "cosine":
            raise ValueError("feature_power is only supported for cosine similarity")

        prototype_result = self.forward(enzyme_embeddings, return_details=True)
        prototypes, prototype_details = prototype_result
        reaction_features = reaction_embeddings.float()
        prototype_features = prototypes.float()
        if similarity == "cosine":
            reaction_features = signed_power_transform(
                reaction_features,
                feature_power,
            )
            prototype_features = signed_power_transform(
                prototype_features,
                feature_power,
            )
            reaction_features = F.normalize(
                reaction_features,
                p=2,
                dim=-1,
                eps=1e-12,
            )
            prototype_features = F.normalize(
                prototype_features,
                p=2,
                dim=-1,
                eps=1e-12,
            )
        component_scores = torch.einsum(
            "qd,tkd->qtk",
            reaction_features,
            prototype_features,
        )
        temperature = self.aggregation_temperature
        routing_logits = (
            component_scores / temperature
            + prototype_details["log_priors"].float().unsqueeze(0)
        )
        scores = temperature * torch.logsumexp(routing_logits, dim=-1)
        if not return_details:
            return scores
        return scores, {
            **prototype_details,
            "component_scores": component_scores,
            "responsibilities": torch.softmax(routing_logits, dim=-1),
        }


class GatedEnzymeFeatureFusion(nn.Module):
    """
    Fuse complementary enzyme representations with trainable per-sample gates.

    The module projects raw sequence mean, SLEEC/attention-pooled residue
    features, frozen hyperbolic tangent features, and optionally a pretrained
    capability vector into a shared output space, then computes a softmax gate.
    """

    def __init__(
        self,
        raw_dim: int,
        pooled_dim: int,
        hyperbolic_dim: int,
        output_dim: int,
        capability_dim: int | None = None,
        gate_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in {
            "raw_dim": raw_dim,
            "pooled_dim": pooled_dim,
            "hyperbolic_dim": hyperbolic_dim,
            "output_dim": output_dim,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if gate_hidden_dim is not None and gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive or None")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in the range [0, 1]")

        self.raw_dim = int(raw_dim)
        self.pooled_dim = int(pooled_dim)
        self.hyperbolic_dim = int(hyperbolic_dim)
        self.capability_dim = None if capability_dim is None else int(capability_dim)
        self.output_dim = int(output_dim)
        gate_hidden_dim = self.output_dim if gate_hidden_dim is None else int(gate_hidden_dim)

        self.raw_projection = nn.Linear(self.raw_dim, self.output_dim)
        self.pooled_projection = nn.Linear(self.pooled_dim, self.output_dim)
        self.hyperbolic_projection = nn.Linear(self.hyperbolic_dim, self.output_dim)
        self.capability_projection = (
            None if self.capability_dim is None else nn.Linear(self.capability_dim, self.output_dim)
        )
        gate_input_dim = self.raw_dim + self.pooled_dim + self.hyperbolic_dim
        gate_output_dim = 3
        if self.capability_dim is not None:
            gate_input_dim += self.capability_dim
            gate_output_dim += 1
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(gate_hidden_dim, gate_output_dim),
        )

    def _validate_feature(self, name: str, value: torch.Tensor, expected_dim: int) -> None:
        if value.ndim != 2:
            raise ValueError(f"{name} must be rank-2, got shape={tuple(value.shape)}")
        if value.shape[-1] != expected_dim:
            raise ValueError(f"{name} final dim must be {expected_dim}, got {value.shape[-1]}")

    def forward(
        self,
        raw_mean: torch.Tensor,
        pooled: torch.Tensor,
        hyperbolic_tangent: torch.Tensor,
        capability_vector: torch.Tensor | None = None,
        capability_mask: torch.Tensor | None = None,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_feature("raw_mean", raw_mean, self.raw_dim)
        self._validate_feature("pooled", pooled, self.pooled_dim)
        self._validate_feature(
            "hyperbolic_tangent",
            hyperbolic_tangent,
            self.hyperbolic_dim,
        )
        if raw_mean.shape[0] != pooled.shape[0] or raw_mean.shape[0] != hyperbolic_tangent.shape[0]:
            raise ValueError("All enzyme fusion inputs must have the same batch size")
        if self.capability_dim is not None:
            if capability_vector is None:
                raise ValueError("capability_vector is required for capability fusion")
            self._validate_feature("capability_vector", capability_vector, self.capability_dim)
            if raw_mean.shape[0] != capability_vector.shape[0]:
                raise ValueError("Capability vectors must have the same batch size")
            if capability_mask is not None:
                if capability_mask.ndim > 1:
                    capability_mask = capability_mask.reshape(capability_mask.shape[0], -1).any(
                        dim=1
                    )
                if capability_mask.shape != (raw_mean.shape[0],):
                    raise ValueError(
                        "capability_mask must have shape [batch], got "
                        f"{tuple(capability_mask.shape)}"
                    )
        elif capability_vector is not None:
            raise ValueError("capability_vector was provided but capability_dim is not configured")
        elif capability_mask is not None:
            raise ValueError("capability_mask was provided but capability_dim is not configured")

        gate_inputs_list = [raw_mean, pooled, hyperbolic_tangent]
        projected_list = [
            self.raw_projection(raw_mean),
            self.pooled_projection(pooled),
            self.hyperbolic_projection(hyperbolic_tangent),
        ]
        if capability_vector is not None:
            gate_inputs_list.append(capability_vector)
            if self.capability_projection is None:
                raise RuntimeError("capability_projection was not initialized")
            projected_list.append(self.capability_projection(capability_vector))
        gate_inputs = torch.cat(gate_inputs_list, dim=-1)
        gate_logits = self.gate(gate_inputs)
        if capability_mask is not None:
            branch_mask = torch.ones(
                gate_logits.shape,
                dtype=torch.bool,
                device=gate_logits.device,
            )
            branch_mask[:, -1] = capability_mask.to(device=gate_logits.device, dtype=torch.bool)
            gate_logits = gate_logits.masked_fill(~branch_mask, torch.finfo(gate_logits.dtype).min)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        projected = torch.stack(projected_list, dim=1)
        fused = torch.sum(gate_weights.unsqueeze(-1) * projected, dim=1)
        if return_gates:
            return fused, gate_weights
        return fused


class BlockwiseEnzymeFeatureFusion(nn.Module):
    """
    Build an explicitly partitioned enzyme embedding from independent branches.

    Each branch is projected into its own subspace, L2-normalized, weighted by
    ``sqrt(weight)``, concatenated, and normalized again. This preserves the
    intended latent allocation better than a weighted sum over branch vectors.
    """

    BRANCH_ORDER = ("core", "site", "ec", "capability", "cofactor", "center", "transition")
    FACTORIZED_CAPABILITY_BRANCHES = ("cofactor", "center", "transition")

    def __init__(
        self,
        raw_dim: int,
        pooled_dim: int,
        output_dim: int,
        hyperbolic_dim: int | None = None,
        capability_dim: int | None = None,
        factorized_capability_dims: dict[str, int] | None = None,
        factorized_capability_use_masks: bool = True,
        block_dims: dict[str, int] | None = None,
        block_weights: dict[str, float] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if raw_dim <= 0 or pooled_dim <= 0 or output_dim <= 0:
            raise ValueError("raw_dim, pooled_dim, and output_dim must be positive")
        if hyperbolic_dim is not None and hyperbolic_dim <= 0:
            raise ValueError("hyperbolic_dim must be positive or None")
        if capability_dim is not None and capability_dim <= 0:
            raise ValueError("capability_dim must be positive or None")
        factorized_capability_dims = (
            None
            if factorized_capability_dims is None
            else {str(name): int(dim) for name, dim in factorized_capability_dims.items()}
        )
        if factorized_capability_dims is not None:
            missing_factor_dims = [
                name
                for name in self.FACTORIZED_CAPABILITY_BRANCHES
                if name not in factorized_capability_dims
            ]
            if missing_factor_dims:
                raise ValueError(
                    "Missing factorized capability input dimensions: " f"{missing_factor_dims}"
                )
            unexpected_factor_dims = sorted(
                set(factorized_capability_dims) - set(self.FACTORIZED_CAPABILITY_BRANCHES)
            )
            if unexpected_factor_dims:
                raise ValueError(
                    "Unexpected factorized capability input dimensions: "
                    f"{unexpected_factor_dims}"
                )
            for name, dim in factorized_capability_dims.items():
                if dim <= 0:
                    raise ValueError(f"factorized capability dim for {name} must be positive")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in the range [0, 1]")

        default_dims = {"core": output_dim}
        if hyperbolic_dim is None and capability_dim is None:
            default_dims = {"core": int(round(output_dim * 0.75)), "site": output_dim}
            default_dims["site"] -= default_dims["core"]
        elif capability_dim is None and factorized_capability_dims is None:
            default_dims = {"core": 224, "site": 96, "ec": output_dim - 320}
        elif factorized_capability_dims is None:
            default_dims = {
                "core": 224,
                "site": 96,
                "ec": 96,
                "capability": output_dim - 416,
            }
        else:
            default_dims = {
                "core": 192,
                "site": 80,
                "ec": 80,
                "cofactor": 56,
                "center": 48,
                "transition": output_dim - 456,
            }
        block_dims = dict(default_dims if block_dims is None else block_dims)
        active = ["core", "site"]
        if hyperbolic_dim is not None:
            active.append("ec")
        if capability_dim is not None:
            active.append("capability")
        if factorized_capability_dims is not None:
            active.extend(self.FACTORIZED_CAPABILITY_BRANCHES)
        unexpected = sorted(set(block_dims) - set(active))
        if unexpected:
            raise ValueError(f"Unexpected enzyme block dimensions: {unexpected}")
        missing = [name for name in active if name not in block_dims]
        if missing:
            raise ValueError(f"Missing enzyme block dimensions: {missing}")
        for name in active:
            if not isinstance(block_dims[name], int) or block_dims[name] <= 0:
                raise ValueError(f"Block dimension for {name} must be a positive integer")
        total_dim = sum(block_dims[name] for name in active)
        if total_dim != output_dim:
            raise ValueError(
                "Enzyme block dimensions must sum to output_dim: "
                f"sum={total_dim}, output_dim={output_dim}"
            )

        default_weights = {name: 1.0 / len(active) for name in active}
        block_weights = dict(default_weights if block_weights is None else block_weights)
        missing_weights = [name for name in active if name not in block_weights]
        if missing_weights:
            raise ValueError(f"Missing enzyme block weights: {missing_weights}")
        weight_sum = 0.0
        for name in active:
            weight = float(block_weights[name])
            if weight < 0.0:
                raise ValueError(f"Block weight for {name} must be non-negative")
            weight_sum += weight
        if weight_sum <= 0.0:
            raise ValueError("At least one enzyme block weight must be positive")

        self.raw_dim = int(raw_dim)
        self.pooled_dim = int(pooled_dim)
        self.hyperbolic_dim = None if hyperbolic_dim is None else int(hyperbolic_dim)
        self.capability_dim = None if capability_dim is None else int(capability_dim)
        self.factorized_capability_dims = factorized_capability_dims
        self.factorized_capability_use_masks = bool(factorized_capability_use_masks)
        self.output_dim = int(output_dim)
        self.active_branches = tuple(active)
        self.block_dims = {name: int(block_dims[name]) for name in active}
        self.block_weights = {name: float(block_weights[name]) / weight_sum for name in active}
        self.projections = nn.ModuleDict(
            {
                "core": self._projection(self.raw_dim, self.block_dims["core"], dropout),
                "site": self._projection(self.pooled_dim, self.block_dims["site"], dropout),
            }
        )
        if self.hyperbolic_dim is not None:
            self.projections["ec"] = self._projection(
                self.hyperbolic_dim,
                self.block_dims["ec"],
                dropout,
            )
        if self.capability_dim is not None:
            self.projections["capability"] = self._projection(
                self.capability_dim,
                self.block_dims["capability"],
                dropout,
            )
        if self.factorized_capability_dims is not None:
            for name in self.FACTORIZED_CAPABILITY_BRANCHES:
                self.projections[name] = self._projection(
                    self.factorized_capability_dims[name],
                    self.block_dims[name],
                    dropout,
                )

    @staticmethod
    def _projection(input_dim: int, output_dim: int, dropout: float) -> nn.Module:
        layers: list[nn.Module] = [nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim)]
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        return nn.Sequential(*layers)

    @staticmethod
    def _validate_feature(name: str, value: torch.Tensor, expected_dim: int) -> None:
        if value.ndim != 2:
            raise ValueError(f"{name} must be rank-2, got shape={tuple(value.shape)}")
        if value.shape[-1] != expected_dim:
            raise ValueError(f"{name} final dim must be {expected_dim}, got {value.shape[-1]}")

    def forward(
        self,
        raw_mean: torch.Tensor,
        pooled: torch.Tensor,
        hyperbolic_tangent: torch.Tensor | None = None,
        capability_vector: torch.Tensor | None = None,
        capability_mask: torch.Tensor | None = None,
        factorized_capability_vectors: dict[str, torch.Tensor] | None = None,
        factorized_capability_masks: dict[str, torch.Tensor] | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_feature("raw_mean", raw_mean, self.raw_dim)
        self._validate_feature("pooled", pooled, self.pooled_dim)
        if raw_mean.shape[0] != pooled.shape[0]:
            raise ValueError("raw_mean and pooled must have the same batch size")
        inputs = {"core": raw_mean, "site": pooled}
        if "ec" in self.active_branches:
            if hyperbolic_tangent is None or self.hyperbolic_dim is None:
                raise ValueError("hyperbolic_tangent is required for the ec block")
            self._validate_feature("hyperbolic_tangent", hyperbolic_tangent, self.hyperbolic_dim)
            if hyperbolic_tangent.shape[0] != raw_mean.shape[0]:
                raise ValueError("hyperbolic_tangent batch size must match raw_mean")
            inputs["ec"] = hyperbolic_tangent
        elif hyperbolic_tangent is not None:
            raise ValueError("hyperbolic_tangent was provided but no ec block is configured")
        if "capability" in self.active_branches:
            if capability_vector is None or self.capability_dim is None:
                raise ValueError("capability_vector is required for the capability block")
            self._validate_feature("capability_vector", capability_vector, self.capability_dim)
            if capability_vector.shape[0] != raw_mean.shape[0]:
                raise ValueError("capability_vector batch size must match raw_mean")
            inputs["capability"] = capability_vector
        elif capability_vector is not None:
            raise ValueError("capability_vector was provided but no capability block is configured")
        if self.factorized_capability_dims is not None:
            if factorized_capability_vectors is None:
                raise ValueError("factorized capability vectors are required")
            for name in self.FACTORIZED_CAPABILITY_BRANCHES:
                value = factorized_capability_vectors.get(name)
                if value is None:
                    raise ValueError(f"factorized capability vector '{name}' is required")
                self._validate_feature(
                    f"capability_{name}_vector",
                    value,
                    self.factorized_capability_dims[name],
                )
                if value.shape[0] != raw_mean.shape[0]:
                    raise ValueError(f"capability_{name}_vector batch size must match raw_mean")
                inputs[name] = value
        elif factorized_capability_vectors:
            raise ValueError(
                "factorized capability vectors were provided but no factorized "
                "capability block is configured"
            )

        details: dict[str, torch.Tensor] = {}
        blocks = []
        for name in self.active_branches:
            projected = self.projections[name](inputs[name])
            projected = F.normalize(projected, p=2, dim=-1, eps=1e-12)
            if name == "capability" and capability_mask is not None:
                mask = capability_mask
                if mask.ndim > 1:
                    mask = mask.reshape(mask.shape[0], -1).any(dim=1)
                if mask.shape != (raw_mean.shape[0],):
                    raise ValueError(
                        "capability_mask must have shape [batch], got " f"{tuple(mask.shape)}"
                    )
                projected = projected * mask.to(
                    device=projected.device, dtype=projected.dtype
                ).unsqueeze(-1)
                if return_details:
                    details["enzyme_block_active_capability"] = mask.to(dtype=projected.dtype)
            if (
                name in self.FACTORIZED_CAPABILITY_BRANCHES
                and factorized_capability_masks is not None
                and self.factorized_capability_use_masks
            ):
                mask = factorized_capability_masks.get(name)
                if mask is not None:
                    if mask.ndim > 1:
                        mask = mask.reshape(mask.shape[0], -1).any(dim=1)
                    if mask.shape != (raw_mean.shape[0],):
                        raise ValueError(
                            f"capability_{name}_mask must have shape [batch], got "
                            f"{tuple(mask.shape)}"
                        )
                    projected = projected * mask.to(
                        device=projected.device,
                        dtype=projected.dtype,
                    ).unsqueeze(-1)
                    if return_details:
                        details[f"enzyme_block_active_{name}"] = mask.to(dtype=projected.dtype)
            weighted = projected * (self.block_weights[name] ** 0.5)
            blocks.append(weighted)
            if return_details:
                details[f"enzyme_block_norm_{name}"] = projected.norm(dim=-1)
                details[f"enzyme_block_weight_{name}"] = torch.full(
                    (raw_mean.shape[0],),
                    self.block_weights[name],
                    dtype=projected.dtype,
                    device=projected.device,
                )
        fused = F.normalize(torch.cat(blocks, dim=-1), p=2, dim=-1, eps=1e-12)
        if return_details:
            return fused, details
        return fused


class EnzymeBioFPSplitEncoder(nn.Module):
    """No-Lorentz enzyme encoder with an inline predicted BioFP subspace."""

    FAMILIES = ("center", "cofactor", "transition")

    def __init__(
        self,
        residue_dim: int,
        output_dim: int = 512,
        seq_dim: int = 384,
        bio_dim: int = 128,
        hidden_dim: int = 512,
        family_output_dims: dict[str, int] | None = None,
        seq_weight: float = 0.75,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if output_dim <= 0 or seq_dim <= 0 or bio_dim <= 0 or hidden_dim <= 0:
            raise ValueError("BioFP split dimensions must be positive")
        if seq_dim + bio_dim != output_dim:
            raise ValueError(
                "BioFP split dimensions must sum to output_dim: "
                f"seq_dim={seq_dim}, bio_dim={bio_dim}, output_dim={output_dim}"
            )
        if not (0.0 <= seq_weight <= 1.0):
            raise ValueError("seq_weight must be in [0, 1]")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in [0, 1]")
        family_output_dims = family_output_dims or {}
        for family in self.FAMILIES:
            value = int(family_output_dims.get(family, 0))
            if value < 0:
                raise ValueError(f"BioFP output dim for {family} must be non-negative")

        self.residue_dim = int(residue_dim)
        self.output_dim = int(output_dim)
        self.seq_dim = int(seq_dim)
        self.bio_dim = int(bio_dim)
        self.hidden_dim = int(hidden_dim)
        self.seq_weight = float(seq_weight)
        self.bio_weight = 1.0 - self.seq_weight

        self.raw_projection = nn.Sequential(
            nn.LayerNorm(self.residue_dim),
            nn.Linear(self.residue_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.pooled_projection = nn.Sequential(
            nn.LayerNorm(self.residue_dim),
            nn.Linear(self.residue_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.sequence_gate = nn.Sequential(
            nn.Linear(2 * self.residue_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 2),
        )
        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.seq_dim),
        )

        self.residue_adapter = nn.Sequential(
            nn.LayerNorm(self.residue_dim),
            nn.Linear(self.residue_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.bio_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bio_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bio_queries = nn.Parameter(torch.empty(len(self.FAMILIES), self.hidden_dim))
        self.bio_dropout = nn.Dropout(float(dropout))
        self.bio_trunk = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * len(self.FAMILIES)),
            nn.Linear(self.hidden_dim * len(self.FAMILIES), self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.bio_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.bio_dim),
        )
        self.biofp_heads = nn.ModuleDict(
            {
                family: nn.Linear(self.hidden_dim, int(family_output_dims.get(family, 0)))
                for family in self.FAMILIES
                if int(family_output_dims.get(family, 0)) > 0
            }
        )
        nn.init.normal_(self.bio_queries, mean=0.0, std=self.hidden_dim**-0.5)

    def _validate_feature(self, name: str, value: torch.Tensor) -> None:
        if value.ndim != 2:
            raise ValueError(f"{name} must be rank-2, got shape={tuple(value.shape)}")
        if value.shape[-1] != self.residue_dim:
            raise ValueError(f"{name} final dim must be {self.residue_dim}, got {value.shape[-1]}")

    def forward(
        self,
        raw_mean: torch.Tensor,
        pooled: torch.Tensor,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_feature("raw_mean", raw_mean)
        self._validate_feature("pooled", pooled)
        if residue_embeddings.ndim != 3 or residue_embeddings.shape[-1] != self.residue_dim:
            raise ValueError(
                "residue_embeddings must have shape [batch, residues, residue_dim], "
                f"got {tuple(residue_embeddings.shape)}"
            )
        if raw_mean.shape[0] != pooled.shape[0] or raw_mean.shape[0] != residue_embeddings.shape[0]:
            raise ValueError("All BioFP encoder inputs must have the same batch size")

        raw_token = self.raw_projection(raw_mean)
        pooled_token = self.pooled_projection(pooled)
        gate_logits = self.sequence_gate(torch.cat([raw_mean, pooled], dim=-1))
        gate_weights = torch.softmax(gate_logits, dim=-1)
        sequence_feature = gate_weights[:, :1] * raw_token + gate_weights[:, 1:2] * pooled_token
        z_seq = F.normalize(self.sequence_projection(sequence_feature), p=2, dim=-1, eps=1e-12)

        adapted = self.residue_adapter(residue_embeddings.float())
        keys = self.bio_key(adapted)
        values = self.bio_value(adapted)
        scores = torch.einsum("bld,fd->bfl", keys, self.bio_queries) / (self.hidden_dim**0.5)
        if residue_padding_mask is not None:
            if residue_padding_mask.shape != residue_embeddings.shape[:2]:
                raise ValueError(
                    "residue_padding_mask must match residue embeddings first two dims: "
                    f"mask={tuple(residue_padding_mask.shape)}, "
                    f"residues={tuple(residue_embeddings.shape)}"
                )
            scores = scores.masked_fill(
                residue_padding_mask.to(device=scores.device, dtype=torch.bool).unsqueeze(1),
                torch.finfo(scores.dtype).min,
            )
        attention = torch.softmax(scores, dim=-1)
        pooled_families = torch.einsum("bfl,bld->bfd", self.bio_dropout(attention), values)
        bio_hidden = self.bio_trunk(pooled_families.reshape(pooled_families.shape[0], -1))
        z_bio = F.normalize(self.bio_projection(bio_hidden), p=2, dim=-1, eps=1e-12)
        seq_scale = self.seq_weight**0.5
        bio_scale = self.bio_weight**0.5
        output = F.normalize(
            torch.cat([seq_scale * z_seq, bio_scale * z_bio], dim=-1),
            p=2,
            dim=-1,
            eps=1e-12,
        )

        if not return_details:
            return output
        details: dict[str, torch.Tensor] = {
            "biofp_sequence_gate_raw_mean": gate_weights[:, 0],
            "biofp_sequence_gate_pooled": gate_weights[:, 1],
            "biofp_seq_norm": z_seq.norm(dim=-1),
            "biofp_latent_norm": z_bio.norm(dim=-1),
        }
        family_entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(
            dim=-1
        )
        for idx, family in enumerate(self.FAMILIES):
            details[f"biofp_attention_entropy_{family}"] = family_entropy[:, idx]
            if family in self.biofp_heads:
                details[f"biofp_logits_{family}"] = self.biofp_heads[family](bio_hidden)
        return output, details


class TigerTextGatedFusion(nn.Module):
    """Fuse sequence-derived enzyme features with TIGER-style text semantics."""

    def __init__(
        self,
        sequence_dim: int,
        text_dim: int,
        output_dim: int,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in {
            "sequence_dim": sequence_dim,
            "text_dim": text_dim,
            "output_dim": output_dim,
            "fusion_dim": fusion_dim,
            "num_heads": num_heads,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if fusion_dim % num_heads != 0:
            raise ValueError("fusion_dim must be divisible by num_heads")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in the range [0, 1]")

        self.sequence_dim = int(sequence_dim)
        self.text_dim = int(text_dim)
        self.output_dim = int(output_dim)
        self.fusion_dim = int(fusion_dim)

        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(self.sequence_dim),
            nn.Linear(self.sequence_dim, self.fusion_dim),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.fusion_dim),
        )
        self.sequence_to_text_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.text_to_sequence_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.gate = nn.Linear(self.fusion_dim * 2, self.fusion_dim)
        self.fusion_ffn = nn.Sequential(
            nn.LayerNorm(self.fusion_dim * 2),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.fusion_dim, self.output_dim),
        )
        self.sequence_residual = (
            nn.Identity()
            if self.sequence_dim == self.output_dim
            else nn.Linear(self.sequence_dim, self.output_dim)
        )

        final_linear = self.fusion_ffn[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def _validate_feature(self, name: str, value: torch.Tensor, expected_dim: int) -> None:
        if value.ndim != 2:
            raise ValueError(f"{name} must be rank-2, got shape={tuple(value.shape)}")
        if value.shape[-1] != expected_dim:
            raise ValueError(f"{name} final dim must be {expected_dim}, got {value.shape[-1]}")

    def forward(
        self,
        sequence_feature: torch.Tensor,
        text_vector: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_feature("sequence_feature", sequence_feature, self.sequence_dim)
        self._validate_feature("text_vector", text_vector, self.text_dim)
        if sequence_feature.shape[0] != text_vector.shape[0]:
            raise ValueError("sequence_feature and text_vector must have the same batch size")

        residual = self.sequence_residual(sequence_feature)
        if text_mask is None:
            valid_text = torch.ones(
                sequence_feature.shape[0],
                dtype=torch.bool,
                device=sequence_feature.device,
            )
        else:
            if text_mask.ndim > 1:
                text_mask = text_mask.reshape(text_mask.shape[0], -1).any(dim=1)
            if text_mask.shape != (sequence_feature.shape[0],):
                raise ValueError(
                    "text_mask must have shape [batch], got " f"{tuple(text_mask.shape)}"
                )
            valid_text = text_mask.to(device=sequence_feature.device, dtype=torch.bool)

        sequence_token = self.sequence_projection(sequence_feature).unsqueeze(1)
        text_token = self.text_projection(text_vector).unsqueeze(1)
        sequence_attended, _ = self.sequence_to_text_attention(
            sequence_token,
            text_token,
            text_token,
            need_weights=False,
        )
        text_attended, _ = self.text_to_sequence_attention(
            text_token,
            sequence_token,
            sequence_token,
            need_weights=False,
        )
        sequence_attended = sequence_attended.squeeze(1)
        text_attended = text_attended.squeeze(1)
        alpha = torch.sigmoid(self.gate(torch.cat([sequence_attended, text_attended], dim=-1)))
        gated = alpha * sequence_attended + (1.0 - alpha) * text_attended
        delta = self.fusion_ffn(torch.cat([gated, sequence_attended + text_attended], dim=-1))
        fused = residual + delta
        fused = torch.where(valid_text.unsqueeze(-1), fused, residual)

        if return_details:
            return fused, {
                "text_fusion_alpha": alpha,
                "text_fusion_has_text": valid_text.to(dtype=sequence_feature.dtype),
            }
        return fused


class ReactionFingerprintAttentionPool(nn.Module):
    """
    Trainable attention pooling over fixed reaction fingerprint views.

    The standard fingerprint representation is RDKitPlus structural bits
    followed by DRFP bits. RDKitPlus structural fingerprints are split into
    reactant and product halves, giving three attention tokens:
    RDKit reactants, RDKit products, and DRFP.
    """

    token_names = ("rdkit_reactants", "rdkit_products", "drfp")

    def __init__(
        self,
        input_dim: int = 2048,
        rdkit_dim: int = 1024,
        drfp_dim: int = 1024,
        token_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if rdkit_dim <= 0 or drfp_dim <= 0:
            raise ValueError("rdkit_dim and drfp_dim must be positive")
        if rdkit_dim % 2 != 0:
            raise ValueError("rdkit_dim must be even")
        if input_dim != rdkit_dim + drfp_dim:
            raise ValueError(
                "input_dim must equal rdkit_dim + drfp_dim: "
                f"{input_dim} != {rdkit_dim} + {drfp_dim}"
            )
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in the range [0, 1]")

        self.input_dim = int(input_dim)
        self.rdkit_dim = int(rdkit_dim)
        self.drfp_dim = int(drfp_dim)
        self.rdkit_half_dim = self.rdkit_dim // 2
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)

        self.reactant_projection = nn.Linear(self.rdkit_half_dim, self.token_dim)
        self.product_projection = nn.Linear(self.rdkit_half_dim, self.token_dim)
        self.drfp_projection = nn.Linear(self.drfp_dim, self.token_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.token_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1, bias=attention_bias),
        )

    def forward(
        self,
        fingerprints: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if fingerprints.ndim != 2 or fingerprints.shape[-1] != self.input_dim:
            raise ValueError(
                f"fingerprints must have shape [B, {self.input_dim}], "
                f"got {tuple(fingerprints.shape)}"
            )

        rdkit = fingerprints[:, : self.rdkit_dim]
        drfp = fingerprints[:, self.rdkit_dim :]
        reactants = rdkit[:, : self.rdkit_half_dim]
        products = rdkit[:, self.rdkit_half_dim :]
        tokens = torch.stack(
            [
                self.reactant_projection(reactants),
                self.product_projection(products),
                self.drfp_projection(drfp),
            ],
            dim=1,
        )
        logits = self.attention(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.einsum("bt,btd->bd", weights, tokens)
        if not return_details:
            return pooled

        entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=1)
        details = {
            "tokens": tokens,
            "logits": logits,
            "weights": weights,
            "entropy": entropy,
            "weight_rdkit_reactants": weights[:, 0],
            "weight_rdkit_products": weights[:, 1],
            "weight_drfp": weights[:, 2],
        }
        return pooled, details


class UniMol2ReactionAttentionEncoder(BaseModel):
    """
    Encode a reaction from frozen Uni-Mol2 molecule embeddings.

    Reactant and product molecule sets are pooled with learned masked attention
    or masked mean pooling, then composed as
    ``[x_R; x_P; x_P - x_R; abs(x_P - x_R)]`` before a standard Horizyn MLP maps
    the reaction to the shared contrastive embedding space.
    """

    supports_attention_return = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        unimol_dim: int = 768,
        reaction_pooling: str = "attention",
        attention_bias: bool = True,
        separate_side_poolers: bool = True,
        return_attention: bool = False,
        activations: nn.Module | list[nn.Module] = nn.ReLU(),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalise_output: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        expected_input_dim = 4 * unimol_dim
        if input_dim != expected_input_dim:
            raise ValueError(
                f"UniMol2 reaction attention input_dim must be 4 * unimol_dim "
                f"({expected_input_dim}), got {input_dim}"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.unimol_dim = unimol_dim
        if reaction_pooling not in {"attention", "mean"}:
            raise ValueError("reaction_pooling must be one of: attention, mean")
        self.reaction_pooling = reaction_pooling
        self.separate_side_poolers = separate_side_poolers
        self.return_attention = return_attention

        if reaction_pooling == "attention":
            self.reactant_pooling = MoleculeSetAttentionPooling(
                hidden_dim=unimol_dim,
                attention_bias=attention_bias,
                return_attention=return_attention,
            )
            product_pooler = MoleculeSetAttentionPooling(
                hidden_dim=unimol_dim,
                attention_bias=attention_bias,
                return_attention=return_attention,
            )
        else:
            self.reactant_pooling = MoleculeSetMeanPooling(return_attention=return_attention)
            product_pooler = MoleculeSetMeanPooling(return_attention=return_attention)

        if separate_side_poolers:
            self.product_pooling = product_pooler
        else:
            self.product_pooling = self.reactant_pooling

        self.composition_encoder = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            widths=widths,
            activations=activations,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            bias=bias,
            normalise_output=False,
        )
        if normalise_output:
            self.post_nn_layers.append(NormalizeLayer(p=2, dim=-1))

    def _pool_side(
        self,
        pooler: MoleculeSetAttentionPooling | MoleculeSetMeanPooling,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None,
        return_attention: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attention_mask = None if padding_mask is None else ~padding_mask.to(torch.bool)
        return pooler(
            embeddings,
            attention_mask=attention_mask,
            return_attention=return_attention,
        )

    def forward(
        self,
        reactant_embeddings: torch.Tensor,
        reactant_padding_mask: torch.Tensor | None = None,
        product_embeddings: torch.Tensor | None = None,
        product_padding_mask: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if product_embeddings is None:
            raise ValueError("product_embeddings is required")
        if reactant_embeddings.shape[-1] != self.unimol_dim:
            raise ValueError(
                f"Expected reactant embeddings dim {self.unimol_dim}, "
                f"got {reactant_embeddings.shape[-1]}"
            )
        if product_embeddings.shape[-1] != self.unimol_dim:
            raise ValueError(
                f"Expected product embeddings dim {self.unimol_dim}, "
                f"got {product_embeddings.shape[-1]}"
            )

        should_return_attention = (
            self.return_attention if return_attention is None else return_attention
        )
        reactant_result = self._pool_side(
            self.reactant_pooling,
            reactant_embeddings,
            reactant_padding_mask,
            should_return_attention,
        )
        product_result = self._pool_side(
            self.product_pooling,
            product_embeddings,
            product_padding_mask,
            should_return_attention,
        )

        if should_return_attention:
            reactant_pooled, reactant_attention = reactant_result
            product_pooled, product_attention = product_result
        else:
            reactant_pooled = reactant_result
            product_pooled = product_result
            reactant_attention = None
            product_attention = None

        delta = product_pooled - reactant_pooled
        composition = torch.cat(
            [reactant_pooled, product_pooled, delta, delta.abs()],
            dim=-1,
        )
        output = self.composition_encoder(composition)
        for layer in self.post_nn_layers:
            output = layer(output)

        if should_return_attention:
            return output, {
                "reactant": reactant_attention,
                "product": product_attention,
            }
        return output


class HybridReactionEncoder(BaseModel):
    """
    Encode reactions from complete fingerprints plus optional Uni-Mol2 molecule sets.

    The fingerprint branch is always used. The Uni-Mol2 branch is multiplied by
    ``has_unimol2`` before fusion so reactions missing from the Uni-Mol2 HDF5
    cleanly fall back to fingerprint-only features.
    """

    supports_attention_return = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        unimol_dim: int = 768,
        reaction_pooling: str = "attention",
        attention_bias: bool = True,
        separate_side_poolers: bool = True,
        fusion_widths: int | list[int] | None = None,
        return_attention: bool = False,
        activations: nn.Module | list[nn.Module] = nn.ReLU(),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalise_output: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.unimol_dim = unimol_dim
        if reaction_pooling not in {"attention", "mean"}:
            raise ValueError("reaction_pooling must be one of: attention, mean")
        self.reaction_pooling = reaction_pooling
        self.return_attention = return_attention

        self.fingerprint_encoder = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            widths=widths,
            activations=activations,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            bias=bias,
            normalise_output=False,
        )
        self.unimol_encoder = UniMol2ReactionAttentionEncoder(
            input_dim=4 * unimol_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            widths=widths,
            unimol_dim=unimol_dim,
            reaction_pooling=reaction_pooling,
            attention_bias=attention_bias,
            separate_side_poolers=separate_side_poolers,
            return_attention=return_attention,
            activations=activations,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            bias=bias,
            normalise_output=False,
        )

        if fusion_widths is None:
            fusion_widths = [output_dim]
        fusion_num_layers = 1 if isinstance(fusion_widths, int) else len(fusion_widths)
        self.fusion_encoder = MLP(
            input_dim=(2 * output_dim) + 1,
            output_dim=output_dim,
            num_layers=fusion_num_layers,
            widths=fusion_widths,
            activations=activations,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            bias=bias,
            normalise_output=False,
        )
        if normalise_output:
            self.post_nn_layers.append(NormalizeLayer(p=2, dim=-1))

    def forward(
        self,
        fingerprint: torch.Tensor,
        reactant_embeddings: torch.Tensor,
        reactant_padding_mask: torch.Tensor | None = None,
        product_embeddings: torch.Tensor | None = None,
        product_padding_mask: torch.Tensor | None = None,
        has_unimol2: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if fingerprint.ndim != 2:
            raise ValueError(f"fingerprint must be rank-2, got {tuple(fingerprint.shape)}")
        if fingerprint.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected fingerprint dim {self.input_dim}, got {fingerprint.shape[-1]}"
            )

        should_return_attention = (
            self.return_attention if return_attention is None else return_attention
        )
        fingerprint_embedding = self.fingerprint_encoder(fingerprint)
        unimol_result = self.unimol_encoder(
            reactant_embeddings=reactant_embeddings,
            reactant_padding_mask=reactant_padding_mask,
            product_embeddings=product_embeddings,
            product_padding_mask=product_padding_mask,
            return_attention=should_return_attention,
        )
        if should_return_attention:
            unimol_embedding, attention = unimol_result
        else:
            unimol_embedding = unimol_result
            attention = {}

        if has_unimol2 is None:
            has_unimol2 = torch.ones(
                fingerprint.shape[0],
                dtype=fingerprint.dtype,
                device=fingerprint.device,
            )
        has_unimol2 = has_unimol2.to(
            device=fingerprint.device,
            dtype=fingerprint.dtype,
        ).view(-1, 1)
        if has_unimol2.shape[0] != fingerprint.shape[0]:
            raise ValueError(
                "has_unimol2 leading dimension must match fingerprint batch: "
                f"{has_unimol2.shape[0]} vs {fingerprint.shape[0]}"
            )

        fused_input = torch.cat(
            [
                fingerprint_embedding,
                unimol_embedding * has_unimol2,
                has_unimol2,
            ],
            dim=-1,
        )
        output = self.fusion_encoder(fused_input)
        for layer in self.post_nn_layers:
            output = layer(output)

        if should_return_attention:
            return output, attention
        return output


class DualContrastiveModel(BaseModel):
    """
    Dual encoder contrastive learning model for Horizyn.

    This model uses separate encoders for query (reaction) and target (protein)
    inputs, producing normalized embeddings for contrastive learning. This is the
    core architecture of the Horizyn SOTA model.

    The model enforces that both encoders output normalized embeddings by checking
    for a NormalizeLayer as the final layer in each encoder.

    Notes:
        - Dict inputs: Dictionary inputs are forwarded to encoders via keyword
          arguments (i.e., encoder(**inputs)). This only works if the encoder
          classes accept those keyword arguments. The default `MLP` expects a
          tensor input and does not consume dicts.
        - Normalization enforcement: When `enforce_normalisation=True`, both
          encoders must end with a `NormalizeLayer`. Custom encoders should append
          `NormalizeLayer` as the last layer or disable enforcement explicitly.
    """

    def __init__(
        self,
        query_encoder_kwargs: dict[str, Any],
        target_encoder_kwargs: dict[str, Any],
        query_encoder: type[BaseModel] = MLP,
        target_encoder: type[BaseModel] = MLP,
        enforce_normalisation: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the DualContrastiveModel.

        Args:
            query_encoder_kwargs: Keyword arguments for query encoder (reactions).
            target_encoder_kwargs: Keyword arguments for target encoder (proteins).
            query_encoder: Query encoder class (default: MLP).
            target_encoder: Target encoder class (default: MLP).
            enforce_normalisation: Whether to enforce that encoders have normalized outputs.
            *args: Additional arguments (must be empty).
            **kwargs: Additional keyword arguments (must be empty).

        Raises:
            ValueError: If enforce_normalisation is True and encoders don't have
                NormalizeLayer as final layer.

        Example:
            >>> # SOTA configuration
            >>> model = DualContrastiveModel(
            ...     query_encoder_kwargs={
            ...         "input_dim": 2048,
            ...         "output_dim": 512,
            ...         "num_layers": 1,
            ...         "widths": 4096,
            ...         "normalise_output": True,
            ...     },
            ...     target_encoder_kwargs={
            ...         "input_dim": 1024,
            ...         "output_dim": 512,
            ...         "num_layers": 1,
            ...         "widths": 4096,
            ...         "normalise_output": True,
            ...     },
            ... )
        """
        super().__init__(*args, **kwargs)
        self.query_encoder = query_encoder(**query_encoder_kwargs)
        self.target_encoder = target_encoder(**target_encoder_kwargs)

        # Validate that encoders have normalized outputs
        if enforce_normalisation:
            if not isinstance(self.query_encoder, BaseModel):
                raise ValueError("query_encoder must be a BaseModel instance")
            if not isinstance(self.target_encoder, BaseModel):
                raise ValueError("target_encoder must be a BaseModel instance")

            # Check that query encoder has NormalizeLayer as last layer
            if len(self.query_encoder.layers) == 0 or not isinstance(
                self.query_encoder.layers[-1], NormalizeLayer
            ):
                raise ValueError(
                    "query_encoder must have a NormalizeLayer as its last layer. "
                    "Set normalise_output=True in query_encoder_kwargs."
                )

            # Check that target encoder has NormalizeLayer as last layer
            if len(self.target_encoder.layers) == 0 or not isinstance(
                self.target_encoder.layers[-1], NormalizeLayer
            ):
                raise ValueError(
                    "target_encoder must have a NormalizeLayer as its last layer. "
                    "Set normalise_output=True in target_encoder_kwargs."
                )

    def forward(
        self, query_inputs: dict | torch.Tensor, target_inputs: dict | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the dual contrastive model.

        Encodes both query and target inputs through their respective encoders.
        Supports both tensor and dict inputs for flexibility.

        Args:
            query_inputs: Input for query encoder (reactions).
                Can be a tensor or dict of tensors.
            target_inputs: Input for target encoder (proteins).
                Can be a tensor or dict of tensors.

        Returns:
            Tuple of (query_embeddings, target_embeddings), both normalized.

        Raises:
            ValueError: If outputs are not rank-2 or if feature dimensions differ.

        Example:
            >>> query_fps = torch.randn(16, 2048)  # Reaction fingerprints
            >>> target_embs = torch.randn(16, 1024)  # Protein T5 embeddings
            >>> query_out, target_out = model(query_fps, target_embs)
            >>> query_out.shape, target_out.shape
            (torch.Size([16, 512]), torch.Size([16, 512]))
        """
        # Encode query inputs
        query = (
            self.query_encoder(**query_inputs)
            if isinstance(query_inputs, dict)
            else self.query_encoder(query_inputs)
        )

        # Encode target inputs
        target = (
            self.target_encoder(**target_inputs)
            if isinstance(target_inputs, dict)
            else self.target_encoder(target_inputs)
        )

        # Validate output ranks
        if query.ndim != 2 or target.ndim != 2:
            raise ValueError(
                f"Encoders must return rank-2 tensors (batch, features): "
                f"query.shape={tuple(query.shape)}, target.shape={tuple(target.shape)}"
            )

        # Validate output dimensions match
        if query.shape[1] != target.shape[1]:
            raise ValueError(
                f"Query and target encoder output shape mismatch: "
                f"query.shape={tuple(query.shape)} != target.shape={tuple(target.shape)}"
            )

        return query, target


class E2RReactionAdapter(nn.Module):
    """Residual reaction adapter used only for enzyme-to-reaction retrieval."""

    FACTORIZED_DIMS = {
        "reaction_model": 128,
        "unimol2": 192,
        "chiro": 96,
        "reaction_chemistry": 96,
    }
    FACTORIZED_WEIGHTS = {
        "reaction_model": 0.25,
        "unimol2": 0.375,
        "chiro": 0.1875,
        "reaction_chemistry": 0.1875,
    }

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        gate_init: float = 0.1,
        use_factorized_inputs: bool = False,
        use_directional_inputs: bool = False,
        reaction_model_dim: int = 768,
        unimol_dim: int = 768,
        chiro_dim: int = 256,
        chemistry_dim: int | None = None,
        directional_dim: int | None = None,
        directional_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be in the open interval (0, 1)")
        if use_factorized_inputs and (chemistry_dim is None or chemistry_dim <= 0):
            raise ValueError("factorized E2R inputs require a positive chemistry_dim")
        if use_directional_inputs and (directional_dim is None or directional_dim <= 0):
            raise ValueError("directional E2R inputs require a positive directional_dim")

        self.embedding_dim = int(embedding_dim)
        self.use_factorized_inputs = bool(use_factorized_inputs)
        self.use_directional_inputs = bool(use_directional_inputs)
        self.directional_dim = None if directional_dim is None else int(directional_dim)
        self.raw_gate = nn.Parameter(
            torch.logit(torch.tensor(float(gate_init), dtype=torch.float32))
        )

        self.factorized_projections = nn.ModuleDict()
        if self.use_factorized_inputs:
            input_dims = {
                "reaction_model": int(reaction_model_dim),
                "unimol2": int(unimol_dim),
                "chiro": int(chiro_dim),
                "reaction_chemistry": int(chemistry_dim),
            }
            for name, output_dim in self.FACTORIZED_DIMS.items():
                self.factorized_projections[name] = nn.Sequential(
                    nn.LayerNorm(input_dims[name]),
                    nn.Linear(input_dims[name], output_dim),
                    nn.GELU(),
                )
            self.register_buffer(
                "factorized_weights",
                torch.tensor(
                    [self.FACTORIZED_WEIGHTS[name] for name in self.FACTORIZED_DIMS],
                    dtype=torch.float32,
                ),
                persistent=True,
            )

        self.directional_projection: nn.Module | None = None
        adapter_input_dim = self.embedding_dim
        if self.use_factorized_inputs:
            adapter_input_dim += sum(self.FACTORIZED_DIMS.values())
        if self.use_directional_inputs:
            self.directional_projection = nn.Sequential(
                nn.LayerNorm(int(directional_dim)),
                nn.Linear(int(directional_dim), int(directional_hidden_dim)),
                nn.GELU(),
            )
            adapter_input_dim += int(directional_hidden_dim)

        self.delta = nn.Sequential(
            nn.LayerNorm(adapter_input_dim),
            nn.Linear(adapter_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.embedding_dim),
        )

    @staticmethod
    def _masked_set_mean(
        reactants: torch.Tensor,
        products: torch.Tensor,
        reactant_padding_mask: torch.Tensor | None,
        product_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if reactant_padding_mask is None:
            reactant_valid = torch.ones(
                reactants.shape[:2], dtype=torch.bool, device=reactants.device
            )
        else:
            reactant_valid = ~reactant_padding_mask.to(torch.bool)
        if product_padding_mask is None:
            product_valid = torch.ones(products.shape[:2], dtype=torch.bool, device=products.device)
        else:
            product_valid = ~product_padding_mask.to(torch.bool)
        total = (reactants * reactant_valid.unsqueeze(-1)).sum(dim=1) + (
            products * product_valid.unsqueeze(-1)
        ).sum(dim=1)
        count = reactant_valid.sum(dim=1) + product_valid.sum(dim=1)
        return total / count.clamp_min(1).unsqueeze(-1).to(total.dtype)

    def _factorized_input(self, query_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        required = {
            "reaction_embedding",
            "reactant_embeddings",
            "product_embeddings",
            "reactant_chirality_embeddings",
            "product_chirality_embeddings",
            "reaction_chemistry_vector",
        }
        missing = sorted(required - set(query_inputs))
        if missing:
            raise ValueError(f"Missing factorized E2R reaction inputs: {missing}")

        unimol = self._masked_set_mean(
            query_inputs["reactant_embeddings"],
            query_inputs["product_embeddings"],
            query_inputs.get("reactant_padding_mask"),
            query_inputs.get("product_padding_mask"),
        )
        chiro = self._masked_set_mean(
            query_inputs["reactant_chirality_embeddings"],
            query_inputs["product_chirality_embeddings"],
            query_inputs.get("reactant_chirality_padding_mask"),
            query_inputs.get("product_chirality_padding_mask"),
        )
        raw = {
            "reaction_model": query_inputs["reaction_embedding"],
            "unimol2": unimol,
            "chiro": chiro,
            "reaction_chemistry": query_inputs["reaction_chemistry_vector"],
        }
        batch_size = unimol.shape[0]
        device = unimol.device
        chirality_valid = query_inputs.get("has_chirality")
        if chirality_valid is None:
            chirality_valid = query_inputs.get("has_chiro")
        if chirality_valid is None:
            chirality_valid = query_inputs.get("has_chienn")
        valid = {
            "reaction_model": torch.ones(
                batch_size,
                dtype=torch.bool,
                device=device,
            ),
            "unimol2": query_inputs.get("has_unimol2"),
            "chiro": chirality_valid,
            "reaction_chemistry": query_inputs.get("has_reaction_chemistry"),
        }
        modality_mask = torch.stack(
            [
                (
                    torch.ones(batch_size, dtype=torch.bool, device=device)
                    if valid[name] is None
                    else valid[name].to(device=device, dtype=torch.bool)
                )
                for name in self.FACTORIZED_DIMS
            ],
            dim=1,
        )
        base_weights = self.factorized_weights.to(
            device=unimol.device,
            dtype=unimol.dtype,
        ).unsqueeze(0)
        weights = base_weights * modality_mask.to(base_weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        blocks = []
        for index, name in enumerate(self.FACTORIZED_DIMS):
            block = F.normalize(
                self.factorized_projections[name](raw[name]),
                p=2,
                dim=-1,
                eps=1e-12,
            )
            blocks.append(block * weights[:, index].sqrt().unsqueeze(-1))
        return torch.cat(blocks, dim=-1)

    def forward(
        self,
        base_embedding: torch.Tensor,
        query_inputs: dict[str, torch.Tensor] | torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts = [base_embedding]
        if self.use_factorized_inputs:
            if not isinstance(query_inputs, dict):
                raise ValueError("Factorized E2R adapter inputs must be a mapping")
            parts.append(self._factorized_input(query_inputs))
        if self.use_directional_inputs:
            if not isinstance(query_inputs, dict):
                raise ValueError("Directional E2R adapter inputs must be a mapping")
            directional = query_inputs.get("reaction_directional_vector")
            if directional is None or self.directional_projection is None:
                raise ValueError("reaction_directional_vector is required by the E2R adapter")
            directional = self.directional_projection(directional)
            directional_mask = query_inputs.get("has_reaction_directional")
            if directional_mask is not None:
                directional = directional * directional_mask.to(
                    device=directional.device,
                    dtype=directional.dtype,
                ).unsqueeze(-1)
            parts.append(directional)

        gate = torch.sigmoid(self.raw_gate)
        delta = self.delta(torch.cat(parts, dim=-1))
        adapted = F.normalize(
            base_embedding + gate * delta,
            p=2,
            dim=-1,
            eps=1e-12,
        )
        if not return_details:
            return adapted
        return adapted, {
            "e2r_adapter_gate": gate.expand(base_embedding.shape[0]),
            "e2r_adapter_delta_norm": delta.norm(dim=-1),
            "e2r_adapter_base_cosine": F.cosine_similarity(
                adapted,
                base_embedding,
                dim=-1,
            ),
        }


class R2EEnzymeAdapter(nn.Module):
    """Residual enzyme adapter used only for reaction-to-enzyme retrieval."""

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        gate_init: float = 0.1,
        use_factorized_inputs: bool = False,
        block_dims: dict[str, int] | None = None,
        block_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("R2E adapter dimensions must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("R2E adapter dropout must be in [0, 1]")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be in the open interval (0, 1)")

        self.embedding_dim = int(embedding_dim)
        self.use_factorized_inputs = bool(use_factorized_inputs)
        self.block_dims: dict[str, int] = {}
        self.block_projections = nn.ModuleDict()
        if self.use_factorized_inputs:
            if not block_dims or not block_weights:
                raise ValueError("Factorized R2E inputs require block_dims and block_weights")
            self.block_dims = {str(name): int(dim) for name, dim in block_dims.items()}
            if any(dim <= 0 for dim in self.block_dims.values()):
                raise ValueError("R2E adapter block dimensions must be positive")
            if sum(self.block_dims.values()) != self.embedding_dim:
                raise ValueError(
                    "R2E adapter blocks must partition the enzyme embedding: "
                    f"sum={sum(self.block_dims.values())}, embedding_dim={self.embedding_dim}"
                )
            if set(block_weights) != set(self.block_dims):
                raise ValueError("R2E adapter block weights must match block dimensions exactly")
            weight_sum = sum(float(block_weights[name]) for name in self.block_dims)
            if weight_sum <= 0.0 or any(
                float(block_weights[name]) < 0.0 for name in self.block_dims
            ):
                raise ValueError("R2E adapter block weights must be non-negative with positive sum")
            self.register_buffer(
                "block_weights",
                torch.tensor(
                    [float(block_weights[name]) / weight_sum for name in self.block_dims],
                    dtype=torch.float32,
                ),
                persistent=True,
            )
            for name, dim in self.block_dims.items():
                self.block_projections[name] = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.GELU(),
                )

        adapter_input_dim = self.embedding_dim * (2 if self.use_factorized_inputs else 1)
        self.raw_gate = nn.Parameter(
            torch.logit(torch.tensor(float(gate_init), dtype=torch.float32))
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(adapter_input_dim),
            nn.Linear(adapter_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.embedding_dim),
        )

    def _factorized_input(self, base_embedding: torch.Tensor) -> torch.Tensor:
        blocks = torch.split(base_embedding, list(self.block_dims.values()), dim=-1)
        weights = self.block_weights.to(
            device=base_embedding.device,
            dtype=base_embedding.dtype,
        )
        projected = []
        for index, (name, block) in enumerate(zip(self.block_dims, blocks)):
            value = F.normalize(
                self.block_projections[name](block),
                p=2,
                dim=-1,
                eps=1e-12,
            )
            projected.append(value * weights[index].sqrt())
        return torch.cat(projected, dim=-1)

    def forward(
        self,
        base_embedding: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if base_embedding.ndim != 2 or base_embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                "R2E adapter expects [batch, embedding_dim], got " f"{tuple(base_embedding.shape)}"
            )
        parts = [base_embedding]
        if self.use_factorized_inputs:
            parts.append(self._factorized_input(base_embedding))
        gate = torch.sigmoid(self.raw_gate)
        delta = self.delta(torch.cat(parts, dim=-1))
        adapted = F.normalize(
            base_embedding + gate * delta,
            p=2,
            dim=-1,
            eps=1e-12,
        )
        if not return_details:
            return adapted
        return adapted, {
            "r2e_adapter_gate": gate.expand(base_embedding.shape[0]),
            "r2e_adapter_delta_norm": delta.norm(dim=-1),
            "r2e_adapter_base_cosine": F.cosine_similarity(
                adapted,
                base_embedding,
                dim=-1,
            ),
        }


class ReactionConditionedAttentionPooling(nn.Module):
    """
    Reaction-conditioned single-head attention pooling over residue embeddings.

    For reaction embedding ``z_r`` and residue state ``h_i`` this computes:

        alpha_i = softmax_i(z_r^T W h_i)
        x_e(r) = sum_i alpha_i h_i

    When ``attention_rank`` is set, the compatibility score uses a low-rank
    factorization:

        alpha_i = softmax_i((A z_r)^T (B h_i))

    The weighted sum pools the original residue states by default. When
    ``value_dim`` is set, residue states are first projected into that value
    space and the weighted sum pools those projected values.
    """

    def __init__(
        self,
        residue_dim: int,
        embedding_dim: int,
        attention_rank: int | None = None,
        projection_bias: bool = False,
        value_dim: int | None = None,
        value_projection_bias: bool = False,
        return_attention: bool = False,
    ):
        super().__init__()
        self.residue_dim = residue_dim
        self.embedding_dim = embedding_dim
        self.attention_rank = attention_rank
        self.value_dim = value_dim
        self.return_attention = return_attention

        if attention_rank is not None and attention_rank <= 0:
            raise ValueError("attention_rank must be None or a positive integer")
        if value_dim is not None and value_dim <= 0:
            raise ValueError("value_dim must be None or a positive integer")

        compatibility_dim = embedding_dim if attention_rank is None else attention_rank
        self.reaction_projection = (
            None if attention_rank is None else nn.Linear(embedding_dim, attention_rank, bias=False)
        )
        self.residue_projection = nn.Linear(
            residue_dim,
            compatibility_dim,
            bias=projection_bias,
        )
        self.value_projection = (
            None
            if value_dim is None
            else nn.Linear(residue_dim, value_dim, bias=value_projection_bias)
        )

    def forward(
        self,
        reaction_embeddings: torch.Tensor,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor,
        return_attention: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Pool residues for each reaction-target combination.

        Args:
            reaction_embeddings: Tensor with shape ``[R, D]``.
            residue_embeddings: Tensor with shape ``[E, L, H]``.
            residue_padding_mask: Bool tensor with shape ``[E, L]`` where True
                denotes padding.
            return_attention: Override for whether to return attention weights.

        Returns:
            Pooled embeddings with shape ``[R, E, H]`` and optionally attention
            weights with shape ``[R, E, L]``.
        """
        if reaction_embeddings.ndim != 2:
            raise ValueError(
                f"reaction_embeddings must be rank-2, got {tuple(reaction_embeddings.shape)}"
            )
        if residue_embeddings.ndim != 3:
            raise ValueError(
                f"residue_embeddings must be rank-3, got {tuple(residue_embeddings.shape)}"
            )
        if residue_padding_mask.shape != residue_embeddings.shape[:2]:
            raise ValueError(
                "residue_padding_mask shape must match residue_embeddings first two dims: "
                f"mask={tuple(residue_padding_mask.shape)}, "
                f"residues={tuple(residue_embeddings.shape)}"
            )
        if residue_padding_mask.all(dim=1).any():
            raise ValueError("Each protein must have at least one unmasked residue")

        projected_reactions = (
            reaction_embeddings
            if self.reaction_projection is None
            else self.reaction_projection(reaction_embeddings)
        )
        projected_residues = self.residue_projection(residue_embeddings)
        scores = torch.einsum("rd,eld->rel", projected_reactions, projected_residues)
        scores = scores.masked_fill(residue_padding_mask.unsqueeze(0), float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attention = attention.masked_fill(residue_padding_mask.unsqueeze(0), 0.0)
        value_embeddings = (
            residue_embeddings
            if self.value_projection is None
            else self.value_projection(residue_embeddings)
        )
        pooled = torch.einsum("rel,elh->reh", attention, value_embeddings)

        should_return_attention = (
            self.return_attention if return_attention is None else return_attention
        )
        if should_return_attention:
            return pooled, attention
        return pooled


class ReactionConditionedDualModel(BaseModel):
    """
    Dual model that conditions protein pooling on each reaction embedding.

    The model produces a full score matrix rather than reusable target
    embeddings because each enzyme representation depends on the reaction.
    ``score_mode="target_mlp"`` preserves the original V1 architecture. The
    lighter ``score_mode="projected_value"`` skips the per-pair target MLP and
    scores normalized reaction embeddings directly against attention-pooled
    projected residue values.
    """

    def __init__(
        self,
        query_encoder_kwargs: dict[str, Any],
        target_encoder_kwargs: dict[str, Any],
        residue_dim: int = 1024,
        embedding_dim: int = 512,
        attention_rank: int | None = None,
        projection_bias: bool = False,
        score_mode: str = "target_mlp",
        value_projection_bias: bool = False,
        normalize_pooled_values: bool = True,
        return_attention: bool = False,
        query_encoder: type[BaseModel] = MLP,
        target_encoder: type[BaseModel] = MLP,
        enforce_normalisation: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if score_mode not in {"target_mlp", "projected_value"}:
            raise ValueError("score_mode must be one of: target_mlp, projected_value")

        self.score_mode = score_mode
        self.normalize_pooled_values = normalize_pooled_values
        self.query_encoder = query_encoder(**query_encoder_kwargs)
        self.target_encoder = (
            target_encoder(**target_encoder_kwargs) if score_mode == "target_mlp" else None
        )
        self.pooling = ReactionConditionedAttentionPooling(
            residue_dim=residue_dim,
            embedding_dim=embedding_dim,
            attention_rank=attention_rank,
            projection_bias=projection_bias,
            value_dim=embedding_dim if score_mode == "projected_value" else None,
            value_projection_bias=value_projection_bias,
            return_attention=return_attention,
        )
        self.residue_dim = residue_dim
        self.embedding_dim = embedding_dim

        if enforce_normalisation:
            self._validate_normalized_encoder(self.query_encoder, "query_encoder")
            if self.target_encoder is not None:
                self._validate_normalized_encoder(self.target_encoder, "target_encoder")

    @staticmethod
    def _validate_normalized_encoder(encoder: BaseModel, name: str) -> None:
        if not isinstance(encoder, BaseModel):
            raise ValueError(f"{name} must be a BaseModel instance")
        if len(encoder.layers) == 0 or not isinstance(encoder.layers[-1], NormalizeLayer):
            raise ValueError(
                f"{name} must have a NormalizeLayer as its last layer. "
                "Set normalise_output=True in encoder kwargs."
            )

    def encode_queries(self, query_inputs: dict | torch.Tensor) -> torch.Tensor:
        query_embeddings = (
            self.query_encoder(**query_inputs)
            if isinstance(query_inputs, dict)
            else self.query_encoder(query_inputs)
        )
        if query_embeddings.ndim != 2:
            raise ValueError(
                f"Query encoder must return rank-2 tensors, got {tuple(query_embeddings.shape)}"
            )
        return query_embeddings

    def score_matrix(
        self,
        query_inputs: dict | torch.Tensor,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Score every reaction against every target enzyme in the batch.

        Returns:
            Score matrix with shape ``[num_queries, num_targets]`` and optionally
            attention weights with shape ``[num_queries, num_targets, max_len]``.
        """
        query_embeddings = self.encode_queries(query_inputs)
        pooled_result = self.pooling(
            query_embeddings,
            residue_embeddings,
            residue_padding_mask,
            return_attention=return_attention,
        )
        if return_attention:
            pooled_residues, attention = pooled_result
        else:
            pooled_residues = pooled_result
            attention = None

        if self.score_mode == "projected_value":
            target_embeddings = pooled_residues
            if self.normalize_pooled_values:
                target_embeddings = F.normalize(target_embeddings, p=2, dim=-1, eps=1e-12)
        else:
            if self.target_encoder is None:
                raise RuntimeError("target_encoder is required for target_mlp score mode")
            num_queries, num_targets, residue_dim = pooled_residues.shape
            target_embeddings = self.target_encoder(pooled_residues.reshape(-1, residue_dim))
            target_embeddings = target_embeddings.reshape(num_queries, num_targets, -1)

        if target_embeddings.shape[-1] != query_embeddings.shape[-1]:
            raise ValueError(
                f"Query and target encoder output dims differ: "
                f"{query_embeddings.shape[-1]} vs {target_embeddings.shape[-1]}"
            )

        scores = torch.einsum("rd,red->re", query_embeddings, target_embeddings)
        if return_attention:
            return scores, attention
        return scores

    def forward(
        self,
        query_inputs: dict | torch.Tensor,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the reaction-conditioned score matrix."""
        return self.score_matrix(query_inputs, residue_embeddings, residue_padding_mask)
