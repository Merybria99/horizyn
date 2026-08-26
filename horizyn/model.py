"""
Neural network model architectures for Horizyn.

This module contains the base model classes and MLP implementation used in
the Horizyn contrastive learning model.
"""

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from horizyn.checkpoint_utils import extract_query_encoder_state_dict


class BaseModel(nn.Module):
    """
    Base class for all models in Horizyn.

    Provides a structured way to organize model layers into pre-processing,
    main body, and post-processing stages, along with optional output heads.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the base model.

        Args:
            *args: Variable length argument list (should be empty).
            **kwargs: Arbitrary keyword arguments (should be empty).

        Raises:
            ValueError: If extra arguments are provided.
        """
        super(BaseModel, self).__init__()
        if args or kwargs:
            error_msg = (
                f"Extra unused arguments provided to BaseModel: args={args}, kwargs={kwargs}"
            )
            raise ValueError(error_msg)

        # Define the pre-nn layers (preprocessing)
        self.pre_nn_layers = nn.ModuleList()
        # Define the main body of nn layers
        self.main_nn = nn.ModuleList()
        # Define the post-nn layers (postprocessing)
        self.post_nn_layers = nn.ModuleList()
        # Optional output heads for multi-task learning
        self.output_heads = nn.ModuleDict()

    @property
    def model_body(self) -> nn.ModuleList:
        """
        Get the main body of the model (all layers excluding output heads).

        Returns:
            ModuleList containing all pre-processing, main, and post-processing layers.
        """
        return nn.ModuleList([*self.pre_nn_layers, *self.main_nn, *self.post_nn_layers])

    @property
    def layers(self) -> nn.ModuleList:
        """
        Get all layers in the model including output heads.

        Returns:
            ModuleList containing all model layers.
        """
        return nn.ModuleList(
            [
                *self.pre_nn_layers,
                *self.main_nn,
                *self.post_nn_layers,
                *self.output_heads.values(),
            ]
        )

    @property
    def num_parameters(self) -> int:
        """
        Get the total number of parameters in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor, or dict of outputs if output heads are defined.
        """
        for layer in self.model_body:
            x = layer(x)
        # Handle multiple output heads if present
        if len(self.output_heads) > 0:
            return {key: head(x) for key, head in self.output_heads.items()}
        return x


class NormalizeLayer(nn.Module):
    """
    Normalization layer for L2 normalization of tensors.

    This layer normalizes input tensors along a specified dimension using the
    L2 norm (Euclidean distance). Commonly used to normalize embeddings in
    contrastive learning.
    """

    def __init__(self, p: float = 2, dim: int = -1, eps: float = 1e-12):
        """
        Initialize the NormalizeLayer.

        Args:
            p: The p-norm to use for normalization (default: 2 for L2 norm).
            dim: The dimension along which to compute the norm (default: -1, last dimension).
            eps: Small value for numerical stability (default: 1e-12).
        """
        super(NormalizeLayer, self).__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to the input tensor.

        Args:
            x: Input tensor to be normalized.

        Returns:
            Normalized tensor with unit norm along the specified dimension.
        """
        return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)

    def extra_repr(self) -> str:
        """
        Return string representation of layer parameters for printing.

        Returns:
            String describing layer configuration.
        """
        return f"p={self.p}, dim={self.dim}, eps={self.eps}"


class ProteinMeanPooling(nn.Module):
    """
    Masked mean pooling over residue-level protein embeddings.

    ``attention_mask`` uses True for valid residue positions and False for
    padding. The returned attention weights are uniform over valid residues,
    which gives the same interpretability interface as attention pooling.
    """

    def __init__(self, return_attention: bool = False):
        super().__init__()
        self.return_attention = return_attention

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residue_embeddings.ndim != 3:
            raise ValueError(
                "residue_embeddings must have shape [batch, seq_len, hidden_dim], "
                f"got {tuple(residue_embeddings.shape)}"
            )

        batch_size, seq_len, _ = residue_embeddings.shape
        if attention_mask is None:
            valid_mask = torch.ones(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=residue_embeddings.device,
            )
        else:
            if attention_mask.shape != residue_embeddings.shape[:2]:
                raise ValueError(
                    "attention_mask must match residue_embeddings first two dims: "
                    f"mask={tuple(attention_mask.shape)}, "
                    f"residues={tuple(residue_embeddings.shape)}"
                )
            valid_mask = attention_mask.to(dtype=torch.bool, device=residue_embeddings.device)

        if valid_mask.sum(dim=1).eq(0).any():
            raise ValueError("Each protein must have at least one valid residue")

        weights = valid_mask.to(dtype=residue_embeddings.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
        pooled = torch.einsum("bl,blh->bh", weights, residue_embeddings)

        should_return_attention = (
            self.return_attention if return_attention is None else return_attention
        )
        if should_return_attention:
            return pooled, weights
        return pooled


class ProteinAttentionPooling(nn.Module):
    """
    Learned attention pooling over residue-level protein embeddings.

    For residue embeddings ``h_i`` this computes:

        alpha_i = softmax_i(w^T h_i)
        x_e = sum_i alpha_i h_i

    ``attention_mask`` has shape ``[batch, seq_len]`` and uses True for valid
    residue positions. Masked/padded positions receive zero attention weight.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_bias: bool = True,
        return_attention: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.return_attention = return_attention
        self.attention = nn.Linear(hidden_dim, 1, bias=attention_bias)

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residue_embeddings.ndim != 3:
            raise ValueError(
                "residue_embeddings must have shape [batch, seq_len, hidden_dim], "
                f"got {tuple(residue_embeddings.shape)}"
            )
        if residue_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected residue hidden_dim={self.hidden_dim}, "
                f"got {residue_embeddings.shape[-1]}"
            )

        batch_size, seq_len, _ = residue_embeddings.shape
        if attention_mask is None:
            valid_mask = torch.ones(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=residue_embeddings.device,
            )
        else:
            if attention_mask.shape != residue_embeddings.shape[:2]:
                raise ValueError(
                    "attention_mask must match residue_embeddings first two dims: "
                    f"mask={tuple(attention_mask.shape)}, "
                    f"residues={tuple(residue_embeddings.shape)}"
                )
            valid_mask = attention_mask.to(dtype=torch.bool, device=residue_embeddings.device)

        if valid_mask.sum(dim=1).eq(0).any():
            raise ValueError("Each protein must have at least one valid residue")

        scores = self.attention(residue_embeddings).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~valid_mask, 0.0)
        pooled = torch.einsum("bl,blh->bh", weights, residue_embeddings)

        should_return_attention = (
            self.return_attention if return_attention is None else return_attention
        )
        if should_return_attention:
            return pooled, weights
        return pooled


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


class EnzymeBiologicalFactorizedEncoder(nn.Module):
    """Build a fixed-layout enzyme embedding from sequence-derived branches.

    Biological labels supervise prediction heads only. They are never accepted
    as encoder inputs, so training and retrieval inference use the same path.
    """

    RESERVED_BLOCKS = ("core", "site", "ec")

    def __init__(
        self,
        residue_dim: int,
        hyperbolic_dim: int | None,
        output_dim: int,
        family_output_dims: dict[str, int],
        block_dims: dict[str, int],
        block_weights: dict[str, float],
        hidden_dim: int = 512,
        dropout: float = 0.1,
        initial_sleec_bias_scale: float = 1.0,
        train_sleec_bias_scale: bool = True,
        learned_block_weights: bool = False,
    ) -> None:
        super().__init__()
        if residue_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Biological factorized dimensions must be positive")
        if hyperbolic_dim is not None and hyperbolic_dim <= 0:
            raise ValueError("hyperbolic_dim must be positive or None")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in [0, 1]")
        if initial_sleec_bias_scale <= 0.0:
            raise ValueError("initial_sleec_bias_scale must be positive")
        families = tuple(name for name in block_dims if name not in self.RESERVED_BLOCKS)
        expected_blocks = {"core", "site"} | set(families)
        if "ec" in block_dims:
            expected_blocks.add("ec")
        if set(block_dims) != expected_blocks or set(block_weights) != expected_blocks:
            raise ValueError(
                "Biological block dimensions and weights must define core, site, "
                "optional ec, "
                f"and every family; expected={sorted(expected_blocks)}"
            )
        if "ec" in expected_blocks and hyperbolic_dim is None:
            raise ValueError("An ec block requires hyperbolic_dim")
        if sum(int(dim) for dim in block_dims.values()) != output_dim:
            raise ValueError("Biological block dimensions must sum to output_dim")
        if any(int(dim) <= 0 for dim in block_dims.values()):
            raise ValueError("Biological block dimensions must be positive")
        weight_sum = sum(float(weight) for weight in block_weights.values())
        if weight_sum <= 0.0 or any(float(weight) < 0.0 for weight in block_weights.values()):
            raise ValueError("Biological block weights must be non-negative with positive sum")
        unexpected_heads = set(family_output_dims) - set(families)
        if unexpected_heads:
            raise ValueError(f"Biological heads have no matching block: {sorted(unexpected_heads)}")

        self.residue_dim = int(residue_dim)
        self.hyperbolic_dim = None if hyperbolic_dim is None else int(hyperbolic_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.families = families
        self.block_dims = {name: int(dim) for name, dim in block_dims.items()}
        normalized_weights = {
            name: float(weight) / weight_sum for name, weight in block_weights.items()
        }
        self.block_weights = normalized_weights
        self.block_names = tuple(self.block_dims)
        prior = torch.tensor(
            [normalized_weights[name] for name in self.block_names],
            dtype=torch.float32,
        )
        self.learned_block_weights = bool(learned_block_weights)
        self.register_buffer("block_weight_prior", prior, persistent=False)
        self.block_weight_logits = (
            nn.Parameter(prior.clamp_min(1e-12).log()) if self.learned_block_weights else None
        )
        self.core_projection = self._projection(residue_dim, self.block_dims["core"], dropout)
        self.site_projection = self._projection(residue_dim, self.block_dims["site"], dropout)
        self.ec_projection = (
            self._projection(self.hyperbolic_dim, self.block_dims["ec"], dropout)
            if "ec" in self.block_dims and self.hyperbolic_dim is not None
            else None
        )
        self.residue_adapter = nn.Sequential(
            nn.LayerNorm(residue_dim),
            nn.Linear(residue_dim, hidden_dim),
            nn.GELU(),
        )
        self.family_key = nn.Linear(hidden_dim, hidden_dim)
        self.family_value = nn.Linear(hidden_dim, hidden_dim)
        self.family_queries = nn.Parameter(torch.empty(len(families), hidden_dim))
        self.family_projections = nn.ModuleDict(
            {
                family: self._projection(hidden_dim, self.block_dims[family], dropout)
                for family in families
            }
        )
        self.family_heads = nn.ModuleDict(
            {
                family: nn.Linear(
                    self.block_dims[family],
                    int(family_output_dims[family]),
                )
                for family in families
                if int(family_output_dims.get(family, 0)) > 0
            }
        )
        raw_scale = torch.log(torch.expm1(torch.tensor(float(initial_sleec_bias_scale))))
        self.raw_sleec_bias_scale = nn.Parameter(raw_scale.repeat(len(families)))
        self.raw_sleec_bias_scale.requires_grad = bool(train_sleec_bias_scale)
        self.family_dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.family_queries, mean=0.0, std=hidden_dim**-0.5)

    @staticmethod
    def _projection(input_dim: int, output_dim: int, dropout: float) -> nn.Module:
        layers: list[nn.Module] = [nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        return nn.Sequential(*layers)

    @property
    def sleec_bias_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_sleec_bias_scale)

    @property
    def current_block_weights(self) -> torch.Tensor:
        if self.block_weight_logits is None:
            return self.block_weight_prior
        return torch.softmax(self.block_weight_logits, dim=0)

    def block_weight_kl(self) -> torch.Tensor:
        weights = self.current_block_weights.clamp_min(1e-12)
        prior = self.block_weight_prior.clamp_min(1e-12)
        return torch.sum(weights * (weights.log() - prior.log()))

    def forward(
        self,
        raw_mean: torch.Tensor,
        pooled: torch.Tensor,
        hyperbolic_tangent: torch.Tensor | None,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor | None = None,
        sleec_prior: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = raw_mean.shape[0]
        if raw_mean.shape != (batch_size, self.residue_dim):
            raise ValueError("raw_mean has an invalid shape")
        if pooled.shape != (batch_size, self.residue_dim):
            raise ValueError("pooled has an invalid shape")
        if "ec" in self.block_dims:
            if hyperbolic_tangent is None or self.hyperbolic_dim is None:
                raise ValueError("hyperbolic_tangent is required for the ec block")
            if hyperbolic_tangent.shape != (batch_size, self.hyperbolic_dim):
                raise ValueError("hyperbolic_tangent has an invalid shape")
        if (
            residue_embeddings.ndim != 3
            or residue_embeddings.shape[0] != batch_size
            or residue_embeddings.shape[-1] != self.residue_dim
        ):
            raise ValueError("residue_embeddings must have shape [batch, length, residue_dim]")

        attention = None
        family_hidden = None
        if self.families:
            adapted = self.residue_adapter(residue_embeddings.float())
            keys = self.family_key(adapted)
            values = self.family_value(adapted)
            scores = torch.einsum("bld,fd->bfl", keys, self.family_queries) / (self.hidden_dim**0.5)
            if sleec_prior is not None:
                if sleec_prior.shape != residue_embeddings.shape[:2]:
                    raise ValueError(
                        "sleec_prior must match the residue batch and length dimensions"
                    )
                scores = scores + self.sleec_bias_scale.view(1, -1, 1) * sleec_prior.unsqueeze(1)
            if residue_padding_mask is not None:
                if residue_padding_mask.shape != residue_embeddings.shape[:2]:
                    raise ValueError("residue_padding_mask must match residue embeddings")
                scores = scores.masked_fill(
                    residue_padding_mask.to(device=scores.device, dtype=torch.bool).unsqueeze(1),
                    torch.finfo(scores.dtype).min,
                )
            attention = torch.softmax(scores, dim=-1)
            family_hidden = torch.einsum(
                "bfl,bld->bfd",
                self.family_dropout(attention),
                values,
            )

        block_inputs: dict[str, torch.Tensor] = {
            "core": self.core_projection(raw_mean),
            "site": self.site_projection(pooled),
        }
        if "ec" in self.block_dims:
            if self.ec_projection is None or hyperbolic_tangent is None:
                raise RuntimeError("ec projection was not initialized")
            block_inputs["ec"] = self.ec_projection(hyperbolic_tangent)
        for index, family in enumerate(self.families):
            if family_hidden is None:
                raise RuntimeError("Biological family features were not initialized")
            block_inputs[family] = self.family_projections[family](family_hidden[:, index])
        blocks: list[torch.Tensor] = []
        details: dict[str, torch.Tensor] = {}
        current_weights = self.current_block_weights
        for block_index, name in enumerate(self.block_names):
            block = F.normalize(block_inputs[name], p=2, dim=-1, eps=1e-12)
            weight = current_weights[block_index]
            blocks.append(block * weight.sqrt())
            if return_details:
                details[f"enzyme_block_norm_{name}"] = block.norm(dim=-1)
                details[f"enzyme_block_weight_{name}"] = weight.expand(batch_size)
        output = F.normalize(torch.cat(blocks, dim=-1), p=2, dim=-1, eps=1e-12)
        if not return_details:
            return output

        details["enzyme_block_weight_kl"] = self.block_weight_kl().expand(batch_size)
        if attention is not None:
            entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1)
            for index, family in enumerate(self.families):
                details[f"biofp_attention_entropy_{family}"] = entropy[:, index]
                details[f"biofp_sleec_bias_scale_{family}"] = self.sleec_bias_scale[index].expand(
                    batch_size
                )
                if family in self.family_heads:
                    details[f"biofp_logits_{family}"] = self.family_heads[family](
                        block_inputs[family]
                    )
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


class FunctionalResidueScorer(nn.Module):
    """
    Residue-wise functional relevance scorer for SLEEC-style pooling.

    The scorer predicts one finite logit per residue. Padding logits and scores
    are set to zero so downstream logging and optional residue supervision can
    always mask them out safely.
    """

    def __init__(self, hidden_dim: int, scorer_hidden_dim: int = 256):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if scorer_hidden_dim <= 0:
            raise ValueError("scorer_hidden_dim must be positive")

        self.hidden_dim = hidden_dim
        self.scorer_hidden_dim = scorer_hidden_dim
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(scorer_hidden_dim, 1),
        )

    @staticmethod
    def _clean_stage1_key(key: str) -> str:
        for prefix in ("module.", "model.", "classifier.", "stage1_model."):
            while key.startswith(prefix):
                key = key[len(prefix) :]
        return key

    @staticmethod
    def _stage1_state_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise ValueError("SLEEC stage-1 checkpoint must be a dictionary")
        for state_key in (
            "unwrapped_model_state_dict",
            "model_state_dict",
            "state_dict",
            "model",
        ):
            state = checkpoint.get(state_key)
            if isinstance(state, dict):
                return {
                    FunctionalResidueScorer._clean_stage1_key(key): value
                    for key, value in state.items()
                    if torch.is_tensor(value)
                }
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return {
                FunctionalResidueScorer._clean_stage1_key(key): value
                for key, value in checkpoint.items()
            }
        raise ValueError("Could not find a model state dict in SLEEC stage-1 checkpoint")

    def load_stage1_checkpoint(self, checkpoint_path: str) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        stage1_state = self._stage1_state_from_checkpoint(checkpoint)
        self.load_stage1_state_dict(stage1_state)

    def load_stage1_state_dict(self, stage1_state: dict[str, torch.Tensor]) -> None:
        expected = self.scorer.state_dict()
        expected_keys = set(expected.keys())
        clean_state = {
            self._clean_stage1_key(key): value
            for key, value in stage1_state.items()
            if torch.is_tensor(value)
        }

        if expected_keys.issubset(clean_state.keys()):
            mapped_state = {key: clean_state[key] for key in expected_keys}
        else:
            linear_prefixes = [
                key[: -len(".weight")]
                for key, value in clean_state.items()
                if key.endswith(".weight")
                and value.ndim == 2
                and f"{key[:-len('.weight')]}.bias" in clean_state
            ]
            if len(linear_prefixes) != 2:
                raise ValueError(
                    "Expected a two-layer SLEEC stage-1 MLP checkpoint; "
                    f"found linear prefixes: {linear_prefixes}"
                )
            first_prefix, second_prefix = linear_prefixes
            mapped_state = {
                "0.weight": clean_state[f"{first_prefix}.weight"],
                "0.bias": clean_state[f"{first_prefix}.bias"],
                "2.weight": clean_state[f"{second_prefix}.weight"],
                "2.bias": clean_state[f"{second_prefix}.bias"],
            }

        for key, value in mapped_state.items():
            if value.shape != expected[key].shape:
                raise ValueError(
                    "SLEEC stage-1 checkpoint shape mismatch for "
                    f"{key}: expected {tuple(expected[key].shape)}, got {tuple(value.shape)}"
                )
        self.scorer.load_state_dict(mapped_state)

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residue_embeddings.ndim != 3:
            raise ValueError(
                "residue_embeddings must have shape [batch, seq_len, hidden_dim], "
                f"got {tuple(residue_embeddings.shape)}"
            )
        if residue_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected residue hidden_dim={self.hidden_dim}, "
                f"got {residue_embeddings.shape[-1]}"
            )

        batch_size, seq_len, _ = residue_embeddings.shape
        if attention_mask is None:
            valid_mask = torch.ones(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=residue_embeddings.device,
            )
        else:
            if attention_mask.shape != residue_embeddings.shape[:2]:
                raise ValueError(
                    "attention_mask must match residue_embeddings first two dims: "
                    f"mask={tuple(attention_mask.shape)}, "
                    f"residues={tuple(residue_embeddings.shape)}"
                )
            valid_mask = attention_mask.to(dtype=torch.bool, device=residue_embeddings.device)

        logits = self.scorer(residue_embeddings).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, 0.0)
        scores = torch.sigmoid(logits).masked_fill(~valid_mask, 0.0)
        return logits, scores


class SLEECFunctionalPool(nn.Module):
    """
    SLEEC-inspired functional-residue pooling over residue embeddings.

    ``attention_mask`` uses True for valid residues and False for padding.
    Padding positions always receive zero weight. The returned details include
    logits, sigmoid scores, normalized residue weights, and pooling statistics.
    """

    def __init__(
        self,
        hidden_dim: int,
        scorer_hidden_dim: int = 256,
        score_hidden_dim: int | None = None,
        mode: str = "topk",
        topk_fraction: float = 0.2,
        threshold: float = 0.5,
        checkpoint_path: str | None = None,
        freeze_scorer: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__()
        if mode not in {"topk", "soft", "threshold"}:
            raise ValueError("mode must be one of: topk, soft, threshold")
        if not (0.0 < topk_fraction <= 1.0):
            raise ValueError("topk_fraction must be in the range (0, 1]")

        self.hidden_dim = hidden_dim
        self.score_hidden_dim = hidden_dim if score_hidden_dim is None else score_hidden_dim
        self.mode = mode
        self.topk_fraction = float(topk_fraction)
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.scorer = FunctionalResidueScorer(
            hidden_dim=self.score_hidden_dim,
            scorer_hidden_dim=scorer_hidden_dim,
        )
        if checkpoint_path:
            self.scorer.load_stage1_checkpoint(checkpoint_path)
        if freeze_scorer:
            for parameter in self.scorer.parameters():
                parameter.requires_grad = False

    def _valid_mask(
        self,
        residue_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = residue_embeddings.shape
        if attention_mask is None:
            return torch.ones(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=residue_embeddings.device,
            )
        if attention_mask.shape != residue_embeddings.shape[:2]:
            raise ValueError(
                "attention_mask must match residue_embeddings first two dims: "
                f"mask={tuple(attention_mask.shape)}, "
                f"residues={tuple(residue_embeddings.shape)}"
            )
        return attention_mask.to(dtype=torch.bool, device=residue_embeddings.device)

    def _topk_selection(self, scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        selected_mask = torch.zeros_like(valid_mask)
        masked_scores = scores.masked_fill(~valid_mask, -torch.inf)
        valid_lengths = valid_mask.sum(dim=1)
        topk_counts = torch.ceil(valid_lengths.to(dtype=torch.float32) * self.topk_fraction)
        topk_counts = topk_counts.to(dtype=torch.long).clamp_min(1)

        for batch_idx, k_value in enumerate(topk_counts.tolist()):
            k_value = min(k_value, int(valid_lengths[batch_idx].item()))
            topk_indices = torch.topk(masked_scores[batch_idx], k=k_value).indices
            selected_mask[batch_idx, topk_indices] = True
        return selected_mask

    def _masked_average_weights(self, valid_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        weights = valid_mask.to(dtype=dtype)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool | None = None,
        return_details: bool = False,
        score_embeddings: torch.Tensor | None = None,
        score_attention_mask: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ):
        if residue_embeddings.ndim != 3:
            raise ValueError(
                "residue_embeddings must have shape [batch, seq_len, hidden_dim], "
                f"got {tuple(residue_embeddings.shape)}"
            )
        if residue_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected residue hidden_dim={self.hidden_dim}, "
                f"got {residue_embeddings.shape[-1]}"
            )

        valid_mask = self._valid_mask(residue_embeddings, attention_mask)
        if valid_mask.sum(dim=1).eq(0).any():
            raise ValueError("Each protein must have at least one valid residue")

        if score_embeddings is None:
            score_embeddings = residue_embeddings
            score_attention_mask = valid_mask
        if score_embeddings.ndim != 3:
            raise ValueError(
                "score_embeddings must have shape [batch, seq_len, score_hidden_dim], "
                f"got {tuple(score_embeddings.shape)}"
            )
        if score_embeddings.shape[:2] != residue_embeddings.shape[:2]:
            raise ValueError(
                "score_embeddings must align with residue_embeddings in batch and length: "
                f"score={tuple(score_embeddings.shape)}, value={tuple(residue_embeddings.shape)}"
            )
        if score_embeddings.shape[-1] != self.score_hidden_dim:
            raise ValueError(
                f"Expected score hidden_dim={self.score_hidden_dim}, "
                f"got {score_embeddings.shape[-1]}"
            )
        if score_attention_mask is None:
            score_attention_mask = valid_mask
        score_valid_mask = self._valid_mask(score_embeddings, score_attention_mask)
        if not torch.equal(score_valid_mask, valid_mask):
            raise ValueError("score_attention_mask must match value residue validity")

        logits, scores = self.scorer(score_embeddings, attention_mask=valid_mask)
        if self.mode == "soft":
            masked_scores = scores.masked_fill(~valid_mask, -torch.inf)
            weights = torch.softmax(masked_scores, dim=-1).masked_fill(~valid_mask, 0.0)
            selected_mask = valid_mask
        else:
            if self.mode == "topk":
                selected_mask = self._topk_selection(scores, valid_mask)
            else:
                selected_mask = (scores > self.threshold) & valid_mask

            raw_weights = scores * selected_mask.to(dtype=scores.dtype)
            weight_sums = raw_weights.sum(dim=1, keepdim=True)
            average_weights = self._masked_average_weights(valid_mask, dtype=scores.dtype)
            weights = raw_weights / weight_sums.clamp_min(self.eps)
            fallback_rows = weight_sums.squeeze(1) <= self.eps
            if bool(fallback_rows.any()):
                weights = torch.where(fallback_rows.unsqueeze(1), average_weights, weights)

        weights = weights.masked_fill(~valid_mask, 0.0)
        pooled = torch.einsum("bl,blh->bh", weights, residue_embeddings)
        entropy = -(weights.clamp_min(self.eps).log() * weights).sum(dim=1)
        num_selected = weights.gt(0.0).sum(dim=1).to(dtype=weights.dtype)

        details = {
            "logits": logits,
            "scores": scores,
            "weights": weights,
            "selected_mask": selected_mask & weights.gt(0.0),
            "pooling_entropy": entropy,
            "num_selected": num_selected,
        }

        if return_details:
            return pooled, details
        if return_attention:
            return pooled, weights
        return pooled


class SLEECGuidedAttentionPool(nn.Module):
    """
    Trainable residue attention pooling guided by a frozen SLEEC stage-1 scorer.

    The learned attention logit is combined with a centered SLEEC prior before
    softmax:

        a_i = learned_logit_i + softplus(alpha) * (sleec_logit_i - logit(threshold))

    Padding positions always receive zero weight.
    """

    def __init__(
        self,
        hidden_dim: int,
        scorer_hidden_dim: int = 256,
        threshold: float = 0.34,
        checkpoint_path: str | None = None,
        freeze_scorer: bool = True,
        attention_bias: bool = True,
        initial_sleec_bias_scale: float = 1.0,
        train_sleec_bias_scale: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be in the open interval (0, 1)")
        if initial_sleec_bias_scale <= 0.0:
            raise ValueError("initial_sleec_bias_scale must be positive")

        self.hidden_dim = int(hidden_dim)
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.attention = nn.Linear(hidden_dim, 1, bias=attention_bias)
        self.sleec_scorer = FunctionalResidueScorer(
            hidden_dim=hidden_dim,
            scorer_hidden_dim=scorer_hidden_dim,
        )
        if checkpoint_path:
            self.sleec_scorer.load_stage1_checkpoint(checkpoint_path)
        if freeze_scorer:
            for parameter in self.sleec_scorer.parameters():
                parameter.requires_grad = False

        threshold_tensor = torch.tensor(self.threshold, dtype=torch.float32)
        self.register_buffer("threshold_logit", torch.logit(threshold_tensor))
        raw_scale = torch.log(torch.expm1(torch.tensor(float(initial_sleec_bias_scale))))
        self.raw_sleec_bias_scale = nn.Parameter(raw_scale)
        self.raw_sleec_bias_scale.requires_grad = bool(train_sleec_bias_scale)

    def _valid_mask(
        self,
        residue_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = residue_embeddings.shape
        if attention_mask is None:
            return torch.ones(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=residue_embeddings.device,
            )
        if attention_mask.shape != residue_embeddings.shape[:2]:
            raise ValueError(
                "attention_mask must match residue_embeddings first two dims: "
                f"mask={tuple(attention_mask.shape)}, "
                f"residues={tuple(residue_embeddings.shape)}"
            )
        return attention_mask.to(dtype=torch.bool, device=residue_embeddings.device)

    @property
    def sleec_bias_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_sleec_bias_scale)

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if residue_embeddings.ndim != 3:
            raise ValueError(
                "residue_embeddings must have shape [batch, seq_len, hidden_dim], "
                f"got {tuple(residue_embeddings.shape)}"
            )
        if residue_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected residue hidden_dim={self.hidden_dim}, "
                f"got {residue_embeddings.shape[-1]}"
            )

        valid_mask = self._valid_mask(residue_embeddings, attention_mask)
        if valid_mask.sum(dim=1).eq(0).any():
            raise ValueError("Each protein must have at least one valid residue")

        sleec_logits, sleec_scores = self.sleec_scorer(
            residue_embeddings,
            attention_mask=valid_mask,
        )
        learned_logits = self.attention(residue_embeddings).squeeze(-1)
        sleec_prior = sleec_logits - self.threshold_logit.to(
            device=residue_embeddings.device,
            dtype=residue_embeddings.dtype,
        )
        combined_logits = (
            learned_logits + self.sleec_bias_scale.to(dtype=residue_embeddings.dtype) * sleec_prior
        )
        combined_logits = combined_logits.masked_fill(~valid_mask, float("-inf"))
        weights = torch.softmax(combined_logits, dim=-1).masked_fill(~valid_mask, 0.0)
        pooled = torch.einsum("bl,blh->bh", weights, residue_embeddings)

        if not return_details:
            return pooled

        entropy = -(weights.clamp_min(self.eps).log() * weights).sum(dim=1)
        sleec_positive = (sleec_scores > self.threshold) & valid_mask
        attention_mass_sleec_positive = (weights * sleec_positive.to(dtype=weights.dtype)).sum(
            dim=1
        )
        details = {
            "logits": sleec_logits,
            "scores": sleec_scores,
            "weights": weights,
            "learned_logits": learned_logits.masked_fill(~valid_mask, 0.0),
            "combined_logits": combined_logits.masked_fill(~valid_mask, 0.0),
            "pooling_entropy": entropy,
            "attention_mass_sleec_positive": attention_mass_sleec_positive,
            "num_sleec_positive": sleec_positive.sum(dim=1).to(dtype=weights.dtype),
            "sleec_bias_scale": self.sleec_bias_scale.detach(),
        }
        return pooled, details


def _torch_load_checkpoint(path: str) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    return checkpoint


def _stage1_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}
    merged = dict(config)
    for key in ("input_dim", "hyp_dim", "curvature"):
        if key in checkpoint and key not in merged:
            merged[key] = checkpoint[key]
    return merged


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


class MoleculeSetAttentionPooling(nn.Module):
    """
    Learned masked attention pooling over a set of molecule embeddings.

    For molecule embeddings ``h_i`` this computes:

        alpha_i = softmax_i(w^T h_i)
        x = sum_i alpha_i h_i

    ``attention_mask`` uses True for valid molecules and False for padding.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_bias: bool = True,
        return_attention: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.return_attention = return_attention
        self.attention = nn.Linear(hidden_dim, 1, bias=attention_bias)

    def forward(
        self,
        molecule_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if molecule_embeddings.ndim != 3:
            raise ValueError(
                "molecule_embeddings must have shape [batch, num_molecules, hidden_dim], "
                f"got {tuple(molecule_embeddings.shape)}"
            )
        if molecule_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected molecule hidden_dim={self.hidden_dim}, "
                f"got {molecule_embeddings.shape[-1]}"
            )

        batch_size, num_molecules, _ = molecule_embeddings.shape
        if attention_mask is None:
            valid_mask = torch.ones(
                batch_size,
                num_molecules,
                dtype=torch.bool,
                device=molecule_embeddings.device,
            )
        else:
            if attention_mask.shape != molecule_embeddings.shape[:2]:
                raise ValueError(
                    "attention_mask must match molecule_embeddings first two dims: "
                    f"mask={tuple(attention_mask.shape)}, "
                    f"molecules={tuple(molecule_embeddings.shape)}"
                )
            valid_mask = attention_mask.to(dtype=torch.bool, device=molecule_embeddings.device)

        if valid_mask.sum(dim=1).eq(0).any():
            raise ValueError("Each molecule set must contain at least one valid molecule")

        scores = self.attention(molecule_embeddings).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~valid_mask, 0.0)
        pooled = torch.einsum("bm,bmh->bh", weights, molecule_embeddings)

        should_return_attention = (
            self.return_attention if return_attention is None else return_attention
        )
        if should_return_attention:
            return pooled, weights
        return pooled


class MoleculeSetMeanPooling(nn.Module):
    """
    Masked mean pooling over a set of molecule embeddings.

    ``attention_mask`` uses True for valid molecules and False for padding. When
    requested, this returns uniform valid-molecule weights so callers can keep
    the same attention-return interface used by learned molecule pooling.
    """

    def __init__(self, return_attention: bool = False):
        super().__init__()
        self.return_attention = return_attention

    def forward(
        self,
        molecule_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if molecule_embeddings.ndim != 3:
            raise ValueError(
                "molecule_embeddings must have shape [batch, num_molecules, hidden_dim], "
                f"got {tuple(molecule_embeddings.shape)}"
            )

        batch_size, num_molecules, _ = molecule_embeddings.shape
        if attention_mask is None:
            valid_mask = torch.ones(
                batch_size,
                num_molecules,
                dtype=torch.bool,
                device=molecule_embeddings.device,
            )
        else:
            if attention_mask.shape != molecule_embeddings.shape[:2]:
                raise ValueError(
                    "attention_mask must match molecule_embeddings first two dims: "
                    f"mask={tuple(attention_mask.shape)}, "
                    f"molecules={tuple(molecule_embeddings.shape)}"
                )
            valid_mask = attention_mask.to(dtype=torch.bool, device=molecule_embeddings.device)

        if valid_mask.sum(dim=1).eq(0).any():
            raise ValueError("Each molecule set must contain at least one valid molecule")

        weights = valid_mask.to(dtype=molecule_embeddings.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
        pooled = torch.einsum("bm,bmh->bh", weights, molecule_embeddings)

        should_return_attention = (
            self.return_attention if return_attention is None else return_attention
        )
        if should_return_attention:
            return pooled, weights
        return pooled


class ResidualMLPProjection(nn.Module):
    """Geometry-preserving residual projection with a bounded learned gate."""

    def __init__(
        self,
        dim: int,
        widths: int | list[int],
        num_layers: int,
        dropout: float,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be strictly between 0 and 1")
        self.normalization = nn.LayerNorm(dim)
        self.delta = MLP(
            input_dim=dim,
            output_dim=dim,
            num_layers=num_layers,
            widths=widths,
            activations=nn.GELU(),
            dropout=dropout,
            normalise_output=False,
        )
        self.raw_gate = nn.Parameter(
            torch.logit(torch.tensor(float(gate_init), dtype=torch.float32))
        )

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gate)

    def forward(
        self,
        value: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual = self.gate * self.delta(self.normalization(value))
        output = value + residual
        if not return_details:
            return output

        base_norm = value.norm(dim=-1)
        residual_norm = residual.norm(dim=-1)
        return output, {
            "residual_gate": self.gate.expand(value.shape[0]),
            "residual_base_cosine": F.cosine_similarity(output, value, dim=-1),
            "residual_base_norm_ratio": residual_norm / base_norm.clamp_min(1e-12),
            "residual_base_norm": base_norm,
            "residual_delta_norm": residual_norm,
        }


class MultimodalReactionAttentionEncoder(BaseModel):
    """
    Encode reactions from learned reaction, Uni-Mol2, and chirality modalities.

    The fixed inputs are converted to trainable modality tokens:
    reaction-model SMILES embedding, Uni-Mol2 reaction composition, and
    optionally a ChIRo chirality composition. Each modality can use either the
    historical single linear adapter or a Horizyn-style MLP adapter before a
    learned attention pooler combines the tokens.
    """

    supports_attention_return = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        reaction_model_dim: int = 1024,
        unimol_dim: int = 768,
        chienn_dim: int = 256,
        reaction_chemistry_dim: int | None = None,
        reaction_directional_dim: int | None = None,
        reaction_pooling: str = "attention",
        attention_bias: bool = True,
        separate_side_poolers: bool = True,
        use_reaction_model: bool = True,
        use_chienn: bool = True,
        use_reaction_chemistry: bool = False,
        use_reaction_directional: bool = False,
        chirality_modality_name: str = "chiro",
        modality_attention_hidden_dim: int | None = None,
        modality_attention_dropout: float = 0.0,
        modality_dropout: float = 0.0,
        chemistry_dropout: float = 0.0,
        modality_token_layer_norm: bool = False,
        modality_l2_normalize: bool = False,
        modality_encoder_num_layers: int | None = None,
        modality_encoder_widths: int | list[int] | None = None,
        modality_encoder_use_layer_norm: bool = False,
        modality_encoder_dropout: float = 0.0,
        modality_encoder_normalise_output: bool = False,
        side_composition: str = "directional_delta",
        modality_fusion: str = "attention",
        factorized_dims: dict[str, int] | None = None,
        factorized_weights: dict[str, float] | None = None,
        attention_prior_weights: dict[str, float] | None = None,
        attention_adaptation_strength: float = 0.4,
        output_projection: str = "mlp",
        residual_gate_init: float = 0.1,
        directional_gate_init: float = 0.1,
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
        if input_dim <= 0:
            raise ValueError("input_dim/token_dim must be positive")
        if use_reaction_model and reaction_model_dim <= 0:
            raise ValueError("reaction_model_dim must be positive when enabled")
        if unimol_dim <= 0:
            raise ValueError("unimol_dim must be positive")
        if use_chienn and chienn_dim <= 0:
            raise ValueError("chirality_dim/chiro_dim must be positive when chirality is enabled")
        if use_reaction_chemistry and (
            reaction_chemistry_dim is None or int(reaction_chemistry_dim) <= 0
        ):
            raise ValueError(
                "reaction_chemistry_dim must be positive when reaction chemistry " "is enabled"
            )
        if use_reaction_directional and (
            reaction_directional_dim is None or int(reaction_directional_dim) <= 0
        ):
            raise ValueError(
                "reaction_directional_dim must be positive when directional features " "are enabled"
            )
        if reaction_pooling not in {"attention", "mean"}:
            raise ValueError("reaction_pooling must be one of: attention, mean")
        if side_composition not in {"directional_delta", "molecule_set"}:
            raise ValueError("side_composition must be one of: directional_delta, molecule_set")
        if modality_fusion not in {
            "attention",
            "mean",
            "factorized_concat",
            "prior_bounded_attention",
        }:
            raise ValueError(
                "modality_fusion must be one of: attention, mean, factorized_concat, "
                "prior_bounded_attention"
            )
        if output_projection not in {"mlp", "residual_mlp", "identity"}:
            raise ValueError("output_projection must be one of: mlp, residual_mlp, identity")
        if output_projection == "residual_mlp" and input_dim != output_dim:
            raise ValueError("residual_mlp requires equal token and output dimensions")
        if (
            output_projection == "identity"
            and input_dim != output_dim
            and modality_fusion != "factorized_concat"
        ):
            raise ValueError("identity output projection requires equal input/output dimensions")
        if not (0.0 <= attention_adaptation_strength < 1.0):
            raise ValueError("attention_adaptation_strength must be in the range [0, 1)")
        if not (0.0 < directional_gate_init < 1.0):
            raise ValueError("directional_gate_init must be in the open interval (0, 1)")
        if not (0.0 <= modality_attention_dropout <= 1.0):
            raise ValueError("modality_attention_dropout must be in the range [0, 1]")
        if not (0.0 <= modality_dropout <= 1.0):
            raise ValueError("modality_dropout must be in the range [0, 1]")
        if not (0.0 <= chemistry_dropout <= 1.0):
            raise ValueError("chemistry_dropout must be in the range [0, 1]")
        if modality_encoder_num_layers is not None and modality_encoder_num_layers < 0:
            raise ValueError("modality_encoder_num_layers must be non-negative or None")
        if not (0.0 <= modality_encoder_dropout <= 1.0):
            raise ValueError("modality_encoder_dropout must be in the range [0, 1]")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.token_dim = input_dim
        self.reaction_model_dim = reaction_model_dim
        self.unimol_dim = unimol_dim
        self.chienn_dim = chienn_dim
        self.chirality_dim = chienn_dim
        self.reaction_chemistry_dim = (
            None if reaction_chemistry_dim is None else int(reaction_chemistry_dim)
        )
        self.reaction_directional_dim = (
            None if reaction_directional_dim is None else int(reaction_directional_dim)
        )
        self.use_reaction_model = bool(use_reaction_model)
        self.use_chienn = use_chienn
        self.use_chirality = use_chienn
        self.use_reaction_chemistry = bool(use_reaction_chemistry)
        self.use_reaction_directional = bool(use_reaction_directional)
        self.chirality_modality_name = chirality_modality_name
        self.reaction_pooling = reaction_pooling
        self.side_composition = side_composition
        self.modality_fusion = modality_fusion
        self.attention_adaptation_strength = float(attention_adaptation_strength)
        self.output_projection = output_projection
        self.modality_dropout = float(modality_dropout)
        self.chemistry_dropout = float(chemistry_dropout)
        self.modality_l2_normalize = bool(modality_l2_normalize)
        self.return_attention = return_attention
        modality_names = []
        if self.use_reaction_model:
            modality_names.append("reaction_model")
        modality_names.append("unimol2")
        if use_chienn:
            modality_names.append(chirality_modality_name)
        if self.use_reaction_chemistry:
            modality_names.append("reaction_chemistry")
        self.modality_names = tuple(modality_names)

        if modality_fusion == "factorized_concat":
            if factorized_dims is None or factorized_weights is None:
                raise ValueError(
                    "factorized_concat requires factorized_dims and factorized_weights"
                )
            if set(factorized_dims) != set(self.modality_names):
                raise ValueError(
                    "factorized_dims must exactly match enabled modalities: "
                    f"expected {self.modality_names}, got {tuple(factorized_dims)}"
                )
            if set(factorized_weights) != set(self.modality_names):
                raise ValueError("factorized_weights must exactly match enabled modalities")
            if sum(int(value) for value in factorized_dims.values()) != output_dim:
                raise ValueError("factorized_dims must sum to output_dim")
            if any(int(value) <= 0 for value in factorized_dims.values()):
                raise ValueError("factorized_dims must contain positive dimensions")
            weight_sum = sum(float(value) for value in factorized_weights.values())
            if abs(weight_sum - 1.0) > 1e-6 or any(
                float(value) <= 0 for value in factorized_weights.values()
            ):
                raise ValueError("factorized_weights must be positive and sum to 1")
        if modality_fusion == "prior_bounded_attention":
            if attention_prior_weights is None:
                raise ValueError("prior_bounded_attention requires attention_prior_weights")
            if set(attention_prior_weights) != set(self.modality_names):
                raise ValueError(
                    "attention_prior_weights must exactly match enabled modalities: "
                    f"expected {self.modality_names}, got {tuple(attention_prior_weights)}"
                )
            prior_sum = sum(float(value) for value in attention_prior_weights.values())
            if abs(prior_sum - 1.0) > 1e-6 or any(
                float(value) <= 0 for value in attention_prior_weights.values()
            ):
                raise ValueError("attention_prior_weights must be positive and sum to 1")
        self.factorized_dims = {name: int(value) for name, value in (factorized_dims or {}).items()}
        self.register_buffer(
            "factorized_weight_values",
            torch.tensor(
                [float((factorized_weights or {}).get(name, 0.0)) for name in self.modality_names],
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "attention_prior_values",
            torch.tensor(
                [
                    float((attention_prior_weights or {}).get(name, 0.0))
                    for name in self.modality_names
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(
            self._discard_persisted_attention_prior,
            with_module=True,
        )

        projection_dims = {
            name: (
                self.factorized_dims[name]
                if modality_fusion == "factorized_concat"
                else self.token_dim
            )
            for name in self.modality_names
        }

        self.reaction_projection = (
            self._build_modality_projection(
                reaction_model_dim,
                output_dim=projection_dims["reaction_model"],
                modality_encoder_num_layers=modality_encoder_num_layers,
                modality_encoder_widths=modality_encoder_widths,
                activations=activations,
                use_layer_norm=modality_encoder_use_layer_norm,
                dropout=modality_encoder_dropout,
                bias=bias,
                normalise_output=modality_encoder_normalise_output,
            )
            if self.use_reaction_model
            else None
        )
        molecule_composition_multiplier = 4 if side_composition == "directional_delta" else 1
        self.unimol_projection = self._build_modality_projection(
            molecule_composition_multiplier * unimol_dim,
            output_dim=projection_dims["unimol2"],
            modality_encoder_num_layers=modality_encoder_num_layers,
            modality_encoder_widths=modality_encoder_widths,
            activations=activations,
            use_layer_norm=modality_encoder_use_layer_norm,
            dropout=modality_encoder_dropout,
            bias=bias,
            normalise_output=modality_encoder_normalise_output,
        )
        self.chienn_projection = (
            self._build_modality_projection(
                molecule_composition_multiplier * chienn_dim,
                output_dim=projection_dims[chirality_modality_name],
                modality_encoder_num_layers=modality_encoder_num_layers,
                modality_encoder_widths=modality_encoder_widths,
                activations=activations,
                use_layer_norm=modality_encoder_use_layer_norm,
                dropout=modality_encoder_dropout,
                bias=bias,
                normalise_output=modality_encoder_normalise_output,
            )
            if use_chienn
            else None
        )
        self.reaction_chemistry_projection = (
            self._build_modality_projection(
                int(self.reaction_chemistry_dim),
                output_dim=projection_dims["reaction_chemistry"],
                modality_encoder_num_layers=modality_encoder_num_layers,
                modality_encoder_widths=modality_encoder_widths,
                activations=activations,
                use_layer_norm=modality_encoder_use_layer_norm,
                dropout=modality_encoder_dropout,
                bias=bias,
                normalise_output=modality_encoder_normalise_output,
            )
            if self.use_reaction_chemistry
            else None
        )
        self.reaction_directional_projection = (
            MLP(
                input_dim=int(self.reaction_directional_dim),
                output_dim=output_dim,
                num_layers=1,
                widths=output_dim,
                activations=activations,
                use_layer_norm=True,
                dropout=dropout,
                bias=bias,
                normalise_output=True,
            )
            if self.use_reaction_directional
            else None
        )
        self.raw_directional_gate = (
            nn.Parameter(
                torch.logit(torch.tensor(float(directional_gate_init), dtype=torch.float32))
            )
            if self.use_reaction_directional
            else None
        )

        if reaction_pooling == "attention":
            self.unimol_reactant_pooling = MoleculeSetAttentionPooling(
                hidden_dim=unimol_dim,
                attention_bias=attention_bias,
                return_attention=return_attention,
            )
            unimol_product_pooler = MoleculeSetAttentionPooling(
                hidden_dim=unimol_dim,
                attention_bias=attention_bias,
                return_attention=return_attention,
            )
            if use_chienn:
                self.chienn_reactant_pooling = MoleculeSetAttentionPooling(
                    hidden_dim=chienn_dim,
                    attention_bias=attention_bias,
                    return_attention=return_attention,
                )
                chienn_product_pooler = MoleculeSetAttentionPooling(
                    hidden_dim=chienn_dim,
                    attention_bias=attention_bias,
                    return_attention=return_attention,
                )
        else:
            self.unimol_reactant_pooling = MoleculeSetMeanPooling(return_attention=return_attention)
            unimol_product_pooler = MoleculeSetMeanPooling(return_attention=return_attention)
            if use_chienn:
                self.chienn_reactant_pooling = MoleculeSetMeanPooling(
                    return_attention=return_attention
                )
                chienn_product_pooler = MoleculeSetMeanPooling(return_attention=return_attention)

        if separate_side_poolers:
            self.unimol_product_pooling = unimol_product_pooler
            if use_chienn:
                self.chienn_product_pooling = chienn_product_pooler
        else:
            self.unimol_product_pooling = self.unimol_reactant_pooling
            if use_chienn:
                self.chienn_product_pooling = self.chienn_reactant_pooling

        attention_hidden_dim = (
            self.token_dim
            if modality_attention_hidden_dim is None
            else int(modality_attention_hidden_dim)
        )
        if attention_hidden_dim <= 0:
            raise ValueError("modality_attention_hidden_dim must be positive")
        self.modality_attention = nn.Sequential(
            nn.Linear(self.token_dim, attention_hidden_dim),
            nn.ReLU(),
            nn.Dropout(modality_attention_dropout),
            nn.Linear(attention_hidden_dim, 1, bias=attention_bias),
        )
        if modality_fusion == "prior_bounded_attention":
            # Residual logits start at zero, making the initial weights exactly the prior.
            nn.init.zeros_(self.modality_attention[-1].weight)
            if self.modality_attention[-1].bias is not None:
                nn.init.zeros_(self.modality_attention[-1].bias)
        self.modality_token_norm = (
            nn.LayerNorm(self.token_dim)
            if modality_token_layer_norm and modality_fusion != "factorized_concat"
            else nn.Identity()
        )
        if output_projection == "residual_mlp":
            self.projection_encoder = ResidualMLPProjection(
                dim=self.token_dim,
                widths=widths,
                num_layers=num_layers,
                dropout=dropout,
                gate_init=residual_gate_init,
            )
        elif output_projection == "mlp":
            self.projection_encoder = MLP(
                input_dim=self.token_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                widths=widths,
                activations=activations,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
                bias=bias,
                normalise_output=False,
            )
        else:
            self.projection_encoder = nn.Identity()
        if normalise_output:
            self.post_nn_layers.append(NormalizeLayer(p=2, dim=-1))

    def _discard_persisted_attention_prior(
        self,
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        state_dict.pop(f"{prefix}attention_prior_values", None)

    def _build_modality_projection(
        self,
        input_dim: int,
        output_dim: int,
        modality_encoder_num_layers: int | None,
        modality_encoder_widths: int | list[int] | None,
        activations: nn.Module | list[nn.Module],
        use_layer_norm: bool,
        dropout: float,
        bias: bool,
        normalise_output: bool,
    ) -> nn.Module:
        if modality_encoder_widths is None:
            hidden_layers = (
                0 if modality_encoder_num_layers is None else int(modality_encoder_num_layers)
            )
            if hidden_layers == 0:
                return nn.Linear(input_dim, output_dim, bias=bias)
            widths: int | list[int] = output_dim
        else:
            widths = modality_encoder_widths
            hidden_layers = (
                len(widths)
                if isinstance(widths, list)
                else (
                    1 if modality_encoder_num_layers is None else int(modality_encoder_num_layers)
                )
            )

        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=hidden_layers,
            widths=widths,
            activations=activations,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            bias=bias,
            normalise_output=normalise_output,
        )

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

    @staticmethod
    def _compose_side_pair(
        reactant_pooled: torch.Tensor, product_pooled: torch.Tensor
    ) -> torch.Tensor:
        delta = product_pooled - reactant_pooled
        return torch.cat([reactant_pooled, product_pooled, delta, delta.abs()], dim=-1)

    def _apply_modality_dropout(self, modality_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0.0:
            return modality_mask
        keep_mask = (
            torch.rand(
                modality_mask.shape,
                dtype=torch.float32,
                device=modality_mask.device,
            )
            >= self.modality_dropout
        )
        dropped_mask = modality_mask & keep_mask
        has_valid_modality = dropped_mask.any(dim=1, keepdim=True)
        return torch.where(has_valid_modality, dropped_mask, modality_mask)

    def _apply_chemistry_dropout(self, modality_mask: torch.Tensor) -> torch.Tensor:
        if (
            not self.training
            or self.chemistry_dropout <= 0.0
            or "reaction_chemistry" not in self.modality_names
        ):
            return modality_mask
        chemistry_index = self.modality_names.index("reaction_chemistry")
        keep_chemistry = (
            torch.rand(
                modality_mask.shape[0],
                dtype=torch.float32,
                device=modality_mask.device,
            )
            >= self.chemistry_dropout
        )
        dropped_mask = modality_mask.clone()
        dropped_mask[:, chemistry_index] &= keep_chemistry
        has_valid_modality = dropped_mask.any(dim=1, keepdim=True)
        return torch.where(has_valid_modality, dropped_mask, modality_mask)

    def _encode_molecule_modality(
        self,
        reactant_pooler: MoleculeSetAttentionPooling | MoleculeSetMeanPooling,
        product_pooler: MoleculeSetAttentionPooling | MoleculeSetMeanPooling,
        reactant_embeddings: torch.Tensor,
        reactant_padding_mask: torch.Tensor | None,
        product_embeddings: torch.Tensor,
        product_padding_mask: torch.Tensor | None,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        reactant_result = self._pool_side(
            reactant_pooler,
            reactant_embeddings,
            reactant_padding_mask,
            return_attention,
        )
        if self.side_composition == "molecule_set":
            if return_attention:
                reactant_pooled, reactant_attention = reactant_result
            else:
                reactant_pooled = reactant_result
                reactant_attention = None
            return reactant_pooled, reactant_attention, None

        product_result = self._pool_side(
            product_pooler,
            product_embeddings,
            product_padding_mask,
            return_attention,
        )
        if return_attention:
            reactant_pooled, reactant_attention = reactant_result
            product_pooled, product_attention = product_result
        else:
            reactant_pooled = reactant_result
            product_pooled = product_result
            reactant_attention = None
            product_attention = None
        return (
            self._compose_side_pair(reactant_pooled, product_pooled),
            reactant_attention,
            product_attention,
        )

    def forward(
        self,
        reactant_embeddings: torch.Tensor,
        reaction_embedding: torch.Tensor | None = None,
        reactant_padding_mask: torch.Tensor | None = None,
        product_embeddings: torch.Tensor | None = None,
        product_padding_mask: torch.Tensor | None = None,
        has_unimol2: torch.Tensor | None = None,
        reactant_chirality_embeddings: torch.Tensor | None = None,
        reactant_chirality_padding_mask: torch.Tensor | None = None,
        product_chirality_embeddings: torch.Tensor | None = None,
        product_chirality_padding_mask: torch.Tensor | None = None,
        has_chirality: torch.Tensor | None = None,
        has_chiro: torch.Tensor | None = None,
        has_chienn: torch.Tensor | None = None,
        reaction_chemistry_vector: torch.Tensor | None = None,
        has_reaction_chemistry: torch.Tensor | None = None,
        reaction_directional_vector: torch.Tensor | None = None,
        has_reaction_directional: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if product_embeddings is None:
            raise ValueError("product_embeddings is required")
        if self.use_chienn and (
            reactant_chirality_embeddings is None or product_chirality_embeddings is None
        ):
            raise ValueError("ChIRo reactant/product chirality embeddings are required")
        if self.use_reaction_chemistry and reaction_chemistry_vector is None:
            raise ValueError("reaction_chemistry_vector is required")
        if self.use_reaction_directional and reaction_directional_vector is None:
            raise ValueError("reaction_directional_vector is required")
        if self.use_reaction_model:
            if reaction_embedding is None:
                raise ValueError("reaction_embedding is required when ReactionT5 is enabled")
            if (
                reaction_embedding.ndim != 2
                or reaction_embedding.shape[-1] != self.reaction_model_dim
            ):
                raise ValueError(
                    f"reaction_embedding must have shape [B, {self.reaction_model_dim}], "
                    f"got {tuple(reaction_embedding.shape)}"
                )
        if reactant_embeddings.shape[-1] != self.unimol_dim:
            raise ValueError(
                f"Expected Uni-Mol2 reactant dim {self.unimol_dim}, "
                f"got {reactant_embeddings.shape[-1]}"
            )
        if product_embeddings.shape[-1] != self.unimol_dim:
            raise ValueError(
                f"Expected Uni-Mol2 product dim {self.unimol_dim}, "
                f"got {product_embeddings.shape[-1]}"
            )
        if self.use_chienn:
            if reactant_chirality_embeddings.shape[-1] != self.chienn_dim:
                raise ValueError(
                    f"Expected ChIRo reactant dim {self.chienn_dim}, "
                    f"got {reactant_chirality_embeddings.shape[-1]}"
                )
            if product_chirality_embeddings.shape[-1] != self.chienn_dim:
                raise ValueError(
                    f"Expected ChIRo product dim {self.chienn_dim}, "
                    f"got {product_chirality_embeddings.shape[-1]}"
                )
        if self.use_reaction_chemistry:
            if (
                reaction_chemistry_vector.ndim != 2
                or reaction_chemistry_vector.shape[-1] != self.reaction_chemistry_dim
            ):
                raise ValueError(
                    "reaction_chemistry_vector must have shape "
                    f"[B, {self.reaction_chemistry_dim}], got "
                    f"{tuple(reaction_chemistry_vector.shape)}"
                )
        if self.use_reaction_directional:
            if (
                reaction_directional_vector.ndim != 2
                or reaction_directional_vector.shape[-1] != self.reaction_directional_dim
            ):
                raise ValueError(
                    "reaction_directional_vector must have shape "
                    f"[B, {self.reaction_directional_dim}], got "
                    f"{tuple(reaction_directional_vector.shape)}"
                )

        batch_size = reactant_embeddings.shape[0]
        device = reactant_embeddings.device

        should_return_attention = (
            self.return_attention if return_attention is None else return_attention
        )
        unimol_composition, unimol_reactant_attention, unimol_product_attention = (
            self._encode_molecule_modality(
                self.unimol_reactant_pooling,
                self.unimol_product_pooling,
                reactant_embeddings,
                reactant_padding_mask,
                product_embeddings,
                product_padding_mask,
                should_return_attention,
            )
        )
        token_list = []
        modality_valid = []
        if self.use_reaction_model:
            if self.reaction_projection is None:
                raise RuntimeError("reaction_projection was not initialized")
            token_list.append(self.reaction_projection(reaction_embedding))
            modality_valid.append(torch.ones(batch_size, dtype=torch.bool, device=device))
        token_list.append(self.unimol_projection(unimol_composition))
        modality_valid.append(
            (
                torch.ones(
                    batch_size,
                    dtype=torch.bool,
                    device=device,
                )
                if has_unimol2 is None
                else has_unimol2.to(device=device, dtype=torch.bool)
            )
        )
        chienn_reactant_attention = None
        chienn_product_attention = None
        if self.use_chienn:
            chienn_composition, chienn_reactant_attention, chienn_product_attention = (
                self._encode_molecule_modality(
                    self.chienn_reactant_pooling,
                    self.chienn_product_pooling,
                    reactant_chirality_embeddings,
                    reactant_chirality_padding_mask,
                    product_chirality_embeddings,
                    product_chirality_padding_mask,
                    should_return_attention,
                )
            )
            token_list.append(self.chienn_projection(chienn_composition))
            chirality_valid = has_chirality
            if chirality_valid is None:
                chirality_valid = has_chiro
            if chirality_valid is None:
                chirality_valid = has_chienn
            modality_valid.append(
                (
                    torch.ones(
                        batch_size,
                        dtype=torch.bool,
                        device=device,
                    )
                    if chirality_valid is None
                    else chirality_valid.to(device=device, dtype=torch.bool)
                )
            )
        if self.use_reaction_chemistry:
            if self.reaction_chemistry_projection is None:
                raise RuntimeError("reaction_chemistry_projection was not initialized")
            token_list.append(self.reaction_chemistry_projection(reaction_chemistry_vector))
            modality_valid.append(
                (
                    torch.ones(
                        batch_size,
                        dtype=torch.bool,
                        device=device,
                    )
                    if has_reaction_chemistry is None
                    else has_reaction_chemistry.to(
                        device=device,
                        dtype=torch.bool,
                    )
                )
            )

        modality_mask = self._apply_modality_dropout(torch.stack(modality_valid, dim=1))
        modality_mask = self._apply_chemistry_dropout(modality_mask)
        adaptive_weights = None
        normalized_prior_weights = None
        if self.modality_fusion == "factorized_concat":
            base_weights = self.factorized_weight_values.to(
                device=device,
                dtype=token_list[0].dtype,
            ).unsqueeze(0)
            effective_weights = base_weights * modality_mask.to(base_weights.dtype)
            modality_weights = effective_weights / effective_weights.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
            modality_logits = base_weights.expand(batch_size, -1).log()
            scaled_blocks = []
            for index, token in enumerate(token_list):
                normalized = F.normalize(token, p=2, dim=-1)
                scale = modality_weights[:, index].sqrt().unsqueeze(-1)
                scaled_blocks.append(normalized * scale)
            pooled = torch.cat(scaled_blocks, dim=-1)
        else:
            tokens = self.modality_token_norm(torch.stack(token_list, dim=1))
            if self.modality_l2_normalize:
                tokens = F.normalize(tokens, p=2, dim=-1) * math.sqrt(self.token_dim)
            modality_logits = self.modality_attention(tokens).squeeze(-1)
            if self.modality_fusion == "mean":
                modality_weights = modality_mask.to(dtype=tokens.dtype)
                modality_weights = modality_weights / modality_weights.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1.0)
            elif self.modality_fusion == "prior_bounded_attention":
                base_prior = self.attention_prior_values.to(
                    device=device,
                    dtype=tokens.dtype,
                ).unsqueeze(0)
                effective_prior = base_prior * modality_mask.to(base_prior.dtype)
                normalized_prior_weights = effective_prior / effective_prior.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-12)
                modality_logits = base_prior.clamp_min(1e-12).log() + modality_logits
                masked_logits = modality_logits.masked_fill(
                    ~modality_mask,
                    torch.finfo(modality_logits.dtype).min,
                )
                max_logits = masked_logits.max(dim=1, keepdim=True).values
                adaptive_unnormalized = torch.exp(masked_logits - max_logits) * modality_mask.to(
                    masked_logits.dtype
                )
                adaptive_weights = adaptive_unnormalized / adaptive_unnormalized.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-12)
                strength = self.attention_adaptation_strength
                modality_weights = (
                    1.0 - strength
                ) * normalized_prior_weights + strength * adaptive_weights
            else:
                modality_logits = modality_logits.masked_fill(
                    ~modality_mask,
                    torch.finfo(modality_logits.dtype).min,
                )
                modality_weights = torch.softmax(modality_logits, dim=1)
            pooled = torch.einsum("bt,btd->bd", modality_weights, tokens)

        projection_details: dict[str, torch.Tensor] = {}
        if isinstance(self.projection_encoder, ResidualMLPProjection):
            projection_result = self.projection_encoder(
                pooled,
                return_details=should_return_attention,
            )
            if should_return_attention:
                output, projection_details = projection_result
            else:
                output = projection_result
        else:
            output = self.projection_encoder(pooled)
        directional_gate = None
        directional_mask = None
        if self.use_reaction_directional:
            if self.reaction_directional_projection is None or self.raw_directional_gate is None:
                raise RuntimeError("Directional residual modules were not initialized")
            directional = self.reaction_directional_projection(reaction_directional_vector)
            directional_mask = (
                torch.ones(batch_size, dtype=torch.bool, device=device)
                if has_reaction_directional is None
                else has_reaction_directional.to(device=device, dtype=torch.bool)
            )
            directional_gate = torch.sigmoid(self.raw_directional_gate)
            output = output + (
                directional_mask.to(output.dtype).unsqueeze(-1) * directional_gate * directional
            )
        for layer in self.post_nn_layers:
            output = layer(output)

        if not should_return_attention:
            return output

        entropy = -(modality_weights.clamp_min(1e-12).log() * modality_weights).sum(dim=1)
        attention = {
            "modality": modality_weights,
            "modality_logits": modality_logits,
            "modality_mask": modality_mask,
            "modality_entropy": entropy,
            "modality_names": self.modality_names,
            "unimol2_reactant": unimol_reactant_attention,
            "unimol2_product": unimol_product_attention,
        }
        attention.update(projection_details)
        if self.modality_fusion == "factorized_concat":
            attention["modality_block_norms"] = torch.stack(
                [block.norm(dim=-1) for block in scaled_blocks],
                dim=1,
            )
        else:
            attention["modality_token_norms"] = tokens.norm(dim=-1)
        if self.modality_fusion == "prior_bounded_attention":
            attention["modality_prior"] = normalized_prior_weights
            attention["modality_adaptive"] = adaptive_weights
            attention["attention_adaptation_strength"] = torch.full(
                (batch_size,),
                self.attention_adaptation_strength,
                device=output.device,
                dtype=output.dtype,
            )
        if self.use_reaction_directional:
            attention["directional_gate"] = directional_gate.expand(batch_size)
            attention["directional_mask"] = directional_mask
        if self.use_chienn:
            attention.update(
                {
                    "chirality_reactant": chienn_reactant_attention,
                    "chirality_product": chienn_product_attention,
                    f"{self.chirality_modality_name}_reactant": chienn_reactant_attention,
                    f"{self.chirality_modality_name}_product": chienn_product_attention,
                    "chienn_reactant": chienn_reactant_attention,
                    "chienn_product": chienn_product_attention,
                }
            )
        return output, attention


class MLP(BaseModel):
    """
    Multi-Layer Perceptron (MLP) neural network.

    Implements a flexible MLP architecture with customizable layers, activation
    functions, layer normalization, dropout, and optional output normalization.
    This is the primary encoder architecture used in the Horizyn SOTA model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        activations: nn.Module | list[nn.Module] = nn.ReLU(),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalise_output: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the MLP.

        Args:
            input_dim: Dimension of the input features.
            output_dim: Dimension of the output features.
            num_layers: Number of hidden layers (default: 1).
            widths: Width(s) of hidden layers. If int, all layers have same width.
                If list, each element specifies width of corresponding layer.
            activations: Activation function(s). If single Module, used for all layers.
                If list, each element specifies activation for corresponding layer.
            use_layer_norm: Whether to apply layer normalization after each hidden layer.
            dropout: Dropout probability (0.0 means no dropout).
            bias: Whether to include bias terms in linear layers.
            normalise_output: Whether to L2-normalize the final output.
            *args: Additional arguments (must be empty).
            **kwargs: Additional keyword arguments (must be empty).

        Example:
            >>> # SOTA reaction encoder: 2048 → 4096 → 512
            >>> mlp = MLP(
            ...     input_dim=2048,
            ...     output_dim=512,
            ...     num_layers=1,
            ...     widths=4096,
            ...     normalise_output=True
            ... )
        """
        super(MLP, self).__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Validate core hyperparameters early (fail fast)
        if num_layers < 0:
            raise ValueError("num_layers must be >= 0")
        if isinstance(widths, int):
            if widths <= 0 and num_layers > 0:
                raise ValueError("widths must be a positive integer when num_layers > 0")
        else:
            if len(widths) == 0 and num_layers > 0:
                raise ValueError("widths list must be non-empty when num_layers > 0")
            if any(w <= 0 for w in widths):
                raise ValueError("all hidden layer widths must be positive integers")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in the range [0.0, 1.0]")

        # Build the main neural network
        self._build_network(num_layers, widths, activations, use_layer_norm, dropout, bias)

        # Add output normalization if requested
        if normalise_output:
            self.post_nn_layers.append(NormalizeLayer(p=2, dim=-1))

    def _build_network(
        self,
        num_layers: int,
        widths: int | list[int],
        activations: nn.Module | list[nn.Module],
        use_layer_norm: bool,
        dropout: float,
        bias: bool,
    ) -> None:
        """
        Build the main neural network structure.

        Constructs the layers of the MLP based on the provided parameters,
        including linear layers, activations, layer normalization, and dropout.

        Notes:
            - If `widths` is a list, its length defines the number of hidden layers
              and overrides `num_layers`.
            - If `activations` is provided as a single nn.Module instance, a deep copy
              of that instance is used per hidden layer to avoid reusing the same
              module object across layers.

        Args:
            num_layers: Number of hidden layers.
            widths: Width(s) of hidden layers.
            activations: Activation function(s) to use.
            use_layer_norm: Whether to use layer normalization.
            dropout: Dropout probability.
            bias: Whether to include bias in linear layers.
        """
        # Ensure widths is a list
        if isinstance(widths, int):
            widths = [widths] * num_layers
        else:
            num_layers = len(widths)

        # Ensure activations is a list
        if not isinstance(activations, list):
            # Use deep copies so each layer gets its own module instance
            activations = [copy.deepcopy(activations) for _ in range(num_layers)]
        if len(activations) != num_layers:
            raise ValueError("Number of activations must match number of hidden layers")
        if any(not isinstance(act, nn.Module) for act in activations):
            raise ValueError("All activations must be instances of nn.Module")

        prev_dim = self.input_dim

        # Construct hidden layers
        for width, activation in zip(widths, activations):
            self.main_nn.append(nn.Linear(prev_dim, width, bias=bias))
            self.main_nn.append(activation)
            if use_layer_norm:
                self.main_nn.append(nn.LayerNorm(width))
            if dropout > 0:
                self.main_nn.append(nn.Dropout(dropout))
            prev_dim = width

        # Add output layer
        self.main_nn.append(nn.Linear(prev_dim, self.output_dim, bias=bias))


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


class ProteinPooledDualModel(BaseModel):
    """
    Dual encoder that pools residue-level protein embeddings before enzyme encoding.

    This model keeps the Horizyn-style reusable enzyme embedding: protein
    residues are pooled independently of the reaction, then the pooled vector is
    optionally passed through a frozen or trainable Stage 1 hyperbolic EC
    projector before the existing target MLP encoder. ``pooling="mean"``
    reproduces masked mean pooling, ``pooling="attention"`` learns residue
    weights, ``pooling="sleec"`` learns SLEEC-style functional-residue weights,
    and ``pooling="sleec_guided_attention"`` uses the SLEEC-prior attention
    pooler from Stage 1.
    """

    def __init__(
        self,
        query_encoder_kwargs: dict[str, Any],
        target_encoder_kwargs: dict[str, Any],
        residue_dim: int = 1024,
        pooling: str = "mean",
        attention_bias: bool = True,
        sleec_mode: str = "topk",
        sleec_topk_fraction: float = 0.2,
        sleec_threshold: float = 0.5,
        sleec_scorer_hidden_dim: int = 256,
        sleec_score_hidden_dim: int | None = None,
        sleec_checkpoint_path: str | None = None,
        sleec_freeze_scorer: bool = False,
        sleec_guided_initial_bias_scale: float = 1.0,
        sleec_guided_train_bias_scale: bool = True,
        hyperbolic_checkpoint_path: str | None = None,
        hyperbolic_freeze_projector: bool = False,
        hyperbolic_use_tangent: bool = True,
        hyperbolic_hyp_dim: int | None = None,
        hyperbolic_load_attention_pooler: bool = False,
        hyperbolic_freeze_attention_pooler: bool = False,
        enzyme_input_mode: str = "standard",
        enzyme_fusion_hidden_dim: int | None = None,
        enzyme_fusion_dropout: float = 0.0,
        enzyme_block_dims: dict[str, int] | None = None,
        enzyme_block_weights: dict[str, float] | None = None,
        enzyme_block_dropout: float = 0.0,
        enzyme_block_learned_weights: bool = False,
        capability_vector_dim: int | None = None,
        capability_freeze: bool = True,
        capability_adapter: bool = False,
        capability_dropout: float = 0.1,
        factorized_capability_dims: dict[str, int] | None = None,
        factorized_capability_freeze: bool = True,
        factorized_capability_use_masks: bool = True,
        biofp_center_dim: int = 0,
        biofp_cofactor_dim: int = 0,
        biofp_transition_dim: int = 0,
        biofp_family_dims: dict[str, int] | None = None,
        biofp_seq_dim: int = 384,
        biofp_dim: int = 128,
        biofp_hidden_dim: int = 512,
        biofp_seq_weight: float = 0.75,
        biofp_dropout: float = 0.1,
        text_vector_dim: int | None = None,
        text_fusion_dim: int = 512,
        text_num_heads: int = 8,
        text_dropout: float = 0.1,
        text_freeze: bool = True,
        text_adapter: bool = False,
        reaction_hyperbolic_checkpoint_path: str | None = None,
        reaction_hyperbolic_freeze_encoder: bool = True,
        reaction_hyperbolic_freeze_projector: bool = True,
        reaction_hyperbolic_use_tangent: bool = True,
        reaction_fingerprint_attention_enabled: bool = False,
        reaction_fingerprint_attention_input_dim: int = 2048,
        reaction_fingerprint_attention_rdkit_dim: int = 1024,
        reaction_fingerprint_attention_drfp_dim: int = 1024,
        reaction_fingerprint_attention_token_dim: int = 512,
        reaction_fingerprint_attention_hidden_dim: int = 512,
        reaction_fingerprint_attention_dropout: float = 0.0,
        reaction_fingerprint_attention_bias: bool = True,
        e2r_adapter_enabled: bool = False,
        e2r_adapter_hidden_dim: int = 512,
        e2r_adapter_dropout: float = 0.1,
        e2r_adapter_gate_init: float = 0.1,
        e2r_adapter_use_factorized_inputs: bool = False,
        e2r_adapter_use_directional_inputs: bool = False,
        e2r_adapter_directional_dim: int | None = None,
        e2r_adapter_directional_hidden_dim: int = 128,
        e2r_adapter_reaction_model_dim: int = 768,
        e2r_adapter_unimol_dim: int = 768,
        e2r_adapter_chiro_dim: int = 256,
        e2r_adapter_chemistry_dim: int | None = None,
        r2e_adapter_enabled: bool = False,
        r2e_adapter_hidden_dim: int = 512,
        r2e_adapter_dropout: float = 0.1,
        r2e_adapter_gate_init: float = 0.1,
        r2e_adapter_use_factorized_inputs: bool = False,
        r2e_adapter_block_dims: dict[str, int] | None = None,
        r2e_adapter_block_weights: dict[str, float] | None = None,
        return_attention: bool = False,
        query_encoder: type[BaseModel] = MLP,
        target_encoder: type[BaseModel] = MLP,
        enforce_normalisation: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if pooling not in {"mean", "avg", "attention", "sleec", "sleec_guided_attention"}:
            raise ValueError(
                "pooling must be one of: mean, avg, attention, sleec, sleec_guided_attention"
            )
        if pooling == "avg":
            pooling = "mean"
        if enzyme_input_mode not in {
            "standard",
            "raw_sleec_hyperbolic_concat",
            "raw_mean_sleec_hyperbolic_gated",
            "raw_mean_sleec_hyperbolic_capability_gated",
            "raw_mean_sleec_hyperbolic_text_gated",
            "raw_mean_sleec_blockwise",
            "raw_mean_sleec_hyperbolic_blockwise",
            "raw_mean_sleec_hyperbolic_capability_blockwise",
            "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
            "raw_mean_sleec_biofp_split",
            "raw_mean_sleec_biological_factorized",
        }:
            raise ValueError(
                "enzyme_input_mode must be one of: standard, raw_sleec_hyperbolic_concat, "
                "raw_mean_sleec_hyperbolic_gated, "
                "raw_mean_sleec_hyperbolic_capability_gated, "
                "raw_mean_sleec_hyperbolic_text_gated, raw_mean_sleec_blockwise, "
                "raw_mean_sleec_hyperbolic_blockwise, "
                "raw_mean_sleec_hyperbolic_capability_blockwise, "
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise, "
                "raw_mean_sleec_biofp_split, raw_mean_sleec_biological_factorized"
            )
        if (
            enzyme_input_mode
            in {
                "raw_sleec_hyperbolic_concat",
                "raw_mean_sleec_hyperbolic_gated",
                "raw_mean_sleec_hyperbolic_capability_gated",
                "raw_mean_sleec_hyperbolic_text_gated",
                "raw_mean_sleec_hyperbolic_blockwise",
                "raw_mean_sleec_hyperbolic_capability_blockwise",
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
            }
            and not hyperbolic_checkpoint_path
        ):
            raise ValueError(
                f"enzyme_input_mode='{enzyme_input_mode}' requires hyperbolic_checkpoint_path"
            )
        if (
            enzyme_input_mode
            in {
                "raw_sleec_hyperbolic_concat",
                "raw_mean_sleec_hyperbolic_gated",
                "raw_mean_sleec_hyperbolic_capability_gated",
                "raw_mean_sleec_hyperbolic_text_gated",
                "raw_mean_sleec_hyperbolic_blockwise",
                "raw_mean_sleec_hyperbolic_capability_blockwise",
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
                "raw_mean_sleec_biological_factorized",
            }
            and not hyperbolic_use_tangent
        ):
            raise ValueError(
                f"enzyme_input_mode='{enzyme_input_mode}' requires hyperbolic_use_tangent=True"
            )

        stage1_checkpoint = (
            _torch_load_checkpoint(hyperbolic_checkpoint_path)
            if hyperbolic_checkpoint_path
            else None
        )
        stage1_config = _stage1_config(stage1_checkpoint) if stage1_checkpoint else {}
        reaction_stage1_checkpoint = (
            _torch_load_checkpoint(reaction_hyperbolic_checkpoint_path)
            if reaction_hyperbolic_checkpoint_path
            else None
        )
        reaction_stage1_config = (
            _stage1_config(reaction_stage1_checkpoint) if reaction_stage1_checkpoint else {}
        )
        if reaction_fingerprint_attention_enabled and reaction_stage1_checkpoint is not None:
            raise ValueError(
                "reaction_fingerprint_attention cannot be enabled together with "
                "reaction_hyperbolic_encoder.checkpoint_path"
            )
        if reaction_fingerprint_attention_enabled:
            expected_query_input_dim = int(reaction_fingerprint_attention_token_dim)
            if int(query_encoder_kwargs["input_dim"]) != expected_query_input_dim:
                raise ValueError(
                    "Query encoder input_dim must match reaction fingerprint attention "
                    f"token_dim: query input_dim={query_encoder_kwargs['input_dim']}, "
                    f"token_dim={expected_query_input_dim}"
                )

        self.query_encoder = query_encoder(**query_encoder_kwargs)
        self.target_encoder = target_encoder(**target_encoder_kwargs)
        self.pooling_name = pooling
        self.enzyme_input_mode = enzyme_input_mode
        self.residue_dim = residue_dim
        self.hyperbolic_use_tangent = bool(hyperbolic_use_tangent)
        self.hyperbolic_projector: nn.Module | None = None
        self.hyperbolic_tangent_dim: int | None = None
        self.reaction_hyperbolic_use_tangent = bool(reaction_hyperbolic_use_tangent)
        self.reaction_hyperbolic_input_dim: int | None = None
        self.reaction_hyperbolic_encoder: nn.Module | None = None
        self.reaction_hyperbolic_projector: nn.Module | None = None
        self.reaction_fingerprint_attention: ReactionFingerprintAttentionPool | None = None
        self.raw_mean_pooling: ProteinMeanPooling | None = None
        self.enzyme_feature_fusion: GatedEnzymeFeatureFusion | None = None
        self.enzyme_block_fusion: BlockwiseEnzymeFeatureFusion | None = None
        self.blockwise_target_is_embedding = False
        self.biofp_split_encoder: EnzymeBioFPSplitEncoder | None = None
        self.biological_factorized_encoder: EnzymeBiologicalFactorizedEncoder | None = None
        self.text_feature_fusion: TigerTextGatedFusion | None = None
        self.capability_vector_dim = (
            None if capability_vector_dim is None else int(capability_vector_dim)
        )
        self.capability_freeze = bool(capability_freeze)
        self.capability_adapter_enabled = bool(capability_adapter)
        self.capability_adapter: nn.Module | None = None
        self.factorized_capability_dims = (
            None
            if factorized_capability_dims is None
            else {str(name): int(dim) for name, dim in factorized_capability_dims.items()}
        )
        self.factorized_capability_freeze = bool(factorized_capability_freeze)
        self.factorized_capability_use_masks = bool(factorized_capability_use_masks)
        self.text_vector_dim = None if text_vector_dim is None else int(text_vector_dim)
        self.text_freeze = bool(text_freeze)
        self.text_adapter_enabled = bool(text_adapter)
        self.text_adapter: nn.Module | None = None
        self.e2r_adapter = (
            E2RReactionAdapter(
                embedding_dim=int(query_encoder_kwargs["output_dim"]),
                hidden_dim=int(e2r_adapter_hidden_dim),
                dropout=float(e2r_adapter_dropout),
                gate_init=float(e2r_adapter_gate_init),
                use_factorized_inputs=bool(e2r_adapter_use_factorized_inputs),
                use_directional_inputs=bool(e2r_adapter_use_directional_inputs),
                reaction_model_dim=int(e2r_adapter_reaction_model_dim),
                unimol_dim=int(e2r_adapter_unimol_dim),
                chiro_dim=int(e2r_adapter_chiro_dim),
                chemistry_dim=e2r_adapter_chemistry_dim,
                directional_dim=e2r_adapter_directional_dim,
                directional_hidden_dim=int(e2r_adapter_directional_hidden_dim),
            )
            if e2r_adapter_enabled
            else None
        )
        self.r2e_adapter = (
            R2EEnzymeAdapter(
                embedding_dim=int(target_encoder_kwargs["output_dim"]),
                hidden_dim=int(r2e_adapter_hidden_dim),
                dropout=float(r2e_adapter_dropout),
                gate_init=float(r2e_adapter_gate_init),
                use_factorized_inputs=bool(r2e_adapter_use_factorized_inputs),
                block_dims=r2e_adapter_block_dims,
                block_weights=r2e_adapter_block_weights,
            )
            if r2e_adapter_enabled
            else None
        )
        if self.enzyme_input_mode in {
            "raw_mean_sleec_hyperbolic_capability_gated",
            "raw_mean_sleec_hyperbolic_capability_blockwise",
        }:
            if self.capability_vector_dim is None or self.capability_vector_dim <= 0:
                raise ValueError(
                    "capability_vector_dim must be positive for " f"{self.enzyme_input_mode}"
                )
            if not (0.0 <= capability_dropout <= 1.0):
                raise ValueError("capability_dropout must be in the range [0, 1]")
            if self.capability_adapter_enabled:
                self.capability_adapter = nn.Sequential(
                    nn.LayerNorm(self.capability_vector_dim),
                    nn.Linear(self.capability_vector_dim, self.capability_vector_dim),
                    nn.GELU(),
                    nn.Dropout(float(capability_dropout)),
                    nn.Linear(self.capability_vector_dim, self.capability_vector_dim),
                )
        if self.enzyme_input_mode == "raw_mean_sleec_hyperbolic_factorized_capability_blockwise":
            expected_families = set(BlockwiseEnzymeFeatureFusion.FACTORIZED_CAPABILITY_BRANCHES)
            if self.factorized_capability_dims is None:
                raise ValueError(
                    "factorized_capability_dims must be configured for " f"{self.enzyme_input_mode}"
                )
            if set(self.factorized_capability_dims) != expected_families:
                raise ValueError(
                    "factorized_capability_dims must contain exactly "
                    f"{sorted(expected_families)}"
                )
        if self.enzyme_input_mode == "raw_mean_sleec_hyperbolic_text_gated":
            if self.text_vector_dim is None or self.text_vector_dim <= 0:
                raise ValueError(
                    "text_vector_dim must be positive for " "raw_mean_sleec_hyperbolic_text_gated"
                )
            if int(text_fusion_dim) <= 0:
                raise ValueError("text_fusion_dim must be positive")
            if int(text_num_heads) <= 0:
                raise ValueError("text_num_heads must be positive")
            if int(text_fusion_dim) % int(text_num_heads) != 0:
                raise ValueError("text_fusion_dim must be divisible by text_num_heads")
            if not (0.0 <= text_dropout <= 1.0):
                raise ValueError("text_dropout must be in the range [0, 1]")
            if self.text_adapter_enabled:
                self.text_adapter = nn.Sequential(
                    nn.LayerNorm(self.text_vector_dim),
                    nn.Linear(self.text_vector_dim, self.text_vector_dim),
                    nn.GELU(),
                    nn.Dropout(float(text_dropout)),
                    nn.Linear(self.text_vector_dim, self.text_vector_dim),
                )
        if self.enzyme_input_mode == "raw_mean_sleec_biofp_split":
            if int(target_encoder_kwargs["input_dim"]) != int(target_encoder_kwargs["output_dim"]):
                raise ValueError(
                    "raw_mean_sleec_biofp_split expects target_encoder input_dim "
                    "to match output_dim because the BioFP split encoder emits the "
                    "final enzyme embedding"
                )
            if int(biofp_seq_dim) + int(biofp_dim) != int(target_encoder_kwargs["output_dim"]):
                raise ValueError(
                    "model.biofp.seq_dim + model.biofp.dim must equal target "
                    f"embedding dim {target_encoder_kwargs['output_dim']}"
                )
            self.raw_mean_pooling = ProteinMeanPooling(return_attention=False)
            self.biofp_split_encoder = EnzymeBioFPSplitEncoder(
                residue_dim=residue_dim,
                output_dim=int(target_encoder_kwargs["output_dim"]),
                seq_dim=int(biofp_seq_dim),
                bio_dim=int(biofp_dim),
                hidden_dim=int(biofp_hidden_dim),
                family_output_dims={
                    "center": int(biofp_center_dim),
                    "cofactor": int(biofp_cofactor_dim),
                    "transition": int(biofp_transition_dim),
                },
                seq_weight=float(biofp_seq_weight),
                dropout=float(biofp_dropout),
            )
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False

        if self.enzyme_input_mode == "raw_mean_sleec_biological_factorized":
            if enzyme_block_dims is None or enzyme_block_weights is None:
                raise ValueError(
                    "raw_mean_sleec_biological_factorized requires enzyme block dims and weights"
                )
            if int(target_encoder_kwargs["input_dim"]) != int(target_encoder_kwargs["output_dim"]):
                raise ValueError(
                    "raw_mean_sleec_biological_factorized emits the final enzyme embedding"
                )

        if self.enzyme_input_mode == "raw_mean_sleec_blockwise":
            self.raw_mean_pooling = ProteinMeanPooling(return_attention=False)
            self.enzyme_block_fusion = BlockwiseEnzymeFeatureFusion(
                raw_dim=residue_dim,
                pooled_dim=residue_dim,
                output_dim=int(target_encoder_kwargs["output_dim"]),
                block_dims=enzyme_block_dims,
                block_weights=enzyme_block_weights,
                dropout=float(enzyme_block_dropout),
            )
            self.blockwise_target_is_embedding = True
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False

        if reaction_fingerprint_attention_enabled:
            self.reaction_fingerprint_attention = ReactionFingerprintAttentionPool(
                input_dim=int(reaction_fingerprint_attention_input_dim),
                rdkit_dim=int(reaction_fingerprint_attention_rdkit_dim),
                drfp_dim=int(reaction_fingerprint_attention_drfp_dim),
                token_dim=int(reaction_fingerprint_attention_token_dim),
                hidden_dim=int(reaction_fingerprint_attention_hidden_dim),
                dropout=float(reaction_fingerprint_attention_dropout),
                attention_bias=bool(reaction_fingerprint_attention_bias),
            )

        if pooling == "mean":
            self.pooling = ProteinMeanPooling(return_attention=return_attention)
        elif pooling == "attention":
            self.pooling = ProteinAttentionPooling(
                hidden_dim=residue_dim,
                attention_bias=attention_bias,
                return_attention=return_attention,
            )
        elif pooling == "sleec":
            self.pooling = SLEECFunctionalPool(
                hidden_dim=residue_dim,
                scorer_hidden_dim=sleec_scorer_hidden_dim,
                score_hidden_dim=sleec_score_hidden_dim,
                mode=sleec_mode,
                topk_fraction=sleec_topk_fraction,
                threshold=sleec_threshold,
                checkpoint_path=sleec_checkpoint_path,
                freeze_scorer=sleec_freeze_scorer,
            )
        else:
            self.pooling = SLEECGuidedAttentionPool(
                hidden_dim=residue_dim,
                scorer_hidden_dim=sleec_scorer_hidden_dim,
                threshold=sleec_threshold,
                checkpoint_path=(
                    None
                    if hyperbolic_load_attention_pooler and stage1_checkpoint is not None
                    else sleec_checkpoint_path
                ),
                freeze_scorer=sleec_freeze_scorer,
                attention_bias=attention_bias,
                initial_sleec_bias_scale=sleec_guided_initial_bias_scale,
                train_sleec_bias_scale=sleec_guided_train_bias_scale,
            )
            if hyperbolic_load_attention_pooler and stage1_checkpoint is not None:
                pool_state = stage1_checkpoint.get("sleec_guided_attention_pool_state_dict")
                if pool_state is None:
                    raise ValueError(
                        "Requested hyperbolic_load_attention_pooler=True, but "
                        f"{hyperbolic_checkpoint_path} has no "
                        "sleec_guided_attention_pool_state_dict"
                    )
                pool_state = dict(pool_state)
                if "raw_prior_strength" in pool_state and "raw_sleec_bias_scale" not in pool_state:
                    pool_state["raw_sleec_bias_scale"] = pool_state.pop("raw_prior_strength")
                self.pooling.load_state_dict(pool_state)
            if hyperbolic_freeze_attention_pooler:
                for parameter in self.pooling.parameters():
                    parameter.requires_grad = False

        if stage1_checkpoint is not None:
            projector_input_dim = int(stage1_config.get("input_dim", residue_dim))
            projector_hyp_dim = int(
                stage1_config.get("hyp_dim", target_encoder_kwargs["input_dim"])
            )
            is_lorentz_checkpoint = (
                "hyperbolic_projector_state_dict" in stage1_checkpoint
                or "projector_input_normalization" in stage1_config
            )
            if not is_lorentz_checkpoint:
                raise ValueError(
                    "Legacy non-Lorentz hyperbolic enzyme checkpoints are no longer supported. "
                    "Use a Lorentz hyperbolic enzyme checkpoint produced by "
                    "scripts/pretrain_hyperbolic_enzyme.py."
                )
            if projector_input_dim != residue_dim:
                raise ValueError(
                    "Hyperbolic projector input_dim must match residue pooled dim: "
                    f"checkpoint input_dim={projector_input_dim}, residue_dim={residue_dim}"
                )
            expected_target_input_dim = (
                projector_hyp_dim
                if self.hyperbolic_use_tangent or not is_lorentz_checkpoint
                else projector_hyp_dim + 1
            )
            if self.enzyme_input_mode == "raw_sleec_hyperbolic_concat":
                expected_target_input_dim = residue_dim + projector_hyp_dim
            if (
                self.enzyme_input_mode
                not in {
                    "raw_mean_sleec_hyperbolic_gated",
                    "raw_mean_sleec_hyperbolic_capability_gated",
                    "raw_mean_sleec_hyperbolic_text_gated",
                    "raw_mean_sleec_biological_factorized",
                }
                and int(target_encoder_kwargs["input_dim"]) != expected_target_input_dim
            ):
                raise ValueError(
                    "Target encoder input_dim must match hyperbolic hyp_dim when a "
                    "hyperbolic checkpoint is used: "
                    f"target input_dim={target_encoder_kwargs['input_dim']}, "
                    f"expected={expected_target_input_dim}"
                )
            from horizyn.hyperbolic_enzyme import LorentzEnzymeProjector

            self.hyperbolic_projector = LorentzEnzymeProjector(
                input_dim=projector_input_dim,
                hyp_dim=projector_hyp_dim,
                curvature=float(
                    stage1_config.get("curvature", stage1_checkpoint.get("curvature", 0.25))
                ),
                tangent_clip=stage1_config.get("tangent_clip", 5.0),
                tangent_clip_mode=str(stage1_config.get("tangent_clip_mode", "hard")),
                input_normalization=str(stage1_config.get("projector_input_normalization", "none")),
                eps=float(stage1_config.get("eps", 1e-6)),
            )
            projector_state = stage1_checkpoint.get("hyperbolic_projector_state_dict")
            if projector_state is None:
                projector_state = stage1_checkpoint.get("projector_state_dict")
            if projector_state is None:
                raise ValueError(
                    f"Hyperbolic checkpoint has no projector state dict: {hyperbolic_checkpoint_path}"
                )
            self.hyperbolic_projector.load_state_dict(projector_state)
            self.hyperbolic_tangent_dim = projector_hyp_dim
            if hyperbolic_freeze_projector:
                for parameter in self.hyperbolic_projector.parameters():
                    parameter.requires_grad = False
            if self.enzyme_input_mode in {
                "raw_mean_sleec_hyperbolic_gated",
                "raw_mean_sleec_hyperbolic_capability_gated",
                "raw_mean_sleec_hyperbolic_text_gated",
            }:
                self.raw_mean_pooling = ProteinMeanPooling(return_attention=False)
                self.enzyme_feature_fusion = GatedEnzymeFeatureFusion(
                    raw_dim=residue_dim,
                    pooled_dim=residue_dim,
                    hyperbolic_dim=projector_hyp_dim,
                    capability_dim=(
                        self.capability_vector_dim
                        if self.enzyme_input_mode == "raw_mean_sleec_hyperbolic_capability_gated"
                        else None
                    ),
                    output_dim=int(target_encoder_kwargs["input_dim"]),
                    gate_hidden_dim=enzyme_fusion_hidden_dim,
                    dropout=float(enzyme_fusion_dropout),
                )
                if self.enzyme_input_mode == "raw_mean_sleec_hyperbolic_text_gated":
                    if self.text_vector_dim is None:
                        raise RuntimeError("text_vector_dim was not initialized")
                    self.text_feature_fusion = TigerTextGatedFusion(
                        sequence_dim=int(target_encoder_kwargs["input_dim"]),
                        text_dim=self.text_vector_dim,
                        output_dim=int(target_encoder_kwargs["input_dim"]),
                        fusion_dim=int(text_fusion_dim),
                        num_heads=int(text_num_heads),
                        dropout=float(text_dropout),
                    )
            if self.enzyme_input_mode in {
                "raw_mean_sleec_hyperbolic_blockwise",
                "raw_mean_sleec_hyperbolic_capability_blockwise",
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
            }:
                self.raw_mean_pooling = ProteinMeanPooling(return_attention=False)
                self.enzyme_block_fusion = BlockwiseEnzymeFeatureFusion(
                    raw_dim=residue_dim,
                    pooled_dim=residue_dim,
                    hyperbolic_dim=projector_hyp_dim,
                    capability_dim=(
                        self.capability_vector_dim
                        if self.enzyme_input_mode
                        == "raw_mean_sleec_hyperbolic_capability_blockwise"
                        else None
                    ),
                    factorized_capability_dims=(
                        self.factorized_capability_dims
                        if self.enzyme_input_mode
                        == "raw_mean_sleec_hyperbolic_factorized_capability_blockwise"
                        else None
                    ),
                    factorized_capability_use_masks=self.factorized_capability_use_masks,
                    output_dim=int(target_encoder_kwargs["output_dim"]),
                    block_dims=enzyme_block_dims,
                    block_weights=enzyme_block_weights,
                    dropout=float(enzyme_block_dropout),
                )
                self.blockwise_target_is_embedding = True
                for parameter in self.target_encoder.parameters():
                    parameter.requires_grad = False

        if self.enzyme_input_mode == "raw_mean_sleec_biological_factorized":
            uses_ec_block = enzyme_block_dims is not None and "ec" in enzyme_block_dims
            if uses_ec_block and self.hyperbolic_projector is None:
                from horizyn.hyperbolic_enzyme import LorentzEnzymeProjector

                random_hyp_dim = int(hyperbolic_hyp_dim or 128)
                self.hyperbolic_projector = LorentzEnzymeProjector(
                    input_dim=residue_dim,
                    hyp_dim=random_hyp_dim,
                    curvature=0.25,
                    tangent_clip=5.0,
                    tangent_clip_mode="hard",
                    input_normalization="none",
                    eps=1e-6,
                )
                self.hyperbolic_tangent_dim = random_hyp_dim
            if uses_ec_block and self.hyperbolic_tangent_dim is None:
                raise RuntimeError("Hyperbolic tangent dimension was not initialized")
            family_dims = dict(biofp_family_dims or {})
            family_names = [
                name
                for name in (enzyme_block_dims or {})
                if name not in EnzymeBiologicalFactorizedEncoder.RESERVED_BLOCKS
            ]
            if not family_dims and family_names:
                default_family_dims = {
                    "mechanism": int(biofp_transition_dim),
                    "cofactor": int(biofp_cofactor_dim),
                }
                family_dims = {name: default_family_dims.get(name, 0) for name in family_names}
            self.raw_mean_pooling = ProteinMeanPooling(return_attention=False)
            self.biological_factorized_encoder = EnzymeBiologicalFactorizedEncoder(
                residue_dim=residue_dim,
                hyperbolic_dim=(self.hyperbolic_tangent_dim if uses_ec_block else None),
                output_dim=int(target_encoder_kwargs["output_dim"]),
                family_output_dims=family_dims,
                block_dims=enzyme_block_dims,
                block_weights=enzyme_block_weights,
                hidden_dim=int(biofp_hidden_dim),
                dropout=float(biofp_dropout),
                initial_sleec_bias_scale=float(sleec_guided_initial_bias_scale),
                train_sleec_bias_scale=bool(sleec_guided_train_bias_scale),
                learned_block_weights=bool(enzyme_block_learned_weights),
            )
            self.blockwise_target_is_embedding = True
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False

        if reaction_stage1_checkpoint is not None:
            reaction_encoder_dims = reaction_stage1_config.get("reaction_encoder_dims")
            if not isinstance(reaction_encoder_dims, list) or len(reaction_encoder_dims) < 2:
                reaction_encoder_dims = [
                    int(
                        reaction_stage1_config.get(
                            "input_dim", reaction_stage1_checkpoint["input_dim"]
                        )
                    ),
                    int(
                        reaction_stage1_config.get(
                            "reaction_embedding_dim",
                            reaction_stage1_checkpoint.get("reaction_embedding_dim", 512),
                        )
                    ),
                ]
            reaction_encoder_dims = [int(value) for value in reaction_encoder_dims]
            reaction_input_dim = int(
                reaction_stage1_config.get("input_dim", reaction_encoder_dims[0])
            )
            reaction_embedding_dim = int(
                reaction_stage1_config.get(
                    "reaction_embedding_dim",
                    reaction_encoder_dims[-1],
                )
            )
            if reaction_encoder_dims[0] != reaction_input_dim:
                raise ValueError(
                    "Reaction hyperbolic encoder input_dim does not match "
                    f"reaction_encoder_dims: input_dim={reaction_input_dim}, "
                    f"dims[0]={reaction_encoder_dims[0]}"
                )
            if reaction_encoder_dims[-1] != reaction_embedding_dim:
                raise ValueError(
                    "Reaction hyperbolic encoder output dim does not match "
                    f"reaction_embedding_dim={reaction_embedding_dim}, "
                    f"dims[-1]={reaction_encoder_dims[-1]}"
                )
            reaction_hyp_dim = int(
                reaction_stage1_config.get(
                    "hyp_dim",
                    reaction_stage1_checkpoint.get("hyp_dim", query_encoder_kwargs["input_dim"]),
                )
            )
            expected_query_input_dim = (
                reaction_hyp_dim if self.reaction_hyperbolic_use_tangent else reaction_hyp_dim + 1
            )
            if int(query_encoder_kwargs["input_dim"]) != expected_query_input_dim:
                raise ValueError(
                    "Query encoder input_dim must match reaction hyperbolic output dim: "
                    f"query input_dim={query_encoder_kwargs['input_dim']}, "
                    f"expected={expected_query_input_dim}"
                )

            self.reaction_hyperbolic_input_dim = reaction_input_dim
            self.reaction_hyperbolic_encoder = MLP(
                input_dim=reaction_encoder_dims[0],
                output_dim=reaction_encoder_dims[-1],
                num_layers=max(len(reaction_encoder_dims) - 2, 0),
                widths=reaction_encoder_dims[1:-1],
                normalise_output=True,
            )
            self.reaction_hyperbolic_encoder.load_state_dict(
                extract_query_encoder_state_dict(reaction_stage1_checkpoint)
            )

            from horizyn.hyperbolic_enzyme import LorentzEnzymeProjector

            self.reaction_hyperbolic_projector = LorentzEnzymeProjector(
                input_dim=reaction_embedding_dim,
                hyp_dim=reaction_hyp_dim,
                curvature=float(
                    reaction_stage1_config.get(
                        "curvature",
                        reaction_stage1_checkpoint.get("curvature", 0.25),
                    )
                ),
                tangent_clip=reaction_stage1_config.get("tangent_clip", 5.0),
                tangent_clip_mode=str(reaction_stage1_config.get("tangent_clip_mode", "hard")),
                input_normalization=str(
                    reaction_stage1_config.get("projector_input_normalization", "none")
                ),
                eps=float(reaction_stage1_config.get("eps", 1e-6)),
            )
            reaction_projector_state = reaction_stage1_checkpoint.get(
                "hyperbolic_projector_state_dict"
            )
            if reaction_projector_state is None:
                reaction_projector_state = reaction_stage1_checkpoint.get("projector_state_dict")
            if reaction_projector_state is None:
                raise ValueError(
                    "Reaction hyperbolic checkpoint has no projector state dict: "
                    f"{reaction_hyperbolic_checkpoint_path}"
                )
            self.reaction_hyperbolic_projector.load_state_dict(reaction_projector_state)

            if reaction_hyperbolic_freeze_encoder:
                for parameter in self.reaction_hyperbolic_encoder.parameters():
                    parameter.requires_grad = False
            if reaction_hyperbolic_freeze_projector:
                for parameter in self.reaction_hyperbolic_projector.parameters():
                    parameter.requires_grad = False

        if enforce_normalisation:
            ReactionConditionedDualModel._validate_normalized_encoder(
                self.query_encoder,
                "query_encoder",
            )
            ReactionConditionedDualModel._validate_normalized_encoder(
                self.target_encoder,
                "target_encoder",
            )

    def encode_queries(
        self,
        query_inputs: dict | torch.Tensor,
        return_attention: bool = False,
        retrieval_direction: str = "reaction_to_enzyme",
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if retrieval_direction not in {
            "reaction_to_enzyme",
            "enzyme_to_reaction",
        }:
            raise ValueError("retrieval_direction must be reaction_to_enzyme or enzyme_to_reaction")
        adapter_inputs = query_inputs
        query_attention: dict[str, torch.Tensor] = {}
        if self.reaction_fingerprint_attention is not None:
            if isinstance(query_inputs, dict):
                raise ValueError("Reaction fingerprint attention expects fingerprint tensor inputs")
            pooled_result = self.reaction_fingerprint_attention(
                query_inputs,
                return_details=return_attention,
            )
            if return_attention:
                query_inputs, reaction_attention = pooled_result
                query_attention["reaction_fingerprint"] = reaction_attention["weights"]
                query_attention["reaction_fingerprint_logits"] = reaction_attention["logits"]
                query_attention["reaction_fingerprint_entropy"] = reaction_attention["entropy"]
                query_attention["reaction_fingerprint_weight_rdkit_reactants"] = reaction_attention[
                    "weight_rdkit_reactants"
                ]
                query_attention["reaction_fingerprint_weight_rdkit_products"] = reaction_attention[
                    "weight_rdkit_products"
                ]
                query_attention["reaction_fingerprint_weight_drfp"] = reaction_attention[
                    "weight_drfp"
                ]
            else:
                query_inputs = pooled_result

        if self.reaction_hyperbolic_encoder is not None:
            if isinstance(query_inputs, dict):
                raise ValueError(
                    "Reaction hyperbolic encoding currently expects fingerprint tensor inputs"
                )
            if query_inputs.ndim != 2:
                raise ValueError(
                    "Reaction hyperbolic encoding expects rank-2 query tensors, "
                    f"got {tuple(query_inputs.shape)}"
                )
            if (
                self.reaction_hyperbolic_input_dim is not None
                and query_inputs.shape[-1] != self.reaction_hyperbolic_input_dim
            ):
                raise ValueError(
                    "Reaction hyperbolic input dim mismatch: "
                    f"got {query_inputs.shape[-1]}, expected {self.reaction_hyperbolic_input_dim}"
                )
            reaction_stage1 = self.reaction_hyperbolic_encoder(query_inputs)
            z_hyp, z_tangent = self.reaction_hyperbolic_projector(reaction_stage1)
            query_inputs = z_tangent if self.reaction_hyperbolic_use_tangent else z_hyp

        if isinstance(query_inputs, dict):
            if return_attention and getattr(
                self.query_encoder,
                "supports_attention_return",
                False,
            ):
                query_result = self.query_encoder(
                    **query_inputs,
                    return_attention=True,
                )
            else:
                query_result = self.query_encoder(**query_inputs)
        else:
            query_result = self.query_encoder(query_inputs)

        if return_attention and isinstance(query_result, tuple):
            query_embeddings, attention = query_result
            query_attention.update(attention)
        else:
            query_embeddings = query_result[0] if isinstance(query_result, tuple) else query_result

        if query_embeddings.ndim != 2:
            raise ValueError(
                f"Query encoder must return rank-2 tensors, got {tuple(query_embeddings.shape)}"
            )
        if retrieval_direction == "enzyme_to_reaction" and self.e2r_adapter is not None:
            adapter_result = self.e2r_adapter(
                query_embeddings,
                adapter_inputs,
                return_details=return_attention,
            )
            if return_attention:
                query_embeddings, adapter_details = adapter_result
                query_attention.update(adapter_details)
            else:
                query_embeddings = adapter_result
        if return_attention:
            return query_embeddings, query_attention
        return query_embeddings

    def pool_residues(
        self,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        score_residue_embeddings: torch.Tensor | None = None,
        score_residue_padding_mask: torch.Tensor | None = None,
        score_attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_details: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ):
        if attention_mask is not None and residue_padding_mask is not None:
            raise ValueError("Pass either attention_mask or residue_padding_mask, not both")
        if score_attention_mask is not None and score_residue_padding_mask is not None:
            raise ValueError(
                "Pass either score_attention_mask or score_residue_padding_mask, not both"
            )
        if attention_mask is None:
            attention_mask = (
                None if residue_padding_mask is None else ~residue_padding_mask.to(torch.bool)
            )
        if score_attention_mask is None:
            score_attention_mask = (
                None
                if score_residue_padding_mask is None
                else ~score_residue_padding_mask.to(torch.bool)
            )
        if self.pooling_name == "sleec":
            return self.pooling(
                residue_embeddings,
                attention_mask=attention_mask,
                return_attention=return_attention,
                return_details=return_details,
                score_embeddings=score_residue_embeddings,
                score_attention_mask=score_attention_mask,
            )
        if self.pooling_name == "sleec_guided_attention":
            result = self.pooling(
                residue_embeddings,
                attention_mask=attention_mask,
                return_details=return_details or return_attention,
            )
            if return_details:
                return result
            if return_attention:
                pooled, details = result
                return pooled, details["weights"]
            return result
        if return_details:
            pooled, weights = self.pooling(
                residue_embeddings,
                attention_mask=attention_mask,
                return_attention=True,
            )
            return pooled, {"weights": weights}
        return self.pooling(
            residue_embeddings,
            attention_mask=attention_mask,
            return_attention=return_attention,
        )

    def _apply_r2e_adapter(
        self,
        target_embeddings: torch.Tensor,
        retrieval_direction: str,
        details: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if retrieval_direction not in {
            "reaction_to_enzyme",
            "enzyme_to_reaction",
        }:
            raise ValueError("retrieval_direction must be reaction_to_enzyme or enzyme_to_reaction")
        if retrieval_direction != "reaction_to_enzyme" or self.r2e_adapter is None:
            return target_embeddings, details
        adapter_result = self.r2e_adapter(
            target_embeddings,
            return_details=details is not None,
        )
        if details is None:
            return adapter_result, None
        adapted, adapter_details = adapter_result
        details = dict(details)
        details.update(adapter_details)
        return adapted, details

    def encode_targets(
        self,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        score_residue_embeddings: torch.Tensor | None = None,
        score_residue_padding_mask: torch.Tensor | None = None,
        score_attention_mask: torch.Tensor | None = None,
        capability_vectors: torch.Tensor | None = None,
        capability_mask: torch.Tensor | None = None,
        factorized_capability_vectors: dict[str, torch.Tensor] | None = None,
        factorized_capability_masks: dict[str, torch.Tensor] | None = None,
        text_vectors: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_pooling_details: bool = False,
        retrieval_direction: str = "reaction_to_enzyme",
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ):
        internal_pooling_details = return_pooling_details or (
            self.enzyme_input_mode == "raw_mean_sleec_biological_factorized"
            and self.pooling_name == "sleec_guided_attention"
        )
        pooled_result = self.pool_residues(
            residue_embeddings,
            residue_padding_mask=residue_padding_mask,
            attention_mask=attention_mask,
            score_residue_embeddings=score_residue_embeddings,
            score_residue_padding_mask=score_residue_padding_mask,
            score_attention_mask=score_attention_mask,
            return_attention=return_attention,
            return_details=internal_pooling_details,
        )
        if internal_pooling_details:
            pooled, details = pooled_result
            attention = details.get("weights") if return_attention else None
        elif return_attention:
            pooled, attention = pooled_result
            details = None
        else:
            pooled = pooled_result
            details = None
            attention = None

        raw_mean = None
        if self.enzyme_input_mode in {
            "raw_mean_sleec_hyperbolic_gated",
            "raw_mean_sleec_hyperbolic_capability_gated",
            "raw_mean_sleec_hyperbolic_text_gated",
            "raw_mean_sleec_blockwise",
            "raw_mean_sleec_hyperbolic_blockwise",
            "raw_mean_sleec_hyperbolic_capability_blockwise",
            "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
            "raw_mean_sleec_biofp_split",
            "raw_mean_sleec_biological_factorized",
        }:
            if self.raw_mean_pooling is None:
                raise RuntimeError("raw_mean_pooling was not initialized")
            raw_attention_mask = attention_mask
            if raw_attention_mask is None and residue_padding_mask is not None:
                raw_attention_mask = ~residue_padding_mask.to(torch.bool)
            raw_mean = self.raw_mean_pooling(
                residue_embeddings,
                attention_mask=raw_attention_mask,
            )

        if self.enzyme_input_mode == "raw_mean_sleec_biofp_split":
            if raw_mean is None or self.biofp_split_encoder is None:
                raise RuntimeError("BioFP split encoder was not initialized")
            biofp_result = self.biofp_split_encoder(
                raw_mean,
                pooled,
                residue_embeddings,
                residue_padding_mask=residue_padding_mask,
                return_details=return_pooling_details,
            )
            if return_pooling_details:
                target_embeddings, biofp_details = biofp_result
                details = dict(details or {})
                details.update(biofp_details)
            else:
                target_embeddings = biofp_result
            target_embeddings, details = self._apply_r2e_adapter(
                target_embeddings,
                retrieval_direction,
                details if return_pooling_details else None,
            )
            if return_pooling_details:
                return target_embeddings, details
            if return_attention:
                return target_embeddings, attention
            return target_embeddings

        if self.enzyme_input_mode == "raw_mean_sleec_biological_factorized":
            if raw_mean is None or self.biological_factorized_encoder is None:
                raise RuntimeError("Biological factorized encoder was not initialized")
            z_tangent = None
            if "ec" in self.biological_factorized_encoder.block_dims:
                if self.hyperbolic_projector is None:
                    raise RuntimeError("Biological ec block has no hyperbolic projector")
                _z_hyp, z_tangent = self.hyperbolic_projector(pooled)
            sleec_prior = None
            if details is not None:
                candidate_prior = details.get("logits")
                if candidate_prior is not None:
                    sleec_prior = candidate_prior.to(
                        device=residue_embeddings.device,
                        dtype=residue_embeddings.dtype,
                    )
            biological_result = self.biological_factorized_encoder(
                raw_mean,
                pooled,
                z_tangent,
                residue_embeddings,
                residue_padding_mask=residue_padding_mask,
                sleec_prior=sleec_prior,
                return_details=return_pooling_details,
            )
            if return_pooling_details:
                target_embeddings, biological_details = biological_result
                details = dict(details or {})
                details.update(biological_details)
            else:
                target_embeddings = biological_result
            target_embeddings, details = self._apply_r2e_adapter(
                target_embeddings,
                retrieval_direction,
                details if return_pooling_details else None,
            )
            if return_pooling_details:
                return target_embeddings, details
            if return_attention:
                return target_embeddings, attention
            return target_embeddings

        target_input = pooled
        if self.hyperbolic_projector is not None:
            z_hyp, z_tangent = self.hyperbolic_projector(pooled)
            hyperbolic_input = z_tangent if self.hyperbolic_use_tangent else z_hyp
            if self.enzyme_input_mode == "raw_sleec_hyperbolic_concat":
                target_input = torch.cat([pooled, z_tangent], dim=-1)
            elif self.enzyme_input_mode in {
                "raw_mean_sleec_hyperbolic_gated",
                "raw_mean_sleec_hyperbolic_capability_gated",
                "raw_mean_sleec_hyperbolic_text_gated",
                "raw_mean_sleec_hyperbolic_blockwise",
                "raw_mean_sleec_hyperbolic_capability_blockwise",
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
            }:
                if raw_mean is None or self.enzyme_feature_fusion is None:
                    if self.enzyme_block_fusion is None:
                        raise RuntimeError("enzyme feature fusion was not initialized")
                capability_input = None
                capability_branch_mask = None
                factorized_inputs = None
                factorized_masks = None
                if self.enzyme_input_mode in {
                    "raw_mean_sleec_hyperbolic_capability_gated",
                    "raw_mean_sleec_hyperbolic_capability_blockwise",
                }:
                    if capability_vectors is None:
                        raise ValueError(
                            "capability_vectors are required for " f"{self.enzyme_input_mode}"
                        )
                    capability_input = capability_vectors.to(
                        device=pooled.device,
                        dtype=pooled.dtype,
                    )
                    if capability_mask is not None:
                        capability_branch_mask = capability_mask.to(
                            device=pooled.device,
                            dtype=torch.bool,
                        )
                    if self.capability_freeze:
                        capability_input = capability_input.detach()
                    if self.capability_adapter is not None:
                        capability_input = capability_input + self.capability_adapter(
                            capability_input
                        )
                if (
                    self.enzyme_input_mode
                    == "raw_mean_sleec_hyperbolic_factorized_capability_blockwise"
                ):
                    if factorized_capability_vectors is None:
                        raise ValueError(
                            "factorized_capability_vectors are required for "
                            f"{self.enzyme_input_mode}"
                        )
                    factorized_inputs = {
                        name: value.to(device=pooled.device, dtype=pooled.dtype)
                        for name, value in factorized_capability_vectors.items()
                    }
                    if self.factorized_capability_freeze:
                        factorized_inputs = {
                            name: value.detach() for name, value in factorized_inputs.items()
                        }
                    factorized_masks = (
                        None
                        if factorized_capability_masks is None
                        else {
                            name: value.to(device=pooled.device, dtype=torch.bool)
                            for name, value in factorized_capability_masks.items()
                        }
                    )
                if self.enzyme_input_mode in {
                    "raw_mean_sleec_hyperbolic_blockwise",
                    "raw_mean_sleec_hyperbolic_capability_blockwise",
                    "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
                }:
                    if self.enzyme_block_fusion is None:
                        raise RuntimeError("enzyme block fusion was not initialized")
                    block_result = self.enzyme_block_fusion(
                        raw_mean,
                        pooled,
                        z_tangent,
                        capability_vector=capability_input,
                        capability_mask=capability_branch_mask,
                        factorized_capability_vectors=factorized_inputs,
                        factorized_capability_masks=factorized_masks,
                        return_details=details is not None,
                    )
                    if details is not None:
                        target_input, block_details = block_result
                        details = dict(details)
                        details.update(block_details)
                    else:
                        target_input = block_result
                else:
                    if self.enzyme_feature_fusion is None:
                        raise RuntimeError("enzyme feature fusion was not initialized")
                    fusion_result = self.enzyme_feature_fusion(
                        raw_mean,
                        pooled,
                        z_tangent,
                        capability_vector=capability_input,
                        capability_mask=capability_branch_mask,
                        return_gates=details is not None,
                    )
                    if details is not None:
                        target_input, gate_weights = fusion_result
                        details = dict(details)
                        details["enzyme_fusion_gate_raw_mean"] = gate_weights[:, 0]
                        details["enzyme_fusion_gate_pooled"] = gate_weights[:, 1]
                        details["enzyme_fusion_gate_hyperbolic"] = gate_weights[:, 2]
                        if gate_weights.shape[1] > 3:
                            details["enzyme_fusion_gate_capability"] = gate_weights[:, 3]
                    else:
                        target_input = fusion_result
                if self.enzyme_input_mode == "raw_mean_sleec_hyperbolic_text_gated":
                    if self.text_feature_fusion is None:
                        raise RuntimeError("text_feature_fusion was not initialized")
                    if text_vectors is None:
                        raise ValueError(
                            "text_vectors are required for " "raw_mean_sleec_hyperbolic_text_gated"
                        )
                    text_input = text_vectors.to(device=pooled.device, dtype=pooled.dtype)
                    text_branch_mask = (
                        None
                        if text_mask is None
                        else text_mask.to(device=pooled.device, dtype=torch.bool)
                    )
                    if self.text_freeze:
                        text_input = text_input.detach()
                    if self.text_adapter is not None:
                        text_input = text_input + self.text_adapter(text_input)
                    text_fusion_result = self.text_feature_fusion(
                        target_input,
                        text_input,
                        text_mask=text_branch_mask,
                        return_details=details is not None,
                    )
                    if details is not None:
                        target_input, text_details = text_fusion_result
                        details = dict(details)
                        details.update(text_details)
                    else:
                        target_input = text_fusion_result
            else:
                target_input = hyperbolic_input
            if details is not None:
                details = dict(details)
                details["hyperbolic_tangent_norm"] = z_tangent.norm(dim=-1)
                details["hyperbolic_ball_norm"] = z_hyp.norm(dim=-1)

        if self.enzyme_input_mode == "raw_mean_sleec_blockwise":
            if raw_mean is None or self.enzyme_block_fusion is None:
                raise RuntimeError("enzyme block fusion was not initialized")
            block_result = self.enzyme_block_fusion(
                raw_mean,
                pooled,
                return_details=details is not None,
            )
            if details is not None:
                target_input, block_details = block_result
                details = dict(details)
                details.update(block_details)
            else:
                target_input = block_result

        target_embeddings = (
            target_input
            if self.blockwise_target_is_embedding
            else self.target_encoder(target_input)
        )
        target_embeddings, details = self._apply_r2e_adapter(
            target_embeddings,
            retrieval_direction,
            details if return_pooling_details else None,
        )
        if return_pooling_details:
            return target_embeddings, details
        if return_attention:
            return target_embeddings, attention
        return target_embeddings

    def forward(
        self,
        query_inputs: dict | torch.Tensor,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        score_residue_embeddings: torch.Tensor | None = None,
        score_residue_padding_mask: torch.Tensor | None = None,
        score_attention_mask: torch.Tensor | None = None,
        capability_vectors: torch.Tensor | None = None,
        capability_mask: torch.Tensor | None = None,
        factorized_capability_vectors: dict[str, torch.Tensor] | None = None,
        factorized_capability_masks: dict[str, torch.Tensor] | None = None,
        text_vectors: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_pooling_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_embeddings = self.encode_queries(query_inputs)
        target_result = self.encode_targets(
            residue_embeddings,
            residue_padding_mask=residue_padding_mask,
            attention_mask=attention_mask,
            score_residue_embeddings=score_residue_embeddings,
            score_residue_padding_mask=score_residue_padding_mask,
            score_attention_mask=score_attention_mask,
            capability_vectors=capability_vectors,
            capability_mask=capability_mask,
            factorized_capability_vectors=factorized_capability_vectors,
            factorized_capability_masks=factorized_capability_masks,
            text_vectors=text_vectors,
            text_mask=text_mask,
            return_attention=return_attention,
            return_pooling_details=return_pooling_details,
        )
        if return_pooling_details:
            target_embeddings, details = target_result
            return query_embeddings, target_embeddings, details
        if return_attention:
            target_embeddings, attention = target_result
            return query_embeddings, target_embeddings, attention
        return query_embeddings, target_result


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
