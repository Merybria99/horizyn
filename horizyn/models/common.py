"""Shared neural layers and historical checkpoint helpers."""

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from horizyn.checkpoint_utils import extract_query_encoder_state_dict
from horizyn.biological_residual import PromiscuityAwareBiologicalResidual



def signed_power_transform(
    features: torch.Tensor,
    exponent: float,
) -> torch.Tensor:
    """Apply elementwise signed power normalization without changing signs."""

    exponent = float(exponent)
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("signed-power exponent must be finite and positive")
    if exponent == 1.0:
        return features
    return torch.sign(features) * torch.abs(features).pow(exponent)


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


class NormalizedReactionBlockProjection(nn.Module):
    """Project one shared feature into fixed, independently normalized blocks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        block_dims: dict[str, int],
        block_weights: dict[str, float],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if not block_dims or not block_weights:
            raise ValueError("block_dims and block_weights must be non-empty")
        if tuple(block_dims) != tuple(block_weights):
            raise ValueError("block_dims and block_weights must have the same ordered keys")
        if any(not isinstance(dim, int) or dim <= 0 for dim in block_dims.values()):
            raise ValueError("block dimensions must be positive integers")
        if sum(block_dims.values()) != output_dim:
            raise ValueError("block dimensions must sum to output_dim")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        weights = {name: float(block_weights[name]) for name in block_dims}
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()):
            raise ValueError("block weights must be finite and non-negative")
        weight_sum = sum(weights.values())
        if weight_sum <= 0.0:
            raise ValueError("at least one block weight must be positive")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.block_names = tuple(block_dims)
        self.block_dims = {name: int(dim) for name, dim in block_dims.items()}
        self.block_weights = {name: weights[name] / weight_sum for name in self.block_names}
        self.projections = nn.ModuleDict(
            {
                name: self._projection(self.input_dim, self.block_dims[name], dropout)
                for name in self.block_names
            }
        )

    @staticmethod
    def _projection(input_dim: int, output_dim: int, dropout: float) -> nn.Module:
        layers: list[nn.Module] = [nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        return nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must have shape [batch, {self.input_dim}], "
                f"got {tuple(features.shape)}"
            )

        blocks: list[torch.Tensor] = []
        details: dict[str, torch.Tensor] = {}
        for name in self.block_names:
            block = F.normalize(
                self.projections[name](features),
                p=2,
                dim=-1,
                eps=1e-12,
            )
            weight = self.block_weights[name]
            weighted_block = block * math.sqrt(weight)
            blocks.append(weighted_block)
            if return_details:
                details[f"reaction_block_norm_{name}"] = block.norm(dim=-1)
                details[f"reaction_block_weighted_norm_{name}"] = weighted_block.norm(dim=-1)
                details[f"reaction_block_weight_{name}"] = torch.full(
                    (features.shape[0],),
                    weight,
                    dtype=features.dtype,
                    device=features.device,
                )

        output = F.normalize(torch.cat(blocks, dim=-1), p=2, dim=-1, eps=1e-12)
        if return_details:
            return output, details
        return output


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
