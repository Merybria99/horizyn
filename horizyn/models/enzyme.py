"""F3/CIRCEv2 enzyme pooling and functional-site primitives. V4 views are in enzyme_multiview."""

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from horizyn.checkpoint_utils import extract_query_encoder_state_dict
from horizyn.biological_residual import PromiscuityAwareBiologicalResidual



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
                # Expose the normalized, unweighted block so auxiliary
                # objectives can act on the same inference-time coordinates
                # without changing the encoder input path.
                details[f"enzyme_block_{name}"] = block
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
                    details[f"biofp_shared_{family}"] = block_inputs[family]
                    details[f"biofp_logits_{family}"] = self.family_heads[family](
                        block_inputs[family]
                    )
        return output, details


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

    The learned attention and pooled values use ``residue_embeddings``. When
    ``score_hidden_dim`` and ``score_embeddings`` are supplied, only the SLEEC
    scorer uses that aligned external representation. This permits, for
    example, EnzGFM values with a frozen ProtT5-based SLEEC prior.

    The learned attention logit is combined with a centered SLEEC prior before
    softmax:

        a_i = learned_logit_i + softplus(alpha) * (sleec_logit_i - logit(threshold))

    Padding positions always receive zero weight.
    """

    def __init__(
        self,
        hidden_dim: int,
        scorer_hidden_dim: int = 256,
        score_hidden_dim: int | None = None,
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
        self.score_hidden_dim = (
            self.hidden_dim if score_hidden_dim is None else int(score_hidden_dim)
        )
        if self.score_hidden_dim <= 0:
            raise ValueError("score_hidden_dim must be positive")
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.attention = nn.Linear(hidden_dim, 1, bias=attention_bias)
        self.sleec_scorer = FunctionalResidueScorer(
            hidden_dim=self.score_hidden_dim,
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
        score_embeddings: torch.Tensor | None = None,
        score_attention_mask: torch.Tensor | None = None,
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

        sleec_logits, sleec_scores = self.sleec_scorer(
            score_embeddings,
            attention_mask=score_valid_mask,
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
