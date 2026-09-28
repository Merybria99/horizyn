"""Multimodal reaction encoding and molecular-set pooling."""

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from horizyn.checkpoint_utils import extract_query_encoder_state_dict
from horizyn.biological_residual import PromiscuityAwareBiologicalResidual

from .common import (BaseModel, MLP, NormalizeLayer, NormalizedReactionBlockProjection, ResidualMLPProjection)


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


class MoleculeSetInteractionPooling(nn.Module):
    """Two small, position-free interaction layers over frozen molecule vectors.

    Padding is True for *valid* molecules, matching the other set poolers.
    Empty sets return zeros; the enclosing encoder handles missing modalities.
    Output width is unchanged so existing feature, checkpoint and scoring APIs
    need no new representation type.
    """

    def __init__(self, hidden_dim: int, attention_bias: bool = True,
                 return_attention: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.return_attention = return_attention
        self.input_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 256))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(256, 4, dim_feedforward=512, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(2)
        ])
        self.final_norm = nn.LayerNorm(256)
        self.pool = MoleculeSetAttentionPooling(256, attention_bias=attention_bias)
        self.output_projection = nn.Linear(256, hidden_dim)

    def forward(self, molecule_embeddings, attention_mask=None, return_attention=None):
        if molecule_embeddings.ndim != 3 or molecule_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected molecule embeddings [B, M, {self.hidden_dim}]")
        valid = (torch.ones(molecule_embeddings.shape[:2], device=molecule_embeddings.device,
                            dtype=torch.bool) if attention_mask is None
                 else attention_mask.to(device=molecule_embeddings.device, dtype=torch.bool))
        if valid.shape != molecule_embeddings.shape[:2]:
            raise ValueError("Molecule attention_mask shape mismatch")
        nonempty = valid.any(dim=1)
        # Mask before projection: arbitrary padding (including NaNs) cannot leak.
        tokens = self.input_projection(molecule_embeddings.masked_fill(~valid[..., None], 0))
        # A dummy key, valid only for empty rows, avoids all-masked softmax NaNs.
        tokens = torch.cat((tokens, tokens.new_zeros(tokens.shape[0], 1, 256)), dim=1)
        safe_valid = torch.cat((valid, ~nonempty[:, None]), dim=1)
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=~safe_valid)
        pooled, weights = self.pool(self.final_norm(tokens), safe_valid, return_attention=True)
        output = self.output_projection(pooled).masked_fill(~nonempty[:, None], 0)
        if self.return_attention if return_attention is None else return_attention:
            return output, weights[:, :-1]
        return output


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


class MultimodalReactionAttentionEncoder(BaseModel):
    """
    Encode reactions from learned reaction, Uni-Mol2, and chirality modalities.

    The fixed inputs are converted to trainable modality tokens:
    reaction-model SMILES embedding, Uni-Mol2 reaction composition, and
    optionally a ChIRo chirality composition. Each modality can use either the
    historical single linear adapter or a Horizyn-style MLP adapter before
    fusion. Competitive feature/scalar gates mix the same normalized tokens
    into one token, retaining the existing output projection.
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
        output_block_dims: dict[str, int] | None = None,
        output_block_weights: dict[str, float] | None = None,
        output_block_dropout: float = 0.0,
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
        if reaction_pooling not in {"attention", "mean", "interaction"}:
            raise ValueError("reaction_pooling must be one of: attention, mean, interaction")
        if reaction_pooling == "interaction" and (
            side_composition != "molecule_set" or use_reaction_chemistry or use_reaction_directional
            or ((use_reaction_model or use_chienn) and modality_fusion not in {
                "concat", "gated_attention_concat", "feature_gate", "scalar_gate"
            })
        ):
            raise ValueError("interaction pooling requires a molecule_set encoder: UniMol2-only, concat/gated_attention_concat, or feature_gate/scalar_gate")
        if side_composition not in {"directional_delta", "molecule_set", "signed_residual"}:
            raise ValueError("side_composition must be directional_delta, molecule_set or signed_residual")
        if modality_fusion not in {
            "attention",
            "mean",
            "factorized_concat",
            "prior_bounded_attention",
            "concat",
            "gated_attention_concat",
            "feature_gate",
            "scalar_gate",
        }:
            raise ValueError(
                "modality_fusion must be one of: attention, mean, factorized_concat, "
                "prior_bounded_attention, concat, gated_attention_concat, feature_gate, scalar_gate"
            )
        concatenated_fusion = modality_fusion in {"concat", "gated_attention_concat"}
        if concatenated_fusion and output_projection != "mlp":
            raise ValueError("concat fusions require an MLP output projection")
        if output_projection not in {"mlp", "residual_mlp", "identity"}:
            raise ValueError("output_projection must be one of: mlp, residual_mlp, identity")
        if (output_block_dims is None) != (output_block_weights is None):
            raise ValueError("output_block_dims and output_block_weights must be set together")
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
        # A small signed correction keeps the original modality MLP input width.
        # At initialization it is exactly the reactant-only representation.
        def signed_residual(dim):
            if side_composition != "signed_residual":
                return None
            block = nn.Sequential(nn.Linear(dim, min(64, dim), bias=False),
                                  nn.Linear(min(64, dim), dim, bias=False))
            nn.init.zeros_(block[-1].weight)
            return block

        self.unimol_signed_residual = signed_residual(unimol_dim)
        self.chienn_signed_residual = signed_residual(chienn_dim) if use_chienn else None
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

        if reaction_pooling in {"attention", "interaction"}:
            pooler_class = (MoleculeSetInteractionPooling if reaction_pooling == "interaction"
                            else MoleculeSetAttentionPooling)
            self.unimol_reactant_pooling = pooler_class(
                hidden_dim=unimol_dim,
                attention_bias=attention_bias,
                return_attention=return_attention,
            )
            unimol_product_pooler = pooler_class(
                hidden_dim=unimol_dim,
                attention_bias=attention_bias,
                return_attention=return_attention,
            )
            if use_chienn:
                self.chienn_reactant_pooling = pooler_class(
                    hidden_dim=chienn_dim,
                    attention_bias=attention_bias,
                    return_attention=return_attention,
                )
                chienn_product_pooler = pooler_class(
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
                input_dim=self.token_dim * len(self.modality_names) if concatenated_fusion else self.token_dim,
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
        self.output_block_projection = (
            NormalizedReactionBlockProjection(
                input_dim=output_dim,
                output_dim=output_dim,
                block_dims=output_block_dims,
                block_weights=output_block_weights,
                dropout=float(output_block_dropout),
            )
            if output_block_dims is not None and output_block_weights is not None
            else None
        )
        if normalise_output:
            self.post_nn_layers.append(NormalizeLayer(p=2, dim=-1))

        # Construct after all shared modules so same-seed concat controls have
        # exactly the same initial branch and output-projection parameters.
        self.cross_modal_fusion = None
        if modality_fusion == "gated_attention_concat":
            from horizyn.gated_reaction_fusion import GatedReactionFusion
            with torch.random.fork_rng(devices=[]):
                self.cross_modal_fusion = GatedReactionFusion(self.token_dim, len(self.modality_names))
        self.competitive_fusion = None
        if modality_fusion in {"feature_gate", "scalar_gate"}:
            from horizyn.gated_reaction_fusion import CompetitiveReactionFusion

            # Gate width must not perturb shared branch/projector initialization
            # or the random stream subsequently used by the enzyme tower.
            with torch.random.fork_rng(devices=[]):
                self.competitive_fusion = CompetitiveReactionFusion(
                    self.token_dim,
                    len(self.modality_names),
                    featurewise=modality_fusion == "feature_gate",
                )

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
        if self.modality_fusion in {"feature_gate", "scalar_gate"}:
            attention_mask = (
                torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
                if attention_mask is None
                else attention_mask.to(device=embeddings.device)
            )
            embeddings = embeddings.masked_fill(~attention_mask[..., None], 0)
            # Existing F3 attention/mean poolers require a nonempty molecule set.
            # A zero dummy is visible only for empty rows and is hidden in details.
            empty = ~attention_mask.any(dim=1, keepdim=True)
            embeddings = torch.cat(
                (embeddings, embeddings.new_zeros(embeddings.shape[0], 1, embeddings.shape[2])),
                dim=1,
            )
            attention_mask = torch.cat((attention_mask, empty), dim=1)
            result = pooler(embeddings, attention_mask=attention_mask, return_attention=return_attention)
            if return_attention:
                pooled, weights = result
                return pooled.masked_fill(empty, 0), weights[:, :-1]
            return result.masked_fill(empty, 0)
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
        signed_residual: nn.Module | None = None,
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
        if self.side_composition == "signed_residual":
            if signed_residual is None:
                raise RuntimeError("Signed directional residual was not initialized")
            composition = reactant_pooled + signed_residual(product_pooled - reactant_pooled)
        else:
            composition = self._compose_side_pair(reactant_pooled, product_pooled)
        return (
            composition,
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
        if self.modality_fusion in {"feature_gate", "scalar_gate"}:
            # Mask before learned branches: masking projected NaNs later cannot
            # prevent NaN gradients through a Linear layer.
            def mask_unavailable(values, present):
                if present is None:
                    return values
                missing = ~present.to(device=values.device, dtype=torch.bool)
                return values.masked_fill(missing.reshape(-1, *([1] * (values.ndim - 1))), 0)

            reactant_embeddings = mask_unavailable(reactant_embeddings, has_unimol2)
            product_embeddings = mask_unavailable(product_embeddings, has_unimol2)
            chirality_present = has_chirality if has_chirality is not None else (
                has_chiro if has_chiro is not None else has_chienn)
            if self.use_chienn:
                reactant_chirality_embeddings = mask_unavailable(
                    reactant_chirality_embeddings, chirality_present)
                product_chirality_embeddings = mask_unavailable(
                    product_chirality_embeddings, chirality_present)
            if self.use_reaction_chemistry:
                reaction_chemistry_vector = mask_unavailable(
                    reaction_chemistry_vector, has_reaction_chemistry)
            if self.use_reaction_directional:
                reaction_directional_vector = mask_unavailable(
                    reaction_directional_vector, has_reaction_directional)
        if self.modality_fusion in {"concat", "gated_attention_concat"}:
            if has_unimol2 is not None:
                reactant_embeddings = reactant_embeddings.masked_fill(
                    ~has_unimol2.to(device=device, dtype=torch.bool)[:, None, None], 0)
            chirality_present = has_chirality if has_chirality is not None else (
                has_chiro if has_chiro is not None else has_chienn)
            if self.use_chienn and chirality_present is not None:
                reactant_chirality_embeddings = reactant_chirality_embeddings.masked_fill(
                    ~chirality_present.to(device=device, dtype=torch.bool)[:, None, None], 0)
        unimol_composition, unimol_reactant_attention, unimol_product_attention = (
            self._encode_molecule_modality(
                self.unimol_reactant_pooling,
                self.unimol_product_pooling,
                reactant_embeddings,
                reactant_padding_mask,
                product_embeddings,
                product_padding_mask,
                should_return_attention,
                self.unimol_signed_residual,
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
                    self.chienn_signed_residual,
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
        fusion_details = {}
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
        elif self.modality_fusion in {"feature_gate", "scalar_gate"}:
            tokens = torch.stack(token_list, dim=1).masked_fill(~modality_mask[..., None], 0)
            tokens = self.modality_token_norm(tokens).masked_fill(~modality_mask[..., None], 0)
            if self.modality_l2_normalize:
                tokens = F.normalize(tokens, p=2, dim=-1) * math.sqrt(self.token_dim)
            pooled, fusion_details = self.competitive_fusion(
                tokens, modality_mask, should_return_attention)
            # Scalar summaries are the mean of the actual per-feature weights.
            modality_weights = (
                fusion_details["modality_feature_weights"].mean(dim=-1)
                if should_return_attention else None
            )
            modality_logits = (
                fusion_details["modality_feature_logits"].mean(dim=-1)
                if should_return_attention else None
            )
        elif self.modality_fusion in {"concat", "gated_attention_concat"}:
            tokens = torch.stack(token_list, dim=1).masked_fill(~modality_mask[..., None], 0)
            tokens = self.modality_token_norm(tokens).masked_fill(~modality_mask[..., None], 0)
            if self.modality_l2_normalize:
                tokens = F.normalize(tokens, p=2, dim=-1) * math.sqrt(self.token_dim)
            if self.cross_modal_fusion is not None:
                tokens, fusion_details = self.cross_modal_fusion(tokens, modality_mask, should_return_attention)
            pooled = tokens.flatten(start_dim=1)
            # Availability diagnostics only: these are not learned fusion weights.
            modality_weights = modality_mask.to(tokens.dtype)
            modality_weights = modality_weights / modality_weights.sum(dim=1, keepdim=True).clamp_min(1)
            modality_logits = tokens.new_zeros(modality_mask.shape)
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
        if self.output_block_projection is not None:
            block_result = self.output_block_projection(
                output,
                return_details=should_return_attention,
            )
            if should_return_attention:
                output, block_details = block_result
                projection_details.update(block_details)
            else:
                output = block_result
        for layer in self.post_nn_layers:
            output = layer(output)

        if self.modality_fusion in {"concat", "gated_attention_concat", "feature_gate", "scalar_gate"}:
            output = output.masked_fill(~modality_mask.any(dim=1, keepdim=True), 0)

        if not should_return_attention:
            return output

        if "modality_feature_entropy" in fusion_details:
            entropy = fusion_details["modality_feature_entropy"].mean(dim=-1)
        else:
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
        attention.update(fusion_details)
        if self.modality_fusion in {"concat", "gated_attention_concat"}:
            attention["modality_weights_are_availability"] = True
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
