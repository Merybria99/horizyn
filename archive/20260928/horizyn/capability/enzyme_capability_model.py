"""Models for enzyme capability pretraining."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover - import depends on runtime environment
    pl = None

from horizyn.capability.capability_losses import (
    BiologicalFactorSupConLoss,
    CompositeBiologicalSupConLoss,
    multilabel_bce_or_zero,
    multi_positive_contrastive_loss,
    pair_id_positive_masks,
    positive_masks_from_pair_maps,
)

LEGACY_BIOLOGICAL_FAMILY_NAMES = (
    "global",
    "cofactor",
    "reaction_center",
    "substrate",
    "product",
)
STAGE1_BIOLOGICAL_FAMILY_NAMES = (
    "global",
    "cofactor",
    "transition",
    "reaction_center",
    "substrate",
    "product",
)
BIOLOGICAL_FAMILY_NAMES = LEGACY_BIOLOGICAL_FAMILY_NAMES
DEFAULT_STAGE1_BIO_FAMILIES = ("cofactor", "transition", "substrate", "product")
TARGET_TO_BIOLOGICAL_FAMILY = {
    "cofactor_targets": "cofactor",
    "core_cofactor_targets": "cofactor",
    "enzyme_derived_cofactor_targets": "cofactor",
    "enzyme_derived_core_cofactor_targets": "cofactor",
    "combined_cofactor_targets": "cofactor",
    "combined_core_cofactor_targets": "cofactor",
    "cofactor_architecture_targets": "cofactor",
    "cofactor_chemistry_targets": "cofactor",
    "metal_ion_targets": "cofactor",
    "auxiliary_participant_targets": "cofactor",
    "reaction_center_targets": "reaction_center",
    "substrate_product_transition_targets": "transition",
    "substrate_targets": "substrate",
    "product_targets": "product",
}
BIOLOGICAL_FACTOR_TARGET_PREFERENCES = {
    "cofactor": (
        "cofactor_architecture_targets",
        "cofactor_chemistry_targets",
        "combined_core_cofactor_targets",
        "core_cofactor_targets",
        "combined_cofactor_targets",
        "enzyme_derived_core_cofactor_targets",
        "enzyme_derived_cofactor_targets",
        "cofactor_targets",
        "metal_ion_targets",
        "auxiliary_participant_targets",
    ),
    "reaction_center": ("reaction_center_targets",),
    "transition": ("substrate_product_transition_targets",),
    "substrate": ("substrate_targets",),
    "product": ("product_targets",),
}


class FeatureBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiQueryResiduePooler(nn.Module):
    """Learn factor-specific attention pools over ProT5 residue embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        family_names: tuple[str, ...] = BIOLOGICAL_FAMILY_NAMES,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.family_names = tuple(family_names)
        self.hidden_dim = int(hidden_dim)
        self.residue_adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.queries = nn.Parameter(torch.empty(len(self.family_names), self.hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.last_attention_weights: torch.Tensor | None = None
        nn.init.normal_(self.queries, mean=0.0, std=self.hidden_dim**-0.5)

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if residue_embeddings.ndim != 3:
            raise ValueError(
                "residue_embeddings must have shape [batch, residues, dim], "
                f"got {tuple(residue_embeddings.shape)}"
            )
        residues = self.residue_adapter(residue_embeddings.float())
        keys = self.key(residues)
        values = self.value(residues)
        scores = torch.einsum("bld,fd->bfl", keys, self.queries) / (self.hidden_dim**0.5)
        if residue_mask is not None:
            mask = residue_mask.to(device=scores.device, dtype=torch.bool)
            if mask.shape != scores.shape[:1] + scores.shape[2:]:
                raise ValueError(
                    "residue_mask must have shape [batch, residues], "
                    f"got {tuple(mask.shape)} for scores {tuple(scores.shape)}"
                )
            scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention_probs = torch.softmax(scores, dim=-1)
        self.last_attention_weights = attention_probs.detach()
        weights = self.dropout(attention_probs)
        pooled = torch.einsum("bfl,bld->bfd", weights, values)
        return {
            family_name: pooled[:, idx]
            for idx, family_name in enumerate(self.family_names)
        }


class EnzymeCapabilityEncoder(nn.Module):
    """Encode static enzyme-side features into a normalized capability vector."""

    def __init__(
        self,
        input_dims: dict[str, int],
        capability_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        normalize_output: bool = True,
        capability_mode: str = "single",
        family_dim: int = 64,
        residue_input_dim: int | None = None,
        use_residue_multiquery_pooling: bool = False,
        residue_pool_dropout: float = 0.0,
        residue_pool_scale: float = 0.1,
        biological_family_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims must contain at least one enzyme feature")
        if capability_mode not in {"single", "factorized_biological"}:
            raise ValueError("capability_mode must be 'single' or 'factorized_biological'")
        self.input_dims = {str(key): int(value) for key, value in input_dims.items() if value}
        self.capability_dim = int(capability_dim)
        self.hidden_dim = int(hidden_dim)
        self.normalize_output = bool(normalize_output)
        self.capability_mode = str(capability_mode)
        self.family_dim = int(family_dim)
        self.biological_family_names = tuple(
            str(name)
            for name in (
                biological_family_names
                if biological_family_names is not None
                else BIOLOGICAL_FAMILY_NAMES
            )
        )
        self.use_residue_multiquery_pooling = bool(use_residue_multiquery_pooling)
        self.branches = nn.ModuleDict(
            {name: FeatureBranch(dim, self.hidden_dim) for name, dim in self.input_dims.items()}
        )
        self.residue_pooler: MultiQueryResiduePooler | None = None
        if self.use_residue_multiquery_pooling:
            if residue_input_dim is None:
                raise ValueError(
                    "residue_input_dim is required when use_residue_multiquery_pooling=True"
                )
            self.residue_pooler = MultiQueryResiduePooler(
                input_dim=int(residue_input_dim),
                hidden_dim=self.hidden_dim,
                family_names=self.biological_family_names,
                dropout=float(residue_pool_dropout),
            )
            self.residue_pool_scale = nn.Parameter(
                torch.tensor(float(residue_pool_scale), dtype=torch.float32)
            )
        if self.capability_mode == "factorized_biological":
            self.trunk = nn.Sequential(
                nn.Linear(self.hidden_dim * len(self.branches), self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.family_heads = nn.ModuleDict(
                {
                    name: nn.Linear(self.hidden_dim, self.family_dim)
                    for name in self.biological_family_names
                }
            )
            self.family_projection = nn.Sequential(
                nn.LayerNorm(self.family_dim * len(self.biological_family_names)),
                nn.Linear(self.family_dim * len(self.biological_family_names), self.capability_dim),
            )
        else:
            fusion_input_count = len(self.branches) + (
                1 if self.use_residue_multiquery_pooling else 0
            )
            self.fusion = nn.Sequential(
                nn.Linear(self.hidden_dim * fusion_input_count, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.capability_dim),
            )

    def _branch_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        branch_outputs = []
        for name, branch in self.branches.items():
            if name not in features:
                raise KeyError(f"Missing enzyme feature branch '{name}'")
            branch_outputs.append(branch(features[name].float()))
        return torch.cat(branch_outputs, dim=-1)

    def _residue_family_features(
        self,
        residue_embeddings: torch.Tensor | None,
        residue_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if self.residue_pooler is None:
            return {}
        if residue_embeddings is None:
            raise KeyError(
                "prot5_residue_embeddings are required when residue multi-query pooling is enabled"
            )
        return self.residue_pooler(residue_embeddings, residue_mask)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        residue_embeddings: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        return_family_vectors: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branch_features = self._branch_features(features)
        residue_family_features = self._residue_family_features(
            residue_embeddings,
            residue_mask,
        )
        if self.capability_mode == "factorized_biological":
            trunk = self.trunk(branch_features)
            family_vectors = {}
            for name, head in self.family_heads.items():
                family_input = trunk
                residue_context = residue_family_features.get(name)
                if residue_context is not None:
                    family_input = family_input + self.residue_pool_scale * residue_context
                family_vectors[name] = F.normalize(head(family_input), p=2, dim=-1, eps=1e-12)
            ordered = [family_vectors[name] for name in self.biological_family_names]
            output = self.family_projection(torch.cat(ordered, dim=-1))
            if self.normalize_output:
                output = F.normalize(output, p=2, dim=-1, eps=1e-12)
            if return_family_vectors:
                return output, family_vectors
            return output

        if residue_family_features:
            residue_values = [
                residue_family_features[name]
                for name in self.biological_family_names
                if name in residue_family_features
            ]
            residue_summary = torch.stack(residue_values, dim=1).mean(dim=1)
            branch_features = torch.cat(
                [branch_features, self.residue_pool_scale * residue_summary],
                dim=-1,
            )
        output = self.fusion(branch_features)
        if self.normalize_output:
            output = F.normalize(output, p=2, dim=-1, eps=1e-12)
        if return_family_vectors:
            return output, {}
        return output

    def attention_diagnostics(self) -> dict[str, torch.Tensor]:
        if self.residue_pooler is None or self.residue_pooler.last_attention_weights is None:
            return {}
        weights = self.residue_pooler.last_attention_weights
        eps = torch.finfo(weights.dtype).eps
        entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=-1)
        query_vectors = F.normalize(self.residue_pooler.queries.detach(), p=2, dim=-1, eps=1e-12)
        query_similarity = query_vectors @ query_vectors.T
        not_self = ~torch.eye(
            query_similarity.shape[0],
            dtype=torch.bool,
            device=query_similarity.device,
        )
        out: dict[str, torch.Tensor] = {}
        for idx, family_name in enumerate(self.residue_pooler.family_names):
            out[f"attention_entropy_{family_name}"] = entropy[:, idx].mean()
        if bool(not_self.any()):
            out["query_cosine_similarity_mean"] = query_similarity[not_self].mean()
            out["query_cosine_similarity_max"] = query_similarity[not_self].max()
        return out


class ReactionDemandEncoder(nn.Module):
    """Project reaction demand vectors into capability space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        normalize_output: bool = True,
        capability_mode: str = "single",
        family_dim: int = 64,
        biological_family_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__()
        if capability_mode not in {"single", "factorized_biological"}:
            raise ValueError("capability_mode must be 'single' or 'factorized_biological'")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.normalize_output = bool(normalize_output)
        self.capability_mode = str(capability_mode)
        self.family_dim = int(family_dim)
        self.biological_family_names = tuple(
            str(name)
            for name in (
                biological_family_names
                if biological_family_names is not None
                else BIOLOGICAL_FAMILY_NAMES
            )
        )
        if self.capability_mode == "factorized_biological":
            self.trunk = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.family_heads = nn.ModuleDict(
                {
                    name: nn.Linear(hidden_dim, self.family_dim)
                    for name in self.biological_family_names
                }
            )
            self.family_projection = nn.Sequential(
                nn.LayerNorm(self.family_dim * len(self.biological_family_names)),
                nn.Linear(self.family_dim * len(self.biological_family_names), self.output_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.output_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        return_family_vectors: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.capability_mode == "factorized_biological":
            trunk = self.trunk(x.float())
            family_vectors = {
                name: F.normalize(head(trunk), p=2, dim=-1, eps=1e-12)
                for name, head in self.family_heads.items()
            }
            ordered = [family_vectors[name] for name in self.biological_family_names]
            output = self.family_projection(torch.cat(ordered, dim=-1))
            if self.normalize_output:
                output = F.normalize(output, p=2, dim=-1, eps=1e-12)
            if return_family_vectors:
                return output, family_vectors
            return output

        output = self.net(x.float())
        if self.normalize_output:
            output = F.normalize(output, p=2, dim=-1, eps=1e-12)
        if return_family_vectors:
            return output, {}
        return output


class CapabilityAuxiliaryHeads(nn.Module):
    def __init__(
        self,
        capability_dim: int,
        output_dims: dict[str, int],
        family_dim: int | None = None,
        target_to_family: dict[str, str] | None = None,
        available_family_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__()
        self.target_to_family = target_to_family or {}
        self.available_family_names = set(str(name) for name in (available_family_names or ()))
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(
                    family_dim
                    if (
                        family_dim is not None
                        and name in self.target_to_family
                        and self.target_to_family[name] in self.available_family_names
                    )
                    else capability_dim,
                    dim,
                )
                for name, dim in output_dims.items()
                if dim > 0
            }
        )

    def forward(
        self,
        capability: torch.Tensor,
        family_vectors: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        family_vectors = family_vectors or {}
        out = {}
        for name, head in self.heads.items():
            family_name = self.target_to_family.get(name)
            head_input = family_vectors.get(family_name, capability)
            out[name] = head(head_input)
        return out


class _LightningBase(pl.LightningModule if pl is not None else nn.Module):
    pass


class EnzymeCapabilityLitModule(_LightningBase):
    """Lightning module for enzyme capability pretraining."""

    def __init__(
        self,
        enzyme_input_dims: dict[str, int],
        reaction_demand_dim: int,
        label_output_dims: dict[str, int],
        capability_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        temperature: float = 0.07,
        r2e_capability_weight: float = 1.0,
        e2r_capability_weight: float = 0.5,
        cofactor_subspace_weight: float = 0.0,
        reaction_center_subspace_weight: float = 0.0,
        substrate_subspace_weight: float = 0.0,
        product_subspace_weight: float = 0.0,
        family_e2r_weight: float = 0.4,
        cofactor_bce_weight: float = 0.0,
        core_cofactor_bce_weight: float = 0.0,
        enzyme_derived_cofactor_bce_weight: float = 0.0,
        enzyme_derived_core_cofactor_bce_weight: float = 0.0,
        combined_cofactor_bce_weight: float = 0.0,
        combined_core_cofactor_bce_weight: float = 0.0,
        cofactor_architecture_bce_weight: float = 0.10,
        cofactor_chemistry_bce_weight: float = 0.10,
        metal_ion_bce_weight: float = 0.03,
        auxiliary_participant_bce_weight: float = 0.02,
        reaction_center_bce_weight: float = 0.10,
        substrate_product_bce_weight: float = 0.10,
        substrate_product_transition_bce_weight: float = 0.10,
        reaction_type_bce_weight: float = 0.10,
        ec_aux_weight: float = 0.03,
        pretraining_objective: str = "reaction_demand",
        bio_cofactor_weight: float = 1.0,
        bio_transition_weight: float = 1.0,
        bio_reaction_center_weight: float = 1.0,
        bio_substrate_weight: float = 0.7,
        bio_product_weight: float = 0.7,
        bio_global_weight: float = 0.5,
        bio_attribute_weight: float = 0.1,
        bio_composite_weight: float = 1.0,
        composite_family_weights: dict[str, float] | None = None,
        composite_positive_family_names: tuple[str, ...] | list[str] | None = None,
        composite_known_family_names: tuple[str, ...] | list[str] | None = None,
        composite_required_positive_families: tuple[str, ...] | list[str] | None = None,
        bio_selected_families: tuple[str, ...] | list[str] | None = None,
        same_center_hard_negative_weight: float = 0.3,
        composite_positive_threshold: float = 0.20,
        composite_min_positive_families: int = 2,
        composite_min_known_families: int = 2,
        same_center_hard_negative_margin: float = 0.10,
        hard_negative_temperature: float | None = None,
        factor_diversity_weight: float = 0.02,
        reaction_to_enzymes: dict[str, set[str] | list[str]] | None = None,
        enzyme_to_reactions: dict[str, set[str] | list[str]] | None = None,
        reaction_family_masks: dict[str, dict[str, bool]] | None = None,
        enzyme_family_masks: dict[str, dict[str, bool]] | None = None,
        freeze_reaction_encoder_after_epochs: int | None = None,
        deduplicate_entities_before_loss: bool = True,
        capability_mode: str = "single",
        family_dim: int = 64,
        residue_input_dim: int | None = None,
        use_residue_multiquery_pooling: bool = False,
        residue_pool_dropout: float = 0.0,
        residue_pool_scale: float = 0.1,
        biological_family_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__()
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(
                ignore=[
                    "reaction_to_enzymes",
                    "enzyme_to_reactions",
                    "reaction_family_masks",
                    "enzyme_family_masks",
                ]
            )
        self.biological_family_names = tuple(
            str(name)
            for name in (
                biological_family_names
                if biological_family_names is not None
                else BIOLOGICAL_FAMILY_NAMES
            )
        )
        self.enzyme_encoder = EnzymeCapabilityEncoder(
            input_dims=enzyme_input_dims,
            capability_dim=capability_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            capability_mode=capability_mode,
            family_dim=family_dim,
            residue_input_dim=residue_input_dim,
            use_residue_multiquery_pooling=use_residue_multiquery_pooling,
            residue_pool_dropout=residue_pool_dropout,
            residue_pool_scale=residue_pool_scale,
            biological_family_names=self.biological_family_names,
        )
        self.reaction_encoder = ReactionDemandEncoder(
            input_dim=reaction_demand_dim,
            output_dim=capability_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            capability_mode=capability_mode,
            family_dim=family_dim,
            biological_family_names=self.biological_family_names,
        )
        self.aux_heads = CapabilityAuxiliaryHeads(
            capability_dim,
            label_output_dims,
            family_dim=family_dim if capability_mode == "factorized_biological" else None,
            target_to_family=TARGET_TO_BIOLOGICAL_FAMILY,
            available_family_names=self.biological_family_names,
        )
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.temperature = float(temperature)
        self.capability_mode = str(capability_mode)
        self.pretraining_objective = str(pretraining_objective)
        if self.pretraining_objective not in {
            "reaction_demand",
            "enzyme_bio_supcon",
            "enzyme_bio_composite_supcon",
            "reaction_demand_bio_composite",
        }:
            raise ValueError(
                "pretraining_objective must be 'reaction_demand', "
                "'enzyme_bio_supcon', 'enzyme_bio_composite_supcon', "
                "or 'reaction_demand_bio_composite'"
            )
        self.family_e2r_weight = float(family_e2r_weight)
        self.weights = {
            "r2e": float(r2e_capability_weight),
            "e2r": float(e2r_capability_weight),
            "family_cofactor": float(cofactor_subspace_weight),
            "family_transition": 0.0,
            "family_reaction_center": float(reaction_center_subspace_weight),
            "family_substrate": float(substrate_subspace_weight),
            "family_product": float(product_subspace_weight),
            "cofactor": float(cofactor_bce_weight),
            "core_cofactor": float(core_cofactor_bce_weight),
            "enzyme_derived_cofactor": float(enzyme_derived_cofactor_bce_weight),
            "enzyme_derived_core_cofactor": float(enzyme_derived_core_cofactor_bce_weight),
            "combined_cofactor": float(combined_cofactor_bce_weight),
            "combined_core_cofactor": float(combined_core_cofactor_bce_weight),
            "cofactor_architecture": float(cofactor_architecture_bce_weight),
            "cofactor_chemistry": float(cofactor_chemistry_bce_weight),
            "metal_ion": float(metal_ion_bce_weight),
            "auxiliary_participant": float(auxiliary_participant_bce_weight),
            "reaction_center": float(reaction_center_bce_weight),
            "substrate_product": float(substrate_product_bce_weight),
            "substrate_product_transition": float(substrate_product_transition_bce_weight),
            "reaction_type": float(reaction_type_bce_weight),
            "ec": float(ec_aux_weight),
            "bio_cofactor": float(bio_cofactor_weight),
            "bio_transition": float(bio_transition_weight),
            "bio_reaction_center": float(bio_reaction_center_weight),
            "bio_substrate": float(bio_substrate_weight),
            "bio_product": float(bio_product_weight),
            "bio_global": float(bio_global_weight),
            "bio_attribute": float(bio_attribute_weight),
            "bio_composite": float(bio_composite_weight),
            "same_center_hard_negative": float(same_center_hard_negative_weight),
            "factor_diversity": float(factor_diversity_weight),
        }
        self.bio_selected_families = tuple(
            str(name)
            for name in (
                bio_selected_families
                if bio_selected_families is not None
                else ("cofactor", "reaction_center", "substrate", "product")
            )
        )
        self.bio_supcon_loss = BiologicalFactorSupConLoss(temperature=self.temperature)
        self.composite_bio_supcon_loss = CompositeBiologicalSupConLoss(
            temperature=self.temperature,
            hard_negative_temperature=(
                self.temperature
                if hard_negative_temperature is None
                else float(hard_negative_temperature)
            ),
            hard_negative_margin=float(same_center_hard_negative_margin),
            positive_threshold=float(composite_positive_threshold),
            min_positive_families=int(composite_min_positive_families),
            min_known_families=int(composite_min_known_families),
            family_weights=composite_family_weights,
            positive_family_names=(
                None
                if composite_positive_family_names is None
                else tuple(str(name) for name in composite_positive_family_names)
            ),
            known_family_names=(
                None
                if composite_known_family_names is None
                else tuple(str(name) for name in composite_known_family_names)
            ),
            required_positive_families=(
                None
                if composite_required_positive_families is None
                else tuple(str(name) for name in composite_required_positive_families)
            ),
        )
        self.reaction_to_enzymes = {
            str(key): set(str(value) for value in values)
            for key, values in (reaction_to_enzymes or {}).items()
        }
        self.enzyme_to_reactions = {
            str(key): set(str(value) for value in values)
            for key, values in (enzyme_to_reactions or {}).items()
        }
        self.reaction_family_masks = {
            str(reaction_id): {str(name): bool(value) for name, value in masks.items()}
            for reaction_id, masks in (reaction_family_masks or {}).items()
        }
        self.enzyme_family_masks = {
            str(enzyme_id): {str(name): bool(value) for name, value in masks.items()}
            for enzyme_id, masks in (enzyme_family_masks or {}).items()
        }
        self.freeze_reaction_encoder_after_epochs = (
            None
            if freeze_reaction_encoder_after_epochs is None
            else int(freeze_reaction_encoder_after_epochs)
        )
        self.deduplicate_entities_before_loss = bool(deduplicate_entities_before_loss)

    def on_train_epoch_start(self) -> None:
        if self.freeze_reaction_encoder_after_epochs is None:
            return
        if int(getattr(self, "current_epoch", 0)) < self.freeze_reaction_encoder_after_epochs:
            return
        for parameter in self.reaction_encoder.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        batch: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        enzyme_features = {
            name: batch[name] for name in self.enzyme_encoder.input_dims if name in batch
        }
        enzyme_capability, enzyme_families = self.enzyme_encoder(
            enzyme_features,
            residue_embeddings=batch.get("prot5_residue_embeddings"),
            residue_mask=batch.get("prot5_residue_mask"),
            return_family_vectors=True,
        )
        reaction_capability, reaction_families = self.reaction_encoder(
            batch["reaction_demand_vec"],
            return_family_vectors=True,
        )
        aux_logits = self.aux_heads(enzyme_capability, enzyme_families)
        return enzyme_capability, reaction_capability, aux_logits, enzyme_families, reaction_families

    def _distributed_enabled(self) -> bool:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _gather_objects(self, obj: Any) -> list[Any]:
        if not self._distributed_enabled():
            return [obj]
        import torch.distributed as dist

        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, obj)
        return gathered

    def _gather_variable_tensor(self, tensor: torch.Tensor, sync_grads: bool) -> list[torch.Tensor]:
        if not self._distributed_enabled():
            return [tensor]
        local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
        gathered_sizes = self.all_gather(local_size)
        if gathered_sizes.dim() == local_size.dim():
            gathered_sizes = gathered_sizes.unsqueeze(0)
        gathered_sizes = gathered_sizes.reshape(-1)
        max_size = int(gathered_sizes.max().item())
        padded = tensor
        if tensor.shape[0] < max_size:
            pad_shape = (max_size - tensor.shape[0], *tensor.shape[1:])
            padded = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)
        gathered = self.all_gather(padded, sync_grads=sync_grads)
        if gathered.dim() == padded.dim():
            gathered = gathered.unsqueeze(0)
        return [
            gathered[rank, : int(size.item())]
            for rank, size in enumerate(gathered_sizes)
        ]

    @staticmethod
    def _deduplicate(
        embeddings: torch.Tensor,
        ids: list[str],
    ) -> tuple[torch.Tensor, list[str]]:
        id_to_idx: dict[str, int] = {}
        keep_indices: list[int] = []
        keep_ids: list[str] = []
        for idx, item_id in enumerate(ids):
            if item_id not in id_to_idx:
                id_to_idx[item_id] = len(keep_indices)
                keep_indices.append(idx)
                keep_ids.append(item_id)
        keep = torch.tensor(keep_indices, dtype=torch.long, device=embeddings.device)
        return embeddings[keep], keep_ids

    @staticmethod
    def _deduplicate_with_tensors(
        embeddings: torch.Tensor,
        ids: list[str],
        tensors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, list[str], dict[str, torch.Tensor]]:
        id_to_idx: dict[str, int] = {}
        keep_indices: list[int] = []
        keep_ids: list[str] = []
        for idx, item_id in enumerate(ids):
            if item_id not in id_to_idx:
                id_to_idx[item_id] = len(keep_indices)
                keep_indices.append(idx)
                keep_ids.append(item_id)
        keep = torch.tensor(keep_indices, dtype=torch.long, device=embeddings.device)
        deduped = {
            name: tensor.to(device=embeddings.device)[keep]
            for name, tensor in tensors.items()
        }
        return embeddings[keep], keep_ids, deduped

    def _contrastive_inputs(
        self,
        reaction_capability: torch.Tensor,
        enzyme_capability: torch.Tensor,
        reaction_ids: list[str],
        enzyme_ids: list[str],
        sync_grads: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
        if self._distributed_enabled():
            reaction_parts = self._gather_variable_tensor(
                reaction_capability,
                sync_grads=sync_grads,
            )
            enzyme_parts = self._gather_variable_tensor(
                enzyme_capability,
                sync_grads=sync_grads,
            )
            reaction_id_parts = self._gather_objects(reaction_ids)
            enzyme_id_parts = self._gather_objects(enzyme_ids)
            reaction_capability = torch.cat(reaction_parts, dim=0)
            enzyme_capability = torch.cat(enzyme_parts, dim=0)
            reaction_ids = [item for part in reaction_id_parts for item in part]
            enzyme_ids = [item for part in enzyme_id_parts for item in part]
        if self.deduplicate_entities_before_loss:
            reaction_capability, reaction_ids = self._deduplicate(reaction_capability, reaction_ids)
            enzyme_capability, enzyme_ids = self._deduplicate(enzyme_capability, enzyme_ids)
        if self.reaction_to_enzymes and self.enzyme_to_reactions:
            r2e_mask, e2r_mask = positive_masks_from_pair_maps(
                reaction_ids,
                enzyme_ids,
                self.reaction_to_enzymes,
                self.enzyme_to_reactions,
            )
        else:
            r2e_mask, e2r_mask = pair_id_positive_masks(reaction_ids, enzyme_ids)
        return reaction_capability, enzyme_capability, r2e_mask, e2r_mask, reaction_ids, enzyme_ids

    @staticmethod
    def _id_family_mask(
        ids: list[str],
        family_masks: dict[str, dict[str, bool]],
        family_name: str,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                bool(family_masks.get(str(item_id), {}).get(family_name, False))
                for item_id in ids
            ],
            dtype=torch.bool,
            device=device,
        )

    def _family_contrastive_loss(
        self,
        *,
        family_name: str,
        reaction_family_vectors: dict[str, torch.Tensor],
        enzyme_family_vectors: dict[str, torch.Tensor],
        reaction_ids: list[str],
        enzyme_ids: list[str],
        sync_grads: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if family_name not in reaction_family_vectors or family_name not in enzyme_family_vectors:
            device = next(iter(reaction_family_vectors.values())).device if reaction_family_vectors else "cpu"
            zero = torch.zeros((), device=device)
            return zero, zero, zero
        (
            contrastive_reactions,
            contrastive_enzymes,
            r2e_mask,
            e2r_mask,
            contrastive_reaction_ids,
            contrastive_enzyme_ids,
        ) = self._contrastive_inputs(
            reaction_family_vectors[family_name],
            enzyme_family_vectors[family_name],
            reaction_ids,
            enzyme_ids,
            sync_grads=sync_grads,
        )
        reaction_valid = self._id_family_mask(
            contrastive_reaction_ids,
            self.reaction_family_masks,
            family_name,
            contrastive_reactions.device,
        )
        enzyme_valid = self._id_family_mask(
            contrastive_enzyme_ids,
            self.enzyme_family_masks,
            family_name,
            contrastive_enzymes.device,
        )
        r2e_positive = r2e_mask.to(device=contrastive_reactions.device) & reaction_valid.unsqueeze(1)
        r2e_positive = r2e_positive & enzyme_valid.unsqueeze(0)
        e2r_positive = e2r_mask.to(device=contrastive_enzymes.device) & enzyme_valid.unsqueeze(1)
        e2r_positive = e2r_positive & reaction_valid.unsqueeze(0)
        r2e_loss = multi_positive_contrastive_loss(
            contrastive_reactions,
            contrastive_enzymes,
            r2e_positive,
            temperature=self.temperature,
            candidate_mask=enzyme_valid,
        )
        e2r_loss = multi_positive_contrastive_loss(
            contrastive_enzymes,
            contrastive_reactions,
            e2r_positive,
            temperature=self.temperature,
            candidate_mask=reaction_valid,
        )
        valid_anchors = r2e_positive.any(dim=1).sum().to(dtype=contrastive_reactions.dtype)
        return r2e_loss, e2r_loss, valid_anchors

    @staticmethod
    def _mrr_from_scores(scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
        valid = positive_mask.any(dim=1)
        if not bool(valid.any()):
            return scores.new_zeros(())
        positive_scores = scores.masked_fill(~positive_mask, torch.finfo(scores.dtype).min)
        best_positive = positive_scores[valid].max(dim=1).values
        ranks = (scores[valid] > best_positive.unsqueeze(1)).sum(dim=1).to(scores.dtype) + 1.0
        return (1.0 / ranks).mean()

    @staticmethod
    def _recall_at_k_from_scores(
        scores: torch.Tensor,
        positive_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
        valid = positive_mask.any(dim=1)
        if not bool(valid.any()):
            return scores.new_zeros(())
        k = min(int(k), scores.shape[1])
        topk = torch.topk(scores[valid], k=k, dim=1).indices
        gathered = positive_mask[valid].gather(1, topk)
        return gathered.any(dim=1).to(scores.dtype).mean()

    @staticmethod
    def _zero_factor_target(batch: dict[str, Any]) -> tuple[str | None, torch.Tensor, torch.Tensor]:
        reference = next(
            value for value in batch.values()
            if torch.is_tensor(value) and value.ndim > 0
        )
        batch_size = int(reference.shape[0])
        return (
            None,
            torch.zeros(batch_size, 0, dtype=torch.float32, device=reference.device),
            torch.zeros(batch_size, dtype=torch.bool, device=reference.device),
        )

    @staticmethod
    def _label_jaccard_matrix(
        labels: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        if labels.ndim != 2 or labels.shape[1] == 0:
            return labels.new_zeros(labels.shape[0], labels.shape[0])
        labels = (labels > 0).to(dtype=labels.dtype)
        label_mask = label_mask.to(device=labels.device, dtype=torch.bool)
        if label_mask.ndim > 1:
            label_mask = label_mask.reshape(label_mask.shape[0], -1).any(dim=1)
        known = (labels.sum(dim=1) > 0) & label_mask
        overlap = labels @ labels.T
        counts = labels.sum(dim=1)
        union = counts.unsqueeze(1) + counts.unsqueeze(0) - overlap
        jaccard = torch.where(
            union > 0,
            overlap / union.clamp_min(1.0),
            torch.zeros_like(overlap),
        )
        known_pair = known.unsqueeze(0) & known.unsqueeze(1)
        return jaccard.masked_fill(~known_pair, 0.0)

    @staticmethod
    def _knn_label_diagnostics(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        *,
        k: int,
    ) -> dict[str, torch.Tensor]:
        device = embeddings.device
        n = int(embeddings.shape[0])
        zero = embeddings.new_zeros(())
        if n <= 1 or labels.ndim != 2 or labels.shape[1] == 0:
            return {"jaccard": zero, "enrichment": zero, "valid_anchors": zero}
        labels = labels.to(device=device, dtype=embeddings.dtype)
        label_mask = label_mask.to(device=device, dtype=torch.bool)
        if label_mask.ndim > 1:
            label_mask = label_mask.reshape(label_mask.shape[0], -1).any(dim=1)
        known = (labels.sum(dim=1) > 0) & label_mask
        if not bool(known.any()):
            return {"jaccard": zero, "enrichment": zero, "valid_anchors": zero}
        jaccard = EnzymeCapabilityLitModule._label_jaccard_matrix(labels, label_mask)
        not_self = ~torch.eye(n, dtype=torch.bool, device=device)
        scores = embeddings @ embeddings.T
        scores = scores.masked_fill(~not_self, torch.finfo(scores.dtype).min)
        k_eff = min(max(1, int(k)), max(1, n - 1))
        topk = torch.topk(scores, k=k_eff, dim=1).indices
        topk_jaccard = jaccard.gather(1, topk).mean(dim=1)
        valid = known & not_self.any(dim=1)
        if not bool(valid.any()):
            return {"jaccard": zero, "enrichment": zero, "valid_anchors": zero}
        random_baseline = jaccard.masked_fill(~not_self, 0.0).sum(dim=1) / not_self.sum(dim=1).clamp_min(1)
        observed = topk_jaccard[valid].mean()
        baseline = random_baseline[valid].mean()
        enrichment = observed / baseline.clamp_min(torch.finfo(embeddings.dtype).eps)
        return {
            "jaccard": observed,
            "enrichment": enrichment,
            "valid_anchors": valid.sum().to(dtype=embeddings.dtype),
        }

    @staticmethod
    def _embedding_health(embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        if embeddings.ndim != 2 or embeddings.shape[0] <= 1:
            zero = embeddings.sum() * 0.0
            return {
                "embed_norm_mean": zero,
                "embed_norm_std": zero,
                "pairwise_cosine_mean": zero,
                "pairwise_cosine_std": zero,
                "effective_rank": zero,
                "collapse_score": zero,
            }
        norms = embeddings.norm(dim=-1)
        scores = embeddings @ embeddings.T
        not_self = ~torch.eye(scores.shape[0], dtype=torch.bool, device=scores.device)
        offdiag = scores[not_self]
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered.float())
        if singular_values.numel() == 0 or not bool((singular_values > 0).any()):
            effective_rank = embeddings.new_zeros(())
        else:
            probs = singular_values / singular_values.sum().clamp_min(torch.finfo(singular_values.dtype).eps)
            entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).eps).log()).sum()
            effective_rank = entropy.exp().to(device=embeddings.device, dtype=embeddings.dtype)
        return {
            "embed_norm_mean": norms.mean(),
            "embed_norm_std": norms.std(unbiased=False),
            "pairwise_cosine_mean": offdiag.mean(),
            "pairwise_cosine_std": offdiag.std(unbiased=False),
            "effective_rank": effective_rank,
            "collapse_score": offdiag.mean().abs(),
        }

    def _composite_separation_diagnostics(
        self,
        embeddings: torch.Tensor,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        labels_by_family: dict[str, torch.Tensor] = {}
        masks_by_family: dict[str, torch.Tensor] = {}
        for family_name in self.bio_selected_families:
            _target_key, labels, label_mask = self._select_factor_target(batch, family_name)
            if labels.shape[1] == 0:
                continue
            labels_by_family[family_name] = labels.to(device=embeddings.device, dtype=embeddings.dtype)
            masks_by_family[family_name] = label_mask.to(device=embeddings.device, dtype=torch.bool)
        if not labels_by_family or embeddings.shape[0] <= 1:
            zero = embeddings.sum() * 0.0
            return {
                "composite_positive_negative_cosine_gap": zero,
                "composite_positive_cosine": zero,
                "composite_negative_cosine": zero,
            }
        n = embeddings.shape[0]
        not_self = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
        composite = embeddings.new_zeros(n, n)
        available = embeddings.new_zeros(n, n)
        positive_count = torch.zeros(n, n, dtype=torch.long, device=embeddings.device)
        required_any = torch.zeros(n, n, dtype=torch.bool, device=embeddings.device)
        for family_name, labels in labels_by_family.items():
            jaccard = self._label_jaccard_matrix(labels, masks_by_family[family_name])
            weight = float(self.composite_bio_supcon_loss.family_weights.get(family_name, 0.0))
            if weight <= 0:
                continue
            known_pair = jaccard > 0
            composite = composite + weight * jaccard
            available = available + weight * known_pair.to(dtype=embeddings.dtype)
            positive_count = positive_count + known_pair.to(torch.long)
            if family_name in self.composite_bio_supcon_loss.required_positive_families:
                required_any = required_any | known_pair
        composite = torch.where(
            available > 0,
            composite / available.clamp_min(torch.finfo(embeddings.dtype).eps),
            torch.zeros_like(composite),
        )
        positive = (
            not_self
            & (positive_count >= self.composite_bio_supcon_loss.min_positive_families)
            & (composite >= self.composite_bio_supcon_loss.positive_threshold)
        )
        if self.composite_bio_supcon_loss.required_positive_families:
            positive = positive & required_any
        negative = not_self & ~positive
        scores = embeddings @ embeddings.T
        pos_score = scores[positive].mean() if bool(positive.any()) else embeddings.new_zeros(())
        neg_score = scores[negative].mean() if bool(negative.any()) else embeddings.new_zeros(())
        return {
            "composite_positive_negative_cosine_gap": pos_score - neg_score,
            "composite_positive_cosine": pos_score,
            "composite_negative_cosine": neg_score,
        }

    def _placement_diagnostics(
        self,
        enzyme_capability: torch.Tensor,
        enzyme_family_vectors: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        diagnostics = self._embedding_health(enzyme_capability)
        diagnostics.update(self._composite_separation_diagnostics(enzyme_capability, batch))
        for family_name in self.bio_selected_families:
            _target_key, labels, label_mask = self._select_factor_target(batch, family_name)
            embedding = enzyme_family_vectors.get(family_name, enzyme_capability)
            for k in (10, 50):
                stats = self._knn_label_diagnostics(
                    embedding,
                    labels,
                    label_mask,
                    k=k,
                )
                diagnostics[f"{family_name}_jaccard_at_{k}"] = stats["jaccard"]
                diagnostics[f"{family_name}_enrichment_at_{k}"] = stats["enrichment"]
                diagnostics[f"{family_name}_valid_anchors_at_{k}"] = stats["valid_anchors"]
        for k in (10, 50):
            global_labels = []
            global_masks = []
            for family_name in self.bio_selected_families:
                _target_key, labels, label_mask = self._select_factor_target(batch, family_name)
                if labels.shape[1] == 0:
                    continue
                global_labels.append(labels)
                global_masks.append(label_mask)
            if global_labels:
                labels = torch.cat(global_labels, dim=1)
                mask = torch.stack(global_masks, dim=0).any(dim=0)
                stats = self._knn_label_diagnostics(enzyme_capability, labels, mask, k=k)
                diagnostics[f"composite_jaccard_at_{k}"] = stats["jaccard"]
                diagnostics[f"composite_enrichment_at_{k}"] = stats["enrichment"]
        diagnostics.update(self.enzyme_encoder.attention_diagnostics())
        return diagnostics

    def _select_factor_target(
        self,
        batch: dict[str, Any],
        family_name: str,
    ) -> tuple[str | None, torch.Tensor, torch.Tensor]:
        candidates: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for key in BIOLOGICAL_FACTOR_TARGET_PREFERENCES.get(family_name, ()):
            target = batch.get(key)
            if not torch.is_tensor(target) or target.ndim != 2 or target.shape[1] == 0:
                continue
            mask = batch.get(f"{key}_mask")
            if torch.is_tensor(mask):
                mask = mask.to(device=target.device, dtype=torch.bool)
                if mask.ndim > 1:
                    mask = mask.reshape(mask.shape[0], -1).any(dim=1)
            else:
                mask = target.to(dtype=torch.float32).sum(dim=1) > 0
            candidates.append((key, target.float(), mask))
        if not candidates:
            return self._zero_factor_target(batch)
        for candidate in candidates:
            target_has_labels = candidate[1].sum(dim=1) > 0
            if bool((candidate[2] & target_has_labels).any()):
                return candidate
        for candidate in candidates:
            if bool(candidate[2].any()):
                return candidate
        return candidates[0]

    def _global_biological_target(self, batch: dict[str, Any]) -> tuple[str, torch.Tensor, torch.Tensor]:
        targets = []
        masks = []
        for family_name in self.bio_selected_families:
            _key, target, mask = self._select_factor_target(batch, family_name)
            if target.shape[1] == 0:
                continue
            targets.append(target)
            masks.append(mask)
        if not targets:
            _key, target, mask = self._zero_factor_target(batch)
            return "global", target, mask
        return "global", torch.cat(targets, dim=1), torch.stack(masks, dim=0).any(dim=0)

    def _gather_enzyme_factor_inputs(
        self,
        embeddings: torch.Tensor,
        enzyme_ids: list[str],
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        sync_grads: bool,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
        labels = labels.to(device=embeddings.device)
        label_mask = label_mask.to(device=embeddings.device)
        if self._distributed_enabled():
            embedding_parts = self._gather_variable_tensor(embeddings, sync_grads=sync_grads)
            label_parts = self._gather_variable_tensor(labels, sync_grads=False)
            mask_parts = self._gather_variable_tensor(label_mask, sync_grads=False)
            enzyme_id_parts = self._gather_objects(enzyme_ids)
            embeddings = torch.cat(embedding_parts, dim=0)
            labels = torch.cat(label_parts, dim=0)
            label_mask = torch.cat(mask_parts, dim=0)
            enzyme_ids = [item for part in enzyme_id_parts for item in part]
        if self.deduplicate_entities_before_loss:
            embeddings, enzyme_ids, tensors = self._deduplicate_with_tensors(
                embeddings,
                enzyme_ids,
                {"labels": labels, "label_mask": label_mask},
            )
            labels = tensors["labels"]
            label_mask = tensors["label_mask"]
        return embeddings, enzyme_ids, labels, label_mask

    def _gather_composite_bio_inputs(
        self,
        embeddings: torch.Tensor,
        enzyme_ids: list[str],
        labels_by_family: dict[str, torch.Tensor],
        masks_by_family: dict[str, torch.Tensor],
        sync_grads: bool,
    ) -> tuple[
        torch.Tensor,
        list[str],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        labels_by_family = {
            name: labels.to(device=embeddings.device)
            for name, labels in labels_by_family.items()
        }
        masks_by_family = {
            name: mask.to(device=embeddings.device)
            for name, mask in masks_by_family.items()
        }
        if self._distributed_enabled():
            embedding_parts = self._gather_variable_tensor(embeddings, sync_grads=sync_grads)
            enzyme_id_parts = self._gather_objects(enzyme_ids)
            embeddings = torch.cat(embedding_parts, dim=0)
            enzyme_ids = [item for part in enzyme_id_parts for item in part]
            labels_by_family = {
                name: torch.cat(self._gather_variable_tensor(labels, sync_grads=False), dim=0)
                for name, labels in labels_by_family.items()
            }
            masks_by_family = {
                name: torch.cat(self._gather_variable_tensor(mask, sync_grads=False), dim=0)
                for name, mask in masks_by_family.items()
            }
        if self.deduplicate_entities_before_loss:
            tensors: dict[str, torch.Tensor] = {}
            for name, labels in labels_by_family.items():
                tensors[f"labels/{name}"] = labels
            for name, mask in masks_by_family.items():
                tensors[f"mask/{name}"] = mask
            embeddings, enzyme_ids, tensors = self._deduplicate_with_tensors(
                embeddings,
                enzyme_ids,
                tensors,
            )
            labels_by_family = {
                name: tensors[f"labels/{name}"]
                for name in labels_by_family
            }
            masks_by_family = {
                name: tensors[f"mask/{name}"]
                for name in masks_by_family
            }
        return embeddings, enzyme_ids, labels_by_family, masks_by_family

    def _biological_factor_loss(
        self,
        *,
        family_name: str,
        embeddings: torch.Tensor,
        batch: dict[str, Any],
        enzyme_ids: list[str],
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        if family_name == "global":
            target_key, labels, label_mask = self._global_biological_target(batch)
        else:
            target_key, labels, label_mask = self._select_factor_target(batch, family_name)
        del target_key
        embeddings, _ids, labels, label_mask = self._gather_enzyme_factor_inputs(
            embeddings,
            enzyme_ids,
            labels,
            label_mask,
            sync_grads=(prefix == "train"),
        )
        return self.bio_supcon_loss(embeddings, labels, label_mask)

    def _composite_biological_loss(
        self,
        *,
        embeddings: torch.Tensor,
        batch: dict[str, Any],
        enzyme_ids: list[str],
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        labels_by_family: dict[str, torch.Tensor] = {}
        masks_by_family: dict[str, torch.Tensor] = {}
        for family_name in (
            "cofactor",
            "transition",
            "substrate",
            "product",
            "reaction_center",
        ):
            _target_key, labels, label_mask = self._select_factor_target(batch, family_name)
            labels_by_family[family_name] = labels.float()
            masks_by_family[family_name] = label_mask.to(dtype=torch.bool)
        embeddings, _ids, labels_by_family, masks_by_family = self._gather_composite_bio_inputs(
            embeddings,
            enzyme_ids,
            labels_by_family,
            masks_by_family,
            sync_grads=(prefix == "train"),
        )
        return self.composite_bio_supcon_loss(
            embeddings,
            labels_by_family,
            masks_by_family,
        )

    @staticmethod
    def _factor_diversity_loss(family_vectors: dict[str, torch.Tensor]) -> torch.Tensor:
        selected = [
            family_vectors[name]
            for name in DEFAULT_STAGE1_BIO_FAMILIES
            if name in family_vectors
        ]
        if len(selected) < 2:
            device = selected[0].device if selected else "cpu"
            return torch.zeros((), device=device)
        stacked = torch.stack(selected, dim=1)
        similarity = torch.einsum("bfd,bgd->bfg", stacked, stacked)
        eye = torch.eye(len(selected), dtype=torch.bool, device=stacked.device)
        return similarity.masked_select(~eye.unsqueeze(0)).pow(2).mean()

    def _selected_attribute_loss(
        self,
        batch: dict[str, Any],
        aux_logits: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        total = None
        components: dict[str, torch.Tensor] = {}
        masked_missing = None
        for family_name in self.bio_selected_families:
            target_key, target, target_mask = self._select_factor_target(batch, family_name)
            if target_key is None or target_key not in aux_logits:
                continue
            loss = multilabel_bce_or_zero(
                aux_logits.get(target_key),
                target,
                target_mask,
            )
            components[family_name] = loss
            total = loss if total is None else total + loss
            missing = (~target_mask.to(dtype=torch.bool)).sum().to(
                device=target.device,
                dtype=target.dtype,
            )
            masked_missing = missing if masked_missing is None else masked_missing + missing
        if total is None:
            reference = next(iter(aux_logits.values())) if aux_logits else next(
                value for value in batch.values() if torch.is_tensor(value)
            )
            total = reference.new_zeros(())
        if masked_missing is None:
            masked_missing = total.new_zeros(())
        return total, components, masked_missing

    def _bio_supcon_step(
        self,
        *,
        batch: dict[str, Any],
        prefix: str,
        enzyme_capability: torch.Tensor,
        aux_logits: dict[str, torch.Tensor],
        enzyme_family_vectors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        enzyme_ids = [str(value) for value in batch["enzyme_id"]]
        factor_embeddings = {
            "global": enzyme_family_vectors.get("global", enzyme_capability),
            "cofactor": enzyme_family_vectors.get("cofactor", enzyme_capability),
            "transition": enzyme_family_vectors.get("transition", enzyme_capability),
            "reaction_center": enzyme_family_vectors.get("reaction_center", enzyme_capability),
            "substrate": enzyme_family_vectors.get("substrate", enzyme_capability),
            "product": enzyme_family_vectors.get("product", enzyme_capability),
        }
        factor_results = {
            family_name: self._biological_factor_loss(
                family_name=family_name,
                embeddings=embedding,
                batch=batch,
                enzyme_ids=enzyme_ids,
                prefix=prefix,
            )
            for family_name, embedding in factor_embeddings.items()
        }
        composite_result: dict[str, torch.Tensor] | None = None
        if self.pretraining_objective == "enzyme_bio_composite_supcon":
            composite_result = self._composite_biological_loss(
                embeddings=enzyme_capability,
                batch=batch,
                enzyme_ids=enzyme_ids,
                prefix=prefix,
            )
        attr_loss, attr_components, missing_attr = self._selected_attribute_loss(batch, aux_logits)
        diversity_loss = self._factor_diversity_loss(enzyme_family_vectors).to(
            device=enzyme_capability.device
        )
        factor_loss = (
            self.weights["bio_global"] * factor_results["global"]["loss"]
            + self.weights["bio_cofactor"] * factor_results["cofactor"]["loss"]
            + self.weights["bio_transition"] * factor_results["transition"]["loss"]
            + self.weights["bio_reaction_center"] * factor_results["reaction_center"]["loss"]
            + self.weights["bio_substrate"] * factor_results["substrate"]["loss"]
            + self.weights["bio_product"] * factor_results["product"]["loss"]
        )
        if composite_result is None:
            loss = (
                factor_loss
                + self.weights["bio_attribute"] * attr_loss
                + self.weights["factor_diversity"] * diversity_loss
            )
        else:
            loss = (
                self.weights["bio_composite"] * composite_result["loss"]
                + self.weights["same_center_hard_negative"]
                * composite_result["loss/hard_negative"]
                + factor_loss
                + self.weights["bio_attribute"] * attr_loss
                + self.weights["factor_diversity"] * diversity_loss
            )
        if hasattr(self, "log"):
            batch_size = len(enzyme_ids)
            self.log(
                f"{prefix}/loss_total",
                loss,
                on_epoch=True,
                on_step=(prefix == "train"),
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                f"{prefix}/loss",
                loss,
                on_epoch=True,
                on_step=(prefix == "train"),
                batch_size=batch_size,
                sync_dist=True,
            )
            component_logs: dict[str, torch.Tensor] = {
                "loss_factor_supcon": factor_loss,
                "loss_global_supcon": factor_results["global"]["loss"],
                "loss_cofactor_supcon": factor_results["cofactor"]["loss"],
                "loss_transition_supcon": factor_results["transition"]["loss"],
                "loss_reaction_center_supcon": factor_results["reaction_center"]["loss"],
                "loss_substrate_supcon": factor_results["substrate"]["loss"],
                "loss_product_supcon": factor_results["product"]["loss"],
                "loss_attribute_bce": attr_loss,
                "loss_factor_diversity": diversity_loss,
                "num_masked_missing_attribute_labels": missing_attr,
            }
            if composite_result is not None:
                component_logs.update(
                    {
                        "loss_composite_supcon": composite_result["loss"],
                        "loss_same_center_hard_negative": composite_result[
                            "loss/hard_negative"
                        ],
                        "composite_valid_anchors": composite_result[
                            "metrics/valid_anchors"
                        ].to(loss.device),
                        "composite_mean_positives": composite_result[
                            "metrics/mean_positives"
                        ].to(loss.device),
                        "composite_mean_positive_score": composite_result[
                            "metrics/mean_positive_score"
                        ].to(loss.device),
                        "composite_mean_known_families": composite_result[
                            "metrics/mean_known_families"
                        ].to(loss.device),
                        "composite_mean_hard_negatives": composite_result[
                            "metrics/mean_hard_negatives"
                        ].to(loss.device),
                    }
                )
            for family_name, result in factor_results.items():
                component_logs[f"{family_name}_valid_anchors"] = result[
                    "metrics/valid_anchors"
                ].to(loss.device)
                component_logs[f"{family_name}_known_samples"] = result[
                    "metrics/known_samples"
                ].to(loss.device)
                component_logs[f"{family_name}_mean_positives"] = result[
                    "metrics/mean_positives"
                ].to(loss.device)
                component_logs[f"{family_name}_mean_jaccard"] = result[
                    "metrics/mean_jaccard"
                ].to(loss.device)
                component_logs[f"{family_name}_masked_missing"] = result[
                    "metrics/masked_missing"
                ].to(loss.device)
            for family_name, value in attr_components.items():
                component_logs[f"loss_attr_{family_name}"] = value
            for name, value in component_logs.items():
                self.log(
                    f"{prefix}/{name}",
                    value if torch.is_tensor(value) else torch.tensor(float(value), device=loss.device),
                    on_epoch=True,
                    on_step=(prefix == "train" and name.startswith("loss")),
                    batch_size=batch_size,
                    sync_dist=True,
                )
            if prefix == "val":
                for name, value in self._placement_diagnostics(
                    enzyme_capability,
                    enzyme_family_vectors,
                    batch,
                ).items():
                    self.log(
                        f"val/placement/{name}",
                        value.to(device=loss.device) if torch.is_tensor(value) else torch.tensor(float(value), device=loss.device),
                        on_epoch=True,
                        on_step=False,
                        batch_size=batch_size,
                        sync_dist=True,
                    )
        return loss

    def _shared_step(self, batch: dict[str, Any], prefix: str) -> torch.Tensor:
        (
            enzyme_capability,
            reaction_capability,
            aux_logits,
            enzyme_family_vectors,
            reaction_family_vectors,
        ) = self(batch)
        if self.pretraining_objective in {"enzyme_bio_supcon", "enzyme_bio_composite_supcon"}:
            return self._bio_supcon_step(
                batch=batch,
                prefix=prefix,
                enzyme_capability=enzyme_capability,
                aux_logits=aux_logits,
                enzyme_family_vectors=enzyme_family_vectors,
            )
        reaction_ids = [str(value) for value in batch["reaction_id"]]
        enzyme_ids = [str(value) for value in batch["enzyme_id"]]
        (
            contrastive_reactions,
            contrastive_enzymes,
            r2e_mask,
            e2r_mask,
            contrastive_reaction_ids,
            contrastive_enzyme_ids,
        ) = self._contrastive_inputs(
            reaction_capability,
            enzyme_capability,
            reaction_ids,
            enzyme_ids,
            sync_grads=(prefix == "train"),
        )
        r2e_loss = multi_positive_contrastive_loss(
            contrastive_reactions,
            contrastive_enzymes,
            r2e_mask.to(device=contrastive_reactions.device),
            temperature=self.temperature,
        )
        e2r_loss = multi_positive_contrastive_loss(
            contrastive_enzymes,
            contrastive_reactions,
            e2r_mask.to(device=contrastive_enzymes.device),
            temperature=self.temperature,
        )
        family_losses: dict[str, torch.Tensor] = {}
        family_valid_anchors: dict[str, torch.Tensor] = {}
        zero_family = r2e_loss.new_zeros(())
        for family_name in ("cofactor", "transition", "reaction_center", "substrate", "product"):
            if not reaction_family_vectors or not enzyme_family_vectors:
                r2e_family = zero_family
                e2r_family = zero_family
                valid_anchors = zero_family
            else:
                r2e_family, e2r_family, valid_anchors = self._family_contrastive_loss(
                    family_name=family_name,
                    reaction_family_vectors=reaction_family_vectors,
                    enzyme_family_vectors=enzyme_family_vectors,
                    reaction_ids=reaction_ids,
                    enzyme_ids=enzyme_ids,
                    sync_grads=(prefix == "train"),
                )
            family_losses[family_name] = r2e_family + self.family_e2r_weight * e2r_family
            family_losses[f"{family_name}_r2e"] = r2e_family
            family_losses[f"{family_name}_e2r"] = e2r_family
            family_valid_anchors[family_name] = valid_anchors
        composite_result: dict[str, torch.Tensor] | None = None
        bio_factor_results: dict[str, dict[str, torch.Tensor]] = {}
        bio_factor_loss = r2e_loss.new_zeros(())
        selected_attr_loss = r2e_loss.new_zeros(())
        selected_attr_components: dict[str, torch.Tensor] = {}
        selected_missing_attr = r2e_loss.new_zeros(())
        diversity_loss = r2e_loss.new_zeros(())
        if self.pretraining_objective == "reaction_demand_bio_composite":
            composite_result = self._composite_biological_loss(
                embeddings=enzyme_capability,
                batch=batch,
                enzyme_ids=enzyme_ids,
                prefix=prefix,
            )
            bio_factor_embeddings = {
                "global": enzyme_family_vectors.get("global", enzyme_capability),
                "cofactor": enzyme_family_vectors.get("cofactor", enzyme_capability),
                "transition": enzyme_family_vectors.get("transition", enzyme_capability),
                "reaction_center": enzyme_family_vectors.get(
                    "reaction_center",
                    enzyme_capability,
                ),
                "substrate": enzyme_family_vectors.get("substrate", enzyme_capability),
                "product": enzyme_family_vectors.get("product", enzyme_capability),
            }
            bio_factor_results = {
                family_name: self._biological_factor_loss(
                    family_name=family_name,
                    embeddings=embedding,
                    batch=batch,
                    enzyme_ids=enzyme_ids,
                    prefix=prefix,
                )
                for family_name, embedding in bio_factor_embeddings.items()
            }
            bio_factor_loss = (
                self.weights["bio_global"] * bio_factor_results["global"]["loss"]
                + self.weights["bio_cofactor"] * bio_factor_results["cofactor"]["loss"]
                + self.weights["bio_transition"] * bio_factor_results["transition"]["loss"]
                + self.weights["bio_reaction_center"]
                * bio_factor_results["reaction_center"]["loss"]
                + self.weights["bio_substrate"] * bio_factor_results["substrate"]["loss"]
                + self.weights["bio_product"] * bio_factor_results["product"]["loss"]
            )
            selected_attr_loss, selected_attr_components, selected_missing_attr = (
                self._selected_attribute_loss(batch, aux_logits)
            )
            diversity_loss = self._factor_diversity_loss(enzyme_family_vectors).to(
                device=enzyme_capability.device
            )
        cofactor_loss = multilabel_bce_or_zero(
            aux_logits.get("cofactor_targets"),
            batch.get("cofactor_targets"),
            batch.get("cofactor_targets_mask"),
        )
        core_cofactor_loss = multilabel_bce_or_zero(
            aux_logits.get("core_cofactor_targets"),
            batch.get("core_cofactor_targets"),
            batch.get("core_cofactor_targets_mask"),
        )
        enzyme_derived_cofactor_loss = multilabel_bce_or_zero(
            aux_logits.get("enzyme_derived_cofactor_targets"),
            batch.get("enzyme_derived_cofactor_targets"),
            batch.get("enzyme_derived_cofactor_targets_mask"),
        )
        enzyme_derived_core_cofactor_loss = multilabel_bce_or_zero(
            aux_logits.get("enzyme_derived_core_cofactor_targets"),
            batch.get("enzyme_derived_core_cofactor_targets"),
            batch.get("enzyme_derived_core_cofactor_targets_mask"),
        )
        combined_cofactor_loss = multilabel_bce_or_zero(
            aux_logits.get("combined_cofactor_targets"),
            batch.get("combined_cofactor_targets"),
            batch.get("combined_cofactor_targets_mask"),
        )
        combined_core_cofactor_loss = multilabel_bce_or_zero(
            aux_logits.get("combined_core_cofactor_targets"),
            batch.get("combined_core_cofactor_targets"),
            batch.get("combined_core_cofactor_targets_mask"),
        )
        cofactor_architecture_loss = multilabel_bce_or_zero(
            aux_logits.get("cofactor_architecture_targets"),
            batch.get("cofactor_architecture_targets"),
            batch.get("cofactor_architecture_targets_mask"),
        )
        cofactor_chemistry_loss = multilabel_bce_or_zero(
            aux_logits.get("cofactor_chemistry_targets"),
            batch.get("cofactor_chemistry_targets"),
            batch.get("cofactor_chemistry_targets_mask"),
        )
        metal_ion_loss = multilabel_bce_or_zero(
            aux_logits.get("metal_ion_targets"),
            batch.get("metal_ion_targets"),
            batch.get("metal_ion_targets_mask"),
        )
        auxiliary_participant_loss = multilabel_bce_or_zero(
            aux_logits.get("auxiliary_participant_targets"),
            batch.get("auxiliary_participant_targets"),
            batch.get("auxiliary_participant_targets_mask"),
        )
        center_loss = multilabel_bce_or_zero(
            aux_logits.get("reaction_center_targets"),
            batch.get("reaction_center_targets"),
            batch.get("reaction_center_targets_mask"),
        )
        substrate_loss = multilabel_bce_or_zero(
            aux_logits.get("substrate_targets"),
            batch.get("substrate_targets"),
            batch.get("substrate_targets_mask"),
        )
        product_loss = multilabel_bce_or_zero(
            aux_logits.get("product_targets"),
            batch.get("product_targets"),
            batch.get("product_targets_mask"),
        )
        transition_loss = multilabel_bce_or_zero(
            aux_logits.get("substrate_product_transition_targets"),
            batch.get("substrate_product_transition_targets"),
            batch.get("substrate_product_transition_targets_mask"),
        )
        type_loss = multilabel_bce_or_zero(
            aux_logits.get("reaction_type_targets"),
            batch.get("reaction_type_targets"),
            batch.get("reaction_type_targets_mask"),
        )
        ec_loss = multilabel_bce_or_zero(
            aux_logits.get("ec_targets"),
            batch.get("ec_targets"),
            batch.get("ec_targets_mask"),
        )
        loss = (
            self.weights["r2e"] * r2e_loss
            + self.weights["e2r"] * e2r_loss
            + self.weights["family_cofactor"] * family_losses["cofactor"]
            + self.weights.get("family_transition", 0.0) * family_losses["transition"]
            + self.weights["family_reaction_center"] * family_losses["reaction_center"]
            + self.weights["family_substrate"] * family_losses["substrate"]
            + self.weights["family_product"] * family_losses["product"]
            + self.weights["cofactor"] * cofactor_loss
            + self.weights["core_cofactor"] * core_cofactor_loss
            + self.weights["enzyme_derived_cofactor"] * enzyme_derived_cofactor_loss
            + self.weights["enzyme_derived_core_cofactor"] * enzyme_derived_core_cofactor_loss
            + self.weights["combined_cofactor"] * combined_cofactor_loss
            + self.weights["combined_core_cofactor"] * combined_core_cofactor_loss
            + self.weights["cofactor_architecture"] * cofactor_architecture_loss
            + self.weights["cofactor_chemistry"] * cofactor_chemistry_loss
            + self.weights["metal_ion"] * metal_ion_loss
            + self.weights["auxiliary_participant"] * auxiliary_participant_loss
            + self.weights["reaction_center"] * center_loss
            + self.weights["substrate_product"] * (substrate_loss + product_loss)
            + self.weights["substrate_product_transition"] * transition_loss
            + self.weights["reaction_type"] * type_loss
            + self.weights["ec"] * ec_loss
        )
        if composite_result is not None:
            loss = (
                loss
                + self.weights["bio_composite"] * composite_result["loss"]
                + self.weights["same_center_hard_negative"]
                * composite_result["loss/hard_negative"]
                + bio_factor_loss
                + self.weights["bio_attribute"] * selected_attr_loss
                + self.weights["factor_diversity"] * diversity_loss
            )
        if hasattr(self, "log"):
            batch_size = len(reaction_ids)
            self.log(
                f"{prefix}/loss_total",
                loss,
                on_epoch=True,
                on_step=(prefix == "train"),
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                f"{prefix}/loss",
                loss,
                on_epoch=True,
                on_step=(prefix == "train"),
                batch_size=batch_size,
                sync_dist=True,
            )
            component_logs = {
                "loss_cap_r2e": r2e_loss,
                "loss_cap_e2r": e2r_loss,
                "loss_family_cofactor": family_losses["cofactor"],
                "loss_family_transition": family_losses["transition"],
                "loss_family_reaction_center": family_losses["reaction_center"],
                "loss_family_substrate": family_losses["substrate"],
                "loss_family_product": family_losses["product"],
                "loss_family_cofactor_r2e": family_losses["cofactor_r2e"],
                "loss_family_transition_r2e": family_losses["transition_r2e"],
                "loss_family_reaction_center_r2e": family_losses["reaction_center_r2e"],
                "loss_family_substrate_r2e": family_losses["substrate_r2e"],
                "loss_family_product_r2e": family_losses["product_r2e"],
                "loss_family_cofactor_e2r": family_losses["cofactor_e2r"],
                "loss_family_transition_e2r": family_losses["transition_e2r"],
                "loss_family_reaction_center_e2r": family_losses["reaction_center_e2r"],
                "loss_family_substrate_e2r": family_losses["substrate_e2r"],
                "loss_family_product_e2r": family_losses["product_e2r"],
                "loss_attr": (
                    cofactor_loss
                    + core_cofactor_loss
                    + enzyme_derived_cofactor_loss
                    + enzyme_derived_core_cofactor_loss
                    + combined_cofactor_loss
                    + combined_core_cofactor_loss
                    + cofactor_architecture_loss
                    + cofactor_chemistry_loss
                    + metal_ion_loss
                    + auxiliary_participant_loss
                    + center_loss
                    + substrate_loss
                    + product_loss
                    + transition_loss
                    + type_loss
                ),
                "loss_ec": ec_loss,
                "loss_bio_factor_supcon": bio_factor_loss,
                "loss_selected_attribute_bce": selected_attr_loss,
                "loss_factor_diversity": diversity_loss,
                "loss_enzyme_derived_cofactor": enzyme_derived_cofactor_loss,
                "loss_enzyme_derived_core_cofactor": enzyme_derived_core_cofactor_loss,
                "loss_combined_cofactor": combined_cofactor_loss,
                "loss_combined_core_cofactor": combined_core_cofactor_loss,
                "loss_cofactor_architecture": cofactor_architecture_loss,
                "loss_cofactor_chemistry": cofactor_chemistry_loss,
                "num_valid_reactions": torch.tensor(
                    float(len(contrastive_reaction_ids)),
                    device=loss.device,
                ),
                "num_valid_enzymes": torch.tensor(
                    float(len(contrastive_enzyme_ids)),
                    device=loss.device,
                ),
                "mean_positives_per_reaction": r2e_mask.sum(dim=1).float().mean().to(loss.device),
                "mean_reactions_per_enzyme": e2r_mask.sum(dim=1).float().mean().to(loss.device),
                "family_cofactor_valid_reactions": family_valid_anchors["cofactor"].to(loss.device),
                "family_transition_valid_reactions": family_valid_anchors["transition"].to(loss.device),
                "family_reaction_center_valid_reactions": family_valid_anchors[
                    "reaction_center"
                ].to(loss.device),
                "family_substrate_valid_reactions": family_valid_anchors["substrate"].to(loss.device),
                "family_product_valid_reactions": family_valid_anchors["product"].to(loss.device),
                "num_masked_missing_attribute_labels": sum(
                    int((~batch.get(f"{key}_mask", torch.ones(batch_size, dtype=torch.bool))).sum().item())
                    for key in (
                        "cofactor_targets",
                        "core_cofactor_targets",
                        "enzyme_derived_cofactor_targets",
                        "enzyme_derived_core_cofactor_targets",
                        "combined_cofactor_targets",
                        "combined_core_cofactor_targets",
                        "cofactor_architecture_targets",
                        "cofactor_chemistry_targets",
                        "metal_ion_targets",
                        "auxiliary_participant_targets",
                        "reaction_center_targets",
                        "substrate_targets",
                        "product_targets",
                        "substrate_product_transition_targets",
                        "reaction_type_targets",
                        "ec_targets",
                    )
                    if torch.is_tensor(batch.get(f"{key}_mask"))
                ),
            }
            if composite_result is not None:
                component_logs.update(
                    {
                        "loss_composite_supcon": composite_result["loss"],
                        "loss_same_center_hard_negative": composite_result[
                            "loss/hard_negative"
                        ],
                        "composite_valid_anchors": composite_result[
                            "metrics/valid_anchors"
                        ].to(loss.device),
                        "composite_mean_positives": composite_result[
                            "metrics/mean_positives"
                        ].to(loss.device),
                        "composite_mean_positive_score": composite_result[
                            "metrics/mean_positive_score"
                        ].to(loss.device),
                        "composite_mean_known_families": composite_result[
                            "metrics/mean_known_families"
                        ].to(loss.device),
                        "composite_mean_hard_negatives": composite_result[
                            "metrics/mean_hard_negatives"
                        ].to(loss.device),
                        "num_masked_missing_selected_attribute_labels": selected_missing_attr,
                    }
                )
                for family_name, result in bio_factor_results.items():
                    component_logs[f"bio_{family_name}_valid_anchors"] = result[
                        "metrics/valid_anchors"
                    ].to(loss.device)
                    component_logs[f"bio_{family_name}_mean_positives"] = result[
                        "metrics/mean_positives"
                    ].to(loss.device)
                    component_logs[f"bio_{family_name}_mean_jaccard"] = result[
                        "metrics/mean_jaccard"
                    ].to(loss.device)
                for family_name, value in selected_attr_components.items():
                    component_logs[f"loss_selected_attr_{family_name}"] = value
            for name, value in component_logs.items():
                if not torch.is_tensor(value):
                    value = torch.tensor(float(value), device=loss.device)
                self.log(
                    f"{prefix}/{name}",
                    value,
                    on_epoch=True,
                    on_step=(prefix == "train" and name.startswith("loss")),
                    batch_size=batch_size,
                    sync_dist=True,
                )
            if prefix == "val":
                scores = contrastive_reactions @ contrastive_enzymes.T
                diagnostics = {
                    "capability_space/r2e_mrr": self._mrr_from_scores(
                        scores,
                        r2e_mask,
                    ),
                    "capability_space/e2r_mrr": self._mrr_from_scores(
                        scores.T,
                        e2r_mask,
                    ),
                    "capability_space/r2e_recall@10": self._recall_at_k_from_scores(
                        scores,
                        r2e_mask,
                        10,
                    ),
                    "capability_space/r2e_recall@100": self._recall_at_k_from_scores(
                        scores,
                        r2e_mask,
                        100,
                    ),
                    "capability_space/r2e_recall@500": self._recall_at_k_from_scores(
                        scores,
                        r2e_mask,
                        500,
                    ),
                }
                for name, value in diagnostics.items():
                    self.log(
                        name,
                        value,
                        on_epoch=True,
                        on_step=False,
                        batch_size=batch_size,
                        sync_dist=True,
                    )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
