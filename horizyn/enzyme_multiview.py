"""Sequence-only multi-view enzyme encoder with bounded residue attention.

The learned slots are latent views, not claims about biological sites or labels.
SLEEC supplies one separate view; it does not bias every learned slot alike.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from horizyn.config_validation import validate_enzyme_multiview_config
from horizyn.positive_bio import PositiveBiologicalReadout, validate_positive_biological_labels


class EnzymeMultiviewEncoder(nn.Module):
    """Fuse global, SLEEC-site, and learned residue views into one embedding.

    All attention is O(length * num_slots), never quadratic in sequence length.
    A global residual and small uniform mixtures reduce single-view starvation;
    differentiable safeguards are returned for the trainer to weight explicitly.
    """

    def __init__(
        self,
        residue_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_slots: int = 4,
        dropout: float = 0.1,
        uniform_mix: float = 0.05,
        min_effective_residues: float = 4.0,
        diversity_margin: float = 0.9,
        gate_floor: float = 0.05,
        max_logit_scale: float = 10.0,
        initial_residual_scale: float = 0.1,
        biological_labels: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        settings = validate_enzyme_multiview_config({
            "hidden_dim": hidden_dim, "num_slots": num_slots, "dropout": dropout,
            "uniform_mix": uniform_mix, "min_effective_residues": min_effective_residues,
            "diversity_margin": diversity_margin, "gate_floor": gate_floor,
            "max_logit_scale": max_logit_scale,
            "initial_residual_scale": initial_residual_scale,
        })
        for name, value in (("residue_dim", residue_dim), ("output_dim", output_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.residue_dim = residue_dim
        self.output_dim = output_dim
        for name, value in settings.items():
            setattr(self, name, value)
        self.view_names = ("global", "sleec", *(f"slot_{i}" for i in range(num_slots)))
        # K=0 is the trained global + SLEEC control: learned-residue modules
        # are absent, rather than present but suppressed only at inference.
        self.residue_adapter = nn.Sequential(
            nn.LayerNorm(residue_dim), nn.Linear(residue_dim, hidden_dim), nn.GELU(),
        ) if num_slots else None
        self.keys = nn.Linear(hidden_dim, hidden_dim, bias=False) if num_slots else None
        self.values = nn.Linear(hidden_dim, hidden_dim) if num_slots else None
        self.queries = nn.Parameter(torch.empty(num_slots, hidden_dim), requires_grad=bool(num_slots))
        if num_slots:
            nn.init.orthogonal_(self.queries)
        if num_slots == 1:
            # A singleton query can receive an equivalent (1, 1)-stride
            # gradient while its parameter has stride (hidden_dim, 1).
            # Fused AdamW requires identical strides even for singleton axes.
            # Copy storage only: gradient values and optimizer math are unchanged.
            self.queries.register_post_accumulate_grad_hook(self._match_query_gradient_layout)
        # The score scale is always in (0, max_logit_scale), including after updates.
        scale_fraction = min(4.0 / max_logit_scale, 0.5)
        self.raw_logit_scale = nn.Parameter(
            torch.tensor(math.log(scale_fraction / (1.0 - scale_fraction))), requires_grad=bool(num_slots))
        self.view_projections = nn.ModuleList([
            self._ffn(residue_dim if i < 2 else hidden_dim, hidden_dim, hidden_dim * 2, dropout)
            for i in range(num_slots + 2)
        ])
        self.gate = nn.Sequential(
            nn.Linear((num_slots + 2) * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, (num_slots + 2) * hidden_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.global_output = self._ffn(residue_dim, output_dim, hidden_dim * 2, dropout)
        self.fused_output = self._ffn(hidden_dim, output_dim, hidden_dim * 2, dropout)
        # Start with a modest, nonzero update; no normalize(zero) gradient singularity.
        residual_fraction = initial_residual_scale / 0.5
        self.raw_residual_scale = nn.Parameter(torch.tensor(math.log(residual_fraction / (1.0 - residual_fraction))))
        self.biological_labels = validate_positive_biological_labels(biological_labels)
        if not num_slots and self.biological_labels:
            raise ValueError("Biological readouts require at least one learned residue view")
        # Build after retrieval layers so adding supervision does not change
        # the seed-matched initial retrieval representation.
        with torch.random.fork_rng(devices=[]):
            self.biological_readouts = nn.ModuleDict({
                family: PositiveBiologicalReadout(hidden_dim, family, labels, gate_floor)
                for family, labels in self.biological_labels.items()
            })

    @staticmethod
    def _match_query_gradient_layout(parameter: torch.Tensor) -> None:
        gradient = parameter.grad
        if gradient is not None and gradient.stride() != parameter.stride():
            parameter.grad = torch.empty_strided(
                parameter.shape, parameter.stride(),
                dtype=gradient.dtype, device=gradient.device,
            ).copy_(gradient)

    @staticmethod
    def _ffn(input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
        )

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.max_logit_scale * self.raw_logit_scale.sigmoid()

    @property
    def residual_scale(self) -> torch.Tensor:
        return 0.5 * self.raw_residual_scale.sigmoid()

    def _attention_safeguards(
        self, weights: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-protein penalties, entropy and centered head similarity."""
        weights = weights.float()
        if not self.num_slots:
            zero = weights.new_zeros(weights.shape[0])
            return zero, zero, weights.new_empty((weights.shape[0], 0)), zero
        lengths = valid.sum(-1).float()
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
        # A fractional length cap avoids forcing every short protein to use
        # uniform attention while the diversity safeguard asks heads to differ.
        floor = torch.minimum(
            0.5 * lengths.log(), lengths.new_tensor(self.min_effective_residues).log(),
        )
        entropy_loss = F.relu(floor[:, None] - entropy).square().mean(-1)
        uniform = valid.to(weights.dtype) / lengths[:, None]
        centered = (weights - uniform[:, None]) * valid[:, None]
        norms = centered.norm(dim=-1)
        unit = F.normalize(centered, dim=-1, eps=1e-6)
        similarity = torch.einsum("bkl,bjl->bkj", unit, unit)
        pair_mask = torch.triu(torch.ones(
            self.num_slots, self.num_slots, device=weights.device, dtype=torch.bool,
        ), diagonal=1)[None]
        # Uniform heads have no meaningful centered direction. One-residue
        # proteins have no alternative residue to attend to.
        eligible = pair_mask & (norms[:, :, None] > 1e-6) & (norms[:, None, :] > 1e-6)
        eligible = eligible & (lengths[:, None, None] > 1)
        count = eligible.sum((1, 2)).clamp_min(1)
        diversity_loss = (
            F.relu(similarity - self.diversity_margin).square() * eligible
        ).sum((1, 2)) / count
        mean_similarity = (similarity * eligible).sum((1, 2)) / count
        return entropy_loss, diversity_loss, entropy, mean_similarity

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor | None = None,
        sleec_prior: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if residue_embeddings.ndim != 3 or residue_embeddings.shape[-1] != self.residue_dim:
            raise ValueError("residue_embeddings must have shape [batch, length, residue_dim]")
        batch, length, _ = residue_embeddings.shape
        if residue_padding_mask is None:
            valid = torch.ones((batch, length), device=residue_embeddings.device, dtype=torch.bool)
        else:
            if residue_padding_mask.shape != (batch, length):
                raise ValueError("residue_padding_mask must match residue batch and length")
            valid = ~residue_padding_mask.to(device=residue_embeddings.device, dtype=torch.bool)
        if length == 0 or not valid.any(-1).all():
            raise ValueError("Each protein must have at least one valid residue")
        clean = residue_embeddings.float().masked_fill(~valid[..., None], 0.0)
        lengths = valid.sum(-1).float()
        uniform = valid.to(clean.dtype) / lengths[:, None]
        raw_mean = (clean * uniform[..., None]).sum(1)
        if sleec_prior is None:
            prior = clean.new_zeros((batch, length))
        else:
            if sleec_prior.shape != (batch, length):
                raise ValueError("sleec_prior must match residue batch and length")
            prior = sleec_prior.detach().to(device=clean.device, dtype=clean.dtype).masked_fill(~valid, 0.0)
        centered_prior = prior - (prior * uniform).sum(-1, keepdim=True)
        site_logits = centered_prior.clamp(-10.0, 10.0).masked_fill(~valid, -torch.inf)
        site_raw = site_logits.softmax(-1)
        site_weights = (1.0 - self.uniform_mix) * site_raw + self.uniform_mix * uniform
        site = (clean * site_weights[..., None]).sum(1)

        if self.num_slots:
            adapted = self.residue_adapter(clean)
            keys = F.normalize(self.keys(adapted).float(), dim=-1, eps=1e-6)
            queries = F.normalize(self.queries.float(), dim=-1, eps=1e-6)
            scores = self.logit_scale * torch.einsum("bld,kd->bkl", keys, queries)
            raw_weights = scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
            slot_weights = (1.0 - self.uniform_mix) * raw_weights + self.uniform_mix * uniform[:, None]
            slot_values = self.values(adapted)
            slots = torch.einsum("bkl,bld->bkd", slot_weights.to(slot_values.dtype), slot_values)
        else:
            raw_weights = slot_weights = clean.new_empty((batch, 0, length))
            slots = clean.new_empty((batch, 0, self.hidden_dim))
        inputs = [raw_mean, site, *slots.unbind(1)]
        projected_views = [
            F.normalize(projection(value), dim=-1, eps=1e-6)
            for projection, value in zip(self.view_projections, inputs)
        ]
        # This exact tensor feeds retrieval and biology, making gradient
        # diagnostics measure the shared representation, not an unused slice.
        shared_slots = (torch.stack(projected_views[2:], dim=1) if self.num_slots
                        else projected_views[0].new_empty((batch, 0, self.hidden_dim)))
        views = torch.cat((torch.stack(projected_views[:2], dim=1), shared_slots), dim=1)
        gate_logits = self.gate(views.flatten(1)).reshape(batch, len(self.view_names), self.hidden_dim)
        raw_gate = gate_logits.float().softmax(1)
        gate_weights = (1.0 - self.gate_floor) * raw_gate + self.gate_floor / len(self.view_names)
        fused = (views * gate_weights.to(views.dtype)).sum(1)
        global_output = F.normalize(self.global_output(raw_mean).float(), dim=-1, eps=1e-6)
        fused_output = F.normalize(self.fused_output(fused).float(), dim=-1, eps=1e-6)
        output = F.normalize(global_output + self.residual_scale * fused_output, dim=-1, eps=1e-6)
        if not return_details:
            return output

        entropy_loss, diversity_loss, entropy, head_similarity = self._attention_safeguards(raw_weights, valid)
        effective_entropy = -(slot_weights * slot_weights.clamp_min(1e-12).log()).sum(-1)
        entropy_normalizer = lengths.clamp_min(2.0).log()[:, None]
        pair_mask = torch.triu(torch.ones(
            self.num_slots, self.num_slots, device=clean.device, dtype=torch.bool,
        ), diagonal=1)[None] & (lengths[:, None, None] > 1)
        pair_count = pair_mask.sum((1, 2)).clamp_min(1)
        raw_unit = F.normalize(raw_weights, dim=-1, eps=1e-6)
        raw_similarity = torch.einsum("bkl,bjl->bkj", raw_unit, raw_unit)
        centered_norms = (raw_weights - uniform[:, None]).norm(dim=-1)
        centered_pairs = pair_mask & (centered_norms[:, :, None] > 1e-6) & (centered_norms[:, None, :] > 1e-6)
        sleec_scores = prior.sigmoid().masked_fill(~valid, 0.0)
        positives = (sleec_scores > 0.5) & valid
        # Absent learned-view diagnostics are explicitly zero for K=0;
        # averaging an empty view axis would otherwise introduce NaNs.
        def slot_mean(value):
            return value.mean(-1) if self.num_slots else clean.new_zeros(batch)

        details = {
            "weights": site_weights,
            "logits": prior,
            "scores": sleec_scores,
            "pooling_entropy": -(site_weights * site_weights.clamp_min(1e-12).log()).sum(-1),
            "attention_mass_sleec_positive": (site_weights * positives).sum(-1),
            "num_sleec_positive": positives.sum(-1).float(),
            "enzyme_multiview_entropy_loss": entropy_loss,
            "enzyme_multiview_diversity_loss": diversity_loss,
            "enzyme_multiview_entropy": slot_mean(entropy),
            "enzyme_multiview_normalized_entropy": slot_mean(entropy / entropy_normalizer),
            "enzyme_multiview_effective_residues": slot_mean(entropy.exp()),
            "enzyme_multiview_max_weight": slot_mean(raw_weights.amax(-1)),
            "enzyme_multiview_effective_entropy": slot_mean(effective_entropy),
            "enzyme_multiview_effective_normalized_entropy": slot_mean(effective_entropy / entropy_normalizer),
            "enzyme_multiview_effective_support": slot_mean(effective_entropy.exp()),
            "enzyme_multiview_effective_max_weight": slot_mean(slot_weights.amax(-1)),
            "enzyme_multiview_head_similarity": head_similarity,
            "enzyme_multiview_head_similarity_raw": (raw_similarity * pair_mask).sum((1, 2)) / pair_count,
            "enzyme_multiview_head_similarity_eligible_fraction": centered_pairs.sum((1, 2)).float() / pair_count,
            "enzyme_multiview_logit_scale": self.logit_scale.expand(batch) if self.num_slots else clean.new_zeros(batch),
            "enzyme_multiview_residual_scale": self.residual_scale.expand(batch),
            "enzyme_multiview_gate_entropy": -(gate_weights * gate_weights.clamp_min(1e-12).log()).sum(1).mean(-1),
            "enzyme_multiview_raw_attention": raw_weights,
            "enzyme_multiview_attention": slot_weights,
            "enzyme_multiview_gate_weights": gate_weights,
            "enzyme_multiview_global_cosine": (output * global_output).sum(-1),
        }
        for index, name in enumerate(self.view_names):
            details[f"enzyme_multiview_gate_{name}"] = gate_weights[:, index].mean(-1)
        for family, readout in self.biological_readouts.items():
            distance, embedding, readout_weights = readout(shared_slots)
            details[f"biofp_alignment_{family}"] = distance
            details[f"biofp_shared_{family}"] = shared_slots
            details[f"biofp_embedding_{family}"] = embedding
            details[f"biofp_readout_weights_{family}"] = readout_weights
            prefix = f"enzyme_multiview_bio_{family}"
            # Local-batch diagnostics, not a claim about global DDP variance.
            details[f"{prefix}_feature_variance"] = embedding.var(dim=0, unbiased=False).mean().expand(batch)
            details[f"{prefix}_readout_gate_entropy"] = -(readout_weights * readout_weights.clamp_min(1e-12).log()).sum(-1)
            for index in range(self.num_slots):
                details[f"{prefix}_readout_gate_slot_{index}"] = readout_weights[:, index]
        return output, details
