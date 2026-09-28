"""Capability pretraining and capability-aware retrieval losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def positive_mask_from_ids(anchor_ids: list[str], positive_ids: list[str]) -> torch.Tensor:
    mask = torch.zeros(len(anchor_ids), len(positive_ids), dtype=torch.bool)
    lookup: dict[str, list[int]] = {}
    for idx, item_id in enumerate(positive_ids):
        lookup.setdefault(item_id, []).append(idx)
    for row_idx, anchor_id in enumerate(anchor_ids):
        for col_idx in lookup.get(anchor_id, []):
            mask[row_idx, col_idx] = True
    return mask


def positive_masks_from_pair_maps(
    reaction_ids: list[str],
    enzyme_ids: list[str],
    reaction_to_enzymes: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    enzyme_to_reactions: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build R->E and E->R positive masks from all-known train pair maps."""

    r2e = torch.zeros(len(reaction_ids), len(enzyme_ids), dtype=torch.bool)
    e2r = torch.zeros(len(enzyme_ids), len(reaction_ids), dtype=torch.bool)
    reaction_to_enzymes = reaction_to_enzymes or {}
    enzyme_to_reactions = enzyme_to_reactions or {}
    for row, reaction_id in enumerate(reaction_ids):
        positives = set(str(value) for value in reaction_to_enzymes.get(reaction_id, ()))
        if not positives:
            continue
        for col, enzyme_id in enumerate(enzyme_ids):
            if enzyme_id in positives:
                r2e[row, col] = True
    for row, enzyme_id in enumerate(enzyme_ids):
        positives = set(str(value) for value in enzyme_to_reactions.get(enzyme_id, ()))
        if not positives:
            continue
        for col, reaction_id in enumerate(reaction_ids):
            if reaction_id in positives:
                e2r[row, col] = True
    return r2e, e2r


def multi_positive_contrastive_loss(
    anchor_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.07,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if anchor_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError("Embeddings must be rank-2")
    if positive_mask.shape != (anchor_embeddings.shape[0], candidate_embeddings.shape[0]):
        raise ValueError(
            "positive_mask shape must be [num_anchors, num_candidates], got "
            f"{tuple(positive_mask.shape)}"
        )
    positive_mask = positive_mask.to(device=anchor_embeddings.device, dtype=torch.bool)
    if candidate_mask is not None:
        if candidate_mask.shape != (candidate_embeddings.shape[0],):
            raise ValueError(
                "candidate_mask shape must be [num_candidates], got "
                f"{tuple(candidate_mask.shape)}"
            )
        candidate_mask = candidate_mask.to(device=anchor_embeddings.device, dtype=torch.bool)
        positive_mask = positive_mask & candidate_mask.unsqueeze(0)
    valid_rows = positive_mask.any(dim=1)
    if not bool(valid_rows.any()):
        return (anchor_embeddings.sum() + candidate_embeddings.sum()) * 0.0
    logits = anchor_embeddings @ candidate_embeddings.T / float(temperature)
    if candidate_mask is not None:
        logits = logits.masked_fill(~candidate_mask.unsqueeze(0), torch.finfo(logits.dtype).min)
    logits = logits[valid_rows]
    mask = positive_mask[valid_rows]
    log_den = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    log_num = torch.logsumexp(positive_logits, dim=1)
    return -(log_num - log_den).mean()


class BiologicalFactorSupConLoss(nn.Module):
    """Weighted enzyme-enzyme supervised contrastive loss for one label family.

    Labels are multi-hot biological annotations. Samples with missing or empty
    labels are masked out, not treated as all-zero negatives. Positive weights
    are Jaccard overlaps between non-empty label sets.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor | None = None,
        label_weights: torch.Tensor | None = None,
        positive_threshold: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must have shape [batch, dim], got {tuple(embeddings.shape)}")
        if labels.ndim != 2:
            raise ValueError(f"labels must have shape [batch, labels], got {tuple(labels.shape)}")
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "labels first dimension must match embeddings batch size, got "
                f"{tuple(labels.shape)} and {tuple(embeddings.shape)}"
            )
        device = embeddings.device
        zero = embeddings.new_zeros(())
        if labels.shape[1] == 0 or embeddings.shape[0] <= 1:
            loss = embeddings.sum() * 0.0
            return {
                "loss": loss,
                "metrics/valid_anchors": zero.detach(),
                "metrics/known_samples": zero.detach(),
                "metrics/mean_positives": zero.detach(),
                "metrics/mean_jaccard": zero.detach(),
                "metrics/masked_missing": torch.tensor(float(embeddings.shape[0]), device=device),
            }

        labels = (labels.to(device=device, dtype=embeddings.dtype) > 0).to(embeddings.dtype)
        if label_weights is None:
            label_weights = torch.ones(labels.shape[1], device=device, dtype=embeddings.dtype)
        else:
            if label_weights.shape != (labels.shape[1],):
                raise ValueError(
                    "label_weights must have shape [num_labels], got "
                    f"{tuple(label_weights.shape)} for labels {tuple(labels.shape)}"
                )
            label_weights = label_weights.to(device=device, dtype=embeddings.dtype)
        known = labels.sum(dim=1) > 0
        if label_mask is not None:
            label_mask = label_mask.to(device=device, dtype=torch.bool)
            if label_mask.ndim > 1:
                label_mask = label_mask.reshape(label_mask.shape[0], -1).any(dim=1)
            if label_mask.shape[0] != embeddings.shape[0]:
                raise ValueError(
                    "label_mask first dimension must match embeddings batch size, got "
                    f"{tuple(label_mask.shape)} and {tuple(embeddings.shape)}"
                )
            known = known & label_mask

        n = embeddings.shape[0]
        known_pair = known.unsqueeze(0) & known.unsqueeze(1)
        not_self = ~torch.eye(n, dtype=torch.bool, device=device)
        candidate_mask = known_pair & not_self
        label_counts = (labels * label_weights.unsqueeze(0)).sum(dim=1)
        overlap = (labels * label_weights.unsqueeze(0)) @ labels.T
        union = label_counts.unsqueeze(1) + label_counts.unsqueeze(0) - overlap
        jaccard = torch.where(
            union > 0,
            overlap / union.clamp_min(1.0),
            torch.zeros_like(overlap),
        )
        positive_weights = jaccard.masked_fill(~candidate_mask, 0.0)
        if positive_threshold > 0:
            positive_weights = positive_weights.masked_fill(
                positive_weights < float(positive_threshold),
                0.0,
            )
        valid_anchors = known & positive_weights.gt(0).any(dim=1)
        known_count = known.sum().to(dtype=embeddings.dtype)
        masked_missing = (~known).sum().to(dtype=embeddings.dtype)
        if not bool(valid_anchors.any()):
            loss = embeddings.sum() * 0.0
            return {
                "loss": loss,
                "metrics/valid_anchors": zero.detach(),
                "metrics/known_samples": known_count.detach(),
                "metrics/mean_positives": zero.detach(),
                "metrics/mean_jaccard": zero.detach(),
                "metrics/masked_missing": masked_missing.detach(),
            }

        logits = embeddings @ embeddings.T / self.temperature
        min_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~candidate_mask, min_value)
        eps = torch.finfo(logits.dtype).eps
        positive_logits = logits + positive_weights.clamp_min(eps).log()
        positive_logits = positive_logits.masked_fill(positive_weights <= 0, min_value)

        log_den = torch.logsumexp(logits[valid_anchors], dim=1)
        log_num = torch.logsumexp(positive_logits[valid_anchors], dim=1)
        loss = -(log_num - log_den).mean()
        positive_counts = positive_weights.gt(0).sum(dim=1).to(dtype=embeddings.dtype)
        positive_values = positive_weights[positive_weights > 0]
        mean_jaccard = positive_values.mean() if positive_values.numel() else zero
        return {
            "loss": loss,
            "metrics/valid_anchors": valid_anchors.sum().to(dtype=embeddings.dtype).detach(),
            "metrics/known_samples": known_count.detach(),
            "metrics/mean_positives": positive_counts[valid_anchors].mean().detach(),
            "metrics/mean_jaccard": mean_jaccard.detach(),
            "metrics/masked_missing": masked_missing.detach(),
        }


class CompositeBiologicalSupConLoss(nn.Module):
    """Composite biological signature contrastive loss.

    Positives are formed by weighted overlap across multiple biological families,
    not by reaction-center alone. Same-center but context-mismatched pairs can be
    emphasized as hard negatives.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_temperature: float = 0.07,
        hard_negative_margin: float = 0.10,
        positive_threshold: float = 0.20,
        min_positive_families: int = 2,
        min_known_families: int = 2,
        family_weights: dict[str, float] | None = None,
        positive_family_names: tuple[str, ...] | None = None,
        known_family_names: tuple[str, ...] | None = None,
        required_positive_families: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.hard_negative_temperature = float(hard_negative_temperature)
        self.hard_negative_margin = float(hard_negative_margin)
        self.positive_threshold = float(positive_threshold)
        self.min_positive_families = int(min_positive_families)
        self.min_known_families = int(min_known_families)
        self.family_weights = {
            "cofactor": 0.35,
            "reaction_center": 0.25,
            "substrate": 0.20,
            "product": 0.20,
            **(family_weights or {}),
        }
        if positive_family_names is None:
            positive_family_names = tuple(
                name for name, weight in self.family_weights.items() if float(weight) > 0
            )
        if known_family_names is None:
            known_family_names = positive_family_names
        self.positive_family_names = tuple(str(name) for name in positive_family_names)
        self.known_family_names = tuple(str(name) for name in known_family_names)
        self.required_positive_families = tuple(
            str(name) for name in (required_positive_families or ())
        )

    @staticmethod
    def _empty_result(embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = embeddings.sum() * 0.0
        return {
            "loss": zero,
            "loss/hard_negative": zero,
            "metrics/valid_anchors": zero.detach(),
            "metrics/mean_positives": zero.detach(),
            "metrics/mean_positive_score": zero.detach(),
            "metrics/mean_known_families": zero.detach(),
            "metrics/mean_hard_negatives": zero.detach(),
        }

    def _family_jaccard(
        self,
        labels: torch.Tensor,
        mask: torch.Tensor | None,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = labels.device
        labels = (labels > 0).to(dtype=labels.dtype)
        known = labels.sum(dim=1) > 0
        if mask is not None:
            mask = mask.to(device=device, dtype=torch.bool)
            if mask.ndim > 1:
                mask = mask.reshape(mask.shape[0], -1).any(dim=1)
            known = known & mask
        if labels.shape[1] == 0:
            n = labels.shape[0]
            return (
                torch.zeros(n, n, device=device, dtype=labels.dtype),
                torch.zeros(n, n, device=device, dtype=torch.bool),
            )
        if weights is None:
            df = labels.sum(dim=0)
            weights = torch.log((known.sum().to(labels.dtype) + 1.0) / (df + 1.0)).clamp_min(0.05)
        weights = weights.to(device=device, dtype=labels.dtype)
        weighted = labels * weights.unsqueeze(0)
        counts = weighted.sum(dim=1)
        overlap = weighted @ labels.T
        union = counts.unsqueeze(1) + counts.unsqueeze(0) - overlap
        jaccard = torch.where(
            union > 0,
            overlap / union.clamp_min(1.0),
            torch.zeros_like(overlap),
        )
        known_pair = known.unsqueeze(0) & known.unsqueeze(1)
        return jaccard.masked_fill(~known_pair, 0.0), known_pair

    def forward(
        self,
        embeddings: torch.Tensor,
        labels_by_family: dict[str, torch.Tensor],
        masks_by_family: dict[str, torch.Tensor | None] | None = None,
        label_weights_by_family: dict[str, torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must have shape [batch, dim], got {tuple(embeddings.shape)}")
        if embeddings.shape[0] <= 1:
            return self._empty_result(embeddings)
        masks_by_family = masks_by_family or {}
        label_weights_by_family = label_weights_by_family or {}
        device = embeddings.device
        dtype = embeddings.dtype
        n = embeddings.shape[0]
        not_self = ~torch.eye(n, dtype=torch.bool, device=device)

        composite = embeddings.new_zeros(n, n)
        available_weight = embeddings.new_zeros(n, n)
        positive_family_count = torch.zeros(n, n, dtype=torch.long, device=device)
        known_family_count = torch.zeros(n, n, dtype=torch.long, device=device)
        required_positive_any = torch.zeros(n, n, dtype=torch.bool, device=device)
        family_jaccards: dict[str, torch.Tensor] = {}
        family_known_pairs: dict[str, torch.Tensor] = {}

        for family_name, family_weight in self.family_weights.items():
            labels = labels_by_family.get(family_name)
            if labels is None:
                continue
            labels = labels.to(device=device, dtype=dtype)
            jaccard, known_pair = self._family_jaccard(
                labels,
                masks_by_family.get(family_name),
                label_weights_by_family.get(family_name),
            )
            family_jaccards[family_name] = jaccard
            family_known_pairs[family_name] = known_pair
            weight = float(family_weight)
            composite = composite + weight * jaccard
            available_weight = available_weight + weight * known_pair.to(dtype=dtype)
            if family_name in self.positive_family_names:
                positive_family_count = positive_family_count + (jaccard > 0).to(torch.long)
            if family_name in self.known_family_names:
                known_family_count = known_family_count + known_pair.to(torch.long)
            if family_name in self.required_positive_families:
                required_positive_any = required_positive_any | (jaccard > 0)

        if not family_jaccards:
            return self._empty_result(embeddings)
        composite = torch.where(
            available_weight > 0,
            composite / available_weight.clamp_min(torch.finfo(dtype).eps),
            torch.zeros_like(composite),
        )
        candidate_mask = (
            (known_family_count >= self.min_known_families)
            & not_self
        )
        positive_mask = (
            candidate_mask
            & (positive_family_count >= self.min_positive_families)
            & (composite >= self.positive_threshold)
        )
        if self.required_positive_families:
            positive_mask = positive_mask & required_positive_any
        positive_weights = composite.masked_fill(~positive_mask, 0.0)
        valid_anchors = positive_mask.any(dim=1)
        if not bool(valid_anchors.any()):
            result = self._empty_result(embeddings)
            result["metrics/mean_known_families"] = known_family_count[not_self].to(dtype).mean().detach()
            return result

        logits = embeddings @ embeddings.T / self.temperature
        min_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~candidate_mask, min_value)
        eps = torch.finfo(logits.dtype).eps
        positive_logits = logits + positive_weights.clamp_min(eps).log()
        positive_logits = positive_logits.masked_fill(~positive_mask, min_value)
        log_den = torch.logsumexp(logits[valid_anchors], dim=1)
        log_num = torch.logsumexp(positive_logits[valid_anchors], dim=1)
        contrastive_loss = -(log_num - log_den).mean()

        center = family_jaccards.get("reaction_center", embeddings.new_zeros(n, n))
        context = embeddings.new_zeros(n, n)
        context_known = torch.zeros(n, n, dtype=torch.bool, device=device)
        for family_name in ("cofactor", "transition", "substrate", "product"):
            if family_name in family_jaccards:
                context = context + family_jaccards[family_name]
                context_known = context_known | family_known_pairs[family_name]
        hard_negative_mask = (
            not_self
            & (center > 0)
            & context_known
            & (context <= 0)
            & ~positive_mask
        )
        hard_valid = valid_anchors & hard_negative_mask.any(dim=1)
        hard_loss = embeddings.new_zeros(())
        if bool(hard_valid.any()):
            raw_scores = embeddings @ embeddings.T
            pos_scores = raw_scores / self.hard_negative_temperature
            weighted_pos = pos_scores + positive_weights.clamp_min(eps).log()
            weighted_pos = weighted_pos.masked_fill(~positive_mask, min_value)
            positive_reference = torch.logsumexp(weighted_pos[hard_valid], dim=1) * self.hard_negative_temperature
            hard_scores = raw_scores[hard_valid].masked_fill(
                ~hard_negative_mask[hard_valid],
                min_value,
            )
            hard_terms = torch.logsumexp(
                (
                    hard_scores
                    - positive_reference.unsqueeze(1)
                    + self.hard_negative_margin
                )
                / self.hard_negative_temperature,
                dim=1,
            )
            hard_loss = F.softplus(hard_terms).mean()

        positive_counts = positive_mask.sum(dim=1).to(dtype=dtype)
        hard_counts = hard_negative_mask.sum(dim=1).to(dtype=dtype)
        positive_values = positive_weights[positive_weights > 0]
        known_family_values = known_family_count[not_self].to(dtype=dtype)
        return {
            "loss": contrastive_loss,
            "loss/hard_negative": hard_loss,
            "metrics/valid_anchors": valid_anchors.sum().to(dtype=dtype).detach(),
            "metrics/mean_positives": positive_counts[valid_anchors].mean().detach(),
            "metrics/mean_positive_score": (
                positive_values.mean().detach() if positive_values.numel() else embeddings.new_zeros(()).detach()
            ),
            "metrics/mean_known_families": known_family_values.mean().detach(),
            "metrics/mean_hard_negatives": hard_counts[hard_valid].mean().detach()
            if bool(hard_valid.any())
            else embeddings.new_zeros(()).detach(),
        }


def pair_id_positive_masks(
    reaction_ids: list[str],
    enzyme_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    r2e = torch.zeros(len(reaction_ids), len(enzyme_ids), dtype=torch.bool)
    e2r = torch.zeros(len(enzyme_ids), len(reaction_ids), dtype=torch.bool)
    reaction_to_enzymes: dict[str, set[str]] = {}
    enzyme_to_reactions: dict[str, set[str]] = {}
    for reaction_id, enzyme_id in zip(reaction_ids, enzyme_ids):
        reaction_to_enzymes.setdefault(reaction_id, set()).add(enzyme_id)
        enzyme_to_reactions.setdefault(enzyme_id, set()).add(reaction_id)
    for row, reaction_id in enumerate(reaction_ids):
        positives = reaction_to_enzymes[reaction_id]
        for col, enzyme_id in enumerate(enzyme_ids):
            if enzyme_id in positives:
                r2e[row, col] = True
    for row, enzyme_id in enumerate(enzyme_ids):
        positives = enzyme_to_reactions[enzyme_id]
        for col, reaction_id in enumerate(reaction_ids):
            if reaction_id in positives:
                e2r[row, col] = True
    return r2e, e2r


def multilabel_bce_or_zero(
    logits: torch.Tensor | None,
    targets: torch.Tensor | None,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits is None or targets is None or logits.numel() == 0 or targets.numel() == 0:
        device = logits.device if logits is not None else (targets.device if targets is not None else "cpu")
        if logits is not None:
            return logits.sum() * 0.0
        return torch.zeros((), device=device)
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    if sample_mask is not None:
        sample_mask = sample_mask.to(device=logits.device, dtype=torch.bool)
        if sample_mask.ndim > 1:
            sample_mask = sample_mask.reshape(sample_mask.shape[0], -1).any(dim=1)
        if sample_mask.shape[0] != logits.shape[0]:
            raise ValueError(
                "sample_mask first dimension must match logits batch size, got "
                f"{tuple(sample_mask.shape)} and {tuple(logits.shape)}"
            )
        if not bool(sample_mask.any()):
            return logits.sum() * 0.0
        logits = logits[sample_mask]
        targets = targets[sample_mask]
    return F.binary_cross_entropy_with_logits(logits, targets)


class PositiveSetContrastiveLoss(nn.Module):
    """Multi-positive supervised contrastive loss over a score matrix."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        positive_mask: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        loss = multi_positive_contrastive_loss(
            anchor_embeddings,
            candidate_embeddings,
            positive_mask,
            temperature=self.temperature,
            candidate_mask=candidate_mask,
        )
        valid = positive_mask.to(device=anchor_embeddings.device, dtype=torch.bool)
        if candidate_mask is not None:
            valid = valid & candidate_mask.to(device=anchor_embeddings.device, dtype=torch.bool).unsqueeze(0)
        valid_rows = valid.any(dim=1).sum().to(dtype=anchor_embeddings.dtype)
        return {
            "loss": loss,
            "metrics/num_valid_anchors": valid_rows.detach(),
            "metrics/mean_positives_per_anchor": valid.sum(dim=1).float().mean().detach(),
        }


class CapabilityContrastiveLoss(nn.Module):
    """Bidirectional capability-space positive-set contrastive loss."""

    def __init__(
        self,
        temperature: float = 0.1,
        r2e_weight: float = 1.0,
        e2r_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.r2e = PositiveSetContrastiveLoss(temperature=temperature)
        self.e2r = PositiveSetContrastiveLoss(temperature=temperature)
        self.r2e_weight = float(r2e_weight)
        self.e2r_weight = float(e2r_weight)

    def forward(
        self,
        reaction_capability: torch.Tensor,
        enzyme_capability: torch.Tensor,
        r2e_positive_mask: torch.Tensor,
        e2r_positive_mask: torch.Tensor,
        enzyme_candidate_mask: torch.Tensor | None = None,
        reaction_candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        r2e = self.r2e(
            reaction_capability,
            enzyme_capability,
            r2e_positive_mask,
            candidate_mask=enzyme_candidate_mask,
        )
        e2r = self.e2r(
            enzyme_capability,
            reaction_capability,
            e2r_positive_mask,
            candidate_mask=reaction_candidate_mask,
        )
        total = self.r2e_weight * r2e["loss"] + self.e2r_weight * e2r["loss"]
        return {
            "loss": total,
            "loss/cap_r2e": r2e["loss"],
            "loss/cap_e2r": e2r["loss"],
            "metrics/r2e_valid_anchors": r2e["metrics/num_valid_anchors"],
            "metrics/e2r_valid_anchors": e2r["metrics/num_valid_anchors"],
        }


class AttributePredictionLoss(nn.Module):
    """Masked multi-label BCE for enzyme capability attributes."""

    def forward(
        self,
        logits_by_name: dict[str, torch.Tensor],
        targets_by_name: dict[str, torch.Tensor],
        masks_by_name: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        masks_by_name = masks_by_name or {}
        total = None
        out: dict[str, torch.Tensor] = {}
        for name, logits in logits_by_name.items():
            loss = multilabel_bce_or_zero(
                logits,
                targets_by_name.get(name),
                masks_by_name.get(name),
            )
            out[f"loss/{name}"] = loss
            total = loss if total is None else total + loss
        if total is None:
            device = next(iter(logits_by_name.values())).device if logits_by_name else "cpu"
            total = torch.zeros((), device=device)
        out["loss"] = total
        return out


class ECHierarchyLoss(AttributePredictionLoss):
    """Weak EC auxiliary loss. Uses masked multi-label BCE for now."""


class EmbeddingAnchorLoss(nn.Module):
    """Anchor trainable embeddings to a detached reference embedding."""

    def forward(
        self,
        embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        reference_embeddings = reference_embeddings.detach().to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        diff = embeddings - reference_embeddings
        if sample_mask is not None:
            sample_mask = sample_mask.to(device=embeddings.device, dtype=torch.bool)
            if not bool(sample_mask.any()):
                return {"loss": embeddings.new_zeros(())}
            diff = diff[sample_mask]
        return {"loss": diff.pow(2).sum(dim=-1).mean()}


class HardNegativeRankLoss(nn.Module):
    """Optional log-sum-exp hard-negative rank loss over a score matrix."""

    def __init__(self, margin: float = 0.10, tau_rank: float = 0.07) -> None:
        super().__init__()
        self.margin = float(margin)
        self.tau_rank = float(tau_rank)

    def forward(
        self,
        scores: torch.Tensor,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
        hard_negative_mask = hard_negative_mask.to(device=scores.device, dtype=torch.bool)
        valid = positive_mask.any(dim=1) & hard_negative_mask.any(dim=1)
        if not bool(valid.any()):
            return {"loss": scores.new_zeros(())}
        pos_scores = scores.masked_fill(~positive_mask, torch.finfo(scores.dtype).min)
        anchor_score = torch.logsumexp(pos_scores[valid], dim=1)
        hard_scores = scores[valid].masked_fill(
            ~hard_negative_mask[valid],
            torch.finfo(scores.dtype).min,
        )
        rank_terms = torch.logsumexp(
            (hard_scores - anchor_score.unsqueeze(1) + self.margin) / self.tau_rank,
            dim=1,
        )
        return {"loss": torch.log1p(torch.exp(rank_terms)).mean()}
