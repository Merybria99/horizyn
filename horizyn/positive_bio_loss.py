"""Positive-only cosine-anchor supervision; missing labels never repel embeddings."""
import torch
import torch.distributed as dist


def positive_anchor_loss(details, targets, family_weights):
    """Mean over observed labels, then enzymes, then globally active families.

    Each rank supplies its locally deduplicated enzyme rows. DDP's gradient
    average is compensated to produce a global active-row mean. Sparse indices
    avoid materializing thousands of EC columns per enzyme in the dataset.
    """
    total, active_weight, components = None, 0.0, {}
    for family, weight in sorted(family_weights.items()):
        distances = details[f"biofp_alignment_{family}"].float()
        indices = targets.get(f"biofp_{family}_positive_indices")
        if indices is None:
            raise ValueError(f"positive_anchor requires sparse positive indices for {family}")
        indices = indices.to(device=distances.device, dtype=torch.long)
        confidence = targets[f"biofp_{family}_confidence"].to(distances)
        if indices.ndim != 2 or indices.shape[0] != distances.shape[0] or confidence.shape != indices.shape:
            raise ValueError(f"Invalid sparse {family} target shape")
        if bool(((indices < -1) | (indices >= distances.shape[1])).any()):
            raise ValueError(f"Out-of-range {family} positive index")
        if not bool(torch.isfinite(confidence).all()) or bool((confidence < 0).any()):
            raise ValueError(f"Invalid {family} positive confidence")
        valid = indices.ge(0) & confidence.gt(0)
        weights = confidence * valid
        selected = distances.gather(1, indices.clamp_min(0))
        row_mass = weights.sum(1)
        row_loss = (selected * weights).sum(1) / row_mass.clamp_min(1e-12)
        numerator = row_loss.sum()
        active_rows = row_mass.gt(0).sum().to(distances.dtype)
        counts = torch.stack([active_rows, valid.sum().to(distances.dtype), numerator.detach()])
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
            world_size = dist.get_world_size()
        local_scaled_loss = world_size * numerator / counts[0].clamp_min(1)
        # Identical forward values on each rank, with correctly scaled local gradients.
        global_value = counts[2] / counts[0].clamp_min(1)
        loss = local_scaled_loss + (global_value - local_scaled_loss.detach())
        active_weight = active_weight + weight * counts[0].gt(0)
        weighted = weight * loss
        total = weighted if total is None else total + weighted
        components[f"biofp_{family}"] = loss
        components[f"weighted_biofp_{family}"] = weighted
        components[f"biofp_{family}_active"] = counts[0]
        components[f"biofp_{family}_active_labels"] = counts[1]
    if total is None:
        raise ValueError("positive_anchor needs at least one family")
    normalizer = active_weight.clamp_min(1e-12)
    for family in family_weights:
        components[f"weighted_biofp_{family}"] = components[f"weighted_biofp_{family}"] / normalizer
    return total / normalizer, components
