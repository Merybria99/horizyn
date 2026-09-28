"""Training-only biological neighborhood loss; no learned heads or inference state.

Each observed category defines positive within-endpoint relationships. Missing
categories never define negatives. EC prefixes encode ancestry, not sequence
position. Category-balanced pair distances are computed exactly from sums,
without a quadratic matrix or sampled pairs.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def category_pair_distance(embeddings, rows, groups, confidence, category_weight):
    """Weighted mean off-diagonal cosine distance, balanced by category.

Confidence scales evidence; division is by pair COUNT, not confidence mass.
Thus uncertain annotations remain weaker rather than normalizing away their
uncertainty. Singleton categories contribute neither loss nor denominator.
"""
    if rows.numel() == 0:
        return embeddings.sum() * 0
    unit = F.normalize(embeddings.float(), dim=-1)
    n = category_weight.numel()
    count = torch.bincount(groups, minlength=n).to(unit.dtype)
    weight_sum = unit.new_zeros(n).index_add(0, groups, confidence)
    squared_weight = unit.new_zeros(n).index_add(0, groups, confidence.square())
    sums = unit.new_zeros((n, unit.shape[1])).index_add(
        0, groups, unit[rows] * confidence[:, None])
    diagonal = unit.new_zeros(n).index_add(
        0, groups, confidence.square() * unit[rows].square().sum(-1))
    numerator = weight_sum.square() - squared_weight - sums.square().sum(-1) + diagonal
    valid = count > 1
    weights = category_weight * valid
    distance = numerator / (count * (count - 1)).clamp_min(1)
    return (distance * weights).sum() / weights.sum().clamp_min(1e-12)


def category_relative_distance(embeddings, rows, groups, confidence, category_weight, margin=.1):
    """Confidence-weighted within-category preference over annotated background.

Only endpoints annotated in this family enter the background. A background
endpoint lacks this particular observed category; it is not declared an
activity-negative enzyme. The hinge supervises relative separation rather
than absolute within-category compactness; it does not guarantee against collapse.
"""
    if rows.numel() == 0:
        return embeddings.sum() * 0
    unit = F.normalize(embeddings.float(), dim=-1)
    n = category_weight.numel()
    count = torch.bincount(groups, minlength=n).to(unit.dtype)
    weight_sum = unit.new_zeros(n).index_add(0, groups, confidence)
    squared_weight = unit.new_zeros(n).index_add(0, groups, confidence.square())
    sums = unit.new_zeros((n, unit.shape[1])).index_add(0, groups, unit[rows] * confidence[:, None])
    diagonal = unit.new_zeros(n).index_add(0, groups, confidence.square() * unit[rows].square().sum(-1))
    pair_mass = weight_sum.square() - squared_weight
    within = (pair_mass - sums.square().sum(-1) + diagonal) / pair_mass.clamp_min(1e-12)
    row_confidence = unit.new_zeros(len(unit)).scatter_reduce(0, rows, confidence, reduce='amax')
    annotated = row_confidence > 0
    background_count = annotated.sum() - count
    inside_background_weight = unit.new_zeros(n).index_add(0, groups, row_confidence[rows])
    inside_background_sum = unit.new_zeros(sums.shape).index_add(
        0, groups, unit[rows] * row_confidence[rows, None])
    outside_weight = row_confidence.sum() - inside_background_weight
    outside_sum = (unit * row_confidence[:, None]).sum(0)[None] - inside_background_sum
    between_mass = weight_sum * outside_weight
    between = (between_mass - (sums * outside_sum).sum(-1)) / between_mass.clamp_min(1e-12)
    evidence = pair_mass / (count * (count - 1)).clamp_min(1)
    valid = (count > 1) & (background_count > 0) & (pair_mass > 0) & (between_mass > 0)
    weights = category_weight * valid
    penalty = evidence * F.relu(float(margin) + within - between)
    return (weights * penalty).sum() / weights.sum().clamp_min(1e-12)


def biological_distance(mode, embeddings, *args, margin=.1):
    if mode == 'attraction':
        return category_pair_distance(embeddings, *args)
    if mode == 'relative':
        return category_relative_distance(embeddings, *args, margin=margin)
    raise ValueError(f'Unknown biological geometry mode: {mode}')


class BiologicalGeometryLoss:
    """Nonparametric training helper, deliberately not part of either encoder."""

    families = ("ec", "cofactor", "mechanism")

    def __init__(self, payload, reaction_ids, enzyme_ids, device, shuffle_seed=None, mode='attraction', margin=.1):
        if payload['reaction_ids'] != list(reaction_ids) or payload['enzyme_ids'] != list(enzyme_ids):
            raise ValueError('Biological labels and compact training axes disagree')
        self.data = {}
        self.mode, self.margin = mode, margin
        for endpoint, size in [('reaction', len(reaction_ids)), ('enzyme', len(enzyme_ids))]:
            permutation = None
            if shuffle_seed is not None:
                generator = torch.Generator().manual_seed(shuffle_seed)
                permutation = torch.randperm(size, generator=generator)
            for family in self.families:
                block = payload['endpoints'][endpoint][family]
                rows = torch.tensor(block['rows'], dtype=torch.long)
                if permutation is not None:
                    rows = permutation[rows]
                self.data[endpoint, family] = (
                    rows.to(device), torch.tensor(block['groups'], dtype=torch.long, device=device),
                    torch.tensor(block['confidence'], dtype=torch.float32, device=device),
                    torch.tensor(block['category_weight'], dtype=torch.float32, device=device))

    def __call__(self, reactions, enzymes, weights):
        pieces = {}
        for family in self.families:
            pieces[family] = (
                biological_distance(self.mode, reactions, *self.data['reaction', family], margin=self.margin) +
                biological_distance(self.mode, enzymes, *self.data['enzyme', family], margin=self.margin)) / 2
        # Fixed denominator: removing a family does not strengthen the others.
        loss = sum(float(weights.get(f, 0)) * pieces[f] for f in self.families) / 3
        return loss, pieces


class BatchBiologicalGeometryLoss:
    """CPU annotation lookup for the existing minibatch embeddings, no heads."""

    def __init__(self, path, mode='attraction', margin=.1):
        import json
        from pathlib import Path
        payload = json.loads(Path(path).read_text())
        self.mode, self.margin = mode, margin
        self.rows = {}
        self.category_weights = {}
        for endpoint, id_key in [('reaction', 'reaction_ids'), ('enzyme', 'enzyme_ids')]:
            ids = payload[id_key]
            for family in BiologicalGeometryLoss.families:
                block = payload['endpoints'][endpoint][family]
                rows = {key: [] for key in ids}
                for row, group, confidence in zip(block['rows'], block['groups'], block['confidence']):
                    rows[ids[row]].append((group, confidence))
                self.rows[endpoint, family] = rows
                self.category_weights[endpoint, family] = block['category_weight']

    def __call__(self, reactions, enzymes, reaction_ids, enzyme_ids):
        pieces = {}
        for family in BiologicalGeometryLoss.families:
            values = []
            for endpoint, embeddings, ids in [('reaction', reactions, reaction_ids), ('enzyme', enzymes, enzyme_ids)]:
                lookup = self.rows[endpoint, family]
                row_indices, groups, confidence = [], [], []
                remap = {}
                category_weights = []
                for row, key in enumerate(ids):
                    key = str(key)
                    # F3's forward-only loader adds this direction suffix;
                    # the phase-2 catalogs keep the original reaction ID.
                    if endpoint == 'reaction' and key.endswith('_f') and key not in lookup:
                        key = key[:-2]
                    if key not in lookup:
                        raise ValueError(f'Non-training {endpoint} ID in biological training loss: {key}')
                    for group, weight in lookup[key]:
                        if group not in remap:
                            remap[group] = len(remap)
                            category_weights.append(self.category_weights[endpoint, family][group])
                        row_indices.append(row); groups.append(remap[group]); confidence.append(weight)
                device = embeddings.device
                # Explicit FP32 protects this small loss from mixed-precision cancellation.
                with torch.autocast(device_type=device.type, enabled=False):
                    values.append(biological_distance(self.mode, embeddings,
                        torch.tensor(row_indices, dtype=torch.long, device=device),
                        torch.tensor(groups, dtype=torch.long, device=device),
                        torch.tensor(confidence, dtype=torch.float32, device=device),
                        torch.tensor(category_weights, dtype=torch.float32, device=device), margin=self.margin))
            pieces[family] = sum(values) / 2
        return sum(pieces.values()) / 3, pieces
