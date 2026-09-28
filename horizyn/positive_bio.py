"""Fixed biological targets and label-independent positive-only readouts.

The anchors encode category identity (and EC ancestry), not measured chemical
distances. They are fixed targets, not a label imputation model. Missing labels
must be masked by the caller; this module never manufactures negative targets.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


BIOLOGICAL_FAMILIES = ("ec", "cofactor", "mechanism")


def ec_prefixes(label: str) -> tuple[str, ...]:
    """Return only specified EC ancestors, accepting trailing unknown levels."""
    parts = label.split(".")
    if not 1 <= len(parts) <= 4:
        raise ValueError(f"Invalid EC label: {label!r}")
    known: list[str] = []
    unknown = False
    for part in parts:
        if part == "-":
            unknown = True
        elif unknown or re.fullmatch(r"[1-9][0-9]*", part) is None:
            raise ValueError(f"Invalid EC label: {label!r}")
        else:
            known.append(part)
    if not known or int(known[0]) not in range(1, 8):
        raise ValueError(f"Invalid EC label: {label!r}")
    return tuple(".".join(known[:level]) for level in range(1, len(known) + 1))


def validate_positive_biological_labels(
    biological_labels: Mapping[str, Sequence[str]] | None,
) -> dict[str, list[str]]:
    """Validate a vocabulary without changing its target-column order."""
    if biological_labels is None:
        return {}
    if not isinstance(biological_labels, Mapping):
        raise ValueError("positive biological labels must be a mapping")
    unknown = set(biological_labels) - set(BIOLOGICAL_FAMILIES)
    if unknown:
        raise ValueError(f"Unknown biological families: {sorted(unknown)}")
    result = {}
    for family in BIOLOGICAL_FAMILIES:
        if family not in biological_labels:
            continue
        labels = biological_labels[family]
        if not isinstance(labels, (list, tuple)) or not labels:
            raise ValueError(f"{family} labels must be a nonempty list")
        if any(not isinstance(label, str) or not label or label != label.strip() for label in labels):
            raise ValueError(f"{family} labels must be nonempty, trimmed strings")
        identities = [ec_prefixes(label)[-1] if family == "ec" else label for label in labels]
        if len(set(identities)) != len(identities):
            raise ValueError(f"Duplicate {family} labels")
        result[family] = list(labels)
    return result


def _identity_vectors(identities: list[str], dimension: int, namespace: str) -> dict[str, torch.Tensor]:
    """Orthogonal/simplex where possible, reproducible random codes otherwise."""
    names = sorted(set(identities))
    if len(names) <= dimension:
        vectors = torch.eye(len(names), dimension, dtype=torch.float32)
    elif namespace != "ec" and len(names) == dimension + 1:
        # Helmert contrasts give n equidistant simplex vertices in n-1 axes.
        vectors = torch.zeros(len(names), dimension, dtype=torch.float32)
        for column in range(dimension):
            scale = ((column + 1) * (column + 2)) ** -0.5
            vectors[:column + 1, column] = scale
            vectors[column + 1, column] = -(column + 1) * scale
        vectors = F.normalize(vectors, dim=-1)
    else:
        rows = []
        for name in names:
            digest = hashlib.sha256(f"circe-positive-anchor-v1:{namespace}:{name}".encode()).digest()
            generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
            rows.append(torch.randn(dimension, generator=generator, dtype=torch.float32))
        vectors = F.normalize(torch.stack(rows), dim=-1)
    return dict(zip(names, vectors))


def build_positive_anchors(family: str, labels: Sequence[str], dimension: int) -> torch.Tensor:
    """Construct fixed unit targets; an EC target shares all known ancestors.

    When the EC vocabulary exceeds the embedding dimension, ancestry vectors
    are deterministic random codes: common ancestors encourage similarity but
    do not define exact orthogonal distances. No descendant is inferred.
    """
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2:
        raise ValueError("Anchor dimension must be an integer >= 2")
    labels = validate_positive_biological_labels({family: labels})[family]
    paths = [ec_prefixes(label) if family == "ec" else (label,) for label in labels]
    vectors = _identity_vectors([name for path in paths for name in path], dimension, family)
    return F.normalize(torch.stack([
        torch.stack([vectors[name] for name in path]).sum(0) for path in paths
    ]), dim=-1)


class PositiveBiologicalReadout(nn.Module):
    """Pool latent views without observing any target labels at forward time."""

    def __init__(self, dimension: int, family: str, labels: Sequence[str], gate_floor: float = 0.05):
        super().__init__()
        self.gate_floor = gate_floor
        self.register_buffer("anchors", build_positive_anchors(family, labels, dimension))
        self.gate = nn.Sequential(nn.Linear(dimension, dimension), nn.Tanh(), nn.Linear(dimension, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.projection = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension),
        )

    def forward(self, shared_slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = self.gate(shared_slots).squeeze(-1).float().softmax(-1)
        weights = (1.0 - self.gate_floor) * weights + self.gate_floor / shared_slots.shape[1]
        pooled = (shared_slots * weights[..., None].to(shared_slots.dtype)).sum(1)
        embedding = F.normalize(self.projection(pooled).float(), dim=-1, eps=1e-6)
        distance = 1.0 - (embedding @ self.anchors.float().T).clamp(-1.0, 1.0)
        return distance, embedding, weights
