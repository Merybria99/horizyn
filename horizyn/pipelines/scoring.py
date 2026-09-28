"""The shared dictionary-free, bidirectional retrieval score."""

import torch
from torch.nn import functional as F


@torch.inference_mode()
def canonical_dot(reactions, enzymes):
    """Keep the float64 accumulation used by frozen evaluation protocols."""
    return (reactions.double() @ enzymes.double().T).float()


def refine(base, head, modality, residual_cap=0.5):
    """Apply the fitted residual at inference; cap=.5 gives scale=.1 for V4."""
    if modality not in {"enzyme", "reaction"}:
        raise ValueError("modality must be enzyme or reaction")
    if not 0 <= residual_cap <= 1:
        raise ValueError("residual_cap must be in [0, 1]")
    if head is None:
        return base
    return F.normalize(base + residual_cap * head.scale * getattr(head, modality)(base), dim=-1)
