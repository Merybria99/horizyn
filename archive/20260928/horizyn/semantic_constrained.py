"""Globally bounded linear maps of independently encoded semantic endpoints."""
from __future__ import annotations

import torch
from torch import nn
from .semantic_anchors import row_unit


class BoundedSemanticMap(nn.Module):
    """I+A with ||A||_2 <= radius < 1, followed by unit normalization.

    This gives a uniform pre-normalization displacement bound for *every*
    input, including points absent from supervised training. It does not imply
    a retrieval-rank or biological-generalization guarantee.
    """

    def __init__(self, dimension: int, radius: float = .5):
        super().__init__()
        if dimension < 1 or not 0 <= radius < 1:
            raise ValueError("Positive dimension and radius in [0,1) required")
        self.dimension, self.radius = dimension, float(radius)
        self.enzyme_delta = nn.Parameter(torch.zeros(dimension, dimension))
        self.reaction_delta = nn.Parameter(torch.zeros(dimension, dimension))

    def encode_enzymes(self, vectors):
        return row_unit(vectors + vectors @ self.enzyme_delta)

    def encode_reactions(self, vectors):
        return row_unit(vectors + vectors @ self.reaction_delta)

    @torch.no_grad()
    def project(self):
        """Exact spectral-norm projection by common rescaling, no input data."""
        norms = []
        for matrix in (self.enzyme_delta, self.reaction_delta):
            norm = torch.linalg.matrix_norm(matrix.double(), ord=2)
            if norm > self.radius:
                # Tiny slack keeps FP32 rounding below the declared bound.
                matrix.mul_(self.radius * (1 - 1e-6) / norm)
            norms.append(float(torch.linalg.matrix_norm(matrix.double(), ord=2)))
        return norms
