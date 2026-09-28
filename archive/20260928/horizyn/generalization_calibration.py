"""Optional post-evaluation mean-affinity control on already encoded endpoints.

This module does not load, modify or fit any frozen encoder. Means are supplied
from training endpoints only. There is deliberately no normalization after the
two bias coordinates are appended.
"""
from __future__ import annotations
import torch


class MeanAffinityCalibration(torch.nn.Module):
    def __init__(self, reaction_mean, enzyme_mean, gamma_enzyme=0., gamma_reaction=0.):
        super().__init__()
        if reaction_mean.ndim != 1 or enzyme_mean.shape != reaction_mean.shape:
            raise ValueError('Aligned one-dimensional training means required')
        if not torch.isfinite(reaction_mean).all() or not torch.isfinite(enzyme_mean).all():
            raise ValueError('Finite training means required')
        if gamma_enzyme < 0 or gamma_reaction < 0:
            raise ValueError('Nonnegative calibration strengths required')
        self.register_buffer('reaction_mean', reaction_mean.double())
        self.register_buffer('enzyme_mean', enzyme_mean.double())
        self.gamma_enzyme, self.gamma_reaction = float(gamma_enzyme), float(gamma_reaction)
        self.requires_grad_(False).eval()

    def _check(self, endpoints):
        if endpoints.dtype != torch.float32 or endpoints.ndim != 2 or endpoints.shape[1] != len(self.reaction_mean):
            raise ValueError('FP32 endpoint matrix matching the means required')
        if not torch.isfinite(endpoints).all():
            raise ValueError('Nonfinite endpoints')

    @torch.inference_mode()
    def encode_reactions(self, endpoints):
        self._check(endpoints)
        if self.gamma_enzyme == self.gamma_reaction == 0:
            return endpoints
        bias = (-self.gamma_reaction * (endpoints.double() * self.enzyme_mean[None, :]).sum(1)).float()
        return torch.cat((endpoints, bias[:, None], torch.ones_like(bias[:, None])), 1)

    @torch.inference_mode()
    def encode_enzymes(self, endpoints):
        self._check(endpoints)
        if self.gamma_enzyme == self.gamma_reaction == 0:
            return endpoints
        bias = (-self.gamma_enzyme * (endpoints.double() * self.reaction_mean[None, :]).sum(1)).float()
        return torch.cat((endpoints, torch.ones_like(bias[:, None]), bias[:, None]), 1)
