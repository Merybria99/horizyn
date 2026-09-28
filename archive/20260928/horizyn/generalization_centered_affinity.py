"""Late exploratory centered enzyme-affinity correction; no default integration."""
from __future__ import annotations
import math
import torch

class CenteredAffinity(torch.nn.Module):
    def __init__(self, reaction_mean, enzyme_mean, gamma=0.):
        super().__init__()
        if reaction_mean.ndim!=1 or enzyme_mean.shape!=reaction_mean.shape or not torch.isfinite(reaction_mean).all() or not torch.isfinite(enzyme_mean).all():
            raise ValueError('Finite aligned training means required')
        if not math.isfinite(gamma) or gamma<0:raise ValueError('Finite nonnegative gamma required')
        self.register_buffer('reaction_mean',reaction_mean.double())
        self.register_buffer('enzyme_mean',enzyme_mean.double())
        self.gamma=float(gamma);self.requires_grad_(False).eval()

    def _check(self,x):
        if x.ndim!=2 or x.dtype!=torch.float32 or x.shape[1]!=len(self.reaction_mean) or not torch.isfinite(x).all():raise ValueError('Finite aligned FP32 endpoints required')

    @torch.inference_mode()
    def encode_enzymes(self,e):
        self._check(e)
        if self.gamma==0:return e
        b=((e.double()-self.enzyme_mean[None,:])*self.reaction_mean[None,:]).sum(1).float()
        return torch.cat((e,b[:,None]),1)

    @torch.inference_mode()
    def encode_reactions(self,r,unsupported_gate):
        self._check(r)
        if unsupported_gate.shape!=(len(r),) or unsupported_gate.dtype!=torch.float32 or not torch.isfinite(unsupported_gate).all() or bool(((unsupported_gate<0)|(unsupported_gate>1)).any()):raise ValueError('Finite FP32 gate in [0,1] required')
        if self.gamma==0:return r
        coordinate=(-self.gamma*unsupported_gate.double()).float()
        return torch.cat((r,coordinate[:,None]),1)
