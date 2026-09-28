"""Capped raw-feature corrections of P2 neural endpoints; semantics stay fixed."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .generalization_retrieval import GeneralizationDualEncoder


class RawFeatureResidual(nn.Module):
    def __init__(self, protein_raw_dimension=1024, reaction_raw_dimension=2409,
                 hidden=1024, cap=.1):
        super().__init__()
        if cap not in (.1, .3) or hidden != 1024:
            raise ValueError('Fixed caps .1/.3 and hidden width1024 required')
        self.cap=float(cap);self.hidden=hidden
        self.protein_raw_dimension=protein_raw_dimension
        self.reaction_raw_dimension=reaction_raw_dimension
        self.enzyme=self._tower(512+protein_raw_dimension)
        self.reaction=self._tower(512+reaction_raw_dimension)

    @staticmethod
    def _tower(dimension):
        result=nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,1024),
                             nn.GELU(),nn.Linear(1024,512))
        nn.init.zeros_(result[-1].weight);nn.init.zeros_(result[-1].bias)
        return result

    def _encode(self,native,raw,parent,tower,identity=False,return_diagnostics=False):
        if native.ndim!=2 or raw.ndim!=2 or parent.shape!=native.shape or native.shape[1]!=512 or len(raw)!=len(native):
            raise ValueError('Aligned native/raw/weighted-parent endpoint matrices required')
        if native.dtype!=torch.float32 or raw.dtype!=torch.float32 or parent.dtype!=torch.float32:
            raise ValueError('Stored endpoint inputs must be FP32')
        if not all(bool(torch.isfinite(x).all()) for x in (native,raw,parent)):
            raise ValueError('Finite endpoint inputs required')
        if raw.shape[1]+512!=tower[0].normalized_shape[0]:
            raise ValueError('Wrong raw endpoint dimension')
        dtype=tower[1].weight.dtype
        base=parent.to(dtype);norm=base.norm(dim=1,keepdim=True)
        if bool((norm<=1e-8).any()):raise ValueError('Nonzero parent neural endpoint required')
        if identity:
            diagnostics={'relative_residual':norm[:,0]*0,'identity':True}
            return (parent,diagnostics) if return_diagnostics else parent
        delta=tower(torch.cat((native,raw),1).to(dtype))
        factor=(self.cap*norm/delta.norm(dim=1,keepdim=True).clamp_min(torch.finfo(dtype).tiny)).clamp(max=1.)
        clipped=delta*factor
        # Parent is already sqrt(.75)-weighted: restore that same weight.
        output=(math.sqrt(.75)*F.normalize(base+clipped,dim=1,eps=torch.finfo(dtype).tiny)).float()
        diagnostics={'relative_residual':clipped.norm(dim=1)/norm[:,0], 'identity':False}
        return (output,diagnostics) if return_diagnostics else output

    def encode_enzymes(self,native,raw,parent,identity=False,return_diagnostics=False):
        return self._encode(native,raw,parent,self.enzyme,identity,return_diagnostics)

    def encode_reactions(self,native,raw,parent,identity=False,return_diagnostics=False):
        return self._encode(native,raw,parent,self.reaction,identity,return_diagnostics)

    @staticmethod
    def compose(neural,weighted_semantic):
        if len(neural)!=len(weighted_semantic):raise ValueError('Endpoint branches are misaligned')
        return torch.cat((neural,weighted_semantic),1)

    score_index=staticmethod(GeneralizationDualEncoder.score_index)
