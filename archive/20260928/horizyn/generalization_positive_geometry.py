"""Positive-edge alignment with bounded independent residual encoders.

No unlisted pair is an explicit negative. Global moment regularization still
constrains the aggregate embedding distribution; it is not an activity label.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .generalization_residual import FrozenGeometryResidual


class PositiveGeometryResidual(FrozenGeometryResidual):
    def __init__(self, dimension=512, hidden=1024, scale=.2, cap=.1):
        super().__init__(dimension,hidden,scale)
        if not 0 < cap < 1:
            raise ValueError('Relative residual norm cap must lie in (0, 1)')
        self.cap=float(cap)

    def _encode(self, base, tower):
        # Training uses FP32 weights. A separate .double() copy provides the
        # pinned FP64 forward+normalization contract for stable inference.
        if base.ndim != 2 or base.shape[1] != self.dimension or not base.is_floating_point():
            raise ValueError('Native F3 features must be a floating matrix with the configured dimension')
        raw=base.to(tower[1].weight.dtype)
        delta=self.scale*tower(raw)
        factor=(self.cap*raw.norm(dim=1,keepdim=True)/delta.norm(dim=1,keepdim=True).clamp_min(1e-12)).clamp(max=1)
        return F.normalize(raw+factor*delta,dim=1,eps=torch.finfo(raw.dtype).tiny).to(base.dtype)

    def encode_enzymes(self, base):
        return self._encode(base,self.enzyme)

    def encode_reactions(self, base):
        return self._encode(base,self.reaction)


def edge_marginals(reaction_index,enzyme_index,n_reactions,n_enzymes,dtype=torch.float32):
    """Each reaction has total mass1/R, split uniformly across known edges."""
    counts=torch.bincount(reaction_index,minlength=n_reactions).to(dtype)
    if bool((counts==0).any()):
        raise ValueError('Every training reaction must have a recorded positive')
    edge_weight=1/(n_reactions*counts[reaction_index])
    enzyme_weight=torch.zeros(n_enzymes,dtype=dtype,device=reaction_index.device)
    enzyme_weight.scatter_add_(0,enzyme_index,edge_weight)
    if bool((enzyme_weight==0).any()):
        raise ValueError('Every training enzyme must have a recorded positive')
    return edge_weight,enzyme_weight


def weighted_moment_penalty(features,weights):
    """||mu||² + d||C-I/d||F², C=E[zzᵀ]-mu muᵀ, normalized weights.

    For unit features, a constant collapsed population has penalty2, while a
    zero-mean isotropic population has penalty0. No unbiased-sample correction
    is used: these are moments of the declared weighted training population.
    """
    weights=weights/weights.sum()
    mean=(weights[:,None]*features).sum(0)
    covariance=features.T@(weights[:,None]*features)-mean[:,None]*mean[None,:]
    dimension=features.shape[1]
    target=torch.eye(dimension,device=features.device,dtype=features.dtype)/dimension
    mean_loss=mean.square().sum()
    covariance_loss=dimension*(covariance-target).square().sum()
    return mean_loss+covariance_loss,mean_loss,covariance_loss


def positive_geometry_loss(reactions,enzymes,reaction_index,enzyme_index,edge_weight,
                           enzyme_weight,base_reactions,base_enzymes,moment_weight,identity_weight=2.):
    positive=(edge_weight*(1-(reactions[reaction_index]*enzymes[enzyme_index]).sum(1))).sum()
    reaction_weight=torch.full((len(reactions),),1/len(reactions),device=reactions.device,dtype=reactions.dtype)
    # Match the same reaction-balanced graph marginals for identity and moments.
    ew=enzyme_weight/enzyme_weight.sum()
    identity_r=(reaction_weight*(1-(reactions*base_reactions).sum(1))).sum()
    identity_e=(ew*(1-(enzymes*base_enzymes).sum(1))).sum()
    identity=(identity_r+identity_e)/2
    moment_r,mean_r,cov_r=weighted_moment_penalty(reactions,reaction_weight)
    moment_e,mean_e,cov_e=weighted_moment_penalty(enzymes,ew)
    moment=(moment_r+moment_e)/2
    total=positive+identity_weight*identity+moment_weight*moment
    diagnostics=dict(positive=positive.detach(),identity=identity.detach(),moment=moment.detach(),
        reaction_mean=mean_r.detach(),reaction_covariance=cov_r.detach(),
        enzyme_mean=mean_e.detach(),enzyme_covariance=cov_e.detach())
    return total,diagnostics
