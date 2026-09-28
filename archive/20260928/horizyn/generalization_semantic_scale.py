"""Independent query semantic scaling from a fixed training-only enzyme bank."""
from __future__ import annotations
import math
import torch
from horizyn.generalization_retrieval import GeneralizationDualEncoder

class QuerySemanticScale(torch.nn.Module):
    def __init__(self,bank,semantic_mean,beta=None,cap=None,dense_width=512,epsilon=1e-8):
        super().__init__()
        if bank.ndim!=2 or bank.dtype!=torch.float32 or len(bank)<1 or dense_width<1 or bank.shape[1]<=dense_width or not torch.isfinite(bank).all():raise ValueError('Finite FP32 training bank and valid dense width required')
        if semantic_mean.shape!=(bank.shape[1]-dense_width,) or not torch.isfinite(semantic_mean).all():raise ValueError('Finite aligned training semantic mean required')
        if (beta is None)!=(cap is None):raise ValueError('Identity uses both beta and cap absent')
        if beta is not None and (not math.isfinite(beta) or beta<=0 or not math.isfinite(cap) or cap<1):raise ValueError('Finite positive beta and cap>=1 required')
        if epsilon!=1e-8:raise ValueError('Fixed epsilon is1e-8')
        self.register_buffer('bank',bank);self.register_buffer('semantic_mean',semantic_mean.double())
        self.beta,self.cap,self.dense_width,self.epsilon=beta,cap,dense_width,epsilon
        self.requires_grad_(False).eval()

    def _check(self,x):
        if x.ndim!=2 or x.dtype!=torch.float32 or x.shape[1]!=self.bank.shape[1] or not torch.isfinite(x).all():raise ValueError('Finite aligned FP32 P2 endpoints required')

    @torch.inference_mode()
    def statistics(self,reactions,batch_size=64):
        self._check(reactions)
        if not isinstance(batch_size,int) or batch_size<1:raise ValueError('Positive batch size required')
        width=self.dense_width;sd_d=[];sd_s=[]
        bd=self.bank[:,:width].double();bs=self.bank[:,width:].double()
        for start in range(0,len(reactions),batch_size):
            r=reactions[start:start+batch_size].double()
            d=r[:,:width]@bd.T;s=r[:,width:]@bs.T
            sd_d.append(((d-d.mean(1,keepdim=True)).square().mean(1)).sqrt())
            sd_s.append(((s-s.mean(1,keepdim=True)).square().mean(1)).sqrt())
        empty=reactions.new_empty((0,),dtype=torch.float64)
        return (torch.cat(sd_d) if sd_d else empty,torch.cat(sd_s) if sd_s else empty)

    @torch.inference_mode()
    def encode_reactions(self,reactions,batch_size=64,return_diagnostics=False):
        self._check(reactions)
        if self.beta is None:
            diagnostics=dict(multiplier=reactions.new_ones((len(reactions),),dtype=torch.float64),identity=True)
            return (reactions,diagnostics) if return_diagnostics else reactions
        sd_d,sd_s=self.statistics(reactions,batch_size)
        multiplier=(self.beta*sd_d/sd_s.clamp_min(self.epsilon)).clamp(1.,self.cap)
        semantic=reactions[:,self.dense_width:].double()
        mean_score=(semantic*self.semantic_mean[None,:]).sum(1)
        offset=(-(multiplier-1.)*mean_score).float()
        result=torch.cat((reactions[:,:self.dense_width],(multiplier[:,None]*semantic).float(),offset[:,None]),1)
        diagnostics=dict(sd_dense=sd_d,sd_semantic=sd_s,multiplier=multiplier,mean_semantic_score=mean_score,offset=offset,identity=False)
        return (result,diagnostics) if return_diagnostics else result

    @torch.inference_mode()
    def encode_enzymes(self,enzymes):
        self._check(enzymes)
        if self.beta is None:return enzymes
        return torch.cat((enzymes,enzymes.new_ones((len(enzymes),1))),1)

    @torch.inference_mode()
    def encode_enzyme_index(self,enzymes):
        self._check(enzymes)
        anchors=enzymes[:,self.dense_width:]
        if self.beta is not None:anchors=torch.cat((anchors,enzymes.new_ones((len(enzymes),1))),1)
        return dict(dense=enzymes[:,:self.dense_width],anchors=anchors.to_sparse_csr())

    score_index=staticmethod(GeneralizationDualEncoder.score_index)
