"""Train-dictionary evidence using raw and native-F3 within-endpoint geometry."""
from __future__ import annotations

import math
import torch

from .semantic_anchors import (row_unit, centered_unit, reaction_features,
    nearest_training_proteins, enzyme_anchor_features)
from .semantic_smooth import SmoothAnchorDualEncoder, reaction_responses


def hybrid_view(raw_unit, native_unit, eta):
    """Actual FP32 sqrt-weight concatenation; no additional normalization."""
    if eta not in (0., .5, 1.):
        raise ValueError('The prespecified native-view weights are 0, .5, 1')
    if raw_unit.ndim!=2 or native_unit.ndim!=2 or len(raw_unit)!=len(native_unit):
        raise ValueError('Raw and native endpoints must be aligned matrices')
    return torch.cat((math.sqrt(1-eta)*raw_unit,math.sqrt(eta)*native_unit),dim=1)


class HybridAnchorDualEncoder(SmoothAnchorDualEncoder):
    """Semantic evidence helper; final density composition is explicit.

    Use hybrid_enzyme_features/hybrid_reaction_features for these semantic
    vectors. The inherited native-F3 composition is intentionally disabled.
    """
    def __init__(self,dictionary,train_reactions,config,native_train_enzymes,native_train_reactions,
                 eta_enzyme,eta_reaction):
        super().__init__(dictionary,train_reactions,config)
        if eta_enzyme not in (0.,.5,1.) or eta_reaction not in (0.,.5,1.):
            raise ValueError('Unregistered hybrid geometry')
        if len(native_train_enzymes)!=len(self.train_proteins) or len(native_train_reactions)!=len(train_reactions):
            raise ValueError('Native and raw training dictionaries must align')
        self.eta_enzyme=float(eta_enzyme);self.eta_reaction=float(eta_reaction)
        self.register_buffer('native_train_enzymes',native_train_enzymes)
        self.register_buffer('native_train_reactions',native_train_reactions)

    def encode_enzymes(self,*args,**kwargs):
        raise RuntimeError('Use hybrid_enzyme_features and explicitly compose with the frozen density encoder')

    def encode_reactions(self,*args,**kwargs):
        raise RuntimeError('Use hybrid_reaction_features and explicitly compose with the frozen density encoder')

    @torch.inference_mode()
    def enzyme_neighbors(self,base,protein_mean,batch_size=512):
        if len(base)!=len(protein_mean):raise ValueError('Native/raw enzyme rows disagree')
        raw=centered_unit(protein_mean,self.protein_center)
        encoded=raw if self.eta_enzyme==0 else hybrid_view(raw,row_unit(base),self.eta_enzyme)
        anchors=self.train_proteins if self.eta_enzyme==0 else hybrid_view(self.train_proteins,self.native_train_enzymes,self.eta_enzyme)
        return nearest_training_proteins(encoded,anchors,self.config['protein_neighbors'],batch_size)

    @torch.inference_mode()
    def hybrid_enzyme_features(self,base,protein_mean,batch_size=512):
        if self.eta_enzyme==0:
            if len(base)!=len(protein_mean):raise ValueError('Native/raw enzyme rows disagree')
            return super().encode_semantic_enzymes(protein_mean,batch_size)
        values,indices=self.enzyme_neighbors(base,protein_mean,batch_size)
        return enzyme_anchor_features(values,indices,self.adjacency,len(self.train_reactions),self.config['enzyme_temperature'])

    @torch.inference_mode()
    def hybrid_reaction_features(self,base,blocks,masks):
        if self.eta_reaction==0:
            return super().encode_semantic_reactions(blocks,masks)
        centers={key:getattr(self,'reaction_center_'+key) for key in self.modalities}
        raw=reaction_features(blocks,centers,masks,self.modalities)
        if len(base)!=len(raw):raise ValueError('Native/raw reaction rows disagree')
        encoded=hybrid_view(raw,row_unit(base),self.eta_reaction)
        anchors=hybrid_view(self.train_reactions,self.native_train_reactions,self.eta_reaction)
        return reaction_responses(encoded,anchors,self.config['kernel'],self.config['reaction_temperature'],self.config['reaction_neighbors'])

    @classmethod
    def from_checkpoint(cls,path,device='cpu'):
        from .generalization_retrieval import checked_artifact
        state=torch.load(path,map_location=device,weights_only=False)
        dictionary=torch.load(checked_artifact(state['dictionary']),map_location=device,weights_only=False)
        smooth=torch.load(checked_artifact(state['smooth_state']),map_location=device,weights_only=False)
        if dictionary['feature_manifest_sha256']!=state['feature_manifest_sha256'] or smooth['feature_manifest_sha256']!=state['feature_manifest_sha256']:
            raise ValueError('Hybrid state/dictionary/source manifests disagree')
        model=cls(dictionary,smooth['train_reactions'],smooth['config'],state['native_train_enzymes'],
            state['native_train_reactions'],state['eta_enzyme'],state['eta_reaction']).to(device)
        return model,state
