"""Frozen selected Morgan reaction view with unchanged P2 enzyme index."""
from __future__ import annotations
import torch
from .generalization_morgan import MorganReactionAnchor
from .semantic_anchors import reaction_features
from .generalization_retrieval import checked_artifact,GeneralizationDualEncoder

RULE=dict(radius=3,eta_reaction=1.,fp_size=4096,include_chirality=True,alpha=.25,temperature=.03)

class MorganComposedEncoder(torch.nn.Module):
    def __init__(self,parent,state):
        super().__init__()
        if state['schema']!='morgan_training_dictionary_v1' or state['rule']!=RULE or parent.alpha!=.25:
            raise ValueError('Selected Morgan/P2 recipe mismatch')
        self.parent=parent;self.modalities=parent.modalities
        self.anchor=MorganReactionAnchor(parent.smooth.train_reactions,state['train_morgan'].to(parent.smooth.train_reactions.device),1.)
        if len(state['train_ids'])!=len(parent.smooth.train_reactions):raise ValueError('Morgan and raw dictionaries differ')
        self.requires_grad_(False).eval()

    def encode_enzymes(self,*args,**kwargs):return self.parent.encode_enzymes(*args,**kwargs)
    def encode_enzyme_index(self,*args,**kwargs):return self.parent.encode_enzyme_index(*args,**kwargs)
    score_index=staticmethod(GeneralizationDualEncoder.score_index)

    @torch.inference_mode()
    def replace_reaction_semantics(self,parent_reactions,blocks,masks,morgan):
        centers={key:getattr(self.parent.smooth,'reaction_center_'+key) for key in self.modalities}
        raw=reaction_features(blocks,centers,masks,self.modalities)
        semantic=self.anchor.encode_reactions(raw,morgan)
        return torch.cat((parent_reactions[:,:512],.5*semantic),1)

    @torch.inference_mode()
    def encode_reactions(self,base,blocks,masks,morgan,batch_size=512):
        original=self.parent.encode_reactions(base,blocks,masks,batch_size)
        return self.replace_reaction_semantics(original,blocks,masks,morgan)

    @classmethod
    def from_artifacts(cls,parent_record,state_record,device='cpu'):
        from .generalization_phase2 import ComposedPhase2Encoder
        parent,spec=ComposedPhase2Encoder.from_bundle(checked_artifact(parent_record),device)
        state=torch.load(checked_artifact(state_record),map_location=device,weights_only=False)
        if state['feature_manifest_sha256']!=spec['feature_manifest_sha256'] or state['split']!=spec['split']:
            raise ValueError('Morgan training state and parent split/manifest disagree')
        dictionary=torch.load(checked_artifact(spec['smooth_dictionary']),map_location='cpu',weights_only=False)
        if state['train_ids']!=dictionary['train_reaction_ids']:raise ValueError('Training dictionary ID/order mismatch')
        return cls(parent,state),spec
