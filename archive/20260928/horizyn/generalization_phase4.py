"""Frozen phase-four hybrid-reaction evidence with independent endpoints."""
from __future__ import annotations
import json
from pathlib import Path
import torch

from .generalization_phase2 import ComposedPhase2Encoder
from .generalization_retrieval import checked_artifact,sha256
from .semantic_anchors import row_unit,reaction_features
from .semantic_hybrid import hybrid_view
from .semantic_smooth import reaction_responses


def same_record(left,right):
    return Path(left['path']).resolve()==Path(right['path']).resolve() and left['sha256']==right['sha256']


def validate_phase4_bundle(spec,parent):
    if spec.get('schema')!='phase4_hybrid_bundle_v1':raise ValueError('Unknown phase-four bundle schema')
    freeze_path=checked_artifact(spec['phase4_frozen_recipe'],parent)
    freeze=json.loads(freeze_path.read_text())
    if freeze.get('schema')!='phase4_hybrid_frozen_recipe_v1' or not freeze.get('frozen_before_new_external_evaluation'):
        raise ValueError('Phase-four recipe is not frozen')
    for key,expected in [('frozen_recipe','original_frozen_recipe'),('phase2_frozen_recipe','phase2_frozen_recipe')]:
        checked_artifact(spec[key],parent)
        if not same_record(spec[key],freeze[expected]):raise ValueError('Input/parent freeze lineage mismatch')
    variant=spec['variant'];expected_seed={'primary':42,'seed17':17,'seed73':73,'hybrid_anchor_only':42}.get(variant)
    expected_alpha=1. if variant=='hybrid_anchor_only' else .25
    if expected_seed is None or spec['seed']!=expected_seed:raise ValueError('Variant label and residual seed disagree')
    if spec['alpha']!=expected_alpha or spec['eta_enzyme']!=0. or spec['eta_reaction']!=.5:
        raise ValueError('Composition or hybrid geometry is not the frozen recipe')
    if freeze['composition']!=dict(alpha=.25,eta_enzyme=0.,eta_reaction=.5):raise ValueError('Unexpected frozen geometry')
    matches=[row for row in freeze['approved_models'] if row['split']==spec['split'] and row['seed']==expected_seed
        and row['feature_manifest_sha256']==spec['feature_manifest_sha256']
        and all(same_record(row[k],spec[k]) for k in ['phase2_bundle','native_reaction_state','base_checkpoint'])]
    if len(matches)!=1:raise ValueError('Bundle does not match its frozen split/seed/artifact tuple')
    for key in ['phase2_bundle','native_reaction_state','base_checkpoint']:checked_artifact(spec[key],parent)
    previous=json.loads(checked_artifact(spec['phase2_bundle'],parent).read_text())
    if previous['split']!=spec['split'] or previous['seed']!=spec['seed'] or previous['feature_manifest_sha256']!=spec['feature_manifest_sha256']:
        raise ValueError('Frozen phase-two parent has inconsistent split/seed/features')
    if not same_record(previous['base_checkpoint'],spec['base_checkpoint']):raise ValueError('Native F3 checkpoint lineage mismatch')
    if not same_record(previous['phase2_frozen_recipe'],spec['phase2_frozen_recipe']) or not same_record(previous['frozen_recipe'],spec['frozen_recipe']):
        raise ValueError('Parent freeze identity mismatch')
    return freeze


class HybridPhase4Encoder(ComposedPhase2Encoder):
    """Unchanged enzyme map; reaction neighbors use half raw, half native F3."""
    def __init__(self,density,smooth,native_train_reactions,alpha=.25):
        super().__init__(density,smooth,alpha)
        if alpha not in (.25,1.):raise ValueError('Unregistered phase-four composition weight')
        if native_train_reactions.shape!=(len(smooth.train_reactions),512):raise ValueError('Native reaction dictionary shape mismatch')
        if not bool(torch.isfinite(native_train_reactions).all()):raise ValueError('Nonfinite native reaction dictionary')
        if not torch.allclose(native_train_reactions.double().norm(dim=1),torch.ones(len(native_train_reactions),device=native_train_reactions.device,dtype=torch.float64),atol=1e-6,rtol=0):
            raise ValueError('Native reaction dictionary must contain unit endpoints')
        self.register_buffer('native_train_reactions',native_train_reactions)

    @torch.inference_mode()
    def encode_reactions(self,base,blocks,masks,batch_size=512,return_diagnostics=False):
        learned=self.density.encode_reactions(base,blocks,masks,batch_size,return_diagnostics=return_diagnostics)
        if return_diagnostics:learned,diagnostics=learned
        centers={key:getattr(self.smooth,'reaction_center_'+key) for key in self.modalities}
        raw=reaction_features(blocks,centers,masks,self.modalities)
        if len(base)!=len(raw):raise ValueError('Native and raw reaction rows disagree')
        query=hybrid_view(raw,row_unit(base),.5)
        anchors=hybrid_view(self.smooth.train_reactions,self.native_train_reactions,.5)
        config=self.smooth.config
        semantic=reaction_responses(query,anchors,config['kernel'],config['reaction_temperature'],config['reaction_neighbors'])
        result=self._compose(learned,semantic)
        return (result,diagnostics) if return_diagnostics else result

    @classmethod
    def from_bundle(cls,path,device='cpu'):
        path=Path(path);spec=json.loads(path.read_text());validate_phase4_bundle(spec,path.parent)
        previous,previous_spec=ComposedPhase2Encoder.from_bundle(checked_artifact(spec['phase2_bundle'],path.parent),device)
        state=torch.load(checked_artifact(spec['native_reaction_state'],path.parent),map_location=device,weights_only=False)
        if state.get('schema')!='phase4_native_reaction_state_v1' or state['split']!=spec['split'] or state['feature_manifest_sha256']!=spec['feature_manifest_sha256']:
            raise ValueError('Native reaction fit-state lineage mismatch')
        if not same_record(state['base_checkpoint'],spec['base_checkpoint']):raise ValueError('Native reaction state checkpoint mismatch')
        for item in state['sources'].values():checked_artifact(item)
        dictionary=torch.load(checked_artifact(previous_spec['smooth_dictionary']),map_location='cpu',weights_only=False)
        if state['train_reaction_ids']!=dictionary['train_reaction_ids']:raise ValueError('Native/raw reaction dictionary ID order mismatch')
        return cls(previous.density,previous.smooth,state['native_train_reactions'],spec['alpha']).to(device),spec
