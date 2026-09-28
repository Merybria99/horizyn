"""Fixed independent composition of density-gated F3 and smooth anchors."""
from __future__ import annotations

import math
import json
from pathlib import Path
import torch

from .generalization_retrieval import checked_artifact, sha256, GeneralizationDualEncoder


def approved_artifacts(value):
    """Collect explicitly pinned path/hash records from a frozen recipe."""
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield (str(Path(value['path']).resolve()), value['sha256'])
        for item in value.values():
            yield from approved_artifacts(item)
    elif isinstance(value, list):
        for item in value:
            yield from approved_artifacts(item)


def validate_phase2_bundle(specification, parent):
    if specification.get('schema') != 'generalization_phase2_bundle_v1':
        raise ValueError('Unrecognized phase-two bundle schema')
    freeze_path = checked_artifact(specification['phase2_frozen_recipe'], parent)
    frozen = json.loads(freeze_path.read_text())
    if frozen.get('schema') != 'phase2_generalization_frozen_recipe_v1' or not frozen.get('frozen_before_new_external_evaluation'):
        raise ValueError('Phase-two recipe has not been frozen')
    old_path = checked_artifact(specification['frozen_recipe'], parent)
    if frozen['original_frozen_recipe']['sha256'] != sha256(old_path):
        raise ValueError('Phase-two and input-feature freeze identities disagree')
    allowed = set(approved_artifacts(frozen))
    for key in ('density_bundle', 'smooth_dictionary'):
        path = checked_artifact(specification[key], parent)
        if (str(path.resolve()), specification[key]['sha256']) not in allowed:
            raise ValueError(f'Artifact is not approved by the phase-two freeze: {key}')
    if specification['smooth_config'] != frozen['smooth']:
        raise ValueError('Smooth kernel configuration differs from the frozen recipe')
    variant = specification['variant']
    alpha = {'density_only':0., 'smooth_only':1.}.get(variant, frozen['composition']['alpha'])
    if variant not in ('primary', 'seed17', 'seed73', 'density_only', 'smooth_only') or specification['alpha'] != alpha:
        raise ValueError('Composition variant/weight differs from frozen recipe')
    expected_seed = {'seed17':17, 'seed73':73}.get(variant,42)
    if specification.get('seed') != expected_seed:
        raise ValueError('Variant label and residual seed disagree')
    def same_record(left,right):
        return Path(left['path']).resolve()==Path(right['path']).resolve() and left['sha256']==right['sha256']
    matching = [row for row in frozen['approved_models']
        if row['split']==specification['split'] and row['seed']==expected_seed
        and row['feature_manifest_sha256']==specification['feature_manifest_sha256']
        and all(same_record(row[key],specification[key]) for key in
                ('density_bundle','smooth_dictionary','parent_bundle','base_checkpoint'))]
    if len(matching)!=1:
        raise ValueError('Bundle does not match its frozen split/seed/artifact tuple')
    parent_path = checked_artifact(specification['parent_bundle'],parent)
    parent_spec = json.loads(parent_path.read_text())
    if parent_spec['feature_manifest_sha256']!=specification['feature_manifest_sha256'] or Path(parent_spec['base_checkpoint']['path']).resolve()!=Path(specification['base_checkpoint']['path']).resolve():
        raise ValueError('Parent bundle lineage and phase-two bundle disagree')
    return frozen


class ComposedPhase2Encoder(torch.nn.Module):
    def __init__(self, density, smooth, alpha):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError('Mixture alpha must lie in [0, 1]')
        self.density, self.smooth, self.alpha = density, smooth, float(alpha)
        self.modalities = list(smooth.modalities)
        self.requires_grad_(False).eval()

    def _compose(self, learned, semantic):
        # Keep actual endpoint concatenation even for weights 0 and 1, matching
        # the validation score contract and avoiding a new GEMM shape convention.
        return torch.cat((math.sqrt(1-self.alpha)*learned,
                          math.sqrt(self.alpha)*semantic),dim=1)

    @torch.inference_mode()
    def encode_enzymes(self, base, protein_mean, batch_size=512, return_diagnostics=False):
        learned = self.density.encode_enzymes(base,protein_mean,batch_size,return_diagnostics=return_diagnostics) if return_diagnostics else self.density.encode_enzymes(base,protein_mean,batch_size)
        if return_diagnostics:
            learned, diagnostics = learned
        semantic = self.smooth.encode_semantic_enzymes(protein_mean,batch_size)
        result = self._compose(learned,semantic)
        return (result,diagnostics) if return_diagnostics else result

    @torch.inference_mode()
    def encode_reactions(self, base, blocks, masks, batch_size=512, return_diagnostics=False):
        learned = self.density.encode_reactions(base,blocks,masks,batch_size,return_diagnostics=return_diagnostics) if return_diagnostics else self.density.encode_reactions(base,blocks,masks,batch_size)
        if return_diagnostics:
            learned, diagnostics = learned
        semantic = self.smooth.encode_semantic_reactions(blocks,masks)
        result = self._compose(learned,semantic)
        return (result,diagnostics) if return_diagnostics else result

    @torch.inference_mode()
    def encode_enzyme_index(self, base, protein_mean, batch_size=512):
        """Precompute a dense learned block and reusable sparse semantic block."""
        if not isinstance(batch_size,int) or batch_size<=0 or len(base)!=len(protein_mean):
            raise ValueError('Positive integer batch size and aligned protein rows required')
        learned = math.sqrt(1-self.alpha)*self.density.encode_enzymes(base,protein_mean,batch_size)
        pointers=[torch.zeros(1,dtype=torch.int64,device=base.device)]
        columns,weights,total=[],[],0
        for start in range(0,len(base),batch_size):
            semantic=self.smooth.encode_semantic_enzymes(protein_mean[start:start+batch_size],batch_size)
            sparse=(math.sqrt(self.alpha)*semantic).to_sparse_csr()
            pointers.append(sparse.crow_indices()[1:]+total)
            columns.append(sparse.col_indices());weights.append(sparse.values());total+=sparse.values().numel()
        column=torch.cat(columns) if columns else torch.empty(0,dtype=torch.int64,device=base.device)
        weight=torch.cat(weights) if weights else torch.empty(0,dtype=base.dtype,device=base.device)
        sparse=torch.sparse_csr_tensor(torch.cat(pointers),column,weight,
            size=(len(base),len(self.smooth.train_reactions)),device=base.device)
        return dict(dense=learned,anchors=sparse)

    score_index = staticmethod(GeneralizationDualEncoder.score_index)

    @classmethod
    def from_bundle(cls, path, device='cpu'):
        from .generalization_density import DensityGatedEncoder
        from .semantic_smooth import SmoothAnchorDualEncoder
        path = Path(path)
        specification = json.loads(path.read_text())
        validate_phase2_bundle(specification,path.parent)
        density_path = checked_artifact(specification['density_bundle'],path.parent)
        density, state = DensityGatedEncoder.from_bundle(density_path,device=device)
        dictionary = torch.load(checked_artifact(specification['smooth_dictionary'],path.parent),map_location=device,weights_only=False)
        if dictionary['feature_manifest_sha256'] != specification['feature_manifest_sha256']:
            raise ValueError('Smooth dictionary and bundle training manifests disagree')
        density_spec = json.loads(density_path.read_text())
        if density_spec['feature_manifest_sha256'] != specification['feature_manifest_sha256']:
            raise ValueError('Density and smooth training manifests disagree')
        smooth = SmoothAnchorDualEncoder(dictionary,dictionary['train_reactions'],
            dict(specification['smooth_config'],alpha=1.)).to(device)
        return cls(density,smooth,specification['alpha']).to(device), specification
