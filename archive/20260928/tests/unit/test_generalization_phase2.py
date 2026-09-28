import pytest
import torch

from horizyn.generalization_phase2 import ComposedPhase2Encoder
from horizyn.generalization_retrieval import canonical_dot
from horizyn.semantic_smooth import SmoothAnchorDualEncoder


class IdentityDensity(torch.nn.Module):
    def encode_enzymes(self, base, means, batch_size):
        return torch.nn.functional.normalize(base,dim=1)

    def encode_reactions(self, base, blocks, masks, batch_size):
        return torch.nn.functional.normalize(base,dim=1)


@pytest.mark.parametrize('alpha',[0.,.25,.5,1.])
def test_composition_uses_raw_semantics_without_double_baseline(alpha):
    dictionary=dict(modalities=['t5v2'],protein_center=torch.zeros(3),
        train_proteins=torch.eye(3),adjacency=torch.arange(3).reshape(3,1),
        reaction_centers={'t5v2':torch.zeros(3)})
    # Deliberately different child mixture: parent must ignore it and use raw
    # semantic endpoints, not blend this already-mixed child's score again.
    smooth=SmoothAnchorDualEncoder(dictionary,torch.eye(3),dict(alpha=.9,
        protein_neighbors=1,enzyme_temperature=.03,kernel='exponential',
        reaction_neighbors=None,reaction_temperature=.1))
    model=ComposedPhase2Encoder(IdentityDensity(),smooth,alpha)
    base_e=torch.eye(3);base_r=torch.tensor([[0.,0.,1.]])
    means=torch.eye(3);blocks={'t5v2':torch.tensor([[1.,0.,0.]])};masks={'t5v2':torch.tensor([True])}
    e=model.encode_enzymes(base_e,means,batch_size=2)
    r=model.encode_reactions(base_r,blocks,masks,batch_size=1)
    semantic=smooth.encode_semantic_reactions(blocks,masks)
    expected=(1-alpha)*base_r+alpha*semantic
    torch.testing.assert_close(canonical_dot(r,e),expected,atol=1.2e-7,rtol=2e-7)
    torch.testing.assert_close(model.encode_enzymes(base_e[:1],means[:1],1),e[:1],rtol=0,atol=0)
    sparse=model.encode_enzyme_index(base_e,means,batch_size=2)
    torch.testing.assert_close(model.score_index(r,sparse),canonical_dot(r,e),rtol=0,atol=0)
    with pytest.raises(ValueError,match='Positive integer'):
        model.encode_enzyme_index(base_e,means,batch_size=-1)


@pytest.fixture
def pinned_phase2_bundle(tmp_path):
    import json
    from horizyn.generalization_retrieval import sha256
    def artifact(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value))
        return dict(path=str(path),sha256=sha256(path))
    original=artifact('original.json',{'frozen':True})
    checkpoint=artifact('checkpoint.json',{'weights':'fixed'})
    parent=artifact('parent.json',dict(feature_manifest_sha256='train',base_checkpoint=checkpoint))
    density42=artifact('density42.json',{'seed':42});density73=artifact('density73.json',{'seed':73})
    dictionary=artifact('anchors.json',{'source':'train'})
    config=dict(kernel='exponential',reaction_neighbors=None,reaction_temperature=.03,protein_neighbors=32,enzyme_temperature=.03)
    rows=[dict(split='reaction_smi',seed=seed,density_bundle=density,smooth_dictionary=dictionary,
        parent_bundle=parent,base_checkpoint=checkpoint,feature_manifest_sha256='train')
        for seed,density in [(42,density42),(73,density73)]]
    freeze=artifact('freeze.json',dict(schema='phase2_generalization_frozen_recipe_v1',
        frozen_before_new_external_evaluation=True,original_frozen_recipe=original,
        composition={'alpha':.25},smooth=config,approved_models=rows))
    spec=dict(schema='generalization_phase2_bundle_v1',variant='primary',alpha=.25,seed=42,
        split='reaction_smi',phase2_frozen_recipe=freeze,frozen_recipe=original,
        density_bundle=density42,smooth_dictionary=dictionary,parent_bundle=parent,
        base_checkpoint=checkpoint,feature_manifest_sha256='train',smooth_config=config)
    return spec,rows,tmp_path


def test_phase2_guards_accept_frozen_seed_tuple(pinned_phase2_bundle):
    from horizyn.generalization_phase2 import validate_phase2_bundle
    spec,rows,path=pinned_phase2_bundle
    assert validate_phase2_bundle(spec,path)['approved_models']==rows


def test_approved_seed73_cannot_be_mislabeled_primary(pinned_phase2_bundle):
    from horizyn.generalization_phase2 import validate_phase2_bundle
    spec,rows,path=pinned_phase2_bundle
    # Both artifacts are individually approved; the tuple binding must still
    # reject using the seed73 checkpoint while calling the result primary42.
    spec['density_bundle']=rows[1]['density_bundle']
    with pytest.raises(ValueError,match='split/seed/artifact tuple'):
        validate_phase2_bundle(spec,path)


def test_primary_cannot_change_seed_field(pinned_phase2_bundle):
    from horizyn.generalization_phase2 import validate_phase2_bundle
    spec,rows,path=pinned_phase2_bundle;spec['seed']=73
    with pytest.raises(ValueError,match='Variant label and residual seed'):
        validate_phase2_bundle(spec,path)


def test_seed_tuple_cannot_change_training_manifest(pinned_phase2_bundle):
    from horizyn.generalization_phase2 import validate_phase2_bundle
    spec,rows,path=pinned_phase2_bundle;spec['feature_manifest_sha256']='other-split'
    with pytest.raises(ValueError,match='split/seed/artifact tuple'):
        validate_phase2_bundle(spec,path)
