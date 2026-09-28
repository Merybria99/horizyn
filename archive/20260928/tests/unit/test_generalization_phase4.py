import hashlib,json
from pathlib import Path
import pytest,torch
from horizyn.generalization_phase4 import HybridPhase4Encoder,validate_phase4_bundle
from horizyn.generalization_retrieval import canonical_dot
from horizyn.semantic_anchors import row_unit
from horizyn.semantic_smooth import SmoothAnchorDualEncoder


class Density(torch.nn.Module):
    def encode_enzymes(self,base,raw,batch_size=512,return_diagnostics=False):
        return row_unit(base)
    def encode_reactions(self,base,blocks,masks,batch_size=512,return_diagnostics=False):
        return row_unit(base)


@pytest.mark.parametrize('alpha',[.25,1.])
def test_independent_hybrid_endpoint_subsets_and_sparse_index_parity(alpha):
    torch.manual_seed(4)
    train=row_unit(torch.randn(5,3));native=row_unit(torch.randn(5,512))
    dictionary=dict(modalities=['x'],protein_center=torch.zeros(3),train_proteins=train,
        adjacency=torch.arange(5)[:,None],reaction_centers={'x':torch.zeros(3)})
    config=dict(alpha=1.,protein_neighbors=2,enzyme_temperature=.03,kernel='exponential',reaction_temperature=.03,reaction_neighbors=None)
    smooth=SmoothAnchorDualEncoder(dictionary,train,config)
    model=HybridPhase4Encoder(Density(),smooth,native,alpha)
    e=torch.randn(11,512);r=torch.randn(7,512);means=torch.randn(11,3)
    blocks={'x':torch.randn(7,3)};masks={'x':torch.ones(7,dtype=torch.bool)}
    encoded_e=model.encode_enzymes(e,means);encoded_r=model.encode_reactions(r,blocks,masks)
    assert torch.equal(encoded_r[:1],model.encode_reactions(r[:1],{'x':blocks['x'][:1]},{'x':masks['x'][:1]}))
    assert torch.equal(encoded_e[:1],model.encode_enzymes(e[:1],means[:1]))
    index=model.encode_enzyme_index(e,means,batch_size=3)
    assert torch.equal(model.score_index(encoded_r,index),canonical_dot(encoded_r,encoded_e))


def record(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def bundle_fixture(tmp_path):
    def file(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return record(p)
    original=file('original.json',{});phase2freeze=file('phase2freeze.json',{});checkpoint=file('base.pt',{});native=file('native.pt',{})
    previous=file('previous.json',dict(split='reaction_smi',seed=42,feature_manifest_sha256='training',base_checkpoint=checkpoint,phase2_frozen_recipe=phase2freeze,frozen_recipe=original))
    row=dict(split='reaction_smi',seed=42,feature_manifest_sha256='training',phase2_bundle=previous,native_reaction_state=native,base_checkpoint=checkpoint)
    freeze=file('freeze.json',dict(schema='phase4_hybrid_frozen_recipe_v1',frozen_before_new_external_evaluation=True,original_frozen_recipe=original,
        phase2_frozen_recipe=phase2freeze,composition=dict(alpha=.25,eta_enzyme=0.,eta_reaction=.5),approved_models=[row]))
    return dict(schema='phase4_hybrid_bundle_v1',variant='primary',alpha=.25,eta_enzyme=0.,eta_reaction=.5,
        phase4_frozen_recipe=freeze,phase2_frozen_recipe=phase2freeze,frozen_recipe=original,**row)


def test_valid_bundle_exact_tuple_and_seed_guard(tmp_path):
    spec=bundle_fixture(tmp_path);validate_phase4_bundle(spec,tmp_path)
    spec['seed']=73
    with pytest.raises(ValueError,match='seed disagree'):validate_phase4_bundle(spec,tmp_path)


def test_alternative_geometry_or_swapped_training_lineage_rejected(tmp_path):
    spec=bundle_fixture(tmp_path);spec['eta_reaction']=1.
    with pytest.raises(ValueError,match='frozen recipe'):validate_phase4_bundle(spec,tmp_path)
    spec['eta_reaction']=.5;spec['feature_manifest_sha256']='other training split'
    with pytest.raises(ValueError,match='artifact tuple'):validate_phase4_bundle(spec,tmp_path)


def test_native_dictionary_requires_correct_shape_and_units():
    class Smooth(torch.nn.Module):
        modalities=['x'];train_reactions=torch.zeros(3,2)
    with pytest.raises(ValueError,match='shape mismatch'):HybridPhase4Encoder(Density(),Smooth(),torch.ones(2,512))
    with pytest.raises(ValueError,match='unit endpoints'):HybridPhase4Encoder(Density(),Smooth(),torch.ones(3,512))
