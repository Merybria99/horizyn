import pytest
import torch
from horizyn.semantic_anchors import row_unit
from horizyn.semantic_hybrid import hybrid_view,HybridAnchorDualEncoder
from horizyn.semantic_smooth import SmoothAnchorDualEncoder


def fixture_model(eta_e,eta_r):
    raw=row_unit(torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,0.,0.]]))
    reaction=raw[:3]
    dictionary=dict(modalities=['x'],protein_center=torch.zeros(3),train_proteins=raw,
        adjacency=torch.tensor([[0],[1],[2],[2]]),reaction_centers={'x':torch.zeros(3)})
    config=dict(alpha=.25,protein_neighbors=1,enzyme_temperature=.03,kernel='exponential',reaction_temperature=.03,reaction_neighbors=None)
    native=row_unit(torch.tensor([[0.,1.],[1.,0.],[-1.,0.],[0.,-1.]]))
    return HybridAnchorDualEncoder(dictionary,reaction,config,native,native[:3],eta_e,eta_r),dictionary,reaction,config


def test_zero_weights_preserve_original_endpoint_paths_exactly():
    model,dictionary,reaction,config=fixture_model(0,0)
    original=SmoothAnchorDualEncoder(dictionary,reaction,config)
    base=torch.tensor([[1.,0.],[0.,1.]])
    raw=torch.tensor([[1.,.2,0.],[0.,1.,.2]])
    blocks={'x':raw};masks={'x':torch.ones(2,dtype=torch.bool)}
    assert torch.equal(model.hybrid_enzyme_features(base,raw),original.encode_semantic_enzymes(raw))
    assert torch.equal(model.hybrid_reaction_features(base,blocks,masks),original.encode_semantic_reactions(blocks,masks))


def test_native_view_changes_neighbor_evidence_and_ignores_raw_at_one():
    raw=torch.tensor([[1.,0.,0.]])
    base=torch.tensor([[1.,0.]])
    rawmodel,*_=fixture_model(0,0);native,*_=fixture_model(1,1)
    assert rawmodel.hybrid_enzyme_features(base,raw).argmax().item()==0
    assert native.hybrid_enzyme_features(base,raw).argmax().item()==1
    assert torch.equal(native.hybrid_enzyme_features(base,raw),native.hybrid_enzyme_features(base,raw.roll(1,1)))
    with pytest.raises(RuntimeError,match='explicitly compose'):
        native.encode_enzymes(base,raw)


@pytest.mark.parametrize('eta',[0.,.5,1.])
def test_hybrid_geometry_is_fixed_independent_weighted_cosine(eta):
    torch.manual_seed(3);raw=row_unit(torch.randn(7,4));native=row_unit(torch.randn(7,3))
    z=hybrid_view(raw,native,eta)
    expected=(1-eta)*(raw.double()@raw.double().T)+eta*(native.double()@native.double().T)
    torch.testing.assert_close(z.double()@z.double().T,expected,atol=1e-7,rtol=1e-7)
    assert torch.equal(z[[0,4]],hybrid_view(raw[[0,4]],native[[0,4]],eta))
