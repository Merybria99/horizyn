import math
import pytest
import torch
from horizyn.generalization_raw_residual import RawFeatureResidual
from horizyn.generalization_retrieval import canonical_dot
from horizyn.semantic_anchors import row_unit


def fixture(cap=.1):
    torch.manual_seed(42)
    model=RawFeatureResidual(7,9,cap=cap).double()
    native=row_unit(torch.randn(11,512));raw=row_unit(torch.randn(11,7))
    parent=math.sqrt(.75)*row_unit(torch.randn(11,512))
    return model,native,raw,parent


def test_explicit_identity_delegates_exact_weighted_parent():
    model,native,raw,parent=fixture()
    assert model.encode_enzymes(native,raw,parent,identity=True) is parent
    out=model.encode_enzymes(native,raw,parent)
    torch.testing.assert_close(out.double().norm(dim=1),torch.full((11,),math.sqrt(.75),dtype=torch.float64),atol=1e-7,rtol=0)


@pytest.mark.parametrize('cap',[.1,.3])
def test_cap_every_row_and_angular_bound(cap):
    model,native,raw,parent=fixture(cap)
    with torch.no_grad():model.enzyme[-1].weight.normal_(std=20);model.enzyme[-1].bias.normal_(std=20)
    out,diag=model.encode_enzymes(native,raw,parent,return_diagnostics=True)
    assert bool((diag['relative_residual']<=cap+1e-12).all())
    cosine=(row_unit(out).double()*row_unit(parent).double()).sum(1)
    assert bool((cosine>=math.sqrt(1-cap**2)-1e-7).all())


def test_input_batch_subset_singleton_and_sparse_score_parity():
    model,native,raw,parent=fixture()
    with torch.no_grad():model.enzyme[-1].weight.normal_(std=.1)
    output=model.encode_enzymes(native,raw,parent)
    assert torch.equal(output[::3],model.encode_enzymes(native[::3],raw[::3],parent[::3]))
    assert torch.equal(output[:1],model.encode_enzymes(native[:1],raw[:1],parent[:1]))
    semantic=torch.zeros(11,17);semantic[:,3]=.5
    full=model.compose(output,semantic)
    assert torch.equal(canonical_dot(full,full),model.score_index(full,{'dense':output,'anchors':semantic.to_sparse_csr()}))


def test_zero_final_initialization_keeps_first_update_gradient():
    model,native,raw,parent=fixture();model=model.float()
    loss=model.encode_enzymes(native,raw,parent)[:,0].sum();loss.backward()
    assert model.enzyme[-1].weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.enzyme.parameters())


@pytest.mark.parametrize('bad',['nan','dimension','zero_parent'])
def test_invalid_inputs_fail_closed(bad):
    model,native,raw,parent=fixture()
    if bad=='nan':raw[0,0]=float('nan')
    elif bad=='dimension':raw=raw[:,:-1]
    else:parent[0]=0
    with pytest.raises(ValueError):model.encode_enzymes(native,raw,parent)
