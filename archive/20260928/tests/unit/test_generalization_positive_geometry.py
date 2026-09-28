import math
import pytest
import torch
from horizyn.generalization_positive_geometry import (PositiveGeometryResidual,
    edge_marginals,weighted_moment_penalty,positive_geometry_loss)


def test_reaction_balanced_edge_and_enzyme_weights():
    r=torch.tensor([0,0,1]);e=torch.tensor([0,1,1])
    edge,marginal=edge_marginals(r,e,2,2)
    torch.testing.assert_close(edge,torch.tensor([.25,.25,.5]))
    torch.testing.assert_close(marginal,torch.tensor([.25,.75]))


def test_moment_scale_is_zero_at_isotropy_and_two_at_collapse():
    z=torch.cat((torch.eye(3),-torch.eye(3)))
    weights=torch.ones(6)/6
    value,mean,cov=weighted_moment_penalty(z,weights)
    assert value.item()==0
    collapsed=z[:1].repeat(6,1)
    value,mean,cov=weighted_moment_penalty(collapsed,weights)
    torch.testing.assert_close(value,torch.tensor(2.))
    torch.testing.assert_close(mean,torch.tensor(1.))
    torch.testing.assert_close(cov,torch.tensor(1.))


@pytest.mark.parametrize('cap',[.1,.3])
def test_pointwise_cap_implies_uniform_angular_bound_and_stable_inference(cap):
    torch.manual_seed(2)
    model=PositiveGeometryResidual(4,8,cap=cap).double()
    torch.nn.init.normal_(model.enzyme[-1].weight,0,100)
    torch.nn.init.normal_(model.reaction[-1].weight,0,100)
    raw=torch.randn(17,4)*torch.logspace(-20,3,17)[:,None]
    for name in ['encode_enzymes','encode_reactions']:
        encoded=getattr(model,name)(raw)
        identity=torch.nn.functional.normalize(raw.double(),dim=1,eps=torch.finfo(torch.float64).tiny).float()
        cosine=(identity*encoded).sum(1)
        assert bool((cosine>=math.sqrt(1-cap*cap)-2e-7).all())
        torch.testing.assert_close(encoded[:1],getattr(model,name)(raw[:1]),atol=0,rtol=0)


def test_positive_loss_uses_only_declared_edges_and_is_differentiable():
    reactions=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
    enzymes=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
    r=e=torch.arange(2);edge,ew=edge_marginals(r,e,2,2)
    loss,terms=positive_geometry_loss(reactions,enzymes,r,e,edge,ew,reactions.detach(),enzymes.detach(),0.,0.)
    assert loss.item()==0 and terms['positive'].item()==0
    loss.backward()
    torch.testing.assert_close(reactions.grad,-.5*torch.eye(2))
    torch.testing.assert_close(enzymes.grad,-.5*torch.eye(2))
