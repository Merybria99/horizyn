import pytest
import torch
from horizyn.generalization_centered_affinity import CenteredAffinity
from horizyn.generalization_retrieval import canonical_dot

def fixture():
 torch.manual_seed(9)
 return torch.randn(13,19),torch.randn(31,19),torch.randn(19,dtype=torch.float64),torch.randn(19,dtype=torch.float64),torch.rand(13)

def test_zero_is_exact_parent_and_no_dimensions_added():
 r,e,mr,me,g=fixture();m=CenteredAffinity(mr,me,0)
 assert m.encode_reactions(r,g) is r and m.encode_enzymes(e) is e
 assert torch.equal(canonical_dot(r,e),canonical_dot(m.encode_reactions(r,g),m.encode_enzymes(e)))

def test_actual_augmented_coordinates_and_subset_scores():
 r,e,mr,me,g=fixture();m=CenteredAffinity(mr,me,.5)
 rr,ee=m.encode_reactions(r,g),m.encode_enzymes(e)
 assert torch.equal(rr[:,-1],(-.5*g.double()).float())
 assert torch.equal(ee[:,-1],((e.double()-me)*mr).sum(1).float())
 assert torch.equal(rr[::3],m.encode_reactions(r[::3],g[::3]))
 assert torch.equal(ee[::4],m.encode_enzymes(e[::4]))
 assert torch.equal(canonical_dot(rr,ee)[::3,::4],canonical_dot(rr[::3],ee[::4]))
 assert torch.equal(canonical_dot(rr,ee)[:1,:1],canonical_dot(rr[:1],ee[:1]))

@pytest.mark.parametrize('gamma',[float('nan'),float('inf'),-1.])
def test_invalid_gamma(gamma):
 with pytest.raises(ValueError):CenteredAffinity(torch.ones(3),torch.ones(3),gamma)

def test_mean_center_removes_mean_coordinate():
 mr=torch.tensor([.25,.5,.75],dtype=torch.float64);me=torch.tensor([1.,2.,3.],dtype=torch.float64)
 m=CenteredAffinity(mr,me,1.)
 assert m.encode_enzymes(me.float()[None,:])[0,-1]==0
