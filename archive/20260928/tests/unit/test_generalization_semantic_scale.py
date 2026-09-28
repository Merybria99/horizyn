import numpy as np
import pytest
import torch
from horizyn.generalization_semantic_scale import QuerySemanticScale
from horizyn.generalization_retrieval import canonical_dot

def example():
 torch.manual_seed(81)
 bank=torch.randn(32,13);bank[:,5:]*=.01
 r=torch.randn(17,13);e=torch.randn(23,13)
 return bank,bank[:,5:].double().mean(0),r,e

def test_identity_delegates_exact_parent():
 bank,mean,r,e=example();m=QuerySemanticScale(bank,mean,dense_width=5)
 assert m.encode_reactions(r) is r and m.encode_enzymes(e) is e
 assert torch.equal(canonical_dot(r,e),m.score_index(r,m.encode_enzyme_index(e)))

def test_fixed_bank_population_sd_matches_numpy():
 bank,mean,r,e=example();m=QuerySemanticScale(bank,mean,.5,100.,dense_width=5)
 d,s=m.statistics(r,4);bd=r[:,:5].double().numpy()@bank[:,:5].double().numpy().T;bs=r[:,5:].double().numpy()@bank[:,5:].double().numpy().T
 np.testing.assert_allclose(d.numpy(),bd.std(axis=1,ddof=0),rtol=1e-14)
 np.testing.assert_allclose(s.numpy(),bs.std(axis=1,ddof=0),rtol=1e-14)
 rr,diag=m.encode_reactions(r,return_diagnostics=True)
 assert bool(((diag['multiplier']>=1)&(diag['multiplier']<=100)).all())
 assert torch.equal(rr[:,:5],r[:,:5])
 assert torch.equal(rr[:,-1],(-(diag['multiplier']-1)*(r[:,5:].double()*mean).sum(1)).float())

def test_query_batch_candidate_subset_singleton_and_sparse_scores():
 bank,mean,r,e=example();m=QuerySemanticScale(bank,mean,.25,10.,dense_width=5)
 rr=m.encode_reactions(r,7);ee=m.encode_enzymes(e)
 assert torch.equal(rr[::3],m.encode_reactions(r[::3],2))
 assert torch.equal(rr[:1],m.encode_reactions(r[:1],1))
 assert torch.equal(ee[::4],m.encode_enzymes(e[::4]))
 score=canonical_dot(rr,ee)
 assert torch.equal(score[::3,::4],canonical_dot(m.encode_reactions(r[::3],2),m.encode_enzymes(e[::4])))
 assert torch.equal(score,m.score_index(rr,m.encode_enzyme_index(e)))

def test_centering_preserves_exact_train_mean_for_integer_example():
 bank=torch.tensor([[0.,0.,1.,0.],[8.,0.,0.,1.],[0.,8.,1.,0.],[8.,8.,0.,1.]])
 mu=bank[:,2:].double().mean(0);m=QuerySemanticScale(bank,mu,1.,10.,dense_width=2)
 r=torch.tensor([[1.,1.,1.,0.]])
 before=canonical_dot(r,bank).double().mean();after=canonical_dot(m.encode_reactions(r),m.encode_enzymes(bank)).double().mean()
 assert before==after

@pytest.mark.parametrize('beta,cap',[(float('nan'),10.),(.1,float('inf')),(-1.,10.),(.1,.5)])
def test_invalid_scale_parameters_rejected(beta,cap):
 bank,mean,_,_=example()
 with pytest.raises(ValueError):QuerySemanticScale(bank,mean,beta,cap,dense_width=5)
