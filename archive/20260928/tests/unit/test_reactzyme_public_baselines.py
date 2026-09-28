"""Fidelity checks for matched-data public baseline adapters."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from reactzyme_public_train import classification_pairs,fast_scores,original_scores,upstream_classes
from generalization_metrics import evaluate_scores


@pytest.mark.parametrize('family',['mlp','contrastive','contrastive_corrected','transformer','birnn'])
def test_cached_pair_forward_matches_upstream(family):
    torch.manual_seed(27);torch.set_num_threads(2)
    classes,_=upstream_classes(family)
    model=classes['PretrainedNetwork'](mol_input_dim=32,seq_input_dim=48,
        hidden_dim=128,output_dim=64,dropout=0.).eval()
    r=torch.randn(5,32);p=torch.randn(7,48)
    expected=original_scores(model,r,p,batch=11)
    actual=fast_scores(model,r,p,family,batch=13)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)


def test_negative_sampling_filters_known_edges_and_is_repeatable():
    positives=np.array([[0,0],[1,1],[2,2]])
    known=np.concatenate([positives,[[0,1]]])
    pairs,labels=classification_pairs(positives,known,42)
    repeat,repeat_labels=classification_pairs(positives,known,42)
    np.testing.assert_array_equal(pairs,repeat);np.testing.assert_array_equal(labels,repeat_labels)
    np.testing.assert_array_equal(pairs[:3],positives)
    assert labels.sum()==3 and len(pairs)==15
    assert not (set(map(tuple,pairs[labels==0])) & set(map(tuple,known)))


def test_impossible_negatives_raise_instead_of_hanging():
    with pytest.raises(ValueError,match='No valid'):
        classification_pairs(np.array([[0,0]]),np.array([[0,0]]),42)


def test_shared_metrics_preserve_all_positive_query_average():
    scores=torch.tensor([[.9,.8,.1],[.2,.5,.4]])
    truth={'reaction_index':np.array([0,0,1]),'enzyme_index':np.array([0,1,2])}
    result=evaluate_scores(scores,truth)['summary']
    assert result['reaction_to_enzyme']['all']['reactzyme_mrr']==pytest.approx(.625)
    assert result['enzyme_to_reaction']['all']['reactzyme_mrr']==pytest.approx(1.)


@pytest.mark.parametrize('device',['cpu']+(['cuda:0'] if torch.cuda.is_available() else []))
@pytest.mark.parametrize('checkpoint_ffn',[False,True])
def test_t5_sdpa_preserves_relative_bias_mask_and_gradients(device,checkpoint_ffn):
    from transformers import T5Config,T5EncoderModel
    from reactzyme_public_t5_sdpa import enable_t5_sdpa,enable_t5_ffn_checkpointing
    torch.manual_seed(13);torch.set_num_threads(2)
    a=T5EncoderModel(T5Config(d_model=32,d_kv=8,d_ff=64,num_layers=2,
        num_heads=4,vocab_size=30,dropout_rate=0)).to(device)
    b=copy.deepcopy(a);enable_t5_sdpa(b)
    if checkpoint_ffn:enable_t5_ffn_checkpointing(b)
    x=torch.randint(0,30,(3,17),device=device);mask=torch.ones_like(x);mask[0,9:]=0
    p=a(x,attention_mask=mask).last_hidden_state;q=b(x,attention_mask=mask).last_hidden_state
    torch.testing.assert_close(p,q,atol=3e-6,rtol=3e-5)
    weights=torch.randn_like(p);(p*weights).sum().backward();(q*weights).sum().backward()
    for (name,v),(other,w) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other
        if v.grad is not None:torch.testing.assert_close(v.grad,w.grad,atol=5e-5,rtol=5e-4)
