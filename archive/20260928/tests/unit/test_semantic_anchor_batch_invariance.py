"""Raw semantic endpoints must not depend on inference batch shape."""
import pytest
import torch

from horizyn.semantic_anchors import (
    centered_unit, fit_center, reaction_features, reaction_anchor_features,
    nearest_training_proteins, enzyme_anchor_features, make_protein_reaction_map,
)

DEVICES = ["cpu"] + (["cuda:1"] if torch.cuda.device_count() > 1 else [])


@pytest.mark.parametrize("device", DEVICES)
def test_complete_anchor_endpoints_match_singleton_and_uneven_batches(device):
    generator = torch.Generator().manual_seed(81)
    raw = torch.randn(47,1024,generator=generator).to(device)
    raw[1] = raw[0]
    raw[2] = raw[0] + 1e-7
    tr = torch.arange(31,device=device)
    center = fit_center(raw,tr)
    all_encoded = centered_unit(raw,center)
    one_encoded = torch.cat([centered_unit(raw[i:i+1],center) for i in range(len(raw))])
    assert torch.equal(all_encoded,one_encoded)
    anchors = all_encoded[tr]
    labels = make_protein_reaction_map(tr % 13,tr,len(tr))
    values,indices = nearest_training_proteins(all_encoded,anchors,top_k=16,batch_size=47)
    expected_e = enzyme_anchor_features(values,indices,labels,13,.03)
    pieces = []
    for start in range(0,len(raw),7):
        encoded = centered_unit(raw[start:start+7],center)
        values,indices = nearest_training_proteins(encoded,anchors,top_k=16,batch_size=1)
        pieces.append(enzyme_anchor_features(values,indices,labels,13,.03))
    assert torch.equal(expected_e,torch.cat(pieces))

    blocks = {key:torch.randn(51,dim,generator=generator).to(device)
              for key,dim in zip(("t5v2","unimol2","chiro","chemistry"),(768,768,256,617))}
    masks = {key:torch.ones(51,dtype=torch.bool,device=device) for key in blocks}
    masks["chiro"][::9] = False
    centers = {key:fit_center(value,torch.arange(32,device=device),masks[key]) for key,value in blocks.items()}
    full = reaction_features(blocks,centers,masks,list(blocks))
    expected_q = reaction_anchor_features(full,full[:32],.03,top_k=16)
    pieces = []
    for start in range(0,51,7):
        encoded = reaction_features({k:v[start:start+7] for k,v in blocks.items()},centers,
                                    {k:v[start:start+7] for k,v in masks.items()},list(blocks))
        assert torch.equal(encoded,full[start:start+7])
        pieces.append(reaction_anchor_features(encoded,full[:32],.03,top_k=16))
    assert torch.equal(expected_q,torch.cat(pieces))
