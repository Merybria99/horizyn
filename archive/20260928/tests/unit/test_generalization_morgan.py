import numpy as np
import pytest
import torch
from horizyn.generalization_morgan import canonical_components,participant_morgan,MorganReactionAnchor
from horizyn.semantic_anchors import row_unit
from horizyn.semantic_smooth import reaction_responses

def test_maps_serialization_order_and_self_side_do_not_change_counts():
    a=participant_morgan('[CH3:2][CH2:1]O.O',2)
    assert np.array_equal(a,participant_morgan('O.OCC',2))
    assert np.array_equal(a,participant_morgan('CCO.O>>O.OCC',2))

def test_component_multiplicity_and_stereo_charge_are_retained():
    assert not np.array_equal(participant_morgan('CCO.O',2),participant_morgan('CCO.O.O',2))
    assert not np.array_equal(participant_morgan('C[C@H](O)F',2),participant_morgan('C[C@@H](O)F',2))
    assert not np.array_equal(participant_morgan('N',2),participant_morgan('[NH4+]',2))
    assert canonical_components('[13CH4]')!=canonical_components('C')

@pytest.mark.parametrize('text',['','CCO..O','CCO>>CC=O','C1CC','CCO>O>CCO'])
def test_invalid_input_fails_closed(text):
    with pytest.raises(ValueError):participant_morgan(text,2)

def test_per_component_normalization_precedes_multiplicity_sum():
    x=participant_morgan('CCCCCC',3).astype('f8');y=participant_morgan('O',3).astype('f8')
    expected=x+2*y;expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(participant_morgan('O.CCCCCC.O',3),expected,rtol=2e-7,atol=2e-8)

def test_hybrid_cosine_response_and_batch_subset_are_independent():
    torch.manual_seed(12)
    tr=row_unit(torch.randn(7,9));tm=row_unit(torch.randn(7,4096))
    raw=row_unit(torch.randn(4,9));morgan=row_unit(torch.randn(4,4096))
    model=MorganReactionAnchor(tr,tm,.5)
    actual=model.encode_reactions(raw,morgan)
    expected=reaction_responses(torch.cat((2**-.5*raw,2**-.5*morgan),1),torch.cat((2**-.5*tr,2**-.5*tm),1),'exponential',.03,None)
    assert torch.equal(actual,expected)
    assert torch.equal(actual[::2],model.encode_reactions(raw[::2],morgan[::2]))
    assert torch.equal(actual[:1],model.encode_reactions(raw[:1],morgan[:1]))
