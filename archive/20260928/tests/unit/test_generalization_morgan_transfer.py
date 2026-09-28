import importlib.util,json
from pathlib import Path
import numpy as np
import pytest
import torch
from horizyn.generalization_morgan_transfer import MorganComposedEncoder,RULE
from horizyn.semantic_anchors import row_unit

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('morgan_transfer_test',ROOT/'scripts/generalization_morgan_transfer.py')
transfer=importlib.util.module_from_spec(spec);spec.loader.exec_module(transfer)

class Parent(torch.nn.Module):
    def __init__(self):
        super().__init__();self.alpha=.25;self.modalities=['a'];self.smooth=torch.nn.Module()
        self.smooth.train_reactions=row_unit(torch.randn(7,3));self.smooth.reaction_center_a=torch.zeros(3)
    def encode_enzymes(self,x,*a,**kw):return x
    def encode_enzyme_index(self,x,*a,**kw):return x

def test_selected_wrapper_preserves_enzyme_index_and_dense_coordinates():
    torch.manual_seed(81);parent=Parent();state=dict(schema='morgan_training_dictionary_v1',rule=RULE,train_ids=list(range(7)),train_morgan=row_unit(torch.randn(7,4096)))
    model=MorganComposedEncoder(parent,state);enzyme=torch.randn(2,519)
    assert model.encode_enzymes(enzyme) is enzyme and model.encode_enzyme_index(enzyme) is enzyme
    r=torch.randn(4,519);blocks={'a':torch.randn(4,3)};masks={'a':torch.ones(4,dtype=torch.bool)};fp=row_unit(torch.randn(4,4096))
    result=model.replace_reaction_semantics(r,blocks,masks,fp)
    assert torch.equal(result[:,:512],r[:,:512])
    small=model.replace_reaction_semantics(r[:1],{'a':blocks['a'][:1]},{'a':masks['a'][:1]},fp[:1])
    assert torch.equal(result[:1],small)

def test_wrong_recipe_rejected():
    p=Parent();state=dict(schema='morgan_training_dictionary_v1',rule={**RULE,'radius':2},train_ids=list(range(7)),train_morgan=row_unit(torch.randn(7,4096)))
    with pytest.raises(ValueError):MorganComposedEncoder(p,state)

def test_split_seed_model_lookup_rejects_mislabeled_seed():
    with pytest.raises(ValueError):transfer.model_row({'approved_models':[{'split':'reaction_smi','seed':73}]},'reaction_smi',42)

@pytest.mark.parametrize('declared,approved,passes',[(None,False,False),(None,True,True),('wrong',True,False),('original',False,True)])
def test_optional_feature_freeze_guard_is_exact_allowlist(tmp_path,monkeypatch,declared,approved,passes):
    data=tmp_path/'data';data.write_text('input');rec={'path':str(data),'sha256':'x'}
    receipt=tmp_path/'receipt.json';payload={'schema':'generalization_feature_bundle_receipt_v1','inputs':{k:rec for k in ['catalog','base','protein_means','reaction_features']},'checkpoint':rec,'source_receipts':[rec]}
    if declared is not None:payload['freeze_sha256']=declared
    receipt.write_text(json.dumps(payload));item={'path':str(receipt),'sha256':'receipt'}
    monkeypatch.setattr(transfer,'checked_artifact',lambda item:Path(item['path']))
    frozen={'original_frozen_recipe':{'sha256':'original'},'optional_input_freeze_receipts':[item] if approved else []}
    if passes:assert transfer.authenticate_input_receipt(frozen,item)['schema']==payload['schema']
    else:
        with pytest.raises(ValueError):transfer.authenticate_input_receipt(frozen,item)
