import copy,importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
def load(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ext=load('generalization_morgan_external_evaluate');large=load('generalization_morgan_large_case1_evaluate')
def fixtures():
 f=dict(schema='post_evaluation_morgan_transfer_freeze_v1',frozen_before_new_predictions=True,development_exposed=True,rule=dict(radius=3,eta_reaction=1.,fp_size=4096,include_chirality=True,alpha=.25,temperature=.03))
 p=dict(schema='morgan_external_evaluation_plan_v1',methods=list(ext.METHODS),panels=list(ext.SHAPES),primary='morgan_seed42',training_split='reaction_smi',development_exposed=True,bootstrap_replicates=10000,permutation_replicates=1000,seed=20260920)
 return f,p
def test_fixed_contract():
 f,p=fixtures();ext.validate_contract(f,p)
 assert ext.METHODS['phase2_seed17']==(17,'parent') and ext.METHODS['F3_fp64']==(42,'baseline_fp64')
 assert ext.SHAPES=={'nitrilase':(38,18),'aminotransferase':(18,25),'case1':(1,123),'p450':(191,490)}
@pytest.mark.parametrize('mutation',[lambda f,p:f['rule'].update(radius=2),lambda f,p:p['methods'].pop(),lambda f,p:p.update(primary='morgan_seed17'),lambda f,p:p.update(training_split='time'),lambda f,p:p.update(bootstrap_replicates=100)])
def test_reject_contract_changes(mutation):
 f,p=fixtures();mutation(f,p)
 with pytest.raises(ValueError):ext.validate_contract(f,p)
def test_large_fixed_plan():
 p=dict(schema='morgan_large_case1_evaluation_plan_v1',methods=large.METHODS,cutoffs=large.CUTS,primary_supplement='morgan_seed42',training_split='reaction_smi',seed=42,original_five_method_primary_unchanged=True,development_exposed=True,unknown_candidates_are_unlabeled=True)
 large.validate_plan(p);p=copy.deepcopy(p);p['methods'].remove('circev2')
 with pytest.raises(ValueError):large.validate_plan(p)
def test_audit_gate():
 a=dict(path='/tmp/a',sha256='a');p=dict(path='/tmp/p',sha256='p')
 audit=dict(schema='large_case1_completion_integrity_audit_v1',all_exact=True,complete=a,protocol=p)
 primary=dict(schema='large_case1_evaluation_complete_v1',scan_receipt=a,protocol=p)
 large.validate_original_gate(audit,a,p,primary)
 with pytest.raises(ValueError):large.validate_original_gate(dict(audit,all_exact=False),a,p,primary)
 with pytest.raises(ValueError):large.validate_original_gate(audit,dict(a,sha256='changed'),p,primary)
def test_original_tie_endpoint_with_synthetic_scores():
 old=load('generalization_large_case1_evaluate')
 ranks,_=old.reference_ranks(np.array([3,2,2,2,1],np.float32),[1,2]);s=old.tier_summary(ranks,[0,1],5,[2])
 assert s['cutoffs']['2']['uniform_tie_expected_recovered']==pytest.approx(2/3)
 assert s['cutoffs']['2']['stable_recovered']==1
