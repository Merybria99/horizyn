import copy
import importlib.util
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[2]/'runs/generalization_20260919_2251/large_case1_mean_query/guarded_run.py'
SPEC=importlib.util.spec_from_file_location('mean_guard',PATH)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def fixture():
    protocol=dict(path='/tmp/protocol.json',sha256='protocol')
    evaluator=dict(path='/tmp/evaluator.py',sha256='evaluator')
    receipt=dict(path='/tmp/scan/complete.json',sha256='complete')
    audit_plan=dict(path='/tmp/audit_plan.json',sha256='auditplan')
    plan=dict(schema='large_case1_mean_query_control_v1',methods=['phase2','phase4'],seeds=[42],split='reaction_smi',
              tiers=['primary_papers','primary_papers_and_patents'],cutoffs=[25,100,1000,10000],
              background_scores_or_outcomes_read=False,labels_used_for_control_selection=False,supplemental=True,
              original_primary_protocol_changed=False,checks=dict(phase2_fp64_mean_exactly_replays_prior_training_only_fit=True),
              sources=dict(scan_protocol=protocol,rank_helper=evaluator),created_utc='2026-09-20T01:00:00+00:00')
    amendment=dict(numerical_audit_plan=audit_plan,created_utc='2026-09-20T02:00:00+00:00')
    complete=dict(schema='large_case1_scan_complete_v1',complete_fixed_candidate_coverage=True,labels_used=False,protocol=protocol)
    audit=dict(schema='large_case1_completion_integrity_audit_v1',all_exact=True,labels_read=False,ranks_inspected=False,
               complete=receipt,protocol=protocol,plan=audit_plan)
    primary=dict(schema='large_case1_evaluation_complete_v1',scan_receipt=receipt,protocol=protocol,evaluator=evaluator)
    opened=dict(all_five_scores_and_prescore_candidate_metadata_authenticated=True,activity_negatives_inferred=False,
                protocol=protocol,evaluator=evaluator,opened_at_utc='2026-09-20T03:00:00+00:00')
    return copy.deepcopy([plan,amendment,complete,audit,primary,opened,receipt])


def test_valid_gates_pass():MODULE.validate_gate_records(*fixture())


@pytest.mark.parametrize('index,key,value',[(0,'methods',['phase4','phase2']),(0,'cutoffs',[25]),
    (2,'complete_fixed_candidate_coverage',False),(3,'schema','wrong'),(3,'labels_read',True),
    (3,'ranks_inspected',True),(4,'schema','wrong'),(5,'activity_negatives_inferred',True)])
def test_wrong_scope_or_execution_status_rejected(index,key,value):
    args=fixture();args[index][key]=value
    with pytest.raises(ValueError):MODULE.validate_gate_records(*args)


@pytest.mark.parametrize('index,key',[(3,'complete'),(3,'protocol'),(3,'plan'),(4,'scan_receipt'),(4,'protocol'),(4,'evaluator'),(5,'protocol')])
def test_mismatched_artifact_lineage_rejected(index,key):
    args=fixture();args[index][key]={**args[index][key],'sha256':'swapped'}
    with pytest.raises(ValueError):MODULE.validate_gate_records(*args)


@pytest.mark.parametrize('index',[0,1])
def test_plan_or_amendment_after_label_open_rejected(index):
    args=fixture();args[index]['created_utc']='2026-09-20T03:00:01+00:00'
    with pytest.raises(ValueError,match='precede'):MODULE.validate_gate_records(*args)
