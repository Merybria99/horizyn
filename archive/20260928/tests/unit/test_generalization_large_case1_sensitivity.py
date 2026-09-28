import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[2]/'scripts/generalization_large_case1_sensitivity.py'
SPEC=importlib.util.spec_from_file_location('sensitivity',PATH)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def test_sha_priority_is_id_based_and_uses_declared_framing():
    ids=['WP_z','WP_a','WP_b','WP_c']
    order=MODULE.priority_order(ids)
    expected=sorted(ids,key=lambda value:(hashlib.sha256(b'20260920\0'+value.encode()).digest(),value))
    assert [ids[i] for i in order]==expected
    reverse=ids[::-1]
    assert [reverse[i] for i in MODULE.priority_order(reverse)]==expected


def test_subsets_are_nested_and_retain_source_order_and_all_references():
    order=np.array([3,0,4,1,2])
    small=MODULE.subset_indices(order,2,5,8)
    large=MODULE.subset_indices(order,4,5,8)
    assert small.tolist()==[0,3,5,6,7]
    assert large.tolist()==[0,1,3,4,5,6,7]
    assert set(small)<=set(large)
    assert MODULE.subset_indices(order,5,5,8).tolist()==list(range(8))


def test_input_guards_reject_ambiguous_ids_and_invalid_sizes():
    with pytest.raises(ValueError):MODULE.priority_order(['same','same'])
    with pytest.raises(ValueError):MODULE.priority_order([''])
    with pytest.raises(ValueError):MODULE.subset_indices(np.arange(5),0,5,8)
    with pytest.raises(ValueError):MODULE.subset_indices(np.arange(5),6,5,8)


def test_execution_requires_preregistered_schema_before_artifact_reads():
    with pytest.raises(ValueError,match='Wrong supplemental plan'):
        MODULE.check_prerequisites({})
    with pytest.raises(ValueError,match='input-only'):
        MODULE.check_prerequisites(dict(schema='large_case1_nested_background_sensitivity_plan_v1',
                                        primary_protocol_unchanged=True,scores_read_for_selection=True))


def test_tie_recovery_tracks_actual_nested_candidate_pool():
    helper_path=PATH.with_name('generalization_large_case1_evaluate.py')
    spec=importlib.util.spec_from_file_location('primary_sensitivity_test_helper',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    # Three background candidates then two forcibly retained references; all tied.
    scores=np.ones(5,dtype=np.float32)
    order=np.array([2,0,1])
    full=MODULE.subset_indices(order,3,3,5)
    small=MODULE.subset_indices(order,1,3,5)
    for idx,count,expected in ((small,1,2/3),(full,3,2/5)):
        ranks,_=helper.reference_ranks(scores[idx],np.arange(count,len(idx)))
        summary=helper.tier_summary(ranks,[0,1],len(idx),cuts=(1,))
        assert summary['cutoffs']['1']['uniform_tie_expected_recovered']==pytest.approx(expected)
        assert summary['cutoffs']['1']['stable_recovered']==0
        assert ranks['tie_size'].tolist()==[len(idx),len(idx)]
