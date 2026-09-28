import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[2]/'scripts/generalization_branch_dominance.py'
SPEC=importlib.util.spec_from_file_location('branch_dominance',SOURCE)
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def test_average_ties_and_undefined_constant_correlations():
    ranks,order,tied=M.ranking(np.array([2.,3.,2.,1.]))
    assert ranks.tolist()==[2.5,1.,2.5,4.] and order.tolist()==[1,0,2,3]
    assert tied==.5
    assert M.pearson(np.ones(4),np.arange(4)) is None


def test_tie_overlap_does_not_turn_all_ties_into_apparent_agreement():
    p=M.top_probability(np.ones(10),3)
    assert np.all(p==.3)
    assert np.dot(p,p)/3==pytest.approx(.3)


def test_known_dispersion_ratio_and_exact_variance_decomposition():
    d=np.arange(12,dtype=np.float32);s=d*.01;m=(.75*d.astype(float)+.25*s).astype(np.float32)
    out=M.row_stats(m,d,s)
    assert out['weighted_neural_to_semantic_std_ratio']==pytest.approx(300,rel=1e-6)
    assert out['mixture_neural_spearman']==pytest.approx(1)
    assert out['mixture_semantic_spearman']==pytest.approx(1)
    assert out['ideal_once_rounded_score_unequal_fraction']==0


def test_constant_singleton_does_not_invent_dispersion():
    result=M.row_stats([.5],[.6],[.2])
    assert result['both_branches_constant'] and result['weighted_neural_to_semantic_std_ratio'] is None
    assert result['mixture_neural_spearman'] is None and result['top25_effective_k']==1
