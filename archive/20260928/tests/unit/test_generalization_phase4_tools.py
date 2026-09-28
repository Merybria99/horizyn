import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.generalization_export import atomic_json,digest
from scripts.generalization_phase4_predict import validate_features,validate_frozen_sources
from scripts.generalization_phase4_official_evaluate import checked_scores


def test_predictor_requires_frozen_unchanged_source_closure(tmp_path):
    source=tmp_path/'source.py';source.write_text('fixed')
    freeze=tmp_path/'freeze.json'
    record=dict(path=str(source),sha256=digest(source))
    atomic_json(freeze,dict(frozen_before_phase4_prediction=True,frozen_before_new_external_evaluation=True,implementation_sources=[record]))
    frozen=dict(path=str(freeze),sha256=digest(freeze))
    assert validate_frozen_sources(frozen,[source])==[record]
    with pytest.raises(ValueError,match='outside the freeze'):validate_frozen_sources(frozen,[tmp_path/'another.py'])
    source.write_text('changed')
    with pytest.raises(ValueError):validate_frozen_sources(frozen,[source])


def test_predictor_rejects_unfrozen_recipe(tmp_path):
    freeze=tmp_path/'freeze.json';atomic_json(freeze,dict(frozen_before_phase4_prediction=False,frozen_before_new_external_evaluation=True))
    with pytest.raises(ValueError,match='frozen before'):validate_frozen_sources(dict(path=str(freeze),sha256=digest(freeze)))


def feature_fixture():
    catalog=dict(proteins=['P0','P1']);reactions=['R']
    blocks={k:torch.zeros(1,n) for k,n in [('t5v2',768),('unimol2',768),('chiro',256),('chemistry',617)]}
    masks={k:torch.ones(1,dtype=torch.bool) for k in blocks}
    return [catalog,reactions,torch.zeros(2,512),torch.zeros(1,512),torch.zeros(2,1024),blocks,masks]


def test_predictor_rejects_catalog_misalignment_and_mask_dtype():
    values=feature_fixture();validate_features(*values)
    values[0]['proteins']=['P0','P0']
    with pytest.raises(ValueError,match='unique'):validate_features(*values)
    values=feature_fixture();values[-1]['chiro']=torch.ones(1)
    with pytest.raises(ValueError,match='mask'):validate_features(*values)
    values=feature_fixture();values[2][0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):validate_features(*values)


def score_fixture(tmp_path,monkeypatch,phase='phase4'):
    import horizyn.generalization_phase4 as p4
    import horizyn.generalization_phase2 as p2
    monkeypatch.setattr(p4,'validate_phase4_bundle',lambda *_:None)
    monkeypatch.setattr(p2,'validate_phase2_bundle',lambda *_:None)
    catalog=tmp_path/'catalog.json';atomic_json(catalog,dict(reactions=['R'],proteins=['P0','P1']))
    np.savez(tmp_path/'scores.npz',selected=np.array([[.7,.2]],np.float32),baseline=np.array([[.6,.1]],np.float32),baseline_fp64=np.array([[.6,.1]],np.float32))
    spec=dict(schema='phase4_hybrid_bundle_v1' if phase=='phase4' else 'generalization_phase2_bundle_v1',split='reaction_smi',seed=42,variant='primary',phase2_frozen_recipe=dict(sha256='p2'),frozen_recipe=dict(sha256='p1'))
    if phase=='phase4':spec['phase4_frozen_recipe']=dict(sha256='p4')
    bundle=tmp_path/'bundle.json';atomic_json(bundle,spec)
    inputs={k:dict(path=str(catalog),sha256=digest(catalog)) for k in ['catalog','base','protein_means','reaction_features']}
    input_path=tmp_path/'input_receipt.json';atomic_json(input_path,dict(schema='generalization_feature_bundle_receipt_v1',freeze_sha256='p1',inputs=inputs))
    receipt=dict(output_sha256=digest(tmp_path/'scores.npz'),labels_used=False,bundle=dict(path=str(bundle),sha256=digest(bundle)),input_receipt=dict(path=str(input_path),sha256=digest(input_path)),inputs=inputs,**{k:v for k,v in spec.items() if 'frozen_recipe' in k})
    atomic_json(tmp_path/'complete.json',receipt)
    return dict(path='scores.npz',score_key='selected'),catalog,{'phase4':(None,'p4'),'phase2':(None,'p2'),'original':(None,'p1')}


@pytest.mark.parametrize('phase,label',[('phase4','phase4_seed42'),('phase2','phase2_seed42')])
def test_evaluator_authenticates_new_and_immutable_parent_controls(tmp_path,monkeypatch,phase,label):
    entry,catalog,lineage=score_fixture(tmp_path,monkeypatch,phase)
    result=checked_scores(entry,tmp_path,catalog,(1,2),lineage,{},label,'reaction_smi')
    assert np.array_equal(result,np.array([[.7,.2]],np.float32))
    bad={**lineage,'original':(None,'wrong')}
    with pytest.raises(ValueError,match='lineage'):checked_scores(entry,tmp_path,catalog,(1,2),bad,{},label,'reaction_smi')
    with pytest.raises(ValueError,match='split/seed/variant'):checked_scores(entry,tmp_path,catalog,(1,2),lineage,{},label,'time')


def test_evaluator_rejects_score_relabelling_and_receipt_tampering(tmp_path,monkeypatch):
    entry,catalog,lineage=score_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='score key'):checked_scores(entry,tmp_path,catalog,(1,2),lineage,{},'F3_fp64','reaction_smi')
    with pytest.raises(ValueError,match='split/seed/variant'):checked_scores(entry,tmp_path,catalog,(1,2),lineage,{},'phase4_seed17','reaction_smi')
    receipt=tmp_path/'complete.json';x=json.loads(receipt.read_text());x['labels_used']=True;atomic_json(receipt,x)
    with pytest.raises(ValueError,match='held-out labels'):checked_scores(entry,tmp_path,catalog,(1,2),lineage,{},'phase4_seed42','reaction_smi')


def test_evaluator_authenticates_every_score_before_opening_truth(tmp_path,monkeypatch):
    import sys
    import scripts.generalization_phase4_official_evaluate as module
    features=tmp_path/'features';features.mkdir()
    for split in module.SPLITS:
        folder=features/f'features_test_{split}';folder.mkdir()
        atomic_json(folder/'catalog.json',dict(reactions=['R'],proteins=['P']))
    methods=[dict(label=label,primary=label=='phase4_seed42',seed=spec[1],splits={s:{} for s in module.SPLITS}) for label,spec in module.METHODS.items()]
    manifest=tmp_path/'manifest.json';atomic_json(manifest,dict(freeze={},baseline_method='F3_fp64',methods=methods))
    tiger=tmp_path/'tiger.json';atomic_json(tiger,{})
    monkeypatch.setattr(module,'checked_lineage',lambda *_:(dict(primary_seed=42),{}))
    calls=[]
    def scores(*args):
        label,split=args[-2:];calls.append((label,split))
        if len(calls)==24:raise ValueError('Last score deliberately unauthenticated')
        return np.zeros((1,1),np.float32)
    monkeypatch.setattr(module,'checked_scores',scores)
    def forbidden_truth(*args):raise AssertionError('Official labels opened before all scores were authenticated')
    monkeypatch.setattr(module,'checked_official_truth',forbidden_truth)
    monkeypatch.setattr(sys,'argv',['evaluate','--manifest',str(manifest),'--feature-root',str(features),'--output',str(tmp_path/'output'),'--tiger-reference',str(tiger)])
    with pytest.raises(ValueError,match='Last score'):module.main()
    assert len(calls)==24
