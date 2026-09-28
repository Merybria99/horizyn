import json
from pathlib import Path
import pytest
from scripts.generalization_calibration_transfer import RULE,record,validate_freeze


def test_freeze_requires_exact_rule_and_current_source(tmp_path):
    source=tmp_path/'source.py';source.write_text('source')
    prior=tmp_path/'old.json';prior.write_text('{}')
    data=dict(schema='post_evaluation_calibration_transfer_freeze_v1',frozen_before_new_predictions=True,development_exposed=True,rule=RULE,
              implementation_sources=[record(source)],phase2_frozen_recipe=record(prior),original_frozen_recipe=record(prior),validation_selection=record(prior))
    path=tmp_path/'freeze.json';path.write_text(json.dumps(data));assert validate_freeze(record(path))['rule']==RULE
    data['rule']={**RULE,'gamma_reaction':.5};path.write_text(json.dumps(data))
    with pytest.raises(ValueError):validate_freeze(record(path))
    data['rule']=RULE;data['frozen_before_new_predictions']=False;path.write_text(json.dumps(data))
    with pytest.raises(ValueError):validate_freeze(record(path))
    data['frozen_before_new_predictions']=True;path.write_text(json.dumps(data));source.write_text('changed')
    with pytest.raises(ValueError):validate_freeze(record(path))


def test_official_scores_authenticated_before_any_truth(tmp_path,monkeypatch):
    import numpy as np
    import sys
    import scripts.generalization_calibration_official_evaluate as module
    methods=[dict(label=label,primary=label=='calibrated_seed42',seed=item[1],splits={s:{} for s in module.SPLITS}) for label,item in module.METHODS.items()]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(freeze={},baseline_method='F3_fp64',methods=methods)))
    for split in module.SPLITS:
        p=tmp_path/f'features_test_{split}';p.mkdir();(p/'catalog.json').write_text(json.dumps(dict(reactions=['r'],proteins=['e'])))
    tiger=tmp_path/'tiger.json';tiger.write_text('{}')
    monkeypatch.setattr(module,'checked_lineage',lambda *_:(dict(primary_seed=42),{}));calls=[]
    def check(*args):
        calls.append(args[-2:])
        if len(calls)==24:raise ValueError('Last score deliberately rejected')
        return np.ones((1,1),np.float32)
    monkeypatch.setattr(module,'checked_scores',check)
    def forbidden(*args):raise AssertionError('Labels opened before score authentication')
    monkeypatch.setattr(module,'checked_official_truth',forbidden)
    monkeypatch.setattr(sys,'argv',['evaluate','--manifest',str(manifest),'--feature-root',str(tmp_path),'--output',str(tmp_path/'out'),'--tiger-reference',str(tiger)])
    with pytest.raises(ValueError,match='Last score'):module.main()
    assert len(calls)==24
