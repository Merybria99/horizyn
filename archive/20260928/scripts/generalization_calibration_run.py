#!/usr/bin/env python3
"""Freeze the declared calibration artifacts, then dispatch fixed predictions."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from generalization_calibration_transfer import RUN,OUT,RULE,SPLITS,SEEDS,record,validate_freeze
from horizyn.generalization_retrieval import checked_artifact
from generalization_full_graph import atomic_json
PANELS=SPLITS+['case1','nitrilase','aminotransferase']


def freeze():
    path=OUT/'frozen_recipe.json'
    if path.exists():raise ValueError('Calibration freeze already exists')
    fit=json.loads((OUT/'fit_complete.json').read_text())
    if fit['rule']!=RULE or [(r['split'],r['seed']) for r in fit['approved_models']]!=[(s,n) for s in SPLITS for n in SEEDS]:raise ValueError('Incomplete training transfer')
    for row in fit['approved_models']:
        for key in ['parent_bundle','calibration_state','preflight_check']:checked_artifact(row[key])
        checks=json.loads(Path(row['preflight_check']['path']).read_text())['checks']
        if not checks or not all(checks.values()):raise ValueError('Prefreeze numerical checks failed')
    plan=OUT/'external_evaluation_plan.json';plan_=json.loads(plan.read_text())
    jobs=[]
    for panel in PANELS:
        split=panel if panel in SPLITS else 'reaction_smi'
        for seed in SEEDS:
            receipt_path=RUN/'phase2/predictions'/panel/f'seed{seed}'/'complete.json';receipt=json.loads(receipt_path.read_text())
            if receipt.get('phase')!='exploratory_phase2' or receipt.get('labels_used') is not False:raise ValueError('Invalid parent prediction')
            input_=checked_artifact(receipt['input_receipt']);score_path=receipt_path.parent/'scores.npz'
            # Authenticate existing controls only; compute no new scores here.
            if record(score_path)['sha256']!=receipt['output_sha256']:raise ValueError('Parent scores changed')
            jobs.append(dict(panel=panel,split=split,seed=seed,input_receipt=record(input_),reaction_key=receipt['reaction_key'],parent_receipt=record(receipt_path),parent_scores=record(score_path)))
    old=json.loads((RUN/'phase4/frozen_recipe.json').read_text());sources={r['path']:r for r in old['implementation_sources']}
    for relative in ['horizyn/generalization_calibration.py','scripts/generalization_composed_calibration.py','scripts/generalization_smooth_anchors.py',
        'scripts/generalization_calibration_transfer.py','scripts/generalization_calibration_predict.py','scripts/generalization_calibration_official_evaluate.py','scripts/generalization_calibration_external_evaluate.py',
        'scripts/generalization_calibration_run.py','tests/unit/test_generalization_calibration.py','tests/unit/test_generalization_calibration_transfer.py','tests/test_generalization_calibration_external_evaluate.py']:
        item=record(ROOT/relative);sources[item['path']]=item
    for item in sources.values():checked_artifact(item)
    for item in plan_.get('sources',{}).values():
        if isinstance(item,dict) and 'path' in item and 'sha256' in item:checked_artifact(item)
    data=dict(schema='post_evaluation_calibration_transfer_freeze_v1',frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        frozen_before_new_predictions=True,development_exposed=True,independent_confirmation=False,rule=RULE,primary_seed=42,primary_method='calibrated_seed42',
        validation_selection=record(RUN/'post_evaluation_calibration/complete.json'),validation_protocol=record(RUN/'post_evaluation_calibration/protocol.json'),
        phase2_frozen_recipe=record(RUN/'phase2/frozen_recipe.json'),original_frozen_recipe=record(RUN/'frozen_recipe.json'),previous_phase4_freeze=record(RUN/'phase4/frozen_recipe.json'),
        numeric_contract='Unchanged P2 FP32 endpoints; trainingmeans FP64 fixed512/256chunk accumulation; reactionbias FP64 rowwise sum, storedFP32. Append[R,bias,1],[E,1,0] with no renormalization. Canonical augmentedFP64dot→FP32. gamma0 delegatesP2 exactly. Parent controls must match original scores bitwise.',
        training_fit_receipt=record(OUT/'fit_complete.json'),approved_models=fit['approved_models'],approved_predictions=jobs,
        external_evaluation_plan=record(plan),implementation_sources=list(sources.values()),
        interpretation='All official and external panels were previously evaluated; this fixed calibration transfer is post-evaluation exploratory. E2R candidatebias is scientific comparison; R2E changes can only reflect constant-offset numerical tie effects. No retuning or large-pool method changes.',
        official_evaluation=dict(primary='calibrated_seed42',references=['phase2_seed42','F3_fp64'],methods=['calibrated_seed42','calibrated_seed17','calibrated_seed73','phase2_seed42','phase2_seed17','phase2_seed73','F3_native','F3_fp64'],bootstrap_replicates=10000,seed=20260919,metrics_unchanged=True))
    atomic_json(path,data);validate_freeze(record(path));print(json.dumps(record(path)),flush=True)
    labels={f'calibrated_seed{s}':(s,'selected') for s in SEEDS};labels.update({f'phase2_seed{s}':(s,'parent') for s in SEEDS});labels.update(F3_native=(42,'baseline'),F3_fp64=(42,'baseline_fp64'))
    methods=[dict(label=label,primary=label=='calibrated_seed42',seed=seed,splits={split:dict(path=str((OUT/'predictions'/split/f'seed{seed}'/'scores.npz').resolve()),score_key=key) for split in SPLITS}) for label,(seed,key) in labels.items()]
    atomic_json(OUT/'official_evaluation_manifest.json',dict(freeze=record(path),baseline_method='F3_fp64',methods=methods))


def predict(args):
    freeze_path=OUT/'frozen_recipe.json';data=validate_freeze(record(freeze_path));results=[]
    for row in data['approved_predictions']:
        output=OUT/'predictions'/row['panel']/f"seed{row['seed']}";output.mkdir(parents=True,exist_ok=True)
        if (output/'complete.json').exists():raise ValueError('Do not replace existing prediction: '+str(output))
        cmd=[sys.executable,str(ROOT/'scripts/generalization_calibration_predict.py'),'--freeze',str(freeze_path),'--panel',row['panel'],'--seed',str(row['seed']),'--output',str(output),'--device',args.device]
        atomic_json(output/'command.json',cmd)
        with (output/'run.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Prediction failed; see '+str(output/'run.log'))
        results.append(record(output/'complete.json'));atomic_json(OUT/'prediction_progress.json',results);print(json.dumps(dict(panel=row['panel'],seed=row['seed'],complete=True)),flush=True)
    atomic_json(OUT/'predictions_complete.json',dict(freeze=record(freeze_path),receipts=results))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','predict']);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.action=='freeze':freeze()
    else:predict(args)

if __name__=='__main__':main()
