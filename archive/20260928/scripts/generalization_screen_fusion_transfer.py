#!/usr/bin/env python3
"""Check a ReactZyme-motivated fusion adjustment on full-pool EnzymeMap validation."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch

from generalization_clipzyme_ablation_watch import evaluate
from generalization_clipzyme_test_queue import run_job,write_json
from generalization_screen_replication import ROOT,sha,invoke
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    out=a.campaign.resolve();cross=out.parent;plan=json.loads((out/'protocol.json').read_text())
    torch.set_num_threads(4)
    for task in plan['tasks']:
        run=out/f"seed{task['seed']}"
        if a.resume and (run/'complete.json').exists():continue
        run.mkdir(exist_ok=a.resume)
        source=Path(task['source']);source_phase=source/'phase2';gpu=task['gpu']
        (run/'configs').mkdir(exist_ok=a.resume)
        if not (run/'configs/train.yaml').exists():(run/'configs/train.yaml').symlink_to(source/'configs/train.yaml')
        checkpoint=source/'checkpoints/screen_selection/screen-epoch=17.ckpt'
        new_checkpoint=run/'checkpoints/screen_selection/screen-epoch=17.ckpt'
        if a.resume and (run/'calibration_checkpoint.json').exists():
            receipt=json.loads((run/'calibration_checkpoint.json').read_text())
            from horizyn.inference_fusion_lineage import verify_refiner_base
            verify_refiner_base(sha(checkpoint),sha(new_checkpoint),run/'calibration_checkpoint.json',new_checkpoint)
        else:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            key,=[k for k in saved['state_dict'] if k.endswith('multiview_encoder.raw_residual_scale')]
            old=saved['state_dict'][key].clone();original_scale=float(.5*old.sigmoid())
            desired=original_scale*plan['multiplier']
            if not 0<desired<.5:raise ValueError('Fusion adjustment outside model domain')
            saved['state_dict'][key]=old.new_tensor(math.log((desired/.5)/(1-desired/.5)))
            receipt=dict(source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),
                execution_source_sha256=sha(Path(__file__)),
                parameter=key,old_parameter=float(old),new_parameter=float(saved['state_dict'][key]),
                original_scale=original_scale,requested_multiplier=plan['multiplier'],
                new_scale=float(.5*saved['state_dict'][key].sigmoid()),
                training_performed=False,inference_only=True,other_parameters_changed=False,
                selection_rule='Accept only if full-library validation BEDROC85 improves and the other three Table1 metrics do not decline')
            saved['inference_fusion_calibration']=receipt
            for name in ('optimizer_states','lr_schedulers','loops','callbacks'):saved.pop(name,None)
            new_checkpoint.parent.mkdir(parents=True)
            torch.save(saved,new_checkpoint);del saved
            receipt['calibrated_checkpoint_sha256']=sha(new_checkpoint);write_json(run/'calibration_checkpoint.json',receipt)
        write_json(run/'state.json',dict(stage='full_library_validation_export'))
        evaluate(run,17,SimpleNamespace(catalog=cross/'clipzyme_f3_catalog_v1',protocol=cross/'clipzyme_screening_evaluation_protocol_v2',
            manifest=cross/'clipzyme_manifests_v2/manifest.json',epochs=[17],gpus=[0,1,2,3]))
        phase=run/'phase2';phase.mkdir(exist_ok=a.resume)
        for name in ('features','anchors.pt'):
            if not (phase/name).exists():(phase/name).symlink_to(source_phase/name)
        head=source_phase/'training/step0100.pt'
        if not (phase/'validation/summary.json').exists():
            invoke('generalization_clipzyme_f3_validation.py',['--manifest',cross/'clipzyme_manifests_v2/manifest.json',
                '--catalog',cross/'clipzyme_f3_catalog_v1','--embeddings',run/'screen_epoch17/validation_embeddings',
                '--output',phase/'validation','--refiner',head,'--batch-size',64,
                '--fusion-calibration',run/'calibration_checkpoint.json','--calibrated-base-checkpoint',new_checkpoint],
                gpu,run/'phase2_validation.log')
        write_json(phase/'protocol.json',dict(base_config=str(run/'configs/train.yaml'),base_checkpoint=str(new_checkpoint),
            base_checkpoint_sha256=sha(new_checkpoint),validation_summary=str(run/'screen_epoch17/validation_evaluation/summary.json'),
            original_head_training_checkpoint_sha256=receipt['source_checkpoint_sha256'],
            inference_only_calibration=receipt,fixed_recipe=True,seed=task['seed'],test_used_for_selection=False))
        write_json(phase/'validation_selected.json',dict(selected=dict(evaluation=str(phase/'validation/summary.json')),
            selection='Unchanged original phase2 head; inference-only F3 fusion adjustment'))
        staging=phase/'validation_library';staging.mkdir(exist_ok=a.resume)
        if not (staging/'test_embeddings').exists():(staging/'test_embeddings').symlink_to(run/'screen_epoch17/validation_embeddings')
        if not (phase/'selected_test').exists():(phase/'selected_test').symlink_to(staging)
        write_json(run/'state.json',dict(stage='composed_full_library_validation'))
        if not (phase/'fusion_validation/complete.json').exists():
            invoke('generalization_screen_phase2_compose.py',['--campaign',phase,'--cross-root',cross,'--fixed-alpha',.25,
                '--fixed-cap',1,'--output-name','fusion_validation','--validate-only'],gpu,run/'composition_validation.log')
        current=json.loads((phase/'fusion_validation/validation.json').read_text())['records'][-1]
        previous=json.loads((source_phase/'composition_v1/validation.json').read_text())['records'][-1]
        metrics=('bedroc85','bedroc20','ef0.05','ef0.1')
        old_values=previous['summary']['table1'];new_values=current['summary']['table1']
        promoted=(new_values['bedroc85']>old_values['bedroc85']+1e-6 and
                  all(new_values[m]>=old_values[m]-1e-6 for m in metrics))
        freeze=dict(selected_multiplier=plan['multiplier'] if promoted else 1.,original=previous,adjusted=current,
            selected_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(out/'protocol.json'),
            test_used_for_selection=False,retains_sleec=True,ensembles=False)
        write_json(run/'selection.json',freeze)
        if promoted:
            (phase/'selected_test').unlink()
            job=dict(label=f"fusion_calibrated_seed{task['seed']}",config=str(run/'configs/train.yaml'),
                checkpoint=str(new_checkpoint),refiner=str(head),output=str(phase/'test'),
                protein_source=str(run/'screen_epoch17/proteins'),gpu=gpu,
                fusion_calibration=str(run/'calibration_checkpoint.json'),
                policy='Fusion multiplier selected by full-library validation BEDROC85 with four-metric nondegradation guard')
            write_json(run/'state.json',dict(stage='validation_selected_test'))
            run_job(job,dict(created_utc=plan['created_utc'],catalog=str(cross/'clipzyme_f3_catalog_v1'),
                screening_protocol=str(cross/'clipzyme_screening_evaluation_protocol_v2')))
            (phase/'selected_test').symlink_to(phase/'test')
            if not (phase/'composition_v1/test_evaluation/summary.json').exists():
                invoke('generalization_screen_phase2_compose.py',['--campaign',phase,'--cross-root',cross,'--fixed-alpha',.25,
                    '--fixed-cap',1],gpu,run/'composition_test.log')
            test=phase/'composition_v1/test_evaluation/summary.json'
        else:test=source_phase/'composition_v1/test_evaluation/summary.json'
        write_json(run/'complete.json',dict(selected_multiplier=freeze['selected_multiplier'],test_summary=str(test),
            unchanged_reference_reused=not promoted,selection_sha256=sha(run/'selection.json')))
        write_json(run/'state.json',dict(stage='complete'))
        print(json.dumps(dict(seed=task['seed'],selected_multiplier=freeze['selected_multiplier'],test_summary=str(test))),flush=True)
    write_json(out/'complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
