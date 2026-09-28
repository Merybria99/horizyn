#!/usr/bin/env python3
"""Complete the same fixed phase-2 recipe for each already-trained geometry arm."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

from generalization_screen_phase2_campaign import invoke,sha
from generalization_clipzyme_test_queue import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());cross=a.plan.parent.parent
    def arm(task):
        out=Path(task['output']);base=Path(task['base_run'])
        try:
            selection=json.loads((base/'validation_selected.json').read_text())['selected']
            val=Path(selection['evaluation']);screen=val.parent.parent
            epoch=int(screen.name.removeprefix('screen_epoch'))
            checkpoint=base/f'checkpoints/screen_selection/screen-epoch={epoch:02d}.ckpt'
            if sha(checkpoint)!=selection['checkpoint_sha256']:raise ValueError('Selected base checkpoint changed')
            config=base/'configs/train.yaml'
            write_json(out/'protocol.json',dict(base_config=str(config),base_config_sha256=sha(config),
                base_checkpoint=str(checkpoint),base_checkpoint_sha256=sha(checkpoint),validation_summary=str(val),
                fixed_phase2=plan['recipe'],test_used_for_selection=False,base_selection_sha256=sha(base/'validation_selected.json')))
            features=out/'features';features.mkdir()
            source=cross/'sleec_multiview_phase2_v1/features'
            for name in ('catalog.json','pairs.npz','reaction_features.npz'):(features/name).symlink_to((source/name).resolve())
            shutil.copyfile(source/'protein_mean.h5',features/'protein_mean.h5')
            manifest=json.loads((source/'manifest.json').read_text())
            for key in ('checkpoint','f3_export_config','f3_residue_cache'):manifest.pop(key,None)
            manifest['raw_feature_reuse']=dict(source_manifest=str(source/'manifest.json'),sha256=sha(source/'manifest.json'))
            manifest['sources']['config']=dict(path=str(config),sha256=sha(config));write_json(features/'manifest.json',manifest)
            write_json(out/'state.json',dict(stage='feature_export'))
            invoke('generalization_clipzyme_phase2_export.py',['--stage','f3','--catalog',cross/'clipzyme_f3_catalog_v1',
                '--config',config,'--checkpoint',checkpoint,'--output',features,'--batch-size',128,
                '--residue-cache','/tmp/enzymediscovery_f3_20260920/train_validation_prott5.h5'],task['gpu'],out/'features.log')
            write_json(out/'state.json',dict(stage='fixed_phase2_training'))
            invoke('generalization_full_graph.py',['--features',features,'--output',out/'training','--steps',100,
                '--snapshot-every',100,'--selection-method','external_screening','--temperature',.2,
                '--identity-weight',10,'--contrastive-objective','positive_ce','--cpu-threads',4],task['gpu'],out/'training.log')
            invoke('generalization_clipzyme_f3_validation.py',['--manifest',cross/'clipzyme_manifests_v2/manifest.json',
                '--catalog',cross/'clipzyme_f3_catalog_v1','--embeddings',screen/'validation_embeddings',
                '--output',out/'validation','--refiner',out/'training/step0100.pt','--batch-size',64],task['gpu'],out/'validation.log')
            write_json(out/'validation_selected.json',dict(selected=dict(evaluation=str(out/'validation/summary.json')),
                selection='Fixed recipe; this arm is reported regardless of relative validation rank'))
            invoke('generalization_clipzyme_phase2_dictionary.py',['--features',features,'--output',out/'anchors.pt'],
                task['gpu'],out/'dictionary.log')
            (out/'selected_test').mkdir();(out/'selected_test/test_embeddings').symlink_to(screen/'requested_test/test_embeddings',target_is_directory=True)
            write_json(out/'state.json',dict(stage='fixed_semantic_composition_and_test'))
            invoke('generalization_screen_phase2_compose.py',['--campaign',out,'--fixed-alpha','.25','--fixed-cap',1],
                task['gpu'],out/'composition.log')
            write_json(out/'state.json',dict(stage='complete',completed_utc=datetime.now(timezone.utc).isoformat()))
        except Exception as exc:
            write_json(out/'failure.json',dict(error=repr(exc)));raise
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(arm,plan['tasks']))
    write_json(a.plan.parent/'complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
