#!/usr/bin/env python3
"""Evaluate every declared cross-benchmark recipe, including validation failures."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from generalization_reactzyme_architecture_phase2 import (
    ROOT, atomic_json, sha256, evaluate_selected)


def ready(task):
    source=Path(task['source_campaign'])/task['arm']
    return all((source/name).exists() for name in (
        'features/complete.json','training/complete.json','training/step0100.pt','anchors.pt'))


def worker(out,plan,task):
    import torch
    import yaml
    dest=out/task['name'];dest.mkdir(exist_ok=False)
    source=Path(task['source_campaign'])/task['arm']
    source_plan=json.loads((Path(task['source_campaign'])/'protocol.json').read_text())
    source_task=next(t for t in source_plan['tasks'] if t['arm']==task['arm'])
    for key in ('train_config','test_config','checkpoint'):
        if source_task[key]!=task[key]:raise ValueError('Declared source task changed')
    config=yaml.safe_load(Path(task['train_config']).read_text())
    if config['training']['loss']['beta']!=task['beta']:raise ValueError('Base temperature mismatch')
    if config['model']['pooling']!='sleec_guided_attention':raise ValueError('SLEEC missing')
    registry=json.loads((source/'training/registry.json').read_text())
    expected=dict(steps=100,temperature=.2,identity_weight=10,learning_rate=.0001,
        contrastive_objective='positive_ce',selection_method='fixed_last',warm_start=None)
    if registry['test_used'] or any(registry['arguments'][k]!=v for k,v in expected.items()):
        raise ValueError('Source residual does not implement the fixed EnzymeMap recipe')
    receipt=json.loads((source/'features/complete.json').read_text())
    if receipt['checkpoint_sha256']!=sha256(task['checkpoint']):raise ValueError('Checkpoint lineage mismatch')
    run=dest/task['arm'];run.mkdir()
    for name in ('features','training','anchors.pt'):(run/name).symlink_to(source/name)
    freeze=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),recipe=plan['recipe'],task=task,
        checkpoint_sha256=receipt['checkpoint_sha256'],protocol_sha256=sha256(out/'protocol.json'),
        source_protocol_sha256=sha256(Path(task['source_campaign'])/'protocol.json'),
        feature_manifest_sha256=sha256(source/'features/manifest.json'),
        residual_sha256=sha256(source/'training/step0100.pt'),dictionary_sha256=sha256(source/'anchors.pt'),
        validation_used_for_selection=False,test_used_for_selection=False,
        reason='Fixed transfer of the EnzymeMap recipe; report all predeclared arms without parent fallback')
    atomic_json(dest/'selection.json',freeze)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    evaluate_selected(dest,task,dict(recipe=plan['recipe']),'cuda:0',task['split'])
    result=json.loads((dest/'selected_test_summary.json').read_text())
    result.update(validation_used_for_selection=False,fixed_cross_benchmark_recipe=True,beta=task['beta'],epoch=task['epoch'])
    atomic_json(dest/'selected_test_summary.json',result)
    atomic_json(dest/'complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat()))


def report(out,plan,status):
    lines=['# Fixed-recipe ReactZyme transfer','',
        'Each target split trains its own SLEEC residue-view model on its official training associations. '
        'The fixed EnzymeMap phase-2 recipe is transferred unchanged: 100 positive-CE updates, temperature 0.2, '
        'identity weight 10, residual cap 1 and semantic alpha 0.25. No support gate, parent fallback, test selection or ensemble.','',
        'Epochs 10 and 20 and beta 5/10 are all reported. These are exploratory tests after earlier benchmark inspection. '
        'ReactZyme uses batch 512; the EnzymeMap replication used batch 2,048 and 18 epochs. '
        'The Reaction-Sim beta-5 trajectory recovered from its epoch-5 optimizer checkpoint after a memory collision.','',
        '| Split / beta / epoch | State | R→E all-positive MRR | E→R all-positive MRR |',
        '| --- | --- | ---: | ---: |']
    for task in plan['tasks']:
        p=out/task['name']/'selected_test_summary.json'
        result=json.loads(p.read_text()) if p.exists() else None
        values=[f"{result['summary'][d]['all']['reactzyme_mrr']:.6f}" for d in ('reaction_to_enzyme','enzyme_to_reaction')] if result else ['pending']*2
        label=f"{task['split']} / {task['beta']} / {task['epoch']}"
        if result:label=f"[{label}]({task['name']}/selected_test_summary.json)"
        lines.append('| '+label+' | '+status[task['name']]['stage']+' | '+' | '.join(values)+' |')
    lines+=['','[Predeclared complete task list](protocol.json) · [Execution status](status.json). '
        'A larger test score in this table is not a validation-selected winner.','']
    (out/'comparison.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--task');a=p.parse_args();out=a.campaign.resolve()
    plan=json.loads((out/'protocol.json').read_text())
    if a.task:
        worker(out,plan,next(t for t in plan['tasks'] if t['name']==a.task));return
    pending=list(plan['tasks']);active={};status={t['name']:dict(stage='waiting_for_source') for t in pending}
    while pending or active:
        for name,(proc,stream) in list(active.items()):
            code=proc.poll()
            if code is not None:
                stream.close();status[name]=dict(stage='complete' if code==0 else 'failed',exit_code=code,pid=proc.pid)
                del active[name]
        for task in list(pending):
            if len(active)>=2:break
            if not ready(task):continue
            stream=(out/(task['name']+'.log')).open('w')
            proc=subprocess.Popen([sys.executable,'-u',__file__,'--campaign',str(out),'--task',task['name']],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(task['gpu']),OMP_NUM_THREADS='4',
                MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=stream,stderr=subprocess.STDOUT)
            active[task['name']]=(proc,stream);pending.remove(task)
            status[task['name']]=dict(stage='test_evaluation',pid=proc.pid)
        atomic_json(out/'status.json',status);report(out,plan,status)
        if pending or active:time.sleep(10)
    atomic_json(out/'complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        all_tasks_succeeded=all(v['stage']=='complete' for v in status.values())))


if __name__=='__main__':main()
