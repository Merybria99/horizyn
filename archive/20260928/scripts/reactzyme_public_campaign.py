#!/usr/bin/env python3
"""Persistent dependency-aware queue for released ReactZyme classifier runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reactzyme_public_features import ROOT, RUN, SPLITS, save_json, sha
from reactzyme_public_train import SOURCES


class AdoptedProcess:
    def __init__(self,pid,output):self.pid=pid;self.output=output
    def poll(self):
        try:
            state=Path(f'/proc/{self.pid}/stat').read_text().split(') ',1)[1].split()[0]
            if state!='Z':return None
        except FileNotFoundError:pass
        return 0 if (self.output/'complete.json').exists() else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--slots-per-gpu',type=int,default=2)
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--patience',type=int,default=30)
    a=p.parse_args();RUN.mkdir(exist_ok=True);logs=RUN/'logs';logs.mkdir(exist_ok=True)
    lock=(RUN/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=[]
    for protein in ('esm2','saprot'):
        for reaction in ('mat_2d','mat_3d','unimol_2d','unimol_3d'):
            for split in SPLITS:
                for family in SOURCES:
                    tasks.append(dict(name=f'{family}_{protein}_{reaction}_{split}_seed42',
                        family=family,protein=protein,reaction=reaction,split=split,seed=42))
    plan_path=RUN/'official_baselines_protocol.json'
    plan=dict(tasks=tasks,primary_families=['mlp','contrastive','transformer','birnn'],
        sensitivity='contrastive_corrected also run because upstream label convention repels positives',
        epochs=a.epochs,patience=a.patience,batch=1000,seed=42,
        data_audit_sha256=sha(RUN/'features/data_audit.json'),
        test_used_for_selection=False,selection='Validation BCE',
        negative_sampling='4 fixed train-only corruptions per positive; released negative files unavailable',
        precision='FP32 downstream training/scoring; BF16 frozen ESM2 extraction',
        scope='96 official-family configurations plus 24 corrected-loss sensitivity runs; structural FANN extension absent from released training scripts')
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Existing plan differs')
    else:save_json(plan_path,plan)
    active={};failed=set();slots=[(g,s) for s in range(a.slots_per_gpu) for g in range(4)]
    # A supervisor restart must not duplicate still-running training children.
    status=RUN/'official_baselines_status.json'
    if status.exists():
        lookup={t['name']:t for t in tasks}
        for rec in json.loads(status.read_text()).get('active',[]):
            try:cmd=Path(f"/proc/{rec['pid']}/cmdline").read_bytes().decode().split('\0')
            except FileNotFoundError:continue
            task=lookup[rec['name']]
            if not any(x.endswith('/reactzyme_public_train.py') for x in cmd):continue
            if not all(cmd[cmd.index('--'+key)+1]==task[key] for key in ('family','split','protein','reaction')):
                raise ValueError('Process identity mismatch')
            slot=next(s for s in slots if s[0]==rec['gpu'] and s not in active)
            active[slot]=(AdoptedProcess(rec['pid'],RUN/'models'/task['name']),task,
                (logs/f"{task['name']}.log").open('a'))
            print(json.dumps(dict(event='adopted',name=task['name'],pid=rec['pid'])),flush=True)
    while True:
        for slot,(proc,task,log) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();del active[slot]
            if code!=0:failed.add(task['name'])
            print(json.dumps(dict(event='finished',name=task['name'],exit_code=code)),flush=True)
        completed={t['name'] for t in tasks if (RUN/'models'/t['name']/'complete.json').exists()}
        failed-=completed
        running={t['name'] for _,t,_ in active.values()};waiting=[];ready=[]
        for task in tasks:
            if task['name'] in completed|running|failed:continue
            missing=[n for n in (task['protein'],task['reaction']) if not (RUN/'features'/f'{n}.complete.json').exists()]
            if missing:waiting.append(dict(name=task['name'],missing_features=missing))
            else:ready.append(task)
        for slot in slots:
            if slot in active or not ready:continue
            task=ready.pop(0);gpu,_=slot
            cmd=[sys.executable,'-u',str(ROOT/'scripts/reactzyme_public_train.py'),
                 '--family',task['family'],'--split',task['split'],'--protein',task['protein'],
                 '--reaction',task['reaction'],'--gpu',str(gpu),'--epochs',str(a.epochs),
                 '--patience',str(a.patience)]
            if (RUN/'models'/task['name']/'protocol.json').exists():cmd.append('--resume')
            log=(logs/f"{task['name']}.log").open('a')
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
            active[slot]=(proc,task,log)
            print(json.dumps(dict(event='started',name=task['name'],gpu=gpu,pid=proc.pid)),flush=True)
        state=dict(updated_unix=time.time(),total=len(tasks),completed=len(completed),
            active=[dict(name=t['name'],gpu=slot[0],pid=p.pid) for slot,(p,t,_) in active.items()],
            failed=sorted(failed),ready=len(ready),waiting_features=waiting)
        save_json(RUN/'official_baselines_status.json',state)
        rows=[]
        for task in tasks:
            result=RUN/'models'/task['name']/'complete.json'
            if result.exists():
                r=json.loads(result.read_text());v=r['test']
                rows.append(f"| {task['family']} | {task['protein']} | {task['reaction']} | {task['split']} | {r['selected_epoch']} | {v['reaction_to_enzyme']['all']['reactzyme_mrr']:.6f} | {v['enzyme_to_reaction']['all']['reactzyme_mrr']:.6f} |")
        report='# ReactZyme public baseline campaign\n\n'
        report+=f"Completed: {len(completed)}/{len(tasks)}. Running: {len(active)}. Failed: {len(failed)}. Waiting for features: {len(waiting)}.\n\n"
        report+='These are retrained matched-data runs with documented protocol adaptations, not copied paper scores. Seed 42 is the initial comparison; no seed or model is selected using test results.\n\n'
        report+='| Family | Protein | Reaction | Split | Selected epoch | R→E all-positive MRR | E→R all-positive MRR |\n|---|---|---|---|---:|---:|---:|\n'+'\n'.join(rows)+'\n'
        (RUN/'results.md').write_text(report)
        if len(completed|failed)==len(tasks) and not active:break
        time.sleep(15)


if __name__=='__main__':main()
