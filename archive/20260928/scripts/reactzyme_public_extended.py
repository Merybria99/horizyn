#!/usr/bin/env python3
"""Persistent public-competitor queue and consolidated campaign report.

Structural methods stay explicitly pending until complete, matched inputs are
available. They are never silently replaced by sequence-only ablations.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/reactzyme_public_baselines_20260921'
SPLITS=('reaction_smi','enzyme_smi','time')


def write_json(path,value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def find_training(task):
    """Recover this exact task across supervisor restarts without duplicates."""
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:cmd=(entry/'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if not any(v.endswith('/'+task['script']) or v==task['script'] for v in cmd):continue
        if all(flag in cmd and cmd[cmd.index(flag)+1]==value for flag,value in task['identity'].items()):
            if live(int(entry.name)):return int(entry.name)
    return None


def report(status):
    base=json.loads((RUN/'official_baselines_status.json').read_text())
    lines=['# ReactZyme: retrained public competitors','',
        f"Updated UTC: {time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())}.",'',
        f"Official-family matrix: **{base['completed']}/{base['total']} complete**, {len(base['active'])} running; {len(base['failed'])} failed.",
        f"Additional runnable comparisons: **{len(status['completed'])}/{status['total']} complete**, {len(status['active'])} running.",'',
        'All scored rows use the exact V4 ReactZyme splits, complete candidate pools, candidate order, and shared all-positive MRR evaluator. Downstream weights are initialized afresh. Checkpoints are selected on validation only. This is a single-seed initial campaign, not a completed multi-seed architectural superiority claim.','',
        'See [protocol](../../documents/reactzyme_public_baselines_protocol_20260921.md) for frozen backbones, objectives, hyperparameters, input adapters, missing inputs, and source provenance.','',
        '## External comparator test results','',
        '| Method | Split | Selected epoch | R→E MRR | E→R MRR |',
        '|---|---|---:|---:|---:|']
    for path in sorted((RUN/'models').glob('*/complete.json')):
        row=json.loads(path.read_text())
        if not (row.get('method') or row.get('protein')=='enzgfm650'):continue
        label=row.get('method','EnzGFM-650M frozen residue-mean backbone + released ReactZyme Transformer')
        val=row['test'];lines.append(f"| {label} | {row['split']} | {row['selected_epoch']} | {val['reaction_to_enzyme']['all']['reactzyme_mrr']:.6f} | {val['enzyme_to_reaction']['all']['reactzyme_mrr']:.6f} |")
    lines+=['','## Running and waiting','', '| Task | State | Detail |','|---|---|---|']
    for row in status['active']:lines.append(f"| {row['name']} | running | GPU {row['gpu']}, PID {row['pid']} |")
    for row in status['waiting']:lines.append(f"| {row['name']} | waiting | {', '.join(row['dependencies'])} |")
    for name in status['failed']:lines.append(f'| {name} | failed | Inspect its log before retrying |')
    structure_status=RUN/'assets/alphafold/status.json'
    if structure_status.exists():
        s=json.loads(structure_status.read_text())
        lines+=['',f"Public structure acquisition: {s['exact_structures']:,} exact-sequence coordinate files among {s['checked']:,}/{s['total']:,} checked; {s['unresolved']:,} unresolved. Coordinates alone do not complete pockets or native graph preprocessing."]
    lines+=['','## Structural and directional input gaps','',
        '| Competitor | Current status |','|---|---|',
        '| CLIPZyme full structural model | Pending complete full-protein structures and a justified adapter for ReactZyme participant-only reactions; not training. |',
        '| EnzymeCAGE full structural model | Local released pockets match 160,241/178,327 sequences; 18,086 missing. Reaction roles/centers are also absent in the benchmark inputs. Not training. |',
        '| VenusRXN | Public/source implementation tracked; mapped directed reactions required. No matched-input training launched. |',
        '| TIGER / FGW-CLIP | Published-score references; official reproducible release not verified by the source audit. |','',
        'No missing candidates are dropped and no transferred supervised retrieval checkpoints are reported as retrained results.','',
        '## Official ReactZyme matrix','',
        '[All completed official-family test rows](results.md). The 96 official configurations and 24 corrected-contrastive-loss sensitivity configurations are separately labeled.','']
    (RUN/'comparison.md').write_text('\n'.join(lines))


def main():
    lock=(RUN/'extended_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=[]
    for split,gpu in zip(SPLITS,(3,0,1)):
        attention,checkpoint=('eager','block') if split=='reaction_smi' else ('sdpa','ffn')
        tasks.append(dict(name=f'creep_two_modality_{split}_seed42',kind='creep',gpu=gpu,
            script='reactzyme_public_creep.py',identity={'--split':split},
            args=['train','--split',split,'--gpu',str(gpu),'--batch','64','--attention',attention,'--checkpoint-policy',checkpoint],
            dependencies=['features/creep_tokens.complete.json','fidelity_tests.complete.json']))
    for reaction in ('mat_2d','mat_3d','unimol_2d','unimol_3d'):
        for split in SPLITS:
            tasks.append(dict(name=f'transformer_enzgfm650_{reaction}_{split}_seed42',kind='enzgfm',gpu=2,
                script='reactzyme_public_train.py',identity={'--protein':'enzgfm650','--reaction':reaction,'--split':split},
                args=['--family','transformer','--protein','enzgfm650','--reaction',reaction,'--split',split,'--gpu','2'],
                dependencies=['features/enzgfm650.complete.json',f'features/{reaction}.complete.json']))
    # Horizyn is already complete on all splits; include its receipts in totals.
    horizyn=[f'horizyn_participant_set_{s}_seed42' for s in SPLITS]
    write_json(RUN/'extended_protocol.json',dict(tasks=tasks,completed_initial_horizyn=horizyn,
        selection_uses_test=False,pending=['full CLIPZyme','full EnzymeCAGE','VenusRXN'],
        note='CREEP and Horizyn participant-only adapters and EnzGFM preprocessing deviations are disclosed in the protocol.'))
    active={};failed=set()
    for t in tasks:
        pid=find_training(t)
        if pid:active[t['name']]=(pid,t,None)
    while True:
        for name,(pid,t,log) in list(active.items()):
            if live(pid):continue
            if log:log.close()
            del active[name]
            if not (RUN/'models'/name/'complete.json').exists():failed.add(name)
        completed=[t['name'] for t in tasks if (RUN/'models'/t['name']/'complete.json').exists()]
        completed += [n for n in horizyn if (RUN/'models'/n/'complete.json').exists()]
        failed-=set(completed);waiting=[]
        for t in tasks:
            if t['name'] in set(completed)|set(active)|failed:continue
            missing=[v for v in t['dependencies'] if not (RUN/v).exists()]
            if t['kind']=='enzgfm' and any(other['kind']=='enzgfm' for _,other,_ in active.values()):missing+=['single EnzGFM training slot']
            if missing:
                waiting.append(dict(name=t['name'],dependencies=missing));continue
            args=list(t['args']);out=RUN/'models'/t['name']
            if (out/'protocol.json').exists():
                if t['kind']=='creep' and not (out/'last.pt').exists():
                    failed.add(t['name']);continue
                args+=['--resume']
            log=(RUN/'logs'/f"{t['name']}.log").open('a')
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/t['script']),*args],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
            active[t['name']]=(proc.pid,t,log)
            print(json.dumps(dict(event='started',name=t['name'],pid=proc.pid,gpu=t['gpu'])),flush=True)
        status=dict(updated_unix=time.time(),total=len(tasks)+len(horizyn),completed=completed,
            active=[dict(name=name,pid=pid,gpu=t['gpu']) for name,(pid,t,_) in active.items()],
            waiting=waiting,failed=sorted(failed))
        write_json(RUN/'extended_status.json',status);report(status)
        if len(completed)+len(failed)==status['total'] and not active:break
        time.sleep(30)


if __name__=='__main__':main()
