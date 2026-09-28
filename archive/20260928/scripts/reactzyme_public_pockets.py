#!/usr/bin/env python3
"""Extract missing EnzymeCAGE pockets as exact public structures arrive."""
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/reactzyme_public_baselines_20260921'


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    source=ROOT/'.deps/enzymecage_p450_official/scripts/extract_p2rank_pockets.py'
    spec=importlib.util.spec_from_file_location('native_public_cage_pockets',source)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    native.check_java_version('java')
    p2rank=RUN/'assets/p2rank_2.5.1'
    if not (p2rank/'prank').exists():raise FileNotFoundError(p2rank)
    out=RUN/'assets/recovered_pockets';out.mkdir(exist_ok=True)
    import fcntl
    lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    missing=set(json.loads((RUN/'features/enzymecage_pocket_coverage.json').read_text())['missing_ids'])
    completed=set();successful=set();failed=set();next_batch=0
    for receipt in out.glob('batch_*/receipt.json'):
        rec=json.loads(receipt.read_text());completed.update(rec['attempted']);successful.update(rec['successful']);failed.update(rec['failed'])
        next_batch=max(next_batch,int(receipt.parent.name.split('_')[1])+1)
    manifest=RUN/'assets/alphafold/manifest.jsonl';position=0;ready={}
    while True:
        if manifest.exists():
            with manifest.open() as stream:
                stream.seek(position)
                while True:
                    begin=stream.tell();line=stream.readline()
                    if not line:position=stream.tell();break
                    if not line.endswith('\n'):position=begin;break
                    row=json.loads(line);pid=row['protein_id']
                    if pid in missing and pid not in completed and row['exact_sequence_match']:
                        ready[pid]=ROOT/row['path']
        status_path=RUN/'assets/alphafold/status.json'
        acquisition=json.loads(status_path.read_text()) if status_path.exists() else {}
        priority_done=acquisition.get('checked',0)>=len(missing)
        if len(ready)>=128 or (priority_done and ready):
            chosen=sorted(ready)[:128];structures={p:ready.pop(p) for p in chosen}
            batch=out/f'batch_{next_batch:05d}';batch.mkdir(exist_ok=True);next_batch+=1
            save(out/'status.json',dict(stage='extracting',batch=str(batch),attempted=len(completed),
                successful=len(successful),failed=len(failed),current_batch=len(chosen),pending=len(ready)))
            raw=native.run_p2rank(structures,p2rank,batch,threads=16)
            table=native.extract_pockets(structures,raw,batch,pocket_rank=1)
            import csv
            with table.open() as f:ok={r['UniprotID'] for r in csv.DictReader(f)}
            bad=set(chosen)-ok;completed.update(chosen);successful.update(ok);failed.update(bad)
            save(batch/'receipt.json',dict(attempted=chosen,successful=sorted(ok),failed=sorted(bad),
                native_source=str(source),p2rank='2.5.1',configuration='alphafold',pocket_rank=1,
                note='No reaction/activity labels used in pocket prediction'))
            print(json.dumps(dict(batch=next_batch-1,successful=len(ok),failed=len(bad),total_successful=len(successful))),flush=True)
            continue
        done=priority_done and not ready
        save(out/'status.json',dict(stage='priority_complete' if done else 'waiting_for_structures',
            attempted=len(completed),successful=len(successful),failed=len(failed),pending=len(ready),
            public_structure_unresolved=acquisition.get('unresolved'),
            full_reactzyme_pocket_coverage_estimate=160241+len(successful),
            note='Existing pocket residue alignment and downstream graph features still require verification'))
        if done:break
        time.sleep(30)


if __name__=='__main__':main()
