#!/usr/bin/env python3
"""Run recorded epoch extensions after successful source training checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path,data):
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();out=a.campaign.resolve();plan=json.loads((out/'protocol.json').read_text());root=Path(__file__).resolve().parents[1]
    def run(task):
        path=Path(task['run_root']);checkpoint=Path(task['checkpoint']);receipt=Path(task['prerequisite_receipt'])
        write(path/'state.json',dict(stage='waiting_for_source_checkpoint'))
        while not receipt.exists() or not checkpoint.exists():time.sleep(10)
        if json.loads(receipt.read_text()).get('exit_code',0)!=0:
            write(path/'failure.json',dict(error='Source training did not complete successfully'));return False
        if hashlib.sha256(Path(task['config']).read_bytes()).hexdigest()!=task['config_sha256']:
            raise ValueError('Training extension config changed')
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--id',str(task['gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=35000:break
            time.sleep(10)
        command=[sys.executable,'-u',str(root/'scripts/train_protein_pooling_fast_io.py'),
            '--config',task['config'],'--resume',str(checkpoint),'--io-mode','residue',
            '--io-output-dir',str(path/'training'),'--io-prefetch','1']
        with (path/'training.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(task['gpu']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            write(path/'execution.json',dict(pid=process.pid,command=command,started_utc=datetime.now(timezone.utc).isoformat(),
                resumed_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            write(path/'state.json',dict(stage='training',pid=process.pid))
            code=process.wait()
        write(path/'complete.json',dict(exit_code=code,completed_utc=datetime.now(timezone.utc).isoformat()))
        write(path/'state.json',dict(stage='complete' if code==0 else 'failed'))
        return code==0
    with ThreadPoolExecutor(max_workers=len(plan['tasks'])) as pool:results=list(pool.map(run,plan['tasks']))
    write(out/'complete.json',dict(success=all(results),completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
