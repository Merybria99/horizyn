#!/usr/bin/env python3
"""Check exact export parity and time ordered CPU prefetch on a real checkpoint."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from generalization_clipzyme_f3_screen import model_from_checkpoint,export_device_lock,sha256
from generalization_gpu_budget import free_memory_mib
from generalization_parallel_residue_export import batches
from horizyn.training_io import StoragePrecisionResidues
from horizyn.benchmarks.retrieval import encode_residue_targets


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--count',type=int,default=8192);a=p.parse_args()
    a.output.mkdir(exist_ok=False);device=f'cuda:{a.gpu}'
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    keys=json.loads(a.catalog.read_text())['proteins'][:a.count]
    with export_device_lock(device):
        while free_memory_mib(a.gpu)<27000:time.sleep(5)
        model,config=model_from_checkpoint(a.config,a.checkpoint,device)
        data=StoragePrecisionResidues(config.data.protein_residue_embeds_path,
            max_tokens=config.data.max_protein_tokens,truncation=config.data.protein_truncation)
        records=[];reference=None
        for method in ('serial','parallel','parallel','serial'):
            start=time.monotonic()
            if method=='serial':result=encode_residue_targets(model,data,keys,device,128,False).float().cpu()
            else:result=torch.cat([v for _,v in batches(model,data,keys,device,128,4)])
            elapsed=time.monotonic()-start
            if reference is None:reference=result
            error=float((result-reference).abs().max())
            records.append(dict(method=method,seconds=elapsed,bitwise_equal=torch.equal(result,reference),max_abs_error=error))
            print(json.dumps(records[-1]),flush=True)
            if error>0:raise ValueError('Parallel export changed embeddings')
        data.close()
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),proteins=len(keys),batch_size=128,workers=4,
        records=records,checkpoint_sha256=sha256(a.checkpoint),config_sha256=sha256(a.config),
        helper_sha256=sha256(ROOT/'scripts/generalization_parallel_residue_export.py'),
        note='Same candidate order, padding, FP32 inference, exact stored transport. Timings include worker startup; serial/parallel order reversed on repeat.')
    (a.output/'parity.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
