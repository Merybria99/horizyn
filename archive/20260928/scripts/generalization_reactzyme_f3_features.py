#!/usr/bin/env python3
"""Regenerate learned F3 coordinates while reusing exact fit-free ReactZyme inputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import os
import subprocess
import time
from generalization_gpu_budget import free_memory_mib

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from horizyn.benchmarks.retrieval import BenchmarkTask,build_reaction_inputs,encode_reactions,encode_residue_targets
from horizyn.training_io import StoragePrecisionResidues
from generalization_clipzyme_f3_screen import model_from_checkpoint,export_device_lock,sha256
from generalization_export import atomic_json,atomic_npz


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=('train_validation','test'),required=True)
    p.add_argument('--split',choices=('reaction_smi','enzyme_smi','time'),default='reaction_smi')
    p.add_argument('--freeze',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=128)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.scope=='test' and (a.freeze is None or not a.freeze.is_file()):
        raise ValueError('Test export requires a recorded validation selection')
    registry=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope=a.scope,
        checkpoint=dict(path=str(a.checkpoint.resolve()),sha256=sha256(a.checkpoint)),
        config=dict(path=str(a.config.resolve()),sha256=sha256(a.config)),
        template=dict(path=str(a.template.resolve()),manifest_sha256=sha256(a.template/'manifest.json')),
        source_sha256=sha256(Path(__file__)),precision='FP32 inference; exact stored residue transport',
        test_labels_read=False,training_data_changed=False)
    if a.freeze:registry['selection']=dict(path=str(a.freeze.resolve()),sha256=sha256(a.freeze))
    atomic_json(a.output/'export_registry.json',registry)
    catalog=json.loads((a.template/'catalog.json').read_text())
    for name in ('catalog.json','reaction_features.npz','protein_mean.h5'):
        (a.output/name).symlink_to((a.template/name).resolve())
    if a.scope=='train_validation':
        (a.output/'pairs.npz').symlink_to((a.template/'pairs.npz').resolve())
    original=json.loads((a.template/'manifest.json').read_text())
    manifest=dict(schema='reactzyme_target_f3_features_v1',scope=a.scope,
        reused_raw_inputs={name:dict(path=str((a.template/name).resolve()),sha256=sha256(a.template/name))
                          for name in ('catalog.json','reaction_features.npz','protein_mean.h5')},
        input_manifest=dict(path=str(a.template/'manifest.json'),sha256=sha256(a.template/'manifest.json')),
        checkpoint=registry['checkpoint'],f3_export_config=registry['config'],
        training_edges_only=a.scope=='train_validation',test_used=a.scope=='test',
        source_sha256=registry['source_sha256'])
    if a.scope=='train_validation':manifest['training_graph']=dict(path=str(a.template/'pairs.npz'),sha256=sha256(a.template/'pairs.npz'))
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    with export_device_lock(a.device):
        logical=int(str(a.device).partition(':')[2] or 0)
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        physical=visible.split(',')[logical].strip() if visible else str(logical)
        if str(a.device).startswith('cuda'):
            while True:
                free_mib=free_memory_mib(physical)
                if free_mib>=27000:break
                print(json.dumps(dict(waiting_for_export_free_mib=free_mib,required_mib=27000)),flush=True)
                time.sleep(5)
        model,config=model_from_checkpoint(a.config,a.checkpoint,a.device)
        if a.scope=='train_validation':
            for key in ('train_pairs_path','validation_pairs_path','train_reactions_path','validation_reactions_path'):
                path=Path(config.data[key]);path=path if path.is_absolute() else ROOT/path
                if sha256(path)!=original['inputs'][key]['sha256']:
                    raise ValueError(f'Target training/validation source differs: {key}')
        residue_path=Path(config.data.protein_residue_embeds_path)
        cache_receipt=residue_path.with_suffix('.receipt.json')
        manifest['residue_cache']=dict(path=str(residue_path),receipt_sha256=sha256(cache_receipt))
        residue=StoragePrecisionResidues(str(residue_path),max_tokens=config.data.max_protein_tokens,
                                         truncation=config.data.protein_truncation)
        try:
            # Bound export memory and checkpoint progress independently of the
            # trainer. Small blocks also make an interrupted export reviewable.
            proteins=np.zeros((len(catalog['proteins']),512),np.float32)
            for start in range(0,len(proteins),4096):
                keys=catalog['proteins'][start:start+4096]
                proteins[start:start+len(keys)]=encode_residue_targets(model,residue,keys,a.device,a.batch_size,False).float().cpu().numpy()
                atomic_json(a.output/'progress.json',dict(encoded_proteins=start+len(keys),total=len(proteins)))
        finally:residue.close()
        reactions=np.zeros((len(catalog['reactions']),512),np.float32)
        lookup={key:i for i,key in enumerate(catalog['reactions'])}
        train_reactions=None
        for split in (('train','validation') if a.scope=='train_validation' else ('test',)):
            prefix='validation' if split=='test' else split
            config.data.reaction_chemistry_vectors_path=config.data[f'{prefix}_reaction_chemistry_vectors_path']
            task=BenchmarkTask(name=f'reactzyme_{split}',task_type='retrieval',dataset='ReactZyme',
                task_label=a.split,split=split,bidirectional_reactions=True,
                pairs=ROOT/config.data[f'{split}_pairs_path'],
                reactions=ROOT/config.data[f'{split}_reactions_path'],
                reaction_model_embeds_h5=ROOT/config.data[f'{prefix}_reaction_t5v2_embeds_path'],
                reaction_unimol2_embeds_h5=ROOT/config.data[f'{prefix}_reaction_unimol2_embeds_path'],
                reaction_chiro_embeds_h5=ROOT/config.data[f'{prefix}_reaction_chiro_embeds_path'])
            inputs=build_reaction_inputs(task,config)
            keys=catalog[f'{split}_reactions']
            forward_keys=[key+'_f' for key in keys]
            if not set(forward_keys)<=set(inputs.keys):raise ValueError('Missing forward reaction input')
            vectors=encode_reactions(model,inputs,forward_keys,a.device,min(a.batch_size,128)).float().cpu().numpy()
            reactions[[lookup[k] for k in keys]]=vectors
            if split=='train':train_reactions=vectors
    if not np.isfinite(proteins).all() or not np.isfinite(reactions).all():raise ValueError('Nonfinite learned features')
    values=dict(proteins=proteins,reactions=reactions)
    if train_reactions is not None:values['train_reactions']=train_reactions
    atomic_npz(a.output/'f3_features.npz',**values)
    atomic_json(a.output/'manifest.json',manifest)
    atomic_json(a.output/'complete.json',dict(manifest_sha256=sha256(a.output/'manifest.json'),
        f3_features_sha256=sha256(a.output/'f3_features.npz'),checkpoint_sha256=registry['checkpoint']['sha256'],
        test_labels_read=False,completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
