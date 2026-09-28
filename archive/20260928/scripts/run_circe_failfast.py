#!/usr/bin/env python3
"""Bounded CPU-only frozen-feature sensitivity checks; no training or source edits."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import numpy as np
import torch
import yaml
from scripts.activity_panels import digest,rows,write_json,write_rows,verify_bundle,validate_scores,evaluate,summarize
from horizyn.benchmarks.retrieval import (BenchmarkTask,build_reaction_inputs,build_query_inputs,
    encode_residue_targets,load_repo_checkpoint,cosine_scores)
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset

GROUPS={
    'reaction_model':['reaction_embedding'],
    'unimol2':['reactant_embeddings','reactant_padding_mask','product_embeddings','product_padding_mask','has_unimol2'],
    'chiro':['reactant_chirality_embeddings','reactant_chirality_padding_mask','product_chirality_embeddings',
             'product_chirality_padding_mask','has_chirality','has_chiro','has_chienn'],
    'reaction_chemistry':['reaction_chemistry_vector','has_reaction_chemistry'],
}


def path(value):
    p=Path(value);return p if p.is_absolute() else ROOT/p


def permutations(n):
    result={}
    for seed in [42,43,44]:
        rng=np.random.default_rng(seed)
        while True:
            p=rng.permutation(n)
            if np.all(p!=np.arange(n)):break
        result[seed]=p
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/circe_failfast_nitrilase_20260918')
    parser.add_argument('--baseline',type=Path,default=ROOT/'runs/activity_panels_nitrilase_v2')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/benchmarks/activity_panels_nitrilase.yaml')
    args=parser.parse_args(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'started.json').exists():raise ValueError('Use a fresh output directory; previous runs are preserved')
    settings=yaml.safe_load(args.config.read_text());bundle=path(settings['benchmark'])
    manifest=verify_bundle(bundle);base=args.baseline.resolve();feat=base/'features'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
    models=['f3','circe_v2_reactzyme','circe_v2_large']
    proteins=rows(bundle/'proteins.csv');variants=rows(bundle/'reactions.csv')
    if manifest['reaction_variant_count']!=manifest['query_count']:raise ValueError('This bounded diagnostic requires one variant per query')
    qids=[r['reaction_id'] for r in variants];pids=[p['protein_id'] for p in proteins]
    perms=permutations(len(qids));start=time.monotonic()
    signature=dict(script_sha256=digest(__file__),config_sha256=digest(args.config),
        baseline_protocol_sha256=digest(base/'protocol.json'),bundle_sha256=digest(bundle/'manifest.json'),
        runtime=dict(torch=torch.__version__,numpy=np.__version__),device='cpu',threads=2,
        model_code={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'horizyn').rglob('*.py'))},
        helper_code={name:digest(ROOT/'scripts'/name) for name in ['activity_panels.py']},
        permutation_reaction_ids={str(seed):[qids[i] for i in p] for seed,p in perms.items()},
        protocol='Baseline, 4 single-modality masks, chemistry-only, 3 independent permutations per modality; frozen checkpoint CPU inference',
        limitations=['Masking and independent feature permutation change the training distribution; sensitivity does not prove that a modality is harmful.',
                     'Three shuffle seeds characterize sensitivity, not training-seed variation or independent biological replication.',
                     'This small previously inspected panel is exploratory; no activity-label fitting and no training.'])
    write_json(out/'started.json',signature)
    results=[]
    for name in models:
        spec=settings['models'][name];dest=out/name;dest.mkdir()
        source=json.loads((base/'scores'/f'{name}.json').read_text())
        if digest(path(spec['checkpoint']))!=source['signature']['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        if digest(path(spec['config']))!=source['signature']['config_sha256']:raise ValueError('Config changed')
        for filename,sha in source['signature']['features'].items():
            if digest(feat/filename)!=sha:raise ValueError(f'Changed baseline feature {filename}')
        print(f'Loading {name}',flush=True)
        config=load_config(str(path(spec['config'])))
        config.data.reaction_chemistry_vectors_path=str(feat/f'chemistry_{name}.npz')
        for m in ['unimol2','chiro','chemistry']:config.data[f'reaction_allow_missing_{m}']=False
        task=BenchmarkTask(name='failfast',task_type='retrieval',dataset='measured_activity',task_label='within_family',split='external',
            pairs=bundle/'encoding_pairs.csv',reactions=bundle/'reactions.csv',
            reaction_model_embeds_h5=feat/'reactiont5.h5',reaction_unimol2_embeds_h5=feat/'unimol2.h5',reaction_chiro_embeds_h5=feat/'chiro.h5')
        reactions=build_reaction_inputs(task,config); inputs=build_query_inputs(reactions,qids,'cpu')
        if not isinstance(inputs,dict):raise ValueError('Expected named multimodal inputs')
        for value in inputs.values():
            if value.shape[0]!=len(qids):raise ValueError('Unexpected input batch axis')
        module,kind=load_repo_checkpoint(path(spec['checkpoint']),config,'cpu')
        if kind!='residue':raise ValueError('Expected residue model')
        module.eval();model=module.model;encoder=model.query_encoder
        if getattr(model,'e2r_adapter',None) is not None or getattr(model,'r2e_adapter',None) is not None:
            raise ValueError('Direction-specific adapters need separate scoring')
        names=list(encoder.modality_names)
        print(f'{name} modality names: {names}; input keys: {sorted(inputs)}',flush=True)
        # Some historical code calls the ChIRo modality chirality or chienn.
        modality_map={k:k for k in GROUPS}
        modality_map['chiro']=next((k for k in ['chiro','chirality','chienn'] if k in names),'chiro')
        if set(modality_map.values())!=set(names):raise ValueError(f'Unexpected modality names: {names}')
        if getattr(encoder,'use_reaction_directional',False):raise ValueError('Separate directional residual needs its own intervention')
        for group,keys in GROUPS.items():
            if not any(k in inputs for k in keys):raise ValueError(f'Missing group {group}')
        dataset=ResidueEmbedDataset(str(feat/'proteins.h5'),max_tokens=1022,truncation='ends_center')
        try:
            with torch.inference_mode():targets=encode_residue_targets(module,dataset,pids,'cpu',8,False)
        finally:dataset.close()
        reference={(r['query_id'],r['protein_id']):float(r['score']) for r in rows(base/'scores'/f'{name}.csv')}
        original_dropout=encoder._apply_modality_dropout
        cases=[('baseline',None,None,None)]
        cases.extend((f'mask_{g}',g,None,None) for g in GROUPS)
        cases.append(('chemistry_only','all_except_chemistry',None,None))
        cases.extend((f'permute_{g}_{seed}',None,g,seed) for g in GROUPS for seed in perms)
        baseline_matrix=None
        for case,mask_group,shuffle_group,seed in cases:
            changed=dict(inputs)
            if shuffle_group:
                index=torch.tensor(perms[seed],dtype=torch.long)
                for key in GROUPS[shuffle_group]:
                    if key in changed:changed[key]=inputs[key].index_select(0,index)
            masked=[] if mask_group is None else ([m for m in names if m!=modality_map['reaction_chemistry']] if mask_group=='all_except_chemistry' else [modality_map[mask_group]])
            def masking(mask):
                mask=original_dropout(mask).clone()
                for m in masked:mask[:,names.index(m)]=False
                if not mask.any(1).all():raise ValueError('All modalities masked')
                return mask
            encoder._apply_modality_dropout=masking
            encoded=[];attentions=[]
            try:
                with torch.inference_mode():
                    for offset in range(0,len(qids),8):
                        part={k:v[offset:offset+8] for k,v in changed.items()}
                        query,attention=model.encode_queries(part,return_attention=True)
                        if list(attention['modality_names'])!=names:raise ValueError('Attention label mismatch')
                        weights=attention['modality'].cpu()
                        if not torch.allclose(weights.sum(1),torch.ones(len(weights)),atol=1e-5):raise ValueError('Invalid attention normalization')
                        for m in masked:
                            if torch.any(weights[:,names.index(m)]!=0):raise ValueError('Intervention failed to mask feature')
                        encoded.append(query);attentions.append(weights)
                    matrix=cosine_scores(torch.cat(encoded),targets).cpu().numpy()
            finally:encoder._apply_modality_dropout=original_dropout
            records=[dict(query_id=r['query_id'],protein_id=p,score=float(matrix[i,j])) for i,r in enumerate(variants) for j,p in enumerate(pids)]
            validate_scores(bundle,records)
            if case=='baseline':
                error=max(abs(r['score']-reference[r['query_id'],r['protein_id']]) for r in records)
                if error>1e-5:raise ValueError(f'Baseline failed to reproduce: {error}')
                baseline_matrix=matrix.copy()
            attention_matrix=torch.cat(attentions).numpy()
            write_rows(dest/f'{case}.csv',records)
            write_rows(dest/f'{case}.attention.csv',[dict(query_id=r['query_id'],**{m:float(attention_matrix[i,j]) for j,m in enumerate(names)}) for i,r in enumerate(variants)])
            summary=summarize(evaluate(bundle,records))
            changed_top1=float(np.mean(matrix.argmax(1)!=baseline_matrix.argmax(1)))
            value=dict(model=name,case=case,mask_group=mask_group,shuffle_group=shuffle_group,seed=seed,
                r2e_ap=summary['r2e/full']['all_queries']['ap'],e2r_ap=summary['e2r/full']['all_queries']['ap'],
                r2e_active_at_5=summary['r2e/full']['all_queries']['active_at_5'],
                e2r_active_at_5=summary['e2r/full']['all_queries']['active_at_5'],
                r2e_changed_top1_fraction=changed_top1,score_rmse_from_baseline=float(np.sqrt(np.mean((matrix-baseline_matrix)**2))),
                attention_mean={m:float(attention_matrix[:,j].mean()) for j,m in enumerate(names)},
                scores_sha256=digest(dest/f'{case}.csv'))
            if case=='baseline':value['max_abs_error_vs_gpu_baseline']=error
            results.append(value);write_json(dest/f'{case}.json',value)
            print(f'{name} {case}: R→E AP={value["r2e_ap"]:.4f}; E→R AP={value["e2r_ap"]:.4f}',flush=True)
        write_json(dest/'complete.json',dict(checkpoint_sha256=source['signature']['checkpoint_sha256'],
            config_sha256=source['signature']['config_sha256'],baseline_score_sha256=digest(base/'scores'/f'{name}.csv'),
            files={p.name:digest(p) for p in sorted(dest.iterdir()) if p.is_file()}))
        del module,model,encoder,targets,inputs,reactions,original_dropout,masking;gc.collect()
    write_json(out/'summary.json',dict(complete=True,signature=signature,results=results,elapsed_seconds=time.monotonic()-start))
    print(f'Completed {len(results)} frozen scoring conditions in {time.monotonic()-start:.1f}s',flush=True)


if __name__=='__main__':main()
