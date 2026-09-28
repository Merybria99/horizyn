#!/usr/bin/env python3
"""Recover equations without enzyme labels; map, validate and export optional CGRs."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horizyn.chemistry.reaction_recovery import ReactionIndex, build_cgr, read_reactions
from horizyn.generalization_diagnostics import atomic_json


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def recover(args):
    source=args.source_root/args.split
    paths=[args.rhea] + [source/f'{part}_rxns.csv' for part in args.partitions]
    signature=dict(schema=1, split=args.split, partitions=args.partitions,
        inputs={str(p.resolve()):digest(p) for p in paths},
        code={str(p):digest(p) for p in (Path(__file__).resolve(),ROOT/'horizyn/chemistry/reaction_recovery.py')},
        search=dict(max_candidates=args.max_candidates,max_equations=args.max_equations,max_nodes=args.max_nodes))
    manifest=args.output/'manifest.json'
    if manifest.exists():
        previous=load(manifest)
        if previous['signature']!=signature:
            raise ValueError('Inputs/code/options changed; use a new output directory')
        for name, expected in previous['outputs'].items():
            if digest(args.output/name)!=expected: raise ValueError(f'Recovery output changed: {name}')
        print('Reusing verified recovery output',flush=True)
        return
    if (args.output/'recovery.json').exists() or (args.output/'equations.json').exists():
        raise ValueError('Incomplete preparation exists; use a new output directory')
    index=ReactionIndex(args.rhea)
    results, memo, by_id = {}, {}, {}
    for part in args.partitions:
        rows=read_reactions(source/f'{part}_rxns.csv')
        results[part]={}
        for i,(query,smiles) in enumerate(rows.items()):
            if query in by_id and by_id[query]!=smiles:
                raise ValueError(f'Reaction ID has conflicting chemistry across splits: {query}')
            by_id[query]=smiles
            if smiles not in memo:
                memo[smiles]=index.recover(smiles,max_candidates=args.max_candidates,
                    max_equations=args.max_equations,max_nodes=args.max_nodes)
            results[part][query]=memo[smiles]
            if (i+1)%500==0: print(f'Recover {part}: {i+1}/{len(rows)}',flush=True)
    selected={eid for table in results.values() for r in table.values() for eid in r['equation_ids']}
    equations={eid:index.equations[eid] for eid in sorted(selected)}
    atomic_json(args.output/'recovery.json',results)
    atomic_json(args.output/'equations.json',equations)
    atomic_json(args.output/'invalid_rhea_rows.json',index.invalid_sources)
    atomic_json(manifest,dict(signature=signature,
        policy='reaction chemistry only; no protein/EC/pair labels; no splits or source files modified',
        decomposition_policy='bounded hypotheses only; never promoted to usable features',
        outputs={name:digest(args.output/name) for name in
                 ('recovery.json','equations.json','invalid_rhea_rows.json')}))


def verify(args):
    manifest=load(args.output/'manifest.json')
    for path, expected in {**manifest['signature']['inputs'],**manifest['signature']['code']}.items():
        if digest(path)!=expected: raise ValueError(f'Input/code changed: {path}')
    for name, expected in manifest['outputs'].items():
        if digest(args.output/name)!=expected: raise ValueError(f'Recovery changed: {name}')
    return manifest


def map_equations(args):
    verify(args)
    os.environ['CUDA_VISIBLE_DEVICES']=''  # CPU only; never contend with training GPUs.
    import torch
    torch.set_num_threads(args.threads)
    from rxnmapper import RXNMapper
    import rxnmapper
    mapper=RXNMapper()
    # Fingerprint weights + vocabulary: caches must never survive a model change silently.
    model_path=Path(mapper.model_path)
    model_files=[p for p in sorted(model_path.iterdir()) if p.suffix in ('.bin','.json','.txt','.safetensors')]
    if not model_files: raise ValueError('Cannot fingerprint RXNMapper model')
    model_identity=dict(version=getattr(rxnmapper,'__version__','unknown'),
                        files={p.name:digest(p) for p in model_files})
    settings=dict(recovery_manifest=digest(args.output/'manifest.json'), model=model_identity,
                  min_confidence=args.min_confidence,max_atoms=args.max_atoms)
    settings_path=args.output/'mapping_settings.json'
    if settings_path.exists() and load(settings_path)!=settings:
        raise ValueError('Mapping model/options changed; use a new output directory')
    atomic_json(settings_path,settings)
    settings_hash=digest(settings_path)
    results=load(args.output/'recovery.json')
    equations=load(args.output/'equations.json')
    # No mapping of hypotheses/ambiguous assignments: they cannot yet be query features.
    wanted=sorted({r['equation_ids'][0] for table in results.values() for r in table.values()
                   if r['status']=='unique_candidate'})
    destination=args.output/'mapped';destination.mkdir(exist_ok=True)
    attempted=0;start=time.monotonic();last_report=start
    for eid in wanted:
        output=destination/f'{eid}.json'
        if output.exists():
            existing=load(output)
            if existing['settings_sha256']!=settings_hash: raise ValueError('Stale mapping cache')
            if not args.retry_failed or existing['status']!='mapping_failed': continue
        if args.limit is not None and attempted>=args.limit: break
        eq=equations[eid];smiles=eq['reaction_smiles']
        record=dict(equation_id=eid,reaction_smiles=smiles,settings_sha256=settings_hash)
        if eq['checks']['status']!='eligible':
            record.update(status='rejected_equation',reason=eq['checks']['status'])
        elif eq['checks']['heavy_atoms']>args.max_atoms:
            record.update(status='mapping_size_limit',reason='heavy_atom_limit')
        else:
            encoded=mapper.tokenizer(smiles)
            token_limit=mapper.model.config.max_position_embeddings
            # RXNMapper may otherwise silently truncate long reaction SMILES.
            if len(encoded['input_ids'])>token_limit:
                record.update(status='mapping_size_limit',reason='token_limit')
            else:
                try:
                    with torch.inference_mode():
                        result=mapper.get_attention_guided_atom_maps([smiles])[0]
                    confidence=float(result['confidence'])
                    if not math.isfinite(confidence) or not 0<=confidence<=1:
                        raise ValueError('Invalid mapper confidence')
                    record.update(mapped_smiles=result['mapped_rxn'],mapping_confidence=confidence)
                    if confidence<args.min_confidence:
                        record.update(status='low_mapping_confidence')
                    else:
                        try:
                            graph=build_cgr(smiles,result['mapped_rxn'])
                            record.update(status='structurally_valid_cgr',graph=graph)
                        except ValueError as error:
                            record.update(status='rejected_mapping',reason=str(error))
                except Exception as error:
                    record.update(status='mapping_failed',reason=f'{type(error).__name__}: {error}')
        atomic_json(output,record)
        attempted+=1
        if attempted%10==0:
            print(f'Mapping: {attempted} new records, {time.monotonic()-start:.1f}s elapsed',flush=True)
        # Avoid repeated directory scans on shared NFS after every small batch.
        if time.monotonic()-last_report>=300:
            report(args,quiet=True)
            last_report=time.monotonic()
    print(f'Mapping pass ended: {attempted} new records. Rerun to resume.',flush=True)


def report(args,quiet=False):
    manifest=verify(args)
    recovery=load(args.output/'recovery.json');equations=load(args.output/'equations.json')
    settings=args.output/'mapping_settings.json'
    settings_hash=digest(settings) if settings.exists() else None
    mapped={}
    for path in sorted((args.output/'mapped').glob('*.json')):
        row=load(path)
        if row['settings_sha256']!=settings_hash: raise ValueError(f'Stale mapped record: {path}')
        if row['equation_id'] not in equations: raise ValueError(f'Unknown mapped equation: {path}')
        mapped[row['equation_id']]=row
    summary=dict(split=manifest['signature']['split'],partitions={},
        mapping_status_counts=dict(Counter(r['status'] for r in mapped.values())),
        caveat='Structurally valid does not certify atom-mapping accuracy. Relaxed assignments remain candidates.',
        usable_policy='unique candidate + concrete balanced equation + chemistry-preserving bijective heavy-atom mapping + nonempty center',
        direction_policy='Rhea side assignment; physiological direction unknown for unordered queries')
    training={r['equation_ids'][0] for r in recovery.get('train',{}).values() if r['status']=='unique_candidate'}
    for part,table in recovery.items():
        counts=Counter();ready=Counter();masks={};equation_states=Counter();overlap=0
        for query, row in table.items():
            counts[row['status']]+=1
            mask=False;eid=None
            if row['status']=='unique_candidate':
                eid=row['equation_ids'][0]
                state=equations[eid]['checks']['status'];equation_states[state]+=1
                if part!='train' and eid in training: overlap+=1
                mapping=mapped.get(eid,{})
                mask=mapping.get('status')=='structurally_valid_cgr'
                if mask: ready[row['mode']]+=1
            masks[query]=dict(mask=mask,equation_id=eid,
                recovery_status=row['status'], match_mode=row.get('mode'),
                mapping_status=mapped.get(eid,{}).get('status','pending' if eid else 'not_assigned'))
        assert len(masks)==len(table)
        atomic_json(args.output/f'{part}_cgr_index.json',masks)
        summary['partitions'][part]=dict(queries=len(table),recovery_counts=dict(counts),
            equation_check_counts=dict(equation_states),usable_cgr_queries=sum(ready.values()),
            usable_fraction=sum(ready.values())/len(table),usable_by_match_mode=dict(ready),
            unique_candidate_equation_overlap_with_train=overlap if part!='train' else None)
    unique={r['equation_ids'][0] for table in recovery.values() for r in table.values() if r['status']=='unique_candidate'}
    summary['unique_equations_to_process']=len(unique)
    summary['pending_equations']=len(unique-set(mapped))
    summary['mapping_pass_complete']=not summary['pending_equations']
    atomic_json(args.output/'report.json',summary)
    if not quiet: print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')  # Rejections are recorded explicitly in JSON.
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('recover','map','report','all'))
    parser.add_argument('--split',choices=('reaction_smi','enzyme_smi','time'),default='reaction_smi')
    parser.add_argument('--partitions',nargs='+',choices=('train','validation','test'),default=['train','validation'])
    parser.add_argument('--source-root',type=Path,default=ROOT/'data/revised_protocols/reactzyme_paper')
    parser.add_argument('--rhea',type=Path,default=ROOT/'data/paper/reactzyme/raw/rhea_molecules.tsv')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-candidates',type=int,default=32)
    parser.add_argument('--max-equations',type=int,default=4)
    parser.add_argument('--max-nodes',type=int,default=3000)
    parser.add_argument('--max-atoms',type=int,default=200)
    parser.add_argument('--min-confidence',type=float,default=0.0,
                        help='Optional mapper-score filter; not a calibrated probability; structural checks always required')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--limit',type=int,help='Maximum new mapping records this invocation; resumable')
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    if any(x<=0 for x in (args.max_candidates,args.max_equations,args.max_nodes,args.max_atoms,args.threads)):
        parser.error('Limits and thread count must be positive')
    if not 0<=args.min_confidence<=1 or (args.limit is not None and args.limit<0):
        parser.error('Invalid confidence/limit')
    if len(set(args.partitions))!=len(args.partitions): parser.error('Duplicate partitions')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage in ('recover','all'): recover(args)
        if args.stage in ('map','all'): map_equations(args)
        report(args)


if __name__=='__main__': main()
