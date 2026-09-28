#!/usr/bin/env python3
"""Held-out test of validation-selected biology heads and cached original F3.

Alpha and epoch come exclusively from the saved head checkpoint. No test sweep.
All methods use identical candidates, ground truth and stable ranking ties.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_circe_scored_biology import BASE, sha, save_json, save_tensor, children, load_features


def selection(path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not 0 <= state['selected_alpha'] <= 1:
        raise ValueError('Invalid saved validation alpha')
    manifest = path.parent.parent/'manifest.json'
    if sha(manifest) != state['manifest_sha256']:
        raise ValueError('Head checkpoint and training manifest disagree')
    return state, json.loads(manifest.read_text())


def prepare(args):
    import yaml
    from horizyn.capability.scored_biology_targets import positive_pairs
    state, trained = selection(args.heads)
    original = json.loads((Path(trained['frozen_source'])/'manifest.json').read_text()) if 'frozen_source' in trained else trained
    config = yaml.safe_load(args.test_config.read_text())
    train_config_path = Path(original['settings']['config'])
    if sha(train_config_path) != original['source_sha256'][str(train_config_path)]:
        raise ValueError('Base training configuration changed')
    training_config = yaml.safe_load(train_config_path.read_text())
    if config['model'] != training_config['model']:
        raise ValueError('Test model differs from trained model')
    for key in ('protein_residue_embeds_path', 'max_protein_tokens', 'protein_truncation',
                'normalize_molecule_sets_as_self_reactions'):
        if config['data'].get(key) != training_config['data'].get(key):
            raise ValueError(f'Test feature recipe changed: {key}')
    checkpoint = Path(state['base_checkpoint'])
    if sha(checkpoint) != original['source_sha256'][str(checkpoint)]:
        raise ValueError('Frozen base checkpoint changed')
    for relative, digest in original['code'].items():
        if relative.startswith('horizyn/') and sha(ROOT/relative) != digest:
            raise ValueError(f'Model/data code changed: {relative}')
    data = config['data']
    paths = dict(validation_pairs=Path(data['test_pairs_path']),
                 validation_reactions=Path(data['test_reactions_path']),
                 validation_candidates=Path(data['validation_retrieval_candidate_ids_path']))
    f3_manifest = json.loads((args.f3_cache/'manifest.json').read_text())
    f3_inputs = f3_manifest['signature']['inputs']
    for key, path in paths.items():
        if sha(path) != f3_inputs[key]['sha256']:
            raise ValueError(f'F3/test input mismatch: {key}')
    pairs = positive_pairs(paths['validation_pairs'])
    queries = sorted({q for q, _ in pairs})
    proteins = sorted(paths['validation_candidates'].read_text().splitlines())
    if not proteins or '' in proteins or len(set(proteins)) != len(proteins):
        raise ValueError('Invalid test candidate IDs')
    if not {p for _, p in pairs} <= set(proteins):
        raise ValueError('Test positive absent from candidate catalog')
    if f3_manifest['counts']['enzymes'] != len(proteins) or f3_manifest['counts']['validation_reactions'] != len(queries):
        raise ValueError('F3/test candidate counts differ')
    # Adapter for the existing frozen export routine; there is NO training here.
    for split in ('train', 'validation'):
        data[f'{split}_pairs_path'] = str(paths['validation_pairs'])
        data[f'{split}_reactions_path'] = str(paths['validation_reactions'])
        for modality in ('t5v2', 'unimol2', 'chiro'):
            data[f'{split}_reaction_{modality}_embeds_path'] = config['data'][f'validation_reaction_{modality}_embeds_path']
    source_files = [args.heads, args.test_config, args.f3_cache/'manifest.json',
                    args.f3_cache/'enzyme_base.h5', args.f3_cache/'validation_reaction_base.h5', *paths.values()]
    feature_files = {}
    for modality in ('t5v2', 'unimol2', 'chiro'):
        path = Path(data[f'validation_reaction_{modality}_embeds_path'])
        feature_files[str(path)] = sha(path)
    manifest = dict(head_checkpoint=str(args.heads), base_checkpoint=str(checkpoint),
        selected_epoch=state['epoch']+1, selected_alpha=state['selected_alpha'], lambda_weight=state['lambda_weight'],
        heads_training_manifest_sha256=state['manifest_sha256'],
        f3_checkpoint=f3_inputs['checkpoint'], f3_cache=str(args.f3_cache),
        reaction_count=len(queries), enzyme_count=len(proteins), source_sha256={str(p):sha(p) for p in source_files},
        reaction_feature_sha256=feature_files, gpus=args.gpus, cache_batch_size=args.cache_batch_size,
        code={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__), ROOT/'scripts/run_circe_scored_biology.py')},
        metric='query-macro all-positive MRR; arithmetic direction mean',
        ties='stable sorted candidate ID order', selection='validation only; no test alpha search')
    target = args.run_root/'manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Test inputs/settings changed: choose a new output directory')
    save_json(target, manifest)
    config_path = args.run_root/'export_config.yaml'
    with config_path.open('w') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    save_tensor(args.run_root/'data.pt', dict(proteins=proteins, train_q=queries, val_q=[], pairs=pairs))
    print(f'Test prepared: {len(queries)} reactions, {len(proteins)} enzymes; '
          f'head epoch={state["epoch"]+1}, lambda={state["lambda_weight"]}, alpha={state["selected_alpha"]}.', flush=True)
    return config_path, checkpoint


def read_f3(path, ids, reaction=False):
    import h5py
    import torch
    from torch.nn import functional as F
    with h5py.File(path) as handle:
        keys = handle['ids'].asstr()[:].tolist()
        vectors = torch.from_numpy(handle['vectors'][:]).float()
    expected = [key+'_f' for key in ids] if reaction else ids
    if set(expected) != set(keys) or len(keys) != len(expected):
        raise ValueError(f'F3 cached IDs differ: {path}')
    if not torch.isfinite(vectors).all() or (vectors.norm(dim=-1) == 0).any():
        raise ValueError('Invalid F3 cached vectors')
    index = {key:i for i,key in enumerate(keys)}
    return F.normalize(vectors[[index[key] for key in expected]], dim=-1)


def score(args):
    import torch
    from horizyn.scored_biology import ScoredBiology, retrieval_metrics
    manifest = json.loads((args.run_root/'manifest.json').read_text())
    if sha(args.heads) != manifest['source_sha256'][str(args.heads)]:
        raise ValueError('Selected head checkpoint changed during export')
    state, _ = selection(args.heads)
    data = torch.load(args.run_root/'data.pt', weights_only=True)
    ep, up = load_features(args, 'enzyme', data['proteins'])
    rq, uq = load_features(args, 'train', data['train_q'])
    model = ScoredBiology(state['enzyme_dim'], state['reaction_dim'], state['labels']).to(args.device)
    model.load_state_dict(state['model'])
    model.eval()
    with torch.inference_mode():
        bq = model.encode(rq, 'reaction')
        bp = torch.cat([model.encode(ep[i:i+2048], 'enzyme') for i in range(0, len(ep), 2048)])
    qm, pm = ({q:i for i,q in enumerate(data['train_q'])}, {p:i for i,p in enumerate(data['proteins'])})
    r2e, e2r = ([[] for _ in qm], [[] for _ in pm])
    for q, p in data['pairs']:
        r2e[qm[q]].append(pm[p])
        e2r[pm[p]].append(qm[q])
    results = {}
    f3q = read_f3(args.f3_cache/'validation_reaction_base.h5', data['train_q'], reaction=True).to(args.device)
    f3p = read_f3(args.f3_cache/'enzyme_base.h5', data['proteins']).to(args.device)
    for name, q, p, alpha in [('original_f3_epoch30', f3q, f3p, 0.),
                              ('frozen_feature_gate_epoch30', uq, up, 0.),
                              ('scored_biology', uq, up, state['selected_alpha'])]:
        r = retrieval_metrics(q, p, bq, bp, r2e, alpha)
        e = retrieval_metrics(p, q, bp, bq, e2r, alpha)
        results[name] = dict(r2e=r, e2r=e, mean_mrr=(r['mrr']+e['mrr'])/2, alpha=alpha)
    save_json(args.run_root/'metrics.json', dict(manifest=manifest, results=results))
    with (args.run_root/'comparison.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model','r2e_mrr','e2r_mrr','mean_mrr','alpha'])
        for name, result in results.items():
            writer.writerow([name,result['r2e']['mrr'],result['e2r']['mrr'],result['mean_mrr'],result['alpha']])
    save_json(args.run_root/'complete.json', dict(results=results, test_selection=False))
    print(json.dumps(results, indent=2), flush=True)


def command(args, stage):
    result = [sys.executable, str(Path(__file__).resolve()), stage]
    for name in ('heads','test_config','f3_cache','run_root','gpus','cache_batch_size'):
        result += ['--'+name.replace('_','-'), str(getattr(args,name))]
    if args.allow_shared_gpus:
        result.append('--allow-shared-gpus')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['launch','run','prepare','score'])
    parser.add_argument('--heads', type=Path, default=ROOT/'runs/circe_scored_biology_lambda_high_v1/lambda_0.1/best.pt')
    parser.add_argument('--test-config', type=Path, default=BASE/'feature/configs/test.yaml')
    parser.add_argument('--f3-cache', type=Path, default=ROOT/'runs/biological_residual_reaction_smi/cache/reactzyme_test')
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--cache-batch-size', type=int, default=128)
    parser.add_argument('--allow-shared-gpus', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.cache_batch_size <= 0:
        parser.error('cache-batch-size must be positive')
    for name in ('heads','test_config','f3_cache','run_root'):
        setattr(args,name,getattr(args,name).resolve())
    args.run_root.mkdir(parents=True,exist_ok=True)
    if args.stage == 'score':
        score(args)
        return
    from scripts.run_circe_generalization_diagnostics import check_gpus
    gpus = args.gpus.split(',')
    if args.stage == 'launch':
        check_gpus(gpus,allow_shared=args.allow_shared_gpus)
        session = 'scored_bio_test_'+hashlib.sha256(str(args.run_root).encode()).hexdigest()[:10]
        if subprocess.run(['tmux','has-session','-t','='+session],capture_output=True).returncode == 0:
            raise RuntimeError(f'Already running: {session}')
        shell = 'exec '+shlex.join(command(args,'run'))+' >> '+shlex.quote(str(args.run_root/'pipeline.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),
            'env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c '+shlex.quote(shell)],check=True)
        print(f'Detached session: {session}\nLog: {args.run_root/"pipeline.log"}',flush=True)
        return
    with (args.run_root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def interrupted(_signum,_frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupted)
        config, checkpoint = prepare(args)
        if args.stage == 'prepare' or (args.run_root/'complete.json').exists():
            return
        check_gpus(gpus,allow_shared=args.allow_shared_gpus)
        export = argparse.Namespace(**vars(args),config=config,checkpoint=checkpoint,
            lambdas='0',epochs=1,seed=42,batch_size=1536,lr=1e-4,alpha=.1)
        children(export,'export',[['--worker',str(i)] for i in range(len(gpus))],lock.fileno())
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[0],PYTHONUNBUFFERED='1')
        process = subprocess.Popen(command(args,'score'),env=env,cwd=ROOT,start_new_session=True,pass_fds=(lock.fileno(),))
        try:
            if process.wait():
                raise RuntimeError('Test scoring failed; inspect pipeline.log')
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
        print('Test complete: '+str(args.run_root/'comparison.csv'),flush=True)


if __name__ == '__main__':
    main()
