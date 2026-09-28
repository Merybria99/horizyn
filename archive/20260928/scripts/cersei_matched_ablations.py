#!/usr/bin/env python3
"""Current-manuscript K and temperature ablations; immutable, single-factor fits.

prepare freezes configs and reference results. Four workers claim independent
fits. Every fit uses the primary training budget and dictionary-free scorer.
No test-dependent selection, model editing, or automatic failed-fit restart.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/generalization_20260919_2251'
CROSS = RUN / 'cross_paper_retraining'
BASE = CROSS / 'shared_fusion03_beta5_b1024_v1'
REFERENCE = ROOT / 'runs/cersei_horizyn_challenge_20260923/no_dictionary_validation_20260923/reference_shared_test'
TARGETS = ('reaction_smi', 'enzyme_smi', 'time', 'enzymemap')
METRICS = ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')


def stamp(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()
def write(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(d, indent=2) + '\n'); tmp.replace(p)


def expected_config(target, k, beta, seed):
    import yaml
    c = yaml.safe_load((BASE / target / 'configs/train.yaml').read_text())
    c['seed'] = seed
    c['model']['enzyme_multiview']['num_slots'] = k
    c['training']['loss']['beta'] = beta
    c['training']['max_epochs'] = 15 if target == 'enzymemap' else 20
    return c


def comparable(c):
    c = copy.deepcopy(c)
    for key in ('log_dir', 'checkpoint_dir'): c['logging'].pop(key, None)
    return c


def prepare(out):
    import yaml
    out.mkdir(parents=True, exist_ok=False)
    reference = {}
    for t in TARGETS:
        epoch = 15 if t == 'enzymemap' else 20
        phase = BASE / t / f'phase2_followup/epoch{epoch}'
        cp = BASE / t / f'training/checkpoints/screen_selection/screen-epoch={epoch-1:02d}.ckpt'
        summary = REFERENCE / t / 'summary.json'
        receipt = read(REFERENCE / t / 'score_receipt.json') if t == 'enzymemap' else read(summary)
        assert sha(cp) == receipt['checkpoint_sha256']
        assert sha(phase / 'training/step0100.pt') == receipt['head_sha256' if t == 'enzymemap' else 'residual_head_sha256']
        reference[t] = dict(epoch=epoch, checkpoint=str(cp), checkpoint_sha256=sha(cp),
            head=str(phase / 'training/step0100.pt'), phase=str(phase),
            config=str(BASE / t / 'configs/train.yaml'),
            config_sha256=sha(BASE / t / 'configs/train.yaml'),
            summary=str(summary), summary_sha256=sha(summary))
    specifications = [(t, k, 5., 42, 'view_count') for k in (0, 1, 2, 8) for t in TARGETS]
    specifications += [('enzymemap', 4, b, seed, 'temperature')
                       for seed in (42, 17, 73) for b in (5., 10.) if (seed, b) != (42, 5.)]
    tasks = []
    for target, k, beta, seed, study in specifications:
        ident = f'{study}/{target}/k{k}_beta{beta:g}_seed{seed}'
        dest = out / ident; (dest / 'configs').mkdir(parents=True)
        c = expected_config(target, k, beta, seed)
        c['logging']['log_dir'] = str(dest / 'unused/logs')
        c['logging']['checkpoint_dir'] = str(dest / 'unused/checkpoints')
        path = dest / 'configs/train.yaml'; path.write_text(yaml.safe_dump(c, sort_keys=False))
        task = dict(id=ident, target=target, k=k, beta=beta, seed=seed, study=study,
                    root=str(dest), config=str(path), config_sha256=sha(path),
                    epochs=c['training']['max_epochs'])
        if target != 'enzymemap':
            tc = yaml.safe_load((BASE / target / 'configs/test.yaml').read_text())
            tc['model'] = copy.deepcopy(c['model']); tc['seed'] = seed
            tc['logging'] = copy.deepcopy(c['logging'])
            test = dest / 'configs/test.yaml'; test.write_text(yaml.safe_dump(tc, sort_keys=False))
            task.update(test_config=str(test), test_config_sha256=sha(test))
        tasks.append(task)
    code = [*sorted((ROOT / 'horizyn').rglob('*.py')),
            ROOT / 'scripts/train_protein_pooling.py', ROOT / 'scripts/train_protein_pooling_fast_io.py',
            ROOT / 'scripts/generalization_full_graph.py',
            ROOT / 'scripts/generalization_reactzyme_f3_features.py',
            ROOT / 'scripts/generalization_clipzyme_phase2_export.py', Path(__file__).resolve()]
    plan = dict(created_utc=stamp(), reference=reference, tasks=tasks,
        code_sha256={str(p): sha(p) for p in code},
        inference=dict(fusion_multiplier=2., kappa_train=.2, kappa_inf=.1, dictionary_weight=0.),
        phase2=dict(steps=100, temperature=.2, identity_weight=10., learning_rate=.0001,
                    weight_decay=.001, hidden=1024, objective='positive_ce', seed=42),
        K=[0, 1, 2, 4, 8], temperature_seeds=[17, 42, 73], temperature_beta=[5., 10.],
        reference_reuse='Existing exact main-paper K4 beta5 seed42 checkpoint for each target',
        selection='Fixed primary encoder epoch and refinement update100 for every arm; no retuning',
        training_budget='ReactZyme20 / EnzymeMap15 epochs; batch1024; native fusion initialized0.3',
        test_role='Retrospective benchmark ablation, all predefined arms reported; no test selection')
    write(out / 'protocol.json', plan)
    shutil.copy2(__file__, out / 'runner.py')
    check(out)


def check(out):
    import yaml
    p = read(out / 'protocol.json')
    for path, digest in p['code_sha256'].items():
        if sha(path) != digest: raise ValueError(f'Frozen source changed: {path}')
    for t, r in p['reference'].items():
        for key in ('config', 'checkpoint', 'summary'):
            assert sha(r[key]) == r[key + '_sha256'], (t, key)
    for t in p['tasks']:
        c = yaml.safe_load(Path(t['config']).read_text())
        assert sha(t['config']) == t['config_sha256']
        assert comparable(c) == comparable(expected_config(t['target'], t['k'], t['beta'], t['seed']))
        assert c['data']['train_batch_size'] == 1024
        assert c['model']['enzyme_multiview']['initial_residual_scale'] == .3
        for key, value in c['data'].items():
            if key.endswith('_path') and value: assert (ROOT / value).exists(), (key, value)
        if 'test_config' in t: assert sha(t['test_config']) == t['test_config_sha256']
    return p


def invoke(script, args, gpu, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str, args)]
    with Path(log).open('a') as f:
        subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def state(t, phase, **extra):
    value = dict(id=t['id'], phase=phase, utc=stamp(), **extra)
    write(Path(t['root']) / 'state.json', value); print(json.dumps(value), flush=True)


def run_fit(out, t, gpu, plan):
    dest = Path(t['root']); target = t['target']
    state(t, 'training', gpu=gpu)
    invoke('train_protein_pooling_fast_io.py', ['--config', t['config'], '--io-mode', 'residue',
           '--io-output-dir', dest / 'training', '--io-prefetch', 1], gpu, dest / 'training.log')
    cp = dest / f'training/checkpoints/screen_selection/screen-epoch={t["epochs"]-1:02d}.ckpt'
    if not cp.exists(): raise FileNotFoundError(cp)
    write(dest / 'train_complete.json', dict(checkpoint=str(cp), sha256=sha(cp), epoch=t['epochs'], utc=stamp()))
    features = dest / 'features'
    state(t, 'export_training_embeddings')
    if target != 'enzymemap':
        source = RUN / {'reaction_smi':'features', 'enzyme_smi':'features_enzyme_smi', 'time':'features_time'}[target]
        invoke('generalization_reactzyme_f3_features.py', ['--template', source, '--config', t['config'],
            '--checkpoint', cp, '--output', features, '--scope', 'train_validation', '--split', target,
            '--batch-size', 256], gpu, dest / 'features.log')
    else:
        import yaml
        features.mkdir()
        source = Path(plan['reference'][target]['phase']) / 'features'
        for name in ('catalog.json', 'pairs.npz', 'reaction_features.npz'):
            (features / name).symlink_to((source / name).resolve())
        shutil.copy2(source / 'protein_mean.h5', features / 'protein_mean.h5')
        manifest = read(source / 'manifest.json')
        for key in ('checkpoint', 'f3_export_config', 'f3_residue_cache'): manifest.pop(key, None)
        manifest['matched_ablation_raw_reuse'] = dict(source=str(source), manifest_sha256=sha(source/'manifest.json'))
        write(features / 'manifest.json', manifest)
        config = yaml.safe_load(Path(t['config']).read_text())
        invoke('generalization_clipzyme_phase2_export.py', ['--stage', 'f3', '--catalog', CROSS/'clipzyme_f3_catalog_v1',
            '--config', t['config'], '--checkpoint', cp, '--output', features, '--batch-size', 256,
            '--residue-cache', config['data']['protein_residue_embeds_path']], gpu, dest / 'features.log')
    state(t, 'refinement')
    invoke('generalization_full_graph.py', ['--features', features, '--output', dest/'phase2',
        '--steps', 100, '--snapshot-every', 100, '--selection-method',
        'external_screening' if target == 'enzymemap' else 'fixed_last',
        '--temperature', .2, '--identity-weight', 10, '--learning-rate', .0001,
        '--weight-decay', .001, '--scale', .2, '--hidden', 1024, '--seed', 42,
        '--cpu-threads', 4, '--contrastive-objective', 'positive_ce'], gpu, dest/'phase2.log')
    state(t, 'export_test_embeddings')
    if target != 'enzymemap':
        invoke('generalization_reactzyme_f3_features.py', ['--template', RUN/f'features_test_{target}',
            '--config', t['test_config'], '--checkpoint', cp, '--output', dest/'test_features',
            '--scope', 'test', '--split', target, '--freeze', out/'protocol.json', '--batch-size', 256], gpu, dest/'test_features.log')
    else:
        invoke('generalization_clipzyme_f3_screen.py', ['--phase', 'reactions', '--scope', 'test',
            '--catalog', CROSS/'clipzyme_f3_catalog_v1', '--protocol', CROSS/'clipzyme_screening_evaluation_protocol_v2',
            '--config', t['config'], '--checkpoint', cp, '--output', dest/'test_features', '--batch-size', 256], gpu, dest/'test_features.log')
    state(t, 'evaluate_full_test')
    invoke(Path(__file__).name, ['evaluate', '--output', out, '--task', t['id']], gpu, dest/'evaluation.log')
    state(t, 'complete')


def evaluate(out, ident):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from generalization_clipzyme_f3_screen import model_from_checkpoint
    from generalization_multiview_calibration import components
    from generalization_reactzyme_architecture_phase2 import load_head, canonical_dot, evaluate_scores, checked_official_truth
    from generalization_clipzyme_screening_evaluate import evaluate_query
    plan = read(out/'protocol.json'); t = next(x for x in plan['tasks'] if x['id'] == ident)
    dest = Path(t['root']); target = t['target']; device = 'cuda:0'
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    cp = Path(read(dest/'train_complete.json')['checkpoint'])
    model, config = model_from_checkpoint(Path(t.get('test_config', t['config'])), cp, device)
    head = load_head(dest/'phase2/step0100.pt', sha(dest/'features/manifest.json'), device)
    if target != 'enzymemap':
        cat = read(dest/'test_features/catalog.json'); ids = cat['proteins']
        z = np.load(dest/'test_features/f3_features.npz'); br = torch.tensor(z['reactions'], device=device)
    else:
        mapping = list(csv.DictReader((CROSS/'clipzyme_f3_catalog_v1/screening_candidate_map.csv').open()))
        candidates = [x['uniprot_id'] for x in mapping]
        assert len(candidates) == 261907
        # Preserve the primary export's exact sequence order and therefore GEMM dimensions.
        screen = BASE/'enzymemap/screen_epoch14/proteins'
        ids = sum([(screen/f'protein_rank{r:02d}_of_04.ids.txt').read_text().splitlines() for r in range(4)], [])
        assert len(ids) == len(set(ids)) == 222985
        config.data.protein_residue_embeds_path = str(CROSS/'clipzyme_f3_catalog_v1/features/proteins_prott5_residue.local.h5')
        br = torch.tensor(np.load(dest/'test_features/reaction_embeddings.npy'), device=device)
    with torch.inference_mode():
        g, f, scale, native, reconstruction_error = components(model, config, ids, device)
        b = F.normalize(g + (scale*2.)*f, dim=-1)
        e = torch.cat([F.normalize(v + .5*head.scale*head.enzyme(v), dim=-1) for v in b.split(8192)]) if target == 'enzymemap' else F.normalize(b + .5*head.scale*head.enzyme(b), dim=-1)
        r = F.normalize(br + .5*head.scale*head.reaction(br), dim=-1)
        if target != 'enzymemap':
            parent = read(RUN/'phase2/models'/target/'seed42/bundle.json'); provenance = {}
            checked, edges = checked_official_truth(RUN/f'features_test_{target}', target, parent['frozen_recipe']['sha256'], provenance)
            assert checked == cat
            scores = canonical_dot(r, e)
            ev = evaluate_scores(scores, dict(reaction_index=edges[:,0], enzyme_index=edges[:,1]))
            summary = ev['summary']
            np.savez_compressed(dest/'per_query.npz', **{d+'_'+k:v for d,b in ev['per_query'].items() for k,v in b['all'].items()})
        else:
            protocol = CROSS/'clipzyme_screening_evaluation_protocol_v2'; receipt = read(protocol/'receipt.json')
            assert sha(protocol/'queries.csv') == receipt['queries_sha256']
            assert sha(protocol/'train_uniprot_ids.txt') == receipt['train_uniprot_ids_sha256']
            rows = list(csv.DictReader((protocol/'queries.csv').open()))
            queries = (dest/'test_features/query_ids.txt').read_text().splitlines()
            assert queries == [x['reaction_id'] for x in rows]
            lookup = {x:i for i,x in enumerate(ids)}; expanded = np.asarray([lookup[x['protein_id']] for x in mapping])
            cidx = {x:i for i,x in enumerate(candidates)}
            train = set((protocol/'train_uniprot_ids.txt').read_text().splitlines())
            kept = np.asarray([i for i,x in enumerate(candidates) if x not in train])
            positives = [np.asarray([cidx[x] for x in json.loads(row['positive_uniprot_ids_json'])]) for row in rows]
            records = []
            with ThreadPoolExecutor(max_workers=16) as pool:
                for start in range(0, len(r), 32):
                    values = (r[start:start+32] @ e.T).float().cpu().numpy()[:,expanded]
                    records += list(pool.map(evaluate_query, [(start+i,queries[start+i],v,positives[start+i],kept) for i,v in enumerate(values)]))
            summary = {}
            for table, count in [('table1',261907),('table2',252113)]:
                selected = [x[table] for x in records if x[table] is not None]
                summary[table] = dict(queries=len(selected), candidate_ids=count, **{m:float(np.mean([x[m] for x in selected])) for m in METRICS})
            assert summary['table1']['queries'] == 1521 and summary['table2']['queries'] == 1337 and len(kept) == 252113
            (dest/'per_query.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
        write(dest/'complete.json', dict(task=t, summary=summary, learned_scale=scale,
            reconstruction_error=reconstruction_error, checkpoint_sha256=sha(cp),
            head_sha256=sha(dest/'phase2/step0100.pt'), protocol_sha256=sha(out/'protocol.json'),
            test_used_for_selection=False, completed_utc=stamp()))


def worker(out, gpu):
    plan = check(out)
    while True:
        with (out/'queue.lock').open('a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            choices = [t for t in plan['tasks'] if not (Path(t['root'])/'claimed.json').exists()]
            if not choices: return
            t = choices[0]
            write(Path(t['root'])/'claimed.json', dict(gpu=gpu, pid=os.getpid(), utc=stamp()))
        try:
            check(out); run_fit(out, t, gpu, plan)
        except Exception as exc:
            write(Path(t['root'])/'failure.json', dict(utc=stamp(), error=repr(exc)))
            state(t, 'failed', error=repr(exc)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','check','worker','evaluate'])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--gpu', type=int); p.add_argument('--task')
    a = p.parse_args(); out = a.output.resolve()
    if a.action == 'prepare': prepare(out)
    elif a.action == 'check': print('Validated',len(check(out)['tasks']),'single-factor training configurations')
    elif a.action == 'worker': worker(out, a.gpu)
    else: evaluate(out, a.task)
