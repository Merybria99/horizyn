#!/usr/bin/env python3
"""Released EnzymeCAGE P450 baselines versus F3; run from the horizyn checkout.

Stages are explicit and restartable. No source data or checkpoints are modified.
Use the native EnzymeCAGE Python for all stages except f3-features/f3-infer.
"""
import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'

import numpy as np
import pandas as pd
import yaml
from p450_protocol import (REVISION, DATA_SHA, KEYS, digest, write_json, validate_panel,
                          align_scores, apply_official_prior, score_metrics)


def run(command, cwd, env=None):
    command = [str(x) for x in command]
    print('RUN', command, flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def checked_source(args):
    actual = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise ValueError(f'Official source revision changed: {actual}')
    if subprocess.check_output(['git', '-C', str(args.source), 'diff', 'HEAD', '--'], text=True):
        raise ValueError('Official source has tracked modifications')


def panel(args):
    path = args.output / 'data/test_P450.csv'
    if digest(path) != DATA_SHA:
        raise ValueError('Released CSV checksum mismatch')
    frame = pd.read_csv(path)
    validate_panel(frame, released=True)
    return frame


def prepare(args):
    original = args.assets / 'dataset/external-test-set/p450'
    if digest(original / 'test_P450.csv') != DATA_SHA:
        raise ValueError('Unexpected source panel')
    data = args.output / 'data'
    data.mkdir(parents=True, exist_ok=True)
    if not (data / 'test_P450.csv').exists():
        shutil.copy2(original / 'test_P450.csv', data / 'test_P450.csv')
    frame = panel(args)
    # Private copy: the native dataset implementation may cache molecule graphs.
    reaction_dest = data / 'feature/reaction'
    if not reaction_dest.exists():
        shutil.copytree(original / 'feature/reaction', reaction_dest)
    pocket_dir = original / 'pockets/pocket'
    missing = [uid for uid in frame.UniprotID.unique() if not (pocket_dir / f'{uid}.pdb').is_file()]
    if missing:
        raise ValueError(f'Missing released pockets: {missing}')
    configurations = {}
    for variant, template, ckpt_parent, epoch in [
        ('pretrained', 'wo-finetune', 'pretrain', 19),
        ('p450_finetuned', 'with-finetune', 'domain-specific-ft/p450', 9),
    ]:
        for seed in [42, 40, 41, 43, 44]:
            conf = yaml.safe_load((args.source / f'config/infer/external-test-set/p450/{template}.yaml').read_text())
            conf['data_path'] = str(data / 'test_P450.csv')
            conf['ckpt_dir'] = str(args.assets / f'checkpoints/{ckpt_parent}/seed_{seed}')
            conf['result_dir'] = str(args.output / f'scores/{variant}/seed_{seed}')
            conf['seed'] = seed
            for field in ['rxn_fp', 'mol_conformation', 'reaction_center', 'protein_gvp_feat', 'esm_mean_feature', 'esm_node_feature']:
                conf[field] = str(data / 'feature' / conf[field].split('/feature/')[1])
            checkpoint = Path(conf['ckpt_dir']) / f'epoch_{epoch}.pth'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            name = f'{variant}_seed_{seed}'
            path = args.output / 'configs' / f'{name}.yaml'
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(conf, sort_keys=False))
            configurations[name] = dict(config=str(path), config_sha256=digest(path),
                                        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
    # Fixed working directory required by the released evaluation module.
    work = args.output / 'official_evaluation'
    (work / 'scripts').mkdir(parents=True, exist_ok=True)
    (work / 'dataset/RHEA').mkdir(parents=True, exist_ok=True)
    ref = work / 'dataset/RHEA/2023-07-12'
    if not ref.exists():
        ref.symlink_to(args.assets / 'dataset/RHEA/2023-07-12', target_is_directory=True)
    f3 = args.output / 'f3_features'
    f3.mkdir(exist_ok=True)
    raw = frame[KEYS[0]].drop_duplicates().tolist()
    mapping = pd.DataFrame({'reaction_id': [f'p450_{i:03d}' for i in range(len(raw))], 'raw_reaction': raw})
    mapping.to_csv(f3 / 'reaction_mapping.csv', index=False)
    uid = frame.drop_duplicates('UniprotID')
    (f3 / 'proteins.fasta').write_text(''.join(f'>{r.UniprotID}\n{r.sequence}\n' for r in uid.itertuples()))
    # These pairs only register reaction/enzyme IDs with the feature loader.
    pairs = frame[KEYS].rename(columns={KEYS[0]: 'reaction_id', KEYS[1]: 'protein_id'})
    pairs.reaction_id = pairs.reaction_id.map(dict(zip(raw, mapping.reaction_id)))
    pairs.to_csv(f3 / 'encoding_pairs.csv', index=False)
    hashes = {str(p.relative_to(args.source)): digest(p) for p in args.source.rglob('*.py') if '.git' not in p.parts}
    write_json(args.output / 'manifest.json', dict(protocol='released_EnzymeCAGE_P450', source_revision=REVISION,
        panel_sha256=DATA_SHA, counts=validate_panel(frame, True), cutoffs=[4, 14, 24], configurations=configurations,
        official_source_sha256=hashes, pocket_sha256={uid: digest(pocket_dir / f'{uid}.pdb') for uid in frame.UniprotID.unique()},
        reaction_feature_sha256={str(p.relative_to(reaction_dest)): digest(p) for p in reaction_dest.rglob('*') if p.is_file()},
        reference_sha256={n: digest(ref / n) for n in ['enzymes.fasta', 'rhea_rxn2uids.csv']},
        notes=['Released checkpoints; no invented retraining recipe.', 'Five EnzymeCAGE seeds are separate runs, not a P450 ensemble.',
               'F3 checkpoint selected by internal validation, not this P450 test.',
               'Current pinned public release, not proof of bitwise historical paper reproduction.',
               'Label 0 means unlisted positive; this panel does not establish measured inactivity.']))
    print('PREPARED', validate_panel(frame, True), flush=True)


def native_features(args):
    env = os.environ.copy()
    # Trusted published/locally generated feature files contain NumPy arrays.
    # Restore historical torch.load behavior for this child process only.
    env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    run([sys.executable, 'main.py', '--data_path', args.output / 'data/test_P450.csv',
         '--pocket_dir', args.assets / 'dataset/external-test-set/p450/pockets/pocket', '--skip_rxn_feature'],
        args.source / 'feature', env)
    validate_native_features(args)


def validate_native_features(args):
    import torch
    frame = panel(args)
    conf = yaml.safe_load((args.output / 'configs/pretrained_seed_42.yaml').read_text())
    gvp = torch.load(conf['protein_gvp_feat'], map_location='cpu', weights_only=False)
    node = torch.load(conf['esm_node_feature'], map_location='cpu', weights_only=False)
    with open(conf['esm_mean_feature'], 'rb') as handle:
        mean = pickle.load(handle)
    expected = set(frame.UniprotID)
    if not expected <= gvp.keys() & node.keys() or not set(frame.sequence) <= mean.keys():
        raise ValueError('Incomplete native features: refusing candidate deletion or substitute structures')
    for row in frame.drop_duplicates('UniprotID').itertuples():
        uid = row.UniprotID
        if len(gvp[uid][0]) != len(node[uid]) or len(node[uid]) == 0:
            raise ValueError(f'Pocket feature alignment mismatch: {uid}')
        for x in [*gvp[uid], node[uid], mean[row.sequence]]:
            if not torch.isfinite(torch.as_tensor(x)).all():
                raise ValueError(f'Nonfinite feature: {uid}')
    write_json(args.output / 'native_features.validated.json', dict(enzymes=len(expected),
        features={k: digest(conf[k]) for k in ['protein_gvp_feat', 'esm_node_feature', 'esm_mean_feature']}))


def load_external(args, prior_only=False):
    os.chdir(args.output / 'official_evaluation/scripts')
    sys.path.insert(0, str(args.source))
    path = args.source / 'scripts/evaluate_external-test.py'
    if prior_only:
        # Only the evaluator import pulls in the neural-model runtime. Remove
        # that unused import; all prior code and chemistry remain unchanged.
        import types
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'evaluate')]
        module = types.ModuleType('official_external_prior')
        exec(compile(tree, str(path), 'exec'), module.__dict__)
        return module
    spec = importlib.util.spec_from_file_location('official_external', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prior(args):
    from rdkit import rdBase
    if rdBase.rdkitVersion != '2022.09.5':
        raise ValueError(f'Prior requires official RDKit 2022.09.5, found {rdBase.rdkitVersion}')
    mmseqs = args.root / '.deps/p450-mmseqs15/mmseqs/bin/mmseqs'
    mmseqs_version = subprocess.check_output([str(mmseqs), 'version'], text=True).strip()
    if '6f452' not in mmseqs_version:
        raise ValueError(f'Prior requires MMseqs2 15-6f452, found {mmseqs_version}')
    frame = panel(args)
    module = load_external(args, prior_only=True)
    directory = args.output / 'data/analysis/mmseqs15_seq_similarity_to_train'
    directory.mkdir(parents=True, exist_ok=True)
    fasta = directory / 'test_enzymes.fasta'
    unique = frame.drop_duplicates('UniprotID')
    module.save_to_fasta(str(fasta), unique.UniprotID.tolist(), unique.sequence.tolist())
    alignment = directory / 'alnRes.m8'
    if not alignment.exists():
        # Same default search as upstream; limit CPU parallelism only.
        run([mmseqs, 'easy-search', fasta,
             args.assets / 'dataset/RHEA/2023-07-12/enzymes.fasta', alignment,
             directory / 'tmp', '--threads', '4'], args.root)
    maps = module.prepare_for_corr_score(str(args.output / 'data/test_P450.csv'), str(alignment))
    result = module.calc_correlation_score(frame, *maps)
    values = dict(zip(result[KEYS[0]] + '_' + result[KEYS[1]], result.corr_score))
    apply_official_prior(frame.assign(pred=1.), values)
    dest = args.output / 'data/corr_score_map.pkl'
    with open(dest.with_suffix('.tmp'), 'wb') as handle:
        pickle.dump(values, handle)
    dest.with_suffix('.tmp').replace(dest)
    write_json(args.output / 'prior.receipt.json', dict(prior_sha256=digest(dest),
        alignment_sha256=digest(alignment), pythonhashseed=os.environ.get('PYTHONHASHSEED'),
        rdkit_version=rdBase.rdkitVersion, mmseqs_version=mmseqs_version,
        implementation='unmodified prepare_for_corr_score and calc_correlation_score from pinned release',
        operational_difference='MMseqs --threads 4; search parameters otherwise default'))


def native_infer(args):
    frame = panel(args)
    validate_native_features(args)
    manifest = json.loads((args.output / 'manifest.json').read_text())
    for variant, epoch in [('pretrained', 19), ('p450_finetuned', 9)]:
        for seed in args.seeds:
            name = f'{variant}_seed_{seed}'
            config = args.output / 'configs' / f'{name}.yaml'
            output = args.output / f'scores/{variant}/seed_{seed}/test_P450_epoch_{epoch}.csv'
            provenance = manifest['configurations'][name]
            if digest(config) != provenance['config_sha256'] or digest(provenance['checkpoint']) != provenance['checkpoint_sha256']:
                raise ValueError(f'Changed configuration/checkpoint: {name}')
            if output.exists():
                receipt = json.loads(output.with_suffix('.receipt.json').read_text())
                if digest(output) != receipt['scores_sha256'] or receipt['config_sha256'] != digest(config):
                    raise ValueError(f'Existing prediction receipt mismatch: {name}')
            if not output.exists():
                run([sys.executable, args.source / 'infer.py', '--config', config], args.source)
            align_scores(frame, pd.read_csv(output))
            write_json(output.with_suffix('.receipt.json'), dict(complete=True, source_revision=REVISION,
                config_sha256=digest(config), scores_sha256=digest(output), rows=len(frame)))


def f3_features(args):
    sys.path.insert(0, str(args.root))
    from horizyn.chemistry.standardizer import Standardizer
    feature = args.output / 'f3_features'
    mapping = pd.read_csv(feature / 'reaction_mapping.csv')
    standardizer = Standardizer(standardize_uncharge=False)
    reactions = pd.DataFrame({'reaction_id': mapping.reaction_id,
        'reaction_smiles': [standardizer.standardize_reaction(s) for s in mapping.raw_reaction]})
    if reactions.reaction_smiles.isna().any() or (reactions.reaction_smiles == '').any():
        raise ValueError('F3 reaction preprocessing failed')
    reactions.to_csv(feature / 'reactions.csv', index=False)
    # Paths/options below are the training-compatible extractors already used by F3.
    py = sys.executable
    prott5 = args.root / 'data/external/cyp_specificity_2026/models/prott5_safetensors'
    rt5 = args.root.parent / 'hf_cache/hub/models--sagawa--ReactionT5v2-forward/snapshots/933114058cb2604dc1bf536dbebdfcefbe83d4fc'
    protein_args = [py, args.root / 'scripts/extract_prott5_residue_embeddings.py', '--fasta', feature / 'proteins.fasta',
        '--output', feature / 'proteins.h5', '--model-name', prott5, '--batch-size', '8', '--max-tokens-per-batch', '8192',
        '--max-sequence-length', '1022', '--length-sort', '--padded-token-budget', '--resume']
    if not (feature / 'proteins.h5').exists():
        run(protein_args + ['--device', args.device], args.root)
        run(protein_args + ['--merge-only', '--device', 'cpu', '--merge-order', 'shard', '--merge-storage', 'copy'], args.root)
    common = ['--reactions', feature / 'reactions.csv', '--no-bidirectional', '--no-allow-pseudo-reactions', '--no-standardize']
    tasks = [
        ('reactiont5.h5', 'extract_reaction_t5v2_embeddings.py', ['--model-name', rt5, '--device', args.device, '--batch-size', '8', '--max-length', '512', '--pooling', 'mean'], None),
        ('unimol2.h5', 'extract_unimol2_reaction_embeddings.py', ['--batch-size', '16', '--dtype', 'float16', '--skip-invalid-molecules', '--no-skip-invalid-reactions', '--compression', 'none'], 'unimol'),
        ('chiro.h5', 'extract_chiro_reaction_embeddings.py', ['--device', args.device, '--num-workers', '1', '--skip-invalid-molecules', '--no-skip-invalid-reactions', '--log-skipped-molecules', '--batch-size', '16'], 'chiro'),
    ]
    for filename, script, options, kind in tasks:
        if (feature / filename).exists():
            continue
        env = os.environ.copy()
        if kind == 'unimol':
            env['PYTHONPATH'] = ':'.join(map(str, [args.root / '.deps/unimol_tools', args.root.parent / 'env/unimol2_site', args.root]))
            env['UNIMOL_WEIGHT_DIR'] = str(args.root.parent / 'unimol_weights')
        elif kind == 'chiro':
            env['PYTHONPATH'] = ':'.join(map(str, [args.root / '.deps/python', args.root / '.deps/ChIRo', args.root]))
        run([py, args.root / 'scripts' / script, *common, '--output', feature / filename, *options, '--force'], args.root, env)
    if not (feature / 'chemistry_f3.npz').exists():
        chemistry_env = os.environ.copy()
        chemistry_env['PYTHONPATH'] = str(args.root)
        run([py, args.root / 'wet_lab/materialize_chemistry.py', '--reactions', feature / 'reactions.csv',
            '--schema', args.root / 'runs/cyp_enzymecage_f3_epoch03/snapshot/schema.json',
            '--cofactor-dictionary', args.root / 'data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv',
            '--output', feature / 'chemistry_f3.npz'], args.root, chemistry_env)


def f3_infer(args):
    import torch
    sys.path.insert(0, str(args.root))
    from horizyn.benchmarks.retrieval import (BenchmarkTask, build_reaction_inputs, build_query_inputs,
        encode_residue_targets, load_repo_checkpoint, cosine_scores)
    from horizyn.config import load_config
    from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
    torch.set_num_threads(4)
    torch.manual_seed(42)
    frame = panel(args)
    feature = args.output / 'f3_features'
    mapping = pd.read_csv(feature / 'reaction_mapping.csv')
    qids = mapping.reaction_id.tolist()
    pids = frame.UniprotID.drop_duplicates().tolist()
    config_path = args.root / 'runs/cyp_enzymecage_f3_epoch03/snapshot/train.yaml'
    config = load_config(str(config_path))
    config.data.reaction_chemistry_vectors_path = str(feature / 'chemistry_f3.npz')
    for modality in ['unimol2', 'chiro', 'chemistry']:
        config.data[f'reaction_allow_missing_{modality}'] = False
    task = BenchmarkTask(name='official_P450', task_type='retrieval', dataset='EnzymeCAGE_P450',
        task_label='within_family', split='external', pairs=feature / 'encoding_pairs.csv', reactions=feature / 'reactions.csv',
        reaction_model_embeds_h5=feature / 'reactiont5.h5', reaction_unimol2_embeds_h5=feature / 'unimol2.h5',
        reaction_chiro_embeds_h5=feature / 'chiro.h5')
    reactions = build_reaction_inputs(task, config)
    if set(reactions.keys) != set(qids):
        raise ValueError('F3 reaction coverage differs')
    inputs = build_query_inputs(reactions, qids, args.device)
    for k, v in inputs.items():
        if not torch.isfinite(v).all() or (k.startswith('has_') and not v.all()):
            raise ValueError(f'Invalid/missing F3 feature: {k}')
    checkpoint = args.root / 'runs/enzymecage_f3_seed42/checkpoints/protein-pooling-epoch=06.ckpt'
    validation_path = args.root / 'runs/enzymecage_f3_seed42/logs/train/protein_pooling_training/version_1/metrics.csv'
    validation = pd.read_csv(validation_path).dropna(subset=['val/mean_bidirectional_mrr'])
    selected = validation.loc[validation['val/mean_bidirectional_mrr'].idxmax()]
    if int(selected['epoch']) != 6:
        raise ValueError('Recorded internal validation no longer selects the prespecified epoch06')
    module, kind = load_repo_checkpoint(checkpoint, config, args.device)
    if kind != 'residue':
        raise ValueError('Unexpected F3 checkpoint type')
    module.eval()
    if any(getattr(module.model, name, None) is not None for name in ['r2e_adapter', 'e2r_adapter']):
        raise ValueError('Directional model needs a separate score implementation')
    dataset = ResidueEmbedDataset(str(feature / 'proteins.h5'), max_tokens=1022, truncation='ends_center')
    try:
        if set(dataset.keys) != set(pids):
            raise ValueError('F3 protein coverage differs')
        with torch.inference_mode():
            target = encode_residue_targets(module, dataset, pids, args.device, 16, False)
            query = torch.cat([module.model.encode_queries({k: v[i:i+8] for k, v in inputs.items()}).cpu()
                               for i in range(0, len(qids), 8)])
            matrix = cosine_scores(query, target.cpu()).numpy()
    finally:
        dataset.close()
    if not np.isfinite(matrix).all() or np.abs(matrix).max() > 1.00001:
        raise ValueError('Invalid cosine scores')
    qindex = {r: i for i, r in enumerate(mapping.raw_reaction)}
    pindex = {p: i for i, p in enumerate(pids)}
    result = frame.copy()
    result['raw_cosine'] = [float(matrix[qindex[r], pindex[p]]) for r, p in frame[KEYS].itertuples(index=False, name=None)]
    # Strictly increasing, positive transform. The released gate sets excluded
    # scores to zero; applying it to negative cosine scores would invert that gate.
    result['pred'] = (result.raw_cosine.astype('float64') + 2.) / 3.
    align_scores(frame, result)
    dest = args.output / 'scores/f3_enzymecage_seed42_epoch06.csv'
    dest.parent.mkdir(exist_ok=True)
    result.to_csv(dest, index=False)
    write_json(dest.with_suffix('.receipt.json'), dict(complete=True, rows=len(result), scores_sha256=digest(dest),
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), config_sha256=digest(config_path),
        validation_sha256=digest(validation_path), validation_metric=float(selected['val/mean_bidirectional_mrr']),
        checkpoint_selection='epoch06: maximum internal val/mean_bidirectional_mrr; fixed before original P450 evaluation',
        score_adapter='pred=(raw_cosine+2)/3, strictly rank-preserving and positive; raw cosine retained',
        feature_sha256={p.name: digest(p) for p in feature.iterdir() if p.is_file()}))


def report(args):
    frame = panel(args)
    prior_path = args.output / 'data/corr_score_map.pkl'
    prior_receipt = json.loads((args.output / 'prior.receipt.json').read_text())
    if digest(prior_path) != prior_receipt['prior_sha256'] or prior_receipt.get('rdkit_version') != '2022.09.5' or '6f452' not in prior_receipt.get('mmseqs_version', ''):
        raise ValueError('Prior provenance does not match the published chemistry/search versions')
    with open(prior_path, 'rb') as handle:
        values = pickle.load(handle)
    records = []
    per_query = []
    reaction_ids = {r: f'p450_{i:03d}' for i, r in enumerate(frame[KEYS[0]].drop_duplicates())}
    positive_map = {r: set(g.UniprotID) for r, g in frame[frame.Label == 1].groupby(KEYS[0])}
    gated_ones = apply_official_prior(frame.assign(pred=1.), values)
    retained_counts = gated_ones.assign(retained_positive=lambda x: (x.pred > 0) & (x.Label == 1)).groupby(KEYS[0]).retained_positive.sum()
    expected = [('pretrained', s, args.output / f'scores/pretrained/seed_{s}/test_P450_epoch_19.csv') for s in args.seeds]
    expected += [('p450_finetuned', s, args.output / f'scores/p450_finetuned/seed_{s}/test_P450_epoch_9.csv') for s in args.seeds]
    expected += [('f3_enzymecage_epoch06', 42, args.output / 'scores/f3_enzymecage_seed42_epoch06.csv')]
    missing = []
    for method, seed, path in expected:
        if not path.exists():
            missing.append(str(path)); continue
        receipt = json.loads(path.with_suffix('.receipt.json').read_text())
        if not receipt.get('complete') or digest(path) != receipt['scores_sha256']:
            raise ValueError(f'Invalid score receipt: {path}')
        pred = align_scores(frame, pd.read_csv(path))
        if (pred.pred <= 0).any() and method.startswith('f3'):
            raise ValueError('F3 scores must use the documented positive rank-preserving adapter')
        for mode, data in [('raw_diagnostic', pred), ('official_prior', apply_official_prior(pred, values))]:
            records.append(dict(method=method, seed=seed, evaluation=mode, **score_metrics(data, frame, args.source)))
            for reaction, group in data.groupby(KEYS[0]):
                ranked = sorted(zip(group.UniprotID, group.pred), key=lambda x: x[1], reverse=True)
                best = min(i for i, (uid, _) in enumerate(ranked, 1) if uid in positive_map[reaction])
                per_query.append(dict(method=method, seed=seed, evaluation=mode, reaction_id=reaction_ids[reaction],
                    best_positive_rank=best, known_positives=len(positive_map[reaction]),
                    positives_retained_by_prior=int(retained_counts[reaction]),
                    hit_top4=int(best <= 4), hit_top14=int(best <= 14), hit_top24=int(best <= 24)))
    write_json(args.output / 'comparison.json', dict(complete=not missing, missing=missing, results=records,
               prior_diagnostic=dict(reactions_with_retained_positive=int((retained_counts > 0).sum()),
                                     total_reactions=len(retained_counts), retained_positive_pairs=int(retained_counts.sum())),
               prior_sha256=digest(prior_path), panel_sha256=DATA_SHA))
    if records:
        table = pd.DataFrame(records)
        table.to_csv(args.output / 'comparison.csv', index=False)
        table.groupby(['method', 'evaluation'])[['Top 1.0%', 'Top 3.0%', 'Top 5.0%']].agg(['count', 'mean', 'std']).to_csv(args.output / 'comparison_seed_summary.csv')
        pd.DataFrame(per_query).to_csv(args.output / 'comparison_per_query.csv', index=False)
    if missing:
        raise RuntimeError(f'Comparison incomplete: {len(missing)} missing score files; partial output explicitly labelled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'native-features', 'check-features', 'prior', 'native-infer', 'f3-features', 'f3-infer', 'report'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('runs/enzymecage_p450_reproduction_20260918'))
    parser.add_argument('--source', type=Path, default=Path('.deps/enzymecage_p450_official'))
    parser.add_argument('--assets', type=Path, default=Path('../EnzymeCAGE'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 40, 41, 43, 44])
    args = parser.parse_args()
    args.root = args.root.resolve()
    for key in ['output', 'source', 'assets']:
        path = getattr(args, key)
        setattr(args, key, (args.root / path).resolve() if not path.is_absolute() else path.resolve())
    checked_source(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if not set(args.seeds) <= {40, 41, 42, 43, 44}:
        parser.error('Only released seeds 40–44 are supported')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    functions = {'prepare': prepare, 'native-features': native_features, 'check-features': validate_native_features,
                 'prior': prior, 'native-infer': native_infer, 'f3-features': f3_features, 'f3-infer': f3_infer, 'report': report}
    try:
        functions[args.stage](args)
        runtime = {}
        for package in ['numpy', 'pandas', 'rdkit', 'rdkit-pypi', 'torch', 'esm', 'transformers']:
            try:
                runtime[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        write_json(args.output / f'{args.stage}.status.json', dict(status='complete', python=sys.version,
                   runtime=runtime, runner_sha256=digest(__file__), protocol_sha256=digest(Path(__file__).with_name('p450_protocol.py'))))
    except Exception as exc:
        write_json(args.output / f'{args.stage}.status.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
