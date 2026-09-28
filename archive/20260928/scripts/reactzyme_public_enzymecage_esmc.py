#!/usr/bin/env python3
"""Resumable native ESM-C600M inputs for the pending ReactZyme EnzymeCAGE runs.

Preserves full sequences, BOS/EOS embeddings, and the author's all-token mean.
Immutable HDF5 parts keep interrupted extraction resumable without one file per
protein. Pocket indexing and reaction adaptation are separate preparation steps.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/reactzyme_public_baselines_20260921'
REVISION = 'e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=RUN/'features/enzymecage_esmc600m')
    parser.add_argument('--token-budget', type=int, default=8192)
    parser.add_argument('--part-size', type=int, default=128)
    parser.add_argument('--limit', type=int, default=0, help='Smoke check only; separate output required')
    args = parser.parse_args()
    assert args.part_size > 0 and args.token_budget > 0 and args.limit >= 0
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out/'worker.pid').write_text(str(os.getpid()) + '\n')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    started = time.time()
    catalog_path = RUN/'features/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    ids, seqs = catalog['protein_ids'], catalog['protein_sequences']
    order = sorted(range(len(ids)), key=lambda i: (len(seqs[i]), i))
    if args.limit:
        # A range of lengths and padding, not just the shortest peptides.
        order = [order[i] for i in np.linspace(0, len(order)*0.99, args.limit).astype(int)]
    import esm.models.esmc as esmc_module
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    from esm.tokenization import get_esmc_model_tokenizers
    source_root = Path(esmc_module.__file__).resolve().parents[2]
    protocol = dict(model='ESM-C600M', repository='biohub/esmc-600m-2024-12',
        revision=REVISION, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=digest(args.checkpoint), catalog_sha256=digest(catalog_path),
        extractor_sha256=digest(Path(__file__)), esm_source=str(source_root),
        esm_revision=subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'], text=True).strip(),
        source_model_sha256=digest(Path(esmc_module.__file__)), torch_version=torch.__version__,
        full_sequences=True, sequence_normalization='none; native ESM-C tokenizer',
        feature_dim=1152, special_tokens='BOS and EOS retained; padding excluded',
        mean='native tensor mean over L+2 tokens, converted losslessly to float32',
        storage_dtype='float32', inference_dtype='bfloat16', use_flash_attn=False,
        pocket_alignment_applied=False, reaction_features_prepared=False,
        test_labels_used=False, count=len(order), smoke_limit=args.limit, part_size=args.part_size)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise ValueError('Output protocol differs; use a new output directory')
    else:
        save_json(out/'protocol.json', protocol)
    def status(stage, completed=0, failed=0, **extra):
        elapsed = time.time()-started
        value = dict(stage=stage, pid=os.getpid(), visible_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),
            completed=completed, failed=failed, total=len(order), seconds=elapsed,
            updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
        save_json(out/'status.json', value)
        print(json.dumps(value), flush=True)
    status('loading_model')
    model = ESMC(d_model=1152, n_heads=18, n_layers=36,
        tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False)
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model = model.to(device='cuda', dtype=torch.bfloat16).eval()

    @torch.inference_mode()
    def batch_features(indices):
        tokens = model._tokenize([seqs[i] for i in indices])
        for j, i in enumerate(indices):
            assert int((tokens[j] != model.tokenizer.pad_token_id).sum()) == len(seqs[i])+2
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(sequence_tokens=tokens)
        features = []
        for j, i in enumerate(indices):
            v = outputs.embeddings[j, :len(seqs[i])+2]
            mean = v.mean(dim=0).float().cpu().numpy()
            vectors = v.float().cpu().numpy()
            assert vectors.shape == (len(seqs[i])+2, 1152)
            if not np.isfinite(vectors).all() or not np.isfinite(mean).all():
                raise ValueError(f'Nonfinite features for {ids[i]}')
            features.append((i, vectors, mean))
        return features

    # Verify padded batching against the exact API used by the author's extractor.
    probes = [order[0], order[min(len(order)-1, len(order)//2)]]
    batched = batch_features(probes)
    errors = []
    for i, actual, actual_mean in batched:
        with torch.inference_mode():
            native = model.logits(model.encode(ESMProtein(sequence=seqs[i])),
                LogitsConfig(sequence=True, return_embeddings=True)).embeddings[0]
        reference = native.float().cpu().numpy()
        error = float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference), 1e-12))
        cosine = float(np.dot(actual_mean, native.mean(0).float().cpu().numpy()) /
            max(np.linalg.norm(actual_mean)*np.linalg.norm(native.mean(0).float().cpu().numpy()), 1e-12))
        if error > 0.025 or cosine < 0.999:
            raise ValueError(f'Native/batch parity failed: relative L2 {error}, mean cosine {cosine}')
        errors.append(dict(protein_id=ids[i], length=len(seqs[i]), relative_l2=error, mean_cosine=cosine))
    save_json(out/'native_api_validation.json', dict(passed=True, checks=errors))
    del native, reference, batched
    torch.cuda.empty_cache()
    completed = failed = fresh = 0
    failures = []
    extraction_started = time.time()
    part_paths = []

    def safe_features(indices):
        try:
            return batch_features(indices)
        except torch.cuda.OutOfMemoryError:
            pass
        # Leave the except block first so its traceback releases failed tensors.
        torch.cuda.empty_cache()
        if len(indices) > 1:
            mid = len(indices)//2
            return safe_features(indices[:mid]) + safe_features(indices[mid:])
        i = indices[0]
        failures.append(dict(index=i, protein_id=ids[i], length=len(seqs[i]), error='CUDA OOM; no truncation or substitute'))
        return []

    for part_no, start in enumerate(range(0, len(order), args.part_size)):
        indices = order[start:start+args.part_size]
        path = out/f'part_{part_no:05d}.h5'
        part_paths.append(path)
        if path.exists():
            with h5py.File(path, 'r') as f:
                assert list(f['requested_indices'][:]) == indices
                completed += len(f['indices'])
                errors = json.loads(f.attrs['failures'])
                failed += len(errors)
                failures.extend(errors)
            continue
        rows = []
        before_errors = len(failures)
        cursor = 0
        while cursor < len(indices):
            n = min(32, len(indices)-cursor)
            while n > 1 and n*(len(seqs[indices[cursor+n-1]])+2) > args.token_budget:
                n = max(1, n//2)
            rows.extend(safe_features(indices[cursor:cursor+n]))
            cursor += n
        offsets = np.cumsum([0]+[len(v) for _,v,_ in rows], dtype=np.int64)
        tmp = path.with_suffix('.h5.tmp')
        with h5py.File(tmp, 'w') as f:
            f.attrs['failures'] = json.dumps(failures[before_errors:])
            f.attrs['catalog_sha256'] = protocol['catalog_sha256']
            f.attrs['checkpoint_sha256'] = protocol['checkpoint_sha256']
            f.create_dataset('requested_indices', data=indices)
            f.create_dataset('indices', data=[i for i,_,_ in rows], dtype='i8')
            f.create_dataset('ids', data=[ids[i] for i,_,_ in rows], dtype=h5py.string_dtype())
            f.create_dataset('offsets', data=offsets)
            vectors = f.create_dataset('vectors', shape=(int(offsets[-1]),1152), dtype='f4', compression='lzf')
            for j, (_,v,_) in enumerate(rows):
                vectors[offsets[j]:offsets[j+1]] = v
            f.create_dataset('native_mean', data=np.asarray([m for _,_,m in rows], dtype=np.float32).reshape(-1,1152))
        tmp.replace(path)
        completed += len(rows)
        fresh += len(rows)
        failed += len(failures)-before_errors
        status('extracting', completed, failed, last_part=part_no,
            records_per_second=fresh/max(time.time()-extraction_started,1),
            gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
        del rows
    save_json(out/'failures.json', failures)
    save_json(out/'parts.json', dict(parts=[p.name for p in part_paths],
        completed=completed, failed=failed, total=len(order), protocol='protocol.json'))
    if not failed:
        # Exact sequence-keyed dictionary expected by the author's dataset loader.
        seq2feature = {}
        for path in part_paths:
            with h5py.File(path, 'r') as f:
                for i, mean in zip(f['indices'][:], f['native_mean'][:]):
                    seq2feature[seqs[i]] = torch.from_numpy(mean.copy())
        tmp = out/'seq2feature.pkl.tmp'
        with tmp.open('wb') as f:
            pickle.dump(seq2feature, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(out/'seq2feature.pkl')
        save_json(out/'complete.json', dict(completed=completed, total=len(order),
            checkpoint_sha256=protocol['checkpoint_sha256'], smoke_only=bool(args.limit)))
    status('complete' if not failed else 'incomplete', completed, failed)


if __name__ == '__main__':
    main()
