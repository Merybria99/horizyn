#!/usr/bin/env python3
"""Prespecified, resumable, label-free Case1 retrieval on a fixed RefSeq tile sample."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from horizyn.generalization_retrieval import canonical_dot, checked_artifact, sha256
from horizyn.semantic_anchors import row_unit
from generalization_full_graph import atomic_json

RUN = ROOT / 'runs/generalization_20260919_2251'
METHODS = ['phase2', 'phase4', 'f3_native', 'f3_fp64', 'circev2']


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def source_stat(path):
    path = Path(path).resolve(); s = path.stat()
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino)


def check_source_stat(item):
    if source_stat(item['path']) != {k: item[k] for k in ('path', 'size', 'mtime_ns', 'inode')}:
        raise ValueError('Residue source identity changed: ' + item['path'])


def tile_selection(n, tile_size=4096, tile_count=256, seed=20260920):
    total = (n + tile_size - 1) // tile_size
    if n <= 0 or tile_count > total:
        raise ValueError('Invalid tile sample size')
    chosen = np.sort(np.random.default_rng(seed).choice(total, tile_count, replace=False))
    tiles = [dict(tile_index=int(i), start=int(i * tile_size), stop=min(n, int((i + 1) * tile_size))) for i in chosen]
    indices = np.concatenate([np.arange(x['start'], x['stop'], dtype=np.int64) for x in tiles])
    return tiles, indices


def alias_catalog(background_ids, literature, full_match):
    ids = list(background_ids); lookup = {key: i for i, key in enumerate(ids)}
    if len(lookup) != len(ids): raise ValueError('Duplicate selected RefSeq IDs')
    rows = []
    if 'proteins' in literature and [g['representative_id'] for g in literature['groups']]!=literature['proteins']:
        raise ValueError('Literature group order differs from canonical catalog order')
    for group in literature['groups']:
        rep = group['representative_id']; match = full_match.get(rep, {}).get('refseq_ids', [])
        selected = [key for key in match if key in lookup]
        if len(selected) > 1: raise ValueError('Unexpected duplicate full RefSeq sequence')
        present = bool(selected)
        candidate = selected[0] if present else 'LIT_' + rep
        if not present:
            if candidate in lookup: raise ValueError('Appended candidate ID collision')
            lookup[candidate] = len(ids); ids.append(candidate)
        rows.append(dict(representative_id=rep, sequence_sha256=group['sequence_sha256'],
                         all_entry_ids=group['all_entry_ids'], candidate_id=candidate,
                         candidate_index=lookup[candidate], already_in_selected_background=present))
    if len({row['sequence_sha256'] for row in rows}) != len(rows):
        raise ValueError('Literature catalog is not sequence-deduplicated')
    return ids, rows


def prepare(args):
    out = args.output
    if (out / 'protocol.json').exists(): raise ValueError('Immutable protocol already exists')
    out.mkdir(parents=True, exist_ok=True)
    audit_path = RUN / 'large_case1_feasibility/candidate_protocol.json'
    audit = json.loads(audit_path.read_text())
    literature_path = RUN / 'case1_audit/features/catalog.json'
    checked_artifact(audit['literature_catalog'])
    literature = json.loads(literature_path.read_text())
    source = ROOT / 'wet_lab/databases/refseq/prokaryotes/five_shards/proteins_prott5_residue.h5'
    for item in audit['residue_source_metadata']: check_source_stat(item)
    with h5py.File(source, 'r') as f:
        all_ids = f['ids'].asstr()[:]
        offsets = f['offsets'][:]
        if len(all_ids) != 3944613 or len(offsets) != len(all_ids) + 1: raise ValueError('Unexpected fixed RefSeq source size')
        tiles, indices = tile_selection(len(all_ids))
        ids = all_ids[indices].tolist()
        for tile in tiles:
            tile['residue_start'] = int(offsets[tile['start']]); tile['residue_stop'] = int(offsets[tile['stop']])
    candidates, aliases = alias_catalog(ids, literature, audit['matching_literature_aliases'])
    selected = out / 'selected_metadata.npz'
    np.savez(selected, ids=np.asarray(ids), indices=indices,
             offsets=offsets, tile_indices=np.asarray([t['tile_index'] for t in tiles], dtype=np.int64))
    atomic_json(out / 'catalog.json', dict(proteins=candidates, query_ids=literature['query_ids'],
        background_count=len(ids), candidate_count=len(candidates), order='Selected RefSeq in original VDS order, then appended literature in source canonical catalog order'))
    atomic_json(out / 'aliases.json', dict(groups=aliases, literature_catalog=record(literature_path),
        exact_sequence_policy='All 123 full-sequence groups forced included; matching sampled RefSeq sequence is reused. Globally matching but unsampled sequences are appended.'))
    base = RUN / 'case1_audit/features'
    receipts = {name: record(base / filename) for name, filename in (
        ('f3', 'feature_bundle_receipt.json'), ('circev2', 'circev2_feature_bundle_receipt.json'))}
    for item in receipts.values():
        receipt = json.loads(checked_artifact(item).read_text())
        for source_item in list(receipt['inputs'].values()) + [receipt['checkpoint']] + receipt['source_receipts']:
            checked_artifact(source_item)
    sources = [Path(__file__), ROOT / 'scripts/generalization_large_case1_native.py']
    helper_parity=json.loads((RUN/'large_case1_feasibility/circev2_shared_stream/helper_parity.json').read_text())
    if not helper_parity.get('native_source_closure'):raise ValueError('Native reference source closure is required')
    for item in helper_parity['native_source_closure']:sources.append(checked_artifact(item))
    phase4_spec=json.loads((RUN/'phase4/models/reaction_smi/seed42/bundle.json').read_text())
    phase4_freeze_path=checked_artifact(phase4_spec['phase4_frozen_recipe'])
    for item in json.loads(phase4_freeze_path.read_text())['implementation_sources']:
        sources.append(checked_artifact(item,phase4_freeze_path.parent))
    sources=sorted({p.resolve() for p in sources},key=str)
    protocol = dict(schema='large_case1_input_protocol_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        frozen_before_scores=True, labels_read_for_selection=False, sampled_retrieval_authorized=True,
        background_count=len(ids), candidate_count=len(candidates), methods=METHODS,cutoffs=[1,10,25,100,1000,10000],
        sampling=dict(seed=20260920, rng='numpy.random.default_rng; PCG64', numpy_version=np.__version__,
            universe=3944613, tile_size=4096, universe_tiles=(3944613+4095)//4096, tile_count=256,
            without_replacement=True, sorted_for_io=True,
            caveat='A physical block-cluster sample of a pre-existing RefSeq subset, not candidate-IID sampling and not all RefSeq. No full-pool rank extrapolation.'),
        tiles=tiles, artifact_order='Sampled RefSeq original VDS order, then appended literature canonical catalog order',
        artifacts={key: record(out / name) for key, name in (
            ('catalog', 'catalog.json'), ('aliases', 'aliases.json'), ('selected_metadata', 'selected_metadata.npz'))},
        evaluation_source=record(args.evaluation_source), implementation_sources=[record(p) for p in sources],
        evaluation_dependencies=[record(ROOT/'scripts/generalization_external_evaluate.py')],
        evaluation_label_sources=[record(RUN/'case1_audit'/relative) for relative in (
            'candidate_evidence.json','positive_sets.json','source_verification.json','features/catalog.json')],
        candidate_input_audit=record(audit_path), residue_vds=source_stat(source),
        residue_shards=audit['residue_source_metadata'], feature_receipts=receipts,
        phase2_bundle=record(RUN / 'phase2/models/reaction_smi/seed42/bundle.json'),
        phase4_bundle=record(RUN / 'phase4/models/reaction_smi/seed42/bundle.json'),
        native_reference_receipt=record(RUN / 'large_case1_feasibility/circev2_shared_stream/complete.json'),
        native_helper_parity=record(RUN / 'large_case1_feasibility/circev2_shared_stream/helper_parity.json'),
        fixed_native_batch_size=32, fixed_postprocessor_batch_size=512,
        precision='Native FP32 highest matmul, TF32 false, no autocast; FP32 endpoint normalization for native F3/CIRCE; declared FP64 reduction normalization for F3_fp64 and frozen postprocessors; final canonical FP64 dot rounded FP32.',
        native_truncation='ends_center1022 from the stored up-to1024 ProtT5 rows',
        raw_mean='NumPy mean(axis=0,dtype=float32) over ALL stored FP16 residue rows, before native truncation',
        queries='Unchanged, authenticated query_matched features; participant-set self-reaction convention',
        reference_controls=dict(phase2=record(RUN / 'phase2/predictions/case1/seed42/complete.json'),
                                phase4=record(RUN / 'phase4/predictions/case1/seed42/complete.json')),
        enzyme_index='Phase2 and phase4 share exactly the same frozen enzyme endpoint; save FP32 dense block + CSR semantic block per chunk.',
        metrics=dict(primary='Uniform-tie expected known-catalyst recovery and Recall@K for 12 independently rechecked paper-positive unique sequences',
            secondary='Same for 24 paper-plus-patent positives', descriptive='Same for 81 workbook-reported active sequences',
            cuts=[1,10,25,100,1000,10000], per_positive='Stable rank, best/worst tied rank, tie size and expected reciprocal rank',
            aggregate='Macro mean expected reciprocal rank over each positive tier; optional first-positive rank secondary',
            interpretation='Forced inclusion of literature candidates among unknown-activity RefSeq proteins. No AUROC/AP treating unlisted proteins as negative, no candidate-IID confidence intervals, no full-4M extrapolation.'),
        budget=dict(seconds=12600, initial_throughput_gate_tiles=4,
            condition='After first four newly computed background tiles, stop if projected remaining time exceeds remaining admission budget.',
            cancellation='Check deadline between tiles. A hard NFS read may remain in uninterruptible kernel wait; deadline cannot guarantee immediate cancellation.'),
        full_pool_scan=False, candidate_expansion_after_scores=False)
    atomic_json(out / 'protocol.json', protocol)
    print(json.dumps(dict(protocol=record(out/'protocol.json'), background_count=len(ids), candidate_count=len(candidates),
                          selected_residue_bytes=sum((t['residue_stop']-t['residue_start'])*2048 for t in tiles))), flush=True)


def validate_protocol(out):
    path = out / 'protocol.json'; protocol = json.loads(path.read_text())
    if protocol.get('schema') != 'large_case1_input_protocol_v1' or protocol['methods'] != METHODS or not protocol['frozen_before_scores']:
        raise ValueError('Invalid immutable scan protocol')
    for item in protocol['implementation_sources'] + protocol['evaluation_dependencies'] + [protocol['evaluation_source']] + list(protocol['artifacts'].values()): checked_artifact(item)
    checked_artifact(protocol['candidate_input_audit']); checked_artifact(protocol['native_reference_receipt']);checked_artifact(protocol['native_helper_parity'])
    for key in ('phase2_bundle', 'phase4_bundle'): checked_artifact(protocol[key])
    check_source_stat(protocol['residue_vds'])
    for item in protocol['residue_shards']: check_source_stat(item)
    for item in protocol['feature_receipts'].values():
        receipt = json.loads(checked_artifact(item).read_text())
        for r in list(receipt['inputs'].values()) + [receipt['checkpoint']] + receipt['source_receipts']: checked_artifact(r)
    return protocol


def read_tile(protocol, tile, expected_ids):
    start = time.monotonic()
    with h5py.File(protocol['residue_vds']['path'], 'r', rdcc_nbytes=64*1024**2) as f:
        ids = f['ids'].asstr()[tile['start']:tile['stop']].tolist()
        if ids != expected_ids: raise ValueError('Read tile IDs differ from prespecified selected IDs')
        offsets = f['offsets'][tile['start']:tile['stop']+1]
        if int(offsets[0]) != tile['residue_start'] or int(offsets[-1]) != tile['residue_stop']:
            raise ValueError('Read tile offsets differ from protocol')
        raw = f['vectors'][int(offsets[0]):int(offsets[-1])]
    read_seconds = time.monotonic()-start; local = offsets-offsets[0]
    if raw.dtype != np.float16 or raw.shape != (int(local[-1]),1024):
        raise ValueError('Invalid stored residue tile shape or dtype')
    # The native helper checks every residue for finiteness before any encoding.
    t = time.monotonic()
    means = np.stack([raw[a:b].mean(0,dtype=np.float32) for a,b in zip(local[:-1],local[1:])])
    if not np.isfinite(means).all(): raise ValueError('Nonfinite all-stored raw mean')
    mean_seconds=time.monotonic()-t
    digest=hashlib.sha256(memoryview(raw)).hexdigest()
    return raw, local, means, dict(read_seconds=read_seconds, mean_seconds=mean_seconds,
        prepare_seconds=time.monotonic()-start, logical_bytes=raw.nbytes, raw_residue_sha256=digest)


def load_queries(protocol, device):
    pending_spec=json.loads(checked_artifact(protocol['phase4_bundle']).read_text())
    freeze_path=checked_artifact(pending_spec['phase4_frozen_recipe']);freeze=json.loads(freeze_path.read_text())
    if not freeze.get('frozen_before_phase4_prediction') or not freeze.get('frozen_before_new_external_evaluation') or not freeze.get('implementation_sources'):
        raise ValueError('Phase4 source closure must have been frozen before inference')
    for item in freeze['implementation_sources']:checked_artifact(item,freeze_path.parent)
    from horizyn.generalization_phase2 import ComposedPhase2Encoder
    from horizyn.generalization_phase4 import HybridPhase4Encoder
    phase2, spec2 = ComposedPhase2Encoder.from_bundle(checked_artifact(protocol['phase2_bundle']),device)
    phase4, spec4 = HybridPhase4Encoder.from_bundle(checked_artifact(protocol['phase4_bundle']),device)
    if any((spec['variant'],spec['split'],spec['seed'],spec['alpha'])!=('primary','reaction_smi',42,.25) for spec in (spec2,spec4)):
        raise ValueError('Both primary method identities must be frozen Reaction-Sim seed42 alpha0.25')
    if spec4['phase2_bundle']['sha256'] != protocol['phase2_bundle']['sha256'] or phase2.alpha != phase4.alpha or phase2.alpha != .25:
        raise ValueError('Phase2/4 candidate endpoint lineage mismatch')
    receipt = json.loads(checked_artifact(protocol['feature_receipts']['f3']).read_text())
    cr = json.loads(checked_artifact(protocol['feature_receipts']['circev2']).read_text())
    if cr['inputs']['catalog']['sha256']!=receipt['inputs']['catalog']['sha256']:
        raise ValueError('CIRCE and F3 native input catalogs differ')
    if receipt['checkpoint']['sha256'] != spec2['base_checkpoint']['sha256']:
        raise ValueError('F3 native feature checkpoint mismatch')
    with np.load(checked_artifact(receipt['inputs']['base'])) as f:
        f3_query=torch.tensor(f['query_matched'],device=device); f3_literature=f['proteins'].copy()
    with np.load(checked_artifact(cr['inputs']['base'])) as f:
        circe_query=torch.tensor(f['query_matched'],device=device); circe_literature=f['proteins'].copy()
    catalog=json.loads(checked_artifact(receipt['inputs']['catalog']).read_text())
    with h5py.File(checked_artifact(receipt['inputs']['protein_means']),'r') as f:
        if f['ids'].asstr()[:].tolist()!=catalog['proteins']: raise ValueError('Literature raw mean ID mismatch')
        means=f['vectors'][:]
    with np.load(checked_artifact(receipt['inputs']['reaction_features'])) as f:
        blocks={key:torch.tensor(f[key],device=device) for key in phase2.modalities}
        masks={key:torch.tensor(f[key+'_mask'],device=device) for key in phase2.modalities}
    with torch.inference_mode():
        queries=dict(phase2=phase2.encode_reactions(f3_query,blocks,masks), phase4=phase4.encode_reactions(f3_query,blocks,masks),
            f3_native=F.normalize(f3_query,dim=1),f3_fp64=row_unit(f3_query),circev2=F.normalize(circe_query,dim=1))
        if not torch.equal(phase2.encode_enzymes(torch.tensor(f3_literature,device=device),torch.tensor(means,device=device)),
                           phase4.encode_enzymes(torch.tensor(f3_literature,device=device),torch.tensor(means,device=device))):
            raise ValueError('Claimed shared candidate endpoint is not exact')
    return phase2, queries, dict(ids=catalog['proteins'], f3=f3_literature, circev2=circe_literature, raw_mean=means)


@torch.inference_mode()
def score_features(post, queries, f3, circe, means, device):
    f3t=torch.tensor(f3,device=device); ct=torch.tensor(circe,device=device); mt=torch.tensor(means,device=device)
    if f3t.shape!=(len(means),512) or ct.shape!=f3t.shape or mt.shape!=(len(means),1024): raise ValueError('Invalid compact endpoint shapes')
    if any(x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) for x in (f3t,ct,mt)): raise ValueError('Invalid compact endpoint values')
    index=post.encode_enzyme_index(f3t,mt,batch_size=512)
    score=dict(phase2=post.score_index(queries['phase2'],index),phase4=post.score_index(queries['phase4'],index),
        f3_native=canonical_dot(queries['f3_native'],F.normalize(f3t,dim=1)),
        f3_fp64=canonical_dot(queries['f3_fp64'],row_unit(f3t)),
        circev2=canonical_dot(queries['circev2'],F.normalize(ct,dim=1)))
    if any(x.shape!=(1,len(means)) or not bool(torch.isfinite(x).all()) for x in score.values()): raise ValueError('Invalid scores')
    cpu_index=dict(dense=index['dense'].cpu(),anchors=index['anchors'].cpu())
    return {k:v.cpu().numpy()[0] for k,v in score.items()},cpu_index


def verify_reference(protocol, post, queries, literature, device, out):
    scores,_=score_features(post,queries,literature['f3'],literature['circev2'],literature['raw_mean'],device)
    checks={}
    for phase in ('phase2','phase4'):
        receipt_path=checked_artifact(protocol['reference_controls'][phase]);receipt=json.loads(receipt_path.read_text())
        path=receipt_path.parent/'scores.npz'
        if sha256(path)!=receipt['output_sha256']: raise ValueError('Historical reference score integrity mismatch')
        with np.load(path) as f:
            for here,there in ((phase,'selected'),('f3_native','baseline'),('f3_fp64','baseline_fp64')):
                exact=bool(np.array_equal(scores[here],f[there][0]));checks[phase+'_'+here]=exact
                if not exact: raise ValueError('Fixed query/reference parity failed: '+phase+'_'+here)
    ref=json.loads(checked_artifact(protocol['native_reference_receipt']).read_text())
    if not all(row['embeddings_exact'] and row['canonical_native_scores_exact'] for row in ref['parity'].values()):
        raise ValueError('Shared native reference parity was not established')
    atomic_json(out/'reference_check.json',dict(protocol=record(out/'protocol.json'),all_exact=True,checks=checks,
        circev2_shared_native_parity=protocol['native_reference_receipt'],labels_used=False))


def save_chunk(path, ids, f3, circe, means, scores, index, protocol_record, metadata):
    path.mkdir(parents=True,exist_ok=True)
    np.savez(path/'features.npz',ids=np.asarray(ids),f3=f3,circev2=circe,raw_mean=means)
    np.savez(path/'scores.npz',ids=np.asarray(ids),**scores)
    torch.save(index,path/'enzyme_index.pt')
    receipt=dict(schema='large_case1_chunk_v1',protocol=protocol_record,rows=len(ids),
        ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),labels_used=False,
        outputs={name:record(path/file) for name,file in [('features','features.npz'),('scores','scores.npz'),('enzyme_index','enzyme_index.pt')]},
        **metadata)
    atomic_json(path/'complete.json',receipt)
    return record(path/'complete.json')


def verify_chunk(path, ids, protocol_record):
    receipt=json.loads(path.read_text())
    if receipt['protocol']!=protocol_record or receipt['rows']!=len(ids) or receipt['ids_sha256']!=hashlib.sha256('\n'.join(ids).encode()).hexdigest():
        raise ValueError('Chunk resume row/protocol mismatch')
    for item in receipt['outputs'].values():checked_artifact(item)
    with np.load(checked_artifact(receipt['outputs']['scores'])) as f:
        if f['ids'].tolist()!=ids or any(f[k].shape!=(len(ids),) or not np.isfinite(f[k]).all() for k in METHODS):
            raise ValueError('Chunk resume score mismatch')
    return receipt


def run(args):
    out=args.output; protocol=validate_protocol(out);protocol_record=record(out/'protocol.json')
    if (out/'complete.json').exists():raise ValueError('Scan already complete')
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_device(args.device);torch.cuda.reset_peak_memory_stats(args.device);torch.manual_seed(42)
    started=time.monotonic();deadline=started+protocol['budget']['seconds']
    parity=json.loads(checked_artifact(protocol['native_helper_parity']).read_text())
    if not parity.get('native_source_closure'):raise ValueError('Require native source closure from exact reference parity process')
    for item in parity['native_source_closure']:checked_artifact(item)
    from generalization_large_case1_native import SharedNativeEncoders
    native=SharedNativeEncoders(device=args.device)
    for key in ('f3','circev2'):
        receipt=json.loads(checked_artifact(protocol['feature_receipts'][key]).read_text())
        if native.identities[key]['checkpoint']['sha256']!=receipt['checkpoint']['sha256']:
            raise ValueError('Native encoder checkpoint differs from fixed query checkpoint')
    atomic_json(out/'native_load.json',dict(protocol=protocol_record,identities=native.identities,load_seconds=native.load_seconds))
    post,queries,literature=load_queries(protocol,args.device)
    verify_reference(protocol,post,queries,literature,args.device,out)
    with np.load(checked_artifact(protocol['artifacts']['selected_metadata'])) as f: selected_ids=f['ids'].tolist()
    catalog=json.loads(checked_artifact(protocol['artifacts']['catalog']).read_text()); aliases=json.loads(checked_artifact(protocol['artifacts']['aliases']).read_text())
    jobs=[];position=0; receipts=[]
    for tile in protocol['tiles']:
        n=tile['stop']-tile['start'];ids=selected_ids[position:position+n];position+=n
        path=out/'chunks'/f"tile{tile['tile_index']:04d}"
        if (path/'complete.json').exists(): verify_chunk(path/'complete.json',ids,protocol_record);receipts.append(record(path/'complete.json'))
        else:jobs.append((tile,ids,path))
    scan_start=time.monotonic();new_rows=0;status='running';completed_new=0
    # One producer with one tile of lookahead bounds host memory while overlapping I/O/means with GPU work.
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending=executor.submit(read_tile,protocol,jobs[0][0],jobs[0][1]) if jobs else None
        for i,(tile,ids,path) in enumerate(jobs):
            if time.monotonic()>=deadline:status='incomplete_deadline';break
            raw,offsets,means,metadata=pending.result()
            pending=executor.submit(read_tile,protocol,jobs[i+1][0],jobs[i+1][1]) if i+1<len(jobs) else None
            t=time.monotonic(); encoded=native.encode_block(raw,offsets,ids)
            f3=encoded['f3'];circe=encoded['circev2'];torch.cuda.synchronize(args.device)
            metadata['native_encode_seconds']=time.monotonic()-t
            del raw,offsets
            t=time.monotonic();scores,index=score_features(post,queries,f3,circe,means,args.device)
            metadata['postprocess_score_seconds']=time.monotonic()-t
            metadata.update(tile=tile,tile_index=tile['tile_index'],native_timing=encoded['timing'],peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
            receipt=save_chunk(path,ids,f3,circe,means,scores,index,protocol_record,metadata);receipts.append(receipt)
            new_rows+=len(ids);completed_new+=1
            elapsed=time.monotonic()-scan_start
            remaining_rows=sum(j[0]['stop']-j[0]['start'] for j in jobs[i+1:])
            projected_remaining=remaining_rows*elapsed/new_rows
            progress=dict(protocol=protocol_record,status=status,completed_background_tiles=len(protocol['tiles'])-len(jobs)+completed_new,
                planned_background_tiles=len(protocol['tiles']),new_rows=new_rows,elapsed_seconds=time.monotonic()-started,
                projected_remaining_seconds=projected_remaining,latest_chunk=receipt)
            atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            del means,encoded,f3,circe,scores,index
            if completed_new==protocol['budget']['initial_throughput_gate_tiles'] and projected_remaining>deadline-time.monotonic():
                status='incomplete_initial_throughput_gate';break
    if status!='running':
        atomic_json(out/'incomplete.json',dict(protocol=protocol_record,status=status,elapsed_seconds=time.monotonic()-started,
            note='Do not evaluate or reinterpret this incomplete prefix as the fixed sampled pool. Resume is possible with unchanged protocol.'))
        return
    # Forced literature inclusions use the exact previously authenticated reference encodings.
    additions=[r for r in aliases['groups'] if not r['already_in_selected_background']]
    lookup={key:i for i,key in enumerate(literature['ids'])};ix=[lookup[r['representative_id']] for r in additions]
    ids=[r['candidate_id'] for r in additions];path=out/'chunks/literature'
    if (path/'complete.json').exists():verify_chunk(path/'complete.json',ids,protocol_record)
    else:
        f3=literature['f3'][ix];circe=literature['circev2'][ix];means=literature['raw_mean'][ix]
        scores,index=score_features(post,queries,f3,circe,means,args.device)
        save_chunk(path,ids,f3,circe,means,scores,index,protocol_record,dict(tile_index='literature',source='Authenticated existing complete literature input encodings; no score/activity filtering'))
    ordered=[out/'chunks'/f"tile{t['tile_index']:04d}"/'complete.json' for t in protocol['tiles']]+[path/'complete.json']
    all_ids=[];values={k:[] for k in METHODS};cursor=0
    for receipt_path in ordered:
        receipt=json.loads(receipt_path.read_text());n=receipt['rows'];ids=catalog['proteins'][cursor:cursor+n];cursor+=n
        verify_chunk(receipt_path,ids,protocol_record)
        with np.load(checked_artifact(receipt['outputs']['scores'])) as f:
            all_ids.extend(f['ids'].tolist())
            for key in METHODS:values[key].append(f[key].copy())
    if all_ids!=catalog['proteins'] or cursor!=protocol['candidate_count']:raise ValueError('Final candidate coverage/order mismatch')
    np.savez(out/'scores.npz',ids=np.asarray(all_ids),**{k:np.concatenate(v) for k,v in values.items()})
    validate_protocol(out)
    complete=dict(schema='large_case1_scan_complete_v1',protocol=protocol_record,labels_used=False,
        background_count=protocol['background_count'],candidate_count=protocol['candidate_count'],methods=METHODS,
        counts=dict(background_count=protocol['background_count'],candidate_count=protocol['candidate_count'],tile_count=len(protocol['tiles'])),
        outputs={key:record(out/file) for key,file in [('scores','scores.npz'),('catalog','catalog.json'),('aliases','aliases.json'),('selected_metadata','selected_metadata.npz')]},
        chunk_receipts=[record(p) for p in ordered],reference_check=record(out/'reference_check.json'),
        elapsed_seconds=time.monotonic()-started,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
        complete_fixed_candidate_coverage=True,candidate_expansion_after_scores=False)
    atomic_json(out/'complete.json',complete);print(json.dumps(complete),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--output',type=Path,default=RUN/'large_case1_scan');p.add_argument('--device',default='cuda:2')
    p.add_argument('--evaluation-source',type=Path,default=ROOT/'scripts/generalization_large_case1_evaluate.py')
    args=p.parse_args()
    if args.action=='prepare':prepare(args)
    else:
        try:run(args)
        except Exception as error:
            import traceback
            atomic_json(args.output/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),labels_used=False))
            raise


if __name__=='__main__':main()
