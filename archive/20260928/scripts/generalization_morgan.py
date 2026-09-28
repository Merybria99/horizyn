#!/usr/bin/env python3
"""Prespecified five-condition train/validation-only Morgan anchor study."""
from __future__ import annotations
import argparse,csv,datetime,json,math,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from rdkit import rdBase

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from horizyn.generalization_morgan import canonical_components,participant_morgan,MorganReactionAnchor
from horizyn.generalization_retrieval import canonical_dot,checked_artifact,sha256,GeneralizationDualEncoder
from horizyn.semantic_anchors import reaction_features,row_unit
from horizyn.semantic_smooth import reaction_responses
from generalization_full_graph import atomic_json,validation_data,eligible,selection_value
from generalization_smooth_anchors import robust_value
from generalization_metrics import evaluate_scores

RUN=ROOT/'runs/generalization_20260919_2251';OUT=RUN/'post_evaluation_morgan'
GRID=[dict(label='identity',radius=None,eta=0.)]+[dict(label=f'radius{r}_eta{e:g}',radius=r,eta=e) for r in (2,3) for e in (.5,1.)]

def record(path):return dict(path=str(Path(path).resolve()),sha256=sha256(path))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sources():
    f=RUN/'features';p=RUN/'phase2/composition_fp64'
    bundle=json.loads((RUN/'phase2/models/reaction_smi/seed42/bundle.json').read_text())
    manifest=json.loads((f/'manifest.json').read_text())
    return {**{k:f/k for k in ('catalog.json','pairs.npz','manifest.json','f3_features.npz','reaction_features.npz')},
        'parent_bundle':RUN/'phase2/models/reaction_smi/seed42/bundle.json',
        'late_authorization':RUN/'late_representation_authorization.json',
        'phase2_freeze':RUN/'phase2/frozen_recipe.json','phase4_freeze':RUN/'phase4/frozen_recipe.json',
        'dictionary':checked_artifact(bundle['smooth_dictionary']),
        'train_reactions':checked_artifact(manifest['inputs']['train_reactions_path']),
        'validation_reactions':checked_artifact(manifest['inputs']['validation_reactions_path']),
        'parent_validation':p/'selected_validation_features.pt','parent_ranks':p/'density_smooth_alpha0.25_ranks.npz',
        'parent_selection':p/'selection.json','script':Path(__file__),
        'module':ROOT/'horizyn/generalization_morgan.py','tests':ROOT/'tests/unit/test_generalization_morgan.py'}

def closure(paths):
    frozen=json.loads(paths['phase4_freeze'].read_text())
    items={x['path']:x for x in frozen['implementation_sources']}
    for path in [paths[k] for k in ('script','module','tests')]+[ROOT/'scripts'/f for f in ('generalization_full_graph.py','generalization_smooth_anchors.py','generalization_metrics.py')]:
        item=record(path);items[item['path']]=item
    for item in items.values():checked_artifact(item)
    return list(items.values())

def reaction_table(path):
    result={}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            key,value=row['reaction_id'],row['reaction_smiles']
            if key in result and result[key]!=value:raise ValueError('Conflicting reaction ID/string')
            result[key]=value
    return result

def configure(device):
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_device(device)

@torch.inference_mode()
def prepare(args):
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists() or (OUT/'prepared.pt').exists():raise ValueError('Existing immutable preparation')
    paths=sources();inputs={k:record(v) for k,v in paths.items()};implementation=closure(paths)
    atomic_json(OUT/'preparation.json',dict(schema='morgan_training_validation_input_preparation_v1',created_utc=utc(),inputs=inputs,implementation_sources=implementation,grid=GRID,no_external_inputs=True,no_scores_computed=True))
    catalog=json.loads(paths['catalog.json'].read_text())
    with np.load(paths['pairs.npz']) as source:pairs={k:source[k] for k in source.files}
    tr=np.unique(pairs['train'][:,0]);te=np.unique(pairs['train'][:,1]);vr,ve,_=validation_data(catalog,pairs)
    if catalog['train_reactions']!=[catalog['reactions'][i] for i in tr] or catalog['train_proteins']!=[catalog['proteins'][i] for i in te]:raise ValueError('Declared training endpoints disagree with actual train edges')
    strings=reaction_table(paths['train_reactions'])
    for key,value in reaction_table(paths['validation_reactions']).items():
        if key in strings and strings[key]!=value:raise ValueError('Train/validation reaction string conflict')
        strings[key]=value
    ids=catalog['reactions'];invalid=[];component_counts=[]
    for key in ids:
        try:component_counts.append(len(canonical_components(strings[key])))
        except Exception as error:invalid.append(dict(reaction_id=key,error=str(error)))
    audit=dict(schema='morgan_input_parse_audit_v1',reaction_count=len(ids),train_reactions=len(tr),validation_reactions=len(vr),invalid_count=len(invalid),invalid=invalid,rdkit_version=rdBase.rdkitVersion,component_counts=component_counts,no_rows_dropped=True,atom_map_labels_removed_only=True,canonical_isotope_labels_retained=True,standard_descriptor_isotope_insensitivity=True)
    atomic_json(OUT/'input_audit.json',audit)
    if invalid:raise ValueError('Invalid molecular inputs; no feature fit or validation performed')
    fingerprints={radius:np.stack([participant_morgan(strings[key],radius) for key in ids]) for radius in (2,3)}
    np.savez(OUT/'morgan_features.npz',radius2=fingerprints[2],radius3=fingerprints[3])
    atomic_json(OUT/'reaction_catalog.json',dict(reaction_ids=ids,train_indices=tr.tolist(),validation_indices=vr.tolist(),reaction_strings=[strings[key] for key in ids]))
    device=torch.device(args.device);configure(device)
    bundle=json.loads(paths['parent_bundle'].read_text())
    from horizyn.generalization_phase2 import validate_phase2_bundle
    validate_phase2_bundle(bundle,paths['parent_bundle'].parent)
    config=bundle['smooth_config']
    if bundle['alpha']!=.25 or bundle['seed']!=42 or bundle['split']!='reaction_smi' or (config['protein_neighbors'],config['enzyme_temperature'],config['reaction_temperature'],config['kernel'],config['reaction_neighbors'])!=(32,.03,.03,'exponential',None):raise ValueError('Parent is not fixed P2 primary')
    dictionary=torch.load(paths['dictionary'],map_location=device,weights_only=False)
    if dictionary['feature_manifest_sha256']!=inputs['manifest.json']['sha256'] or dictionary['train_reaction_ids']!=catalog['train_reactions'] or dictionary['train_protein_ids']!=catalog['train_proteins']:raise ValueError('Training dictionary/source IDs disagree')
    with np.load(paths['reaction_features.npz']) as source:
        blocks={k:torch.tensor(source[k][vr],device=device) for k in dictionary['modalities']}
        masks={k:torch.tensor(source[k+'_mask'][vr],device=device) for k in dictionary['modalities']}
    raw=reaction_features(blocks,dictionary['reaction_centers'],masks,dictionary['modalities'])
    train_raw=dictionary['train_reactions']
    if len(train_raw)!=len(tr):raise ValueError('Raw training dictionary count mismatch')
    # Geometry is prepared before validation scoring. The frozen endpoint/rank
    # reproduction is required separately at the beginning of run().
    torch.save(dict(schema='morgan_prepared_geometry_v1',raw_validation=raw.cpu(),raw_training=train_raw.cpu(),train_ids=catalog['train_reactions'],validation_ids=catalog['validation_reactions'],inputs=inputs),OUT/'prepared.pt')
    protocol=dict(schema='late_morgan_reaction_view_validation_protocol_v1',created_utc=utc(),grid=GRID,inputs=inputs,implementation_sources=implementation,
        prepared=record(OUT/'prepared.pt'),morgan_features=record(OUT/'morgan_features.npz'),reaction_catalog=record(OUT/'reaction_catalog.json'),input_audit=record(OUT/'input_audit.json'),preparation=record(OUT/'preparation.json'),
        chemistry='Sanitize every component; clear atom-map labels only; canonical isomeric SMILES retaining charge/isotope/defined stereo. Complete dot-participant multiset, sorted with duplicates retained. Identical canonical self-reaction sides accepted once; non-self directional strings rejected.',
        fingerprint='RDKit GetMorganGenerator(radius2or3,fpSize4096,includeChirality=True), GetCountFingerprintAsNumPy. FP64 L2-normalize each component, sum sorted components including repetitions, FP64 L2-normalize total thenFP32. Standard invariants may ignore isotope differences; finite hashes collide. No fitted vocabulary or labels.',
        geometry='concat(FP32(sqrt(1-eta)*unchanged unit raw P2 reaction view),FP32(sqrt(eta)*unit Morgan)), no added normalization. Query and every actual training reaction independently encoded. Canonical FP64 similarity accumulation→FP32; unchanged exponential responses temperature.03, maximum stabilization, positive floor, FP64 unit norm→FP32.',
        composition='Frozen enzyme endpoints and dense reaction coordinates reused bitwise. Replace only final weighted semantic reaction block with .5*Morgan-anchor response. Alpha remains.25; densityheads/gates, protein-neighbor map and adjacency unchanged. Sparse enzyme index unchanged.',
        selection='Equal-weight seen/unseen all-positive MRR in both directions. All/unseen directions within.005 of BOTH F3_fp64 andP2. First grid winner for exactties; strict balancedgain overP2 required. Select bymetrics then numericalacceptance; failure rejects withoutchoosingalternate.',
        numerical='Everyrow full/subset/singleton+chunk7 reactionfeatures and scores; exact dense/CSR fixedenzymeindex parity. Identity delegates originalparent endpoints andpositive ranks exactly.',
        exploration='Late exploratory test after observed external failures and multiple validation analyses; no independent confirmation.',
        external_inputs_or_scores_used=False,large_pool_protocol_or_models_changed=False,external_prediction_authorized=False,
        versions=dict(rdkit=rdBase.rdkitVersion,numpy=np.__version__,torch=torch.__version__),preparation_seconds=time.monotonic()-start)
    atomic_json(OUT/'protocol.json',protocol);print(json.dumps(dict(protocol=record(OUT/'protocol.json'),input_invalid_count=0,seconds=protocol['preparation_seconds'])),flush=True)

def compare(a,b):return dict(exact=bool(torch.equal(a,b)),unequal=int(torch.count_nonzero(a!=b)),max_abs=float((a.double()-b.double()).abs().max()) if a.numel() else 0.)
def ranks(path,result):np.savez_compressed(path,**{d+'_'+k:v for d,block in result['per_positive'].items() for k,v in block.items()})

@torch.inference_mode()
def run(args):
    start=time.monotonic();protocol=json.loads((OUT/'protocol.json').read_text())
    if (OUT/'registry.json').exists():raise ValueError('Existing immutable execution')
    for item in protocol['implementation_sources']:checked_artifact(item)
    for item in protocol['inputs'].values():checked_artifact(item)
    for key in ('prepared','morgan_features','reaction_catalog','input_audit','preparation'):checked_artifact(protocol[key])
    if json.loads(Path(protocol['input_audit']['path']).read_text())['invalid_count']:raise ValueError('Invalid input rows')
    atomic_json(OUT/'registry.json',dict(protocol=record(OUT/'protocol.json'),started_utc=utc()))
    device=torch.device(args.device);configure(device);torch.cuda.reset_peak_memory_stats(device)
    p={k:Path(v['path']) for k,v in protocol['inputs'].items()};catalog=json.loads(p['catalog.json'].read_text())
    with np.load(p['pairs.npz']) as source:pairs={k:source[k] for k in source.files}
    vr,ve,truth=validation_data(catalog,pairs);tr=np.unique(pairs['train'][:,0])
    prepared=torch.load(protocol['prepared']['path'],map_location=device,weights_only=False)
    if prepared['train_ids']!=catalog['train_reactions'] or prepared['validation_ids']!=catalog['validation_reactions']:raise ValueError('Prepared geometry ID order mismatch')
    raw,train_raw=prepared['raw_validation'],prepared['raw_training']
    with np.load(protocol['morgan_features']['path']) as source:fps={radius:torch.tensor(source[f'radius{radius}'],device=device) for radius in (2,3)}
    parent=torch.load(p['parent_validation'],map_location=device,weights_only=False)
    if parent['enzyme_ids']!=catalog['validation_candidates'] or parent['reaction_ids']!=catalog['validation_reactions']:raise ValueError('Parent validation ID order mismatch')
    r,e=parent['reactions'],parent['enzymes'];index=dict(dense=e[:,:512],anchors=e[:,512:].to_sparse_csr())
    old_semantic=reaction_responses(raw,train_raw,'exponential',.03,None)
    if not torch.equal(.5*old_semantic,r[:,512:]):raise ValueError('Prepared raw geometry does not exactly replay P2 semantic branch')
    parent_scores=canonical_dot(r,e);parent_result=evaluate_scores(parent_scores,truth);p2=parent_result['summary']
    with np.load(p['parent_ranks']) as source:
        if not all(np.array_equal(source[d+'_'+k],v) for d,block in parent_result['per_positive'].items() for k,v in block.items()):raise ValueError('P2 rank replay failed')
    if robust_value(p2)!=json.loads(p['parent_selection'].read_text())['selected']['robust_value']:raise ValueError('P2 metric replay failed')
    with np.load(p['f3_features.npz']) as source:fr=torch.tensor(source['reactions'][vr],device=device);fe=torch.tensor(source['proteins'][ve],device=device)
    f3=evaluate_scores(canonical_dot(row_unit(fr),row_unit(fe)),truth)['summary']
    atomic_json(OUT/'baseline_replay.json',dict(p2=p2,f3_fp64=f3,raw_semantic_exact=True,parent_positive_ranks_exact=True))
    ir=torch.tensor(np.unique(np.linspace(0,len(r)-1,31,dtype=int)),device=device);ie=torch.tensor(np.unique(np.linspace(0,len(e)-1,193,dtype=int)),device=device)
    rows=[];selected=None
    for config in protocol['grid']:
        if config['radius'] is None:
            rr=r;small=r[ir];single=r[:1];chunk=r[ir];model=None
        else:
            fp=fps[config['radius']];model=MorganReactionAnchor(train_raw,fp[tr],config['eta']).to(device)
            def encode(indices):return torch.cat((r[indices,:512],.5*model.encode_reactions(raw[indices],fp[vr][indices])),1)
            rr=encode(slice(None));small=encode(ir);single=encode(slice(0,1));chunk=torch.cat([encode(part) for part in ir.split(7)],0)
        scores=canonical_dot(rr,e);sparse=GeneralizationDualEncoder.score_index(rr,index)
        checks=dict(reaction_subset=compare(rr[ir],small),reaction_singleton=compare(rr[:1],single),reaction_chunk7=compare(rr[ir],chunk),dense_reaction_branch=compare(rr[:,:512],r[:,:512]),subset_scores=compare(scores[ir][:,ie],canonical_dot(small,e[ie])),singleton_scores=compare(scores[:1,:1],canonical_dot(single,e[:1])),sparse_dense=compare(scores,sparse),subset_sparse=compare(scores[ir][:,ie],GeneralizationDualEncoder.score_index(small,dict(dense=e[ie,:512],anchors=e[ie,512:].to_sparse_csr()))))
        if model is None:
            checks['identity_scores']=compare(scores,parent_scores);result=parent_result
        else:result=evaluate_scores(scores,truth)
        row=dict(**config,summary=result['summary'],balanced_value=robust_value(result['summary']),aggregate_value=selection_value(result['summary']),metric_guards=eligible(result['summary'],p2,.005) and eligible(result['summary'],f3,.005),numerical_checks=checks,numerical_exact=all(x['exact'] for x in checks.values()))
        rows.append(row);ranks(OUT/(config['label']+'_ranks.npz'),result)
        if row['metric_guards'] and (selected is None or row['balanced_value']>selected['balanced_value']):selected=row
        atomic_json(OUT/'progress.json',rows);print(json.dumps({k:row[k] for k in ('label','balanced_value','aggregate_value','metric_guards','numerical_exact')}),flush=True)
        del rr,scores,sparse
    qualifies=bool(selected and selected['balanced_value']>robust_value(p2) and selected['numerical_exact'])
    atomic_json(OUT/'selected_state.json',dict(schema='morgan_reaction_view_selected_v1',selected=selected,protocol=record(OUT/'protocol.json'),prepared=protocol['prepared'],morgan_features=protocol['morgan_features'],parent_bundle=protocol['inputs']['parent_bundle'],alpha=.25))
    final=dict(schema='late_morgan_validation_complete_v1',protocol=record(OUT/'protocol.json'),candidates=rows,selected=selected,p2_balanced=robust_value(p2),qualifies_for_parent_review=qualifies,external_evaluation_performed=False,external_inputs_used=False,exploratory=True,elapsed_seconds=time.monotonic()-start,peak_vram_gib=torch.cuda.max_memory_allocated(device)/2**30)
    atomic_json(OUT/'complete.json',final)
    for item in protocol['implementation_sources']:checked_artifact(item)
    print(json.dumps(dict(selected=selected['label'],balanced=selected['balanced_value'],qualifies=qualifies,seconds=final['elapsed_seconds'])),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run']);p.add_argument('--device',default='cuda:1');args=p.parse_args()
    try:prepare(args) if args.action=='prepare' else run(args)
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/(args.action+'_failure.json'),dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),external_inputs_used=False));raise

if __name__=='__main__':main()
