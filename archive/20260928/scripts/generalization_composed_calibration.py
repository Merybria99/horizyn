#!/usr/bin/env python3
"""Prespecified training-mean calibration of frozen P2, validation only."""
from __future__ import annotations
import argparse, datetime, itertools, json, shutil, sys, time
from pathlib import Path
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from horizyn.generalization_calibration import MeanAffinityCalibration
from horizyn.generalization_retrieval import canonical_dot,checked_artifact,sha256
from horizyn.semantic_anchors import row_unit
from generalization_full_graph import validation_data,eligible,selection_value,atomic_json
from generalization_smooth_anchors import robust_value
from generalization_metrics import evaluate_scores

RUN=ROOT/'runs/generalization_20260919_2251'
GRID=[0.,.25,.5,1.,2.]
SCHEMES=['uniform_proteins','reaction_balanced_edges']
DIRECTIONS=['reaction_to_enzyme','enzyme_to_reaction']


def record(p):return dict(path=str(Path(p).resolve()),sha256=sha256(p))


def sources():
    feature=RUN/'features';previous=RUN/'phase2/composition_fp64'
    return {**{name:feature/name for name in ['catalog.json','pairs.npz','manifest.json','f3_features.npz','protein_mean.h5','reaction_features.npz']},
        'p2_bundle':RUN/'phase2/models/reaction_smi/seed42/bundle.json','p2_freeze':RUN/'phase2/frozen_recipe.json',
        'p2_validation':previous/'selected_validation_features.pt','p2_selection':previous/'selection.json',
        'p2_ranks':previous/'density_smooth_alpha0.25_ranks.npz',
        'p4_validation':RUN/'phase4/hybrid_anchors/validation.json',
        'prior_native_hubness':RUN/'hubness/config.json',
        'script':Path(__file__),'module':ROOT/'horizyn/generalization_calibration.py',
        'metrics':ROOT/'scripts/generalization_metrics.py','validation_helpers':ROOT/'scripts/generalization_full_graph.py',
        'selection_helpers':ROOT/'scripts/generalization_smooth_anchors.py'}


def authenticate_sources():
    freeze=json.loads((RUN/'phase2/frozen_recipe.json').read_text())
    for item in freeze['implementation_sources']:checked_artifact(item)
    return freeze


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'protocol.json').exists():raise ValueError('Existing immutable protocol')
    authenticate_sources()
    protocol=dict(schema='post_evaluation_composed_calibration_protocol_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        exploratory_after_opened_p1_p2_p4_external_results=True,independent_confirmation=False,
        external_inputs_or_scores_used=False,official_test_used=False,large_pool_scores_used=False,
        parent='Frozen Reaction-Sim P2 seed42, alpha0.25; no encoder updates',
        gamma_enzyme=GRID,gamma_reaction=GRID,enzyme_mean_schemes=SCHEMES,reaction_mean='Uniform unique reactions occurring in training edges',
        enzyme_mean='Uniform unique training proteins OR sum over unique training edges weighted1/(R*reaction_degree)',
        fit_scope='Actual pairs.npz train-edge selectors; no validation-only rows, candidate population or labels used in means. Compact F3 training reaction cache in catalog.train_reactions order.',
        train_precision='Existing train native cache BF16-inferred and storedFP16, with existing fit-free FP32 validation protein overrides; no fresh native extraction. Parent postprocessor unchanged.',
        accumulation='FP64 weighted sums of complete P2 endpoint vectors; ascending global catalog trainID order; fixed512 protein chunks and256 reaction chunks; FP64 means retained without renormalization.',
        endpoint='Raug=[R,FP32(-gammaR*sum_FP64(R*meanE)),1]; Eaug=[E,1,FP32(-gammaE*sum_FP64(E*meanR))]. No normalization after augmentation. Bothgamma0 delegates unchanged P2 vectors/dimension.',
        score='Actual FP32 augmented endpoints, canonical FP64 dot then one FP32 score rounding; never subtract from already-rounded P2 score.',
        selection='Rank directional candidate-bias grids separately; compose50 prespecified mean/gamma tuples. Max equal-weight seen/unseen-reaction all-positiveMRR across both directions; all/unseen in bothdirections within0.005 of BOTH F3_fp64 and P2. Strict gain over P2 required; ties retain first enumerated (zero and uniform first).',
        numerical_acceptance='For selected tuple only, compute actual combined augmented score and bothdirection metrics. Require allguards, strictbalancedgain, exactfull/subset/singleton endpoint and score replay. If actualcombined differs from separable ranks, report allchanges as query-constant final-rounding sensitivity; no alternative gamma is selected after this check.',
        noop_diagnostic='Adding gammaR changes only a rowconstant forR2E; gammaE only acolumnconstant forE2R. Compare selected actualfull ranks against corresponding candidate-only directional ranks, including eachstratum metric delta; distinguish rounding changes from candidate-bias gain.',
        budget_minutes=30,device='GPU1 only',inputs={k:record(v) for k,v in sources().items()},
        grid_evaluations='Four nonzero R2E candidatebias evaluations; eight nonzero E2R (two meanE schemes); one actualcombined selected audit. No retraining or default behavior changes.')
    atomic_json(args.output/'protocol.json',protocol);shutil.copyfile(__file__,args.output/'source.py')
    print(json.dumps(record(args.output/'protocol.json')),flush=True)


def save_result(path,result):
    arrays={f'{d}__{s}__{k}':v for d,block in result['per_query'].items() for s,values in block.items() for k,v in values.items()}
    arrays.update({f'positive__{d}__{k}':v for d,block in result['per_positive'].items() for k,v in block.items()})
    np.savez_compressed(path,**arrays)


def fit_training_means(model,base_e,train_r,blocks,masks,raw_e,train,tr,te,out):
    lookup=np.full(len(base_e),-1,np.int64);lookup[te]=np.arange(len(te))
    degree=np.bincount(train[:,0],minlength=max(int(tr.max())+1,len(blocks['t5v2'])))
    weights=np.bincount(lookup[train[:,1]],weights=1./(len(tr)*degree[train[:,0]]),minlength=len(te))
    uniform=np.full(len(te),1./len(te),np.float64)
    if not np.isclose(weights.sum(),1.,rtol=0,atol=1e-12) or np.any(weights<=0):raise ValueError('Invalid reaction-balanced training marginal')
    dimension=512+len(model.smooth.train_reactions);device=base_e.device
    sums={key:torch.zeros(dimension,dtype=torch.float64,device=device) for key in SCHEMES}
    started=time.monotonic()
    for start in range(0,len(te),512):
        stop=min(start+512,len(te));rows=te[start:stop]
        z=model.encode_enzymes(base_e[rows],raw_e[rows],batch_size=512).double()
        for name,w in [('uniform_proteins',uniform),('reaction_balanced_edges',weights)]:
            sums[name]+=(z*torch.tensor(w[start:stop],device=device)[:,None]).sum(0)
        if start%16384==0:print(json.dumps(dict(fit_proteins=stop,total=len(te),elapsed_seconds=time.monotonic()-started)),flush=True)
    reaction_sum=torch.zeros(dimension,dtype=torch.float64,device=device)
    for start in range(0,len(tr),256):
        stop=min(start+256,len(tr));rows=tr[start:stop]
        z=model.encode_reactions(train_r[start:stop],{k:v[rows] for k,v in blocks.items()},{k:v[rows] for k,v in masks.items()},batch_size=512)
        reaction_sum+=z.double().sum(0)
    mean_r=reaction_sum/len(tr)
    torch.save(dict(reaction_mean=mean_r.cpu(),enzyme_means={k:v.cpu() for k,v in sums.items()},train_reaction_indices=tr,train_enzyme_indices=te,
        uniform_weights=uniform,reaction_balanced_weights=weights,protocol=record(out/'protocol.json'),training_edges=len(train)),out/'train_means.pt')
    return mean_r,sums


def guard(summary,f3,p2):
    return eligible(summary,f3,.005) and eligible(summary,p2,.005)


@torch.inference_mode()
def run(args):
    started=time.monotonic();protocol=json.loads((args.output/'protocol.json').read_text())
    if (args.output/'registry.json').exists():raise ValueError('Existing screen output; use a fresh protocol directory')
    for name,path in sources().items():
        if record(path)!=protocol['inputs'][name]:raise ValueError('Changed protocol input: '+name)
    authenticate_sources();atomic_json(args.output/'registry.json',dict(protocol=record(args.output/'protocol.json'),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device);torch.cuda.set_device(device)
    from horizyn.generalization_phase2 import ComposedPhase2Encoder
    model,_=ComposedPhase2Encoder.from_bundle(sources()['p2_bundle'],device)
    catalog=json.loads(sources()['catalog.json'].read_text())
    with np.load(sources()['pairs.npz']) as f:pairs={k:f[k] for k in f.files}
    train=pairs['train'];tr,te=np.unique(train[:,0]),np.unique(train[:,1]);vr,ve,truth=validation_data(catalog,pairs)
    if len(np.unique(train,axis=0))!=len(train):raise ValueError('Training edge array must already be unique')
    if [catalog['reactions'][i] for i in tr]!=catalog['train_reactions'] or [catalog['proteins'][i] for i in te]!=catalog['train_proteins']:raise ValueError('Actual training selectors/catalog mismatch')
    with np.load(sources()['f3_features.npz']) as f:
        base_e=torch.tensor(f['proteins'],device=device);base_r=torch.tensor(f['reactions'],device=device);train_r=torch.tensor(f['train_reactions'],device=device)
    with np.load(sources()['reaction_features.npz']) as f:
        blocks={k:torch.tensor(f[k],device=device) for k in model.modalities};masks={k:torch.tensor(f[k+'_mask'],device=device) for k in model.modalities}
    with h5py.File(sources()['protein_mean.h5']) as f:
        if f['ids'].asstr()[:].tolist()!=catalog['proteins'] or not f['complete'][:].all():raise ValueError('Rawmean catalog/coverage mismatch')
        raw_e=torch.tensor(f['vectors'][:],device=device)
    for x in [base_e,base_r,train_r,raw_e,*blocks.values()]:
        if x.dtype!=torch.float32 or not torch.isfinite(x).all():raise ValueError('Finite FP32 source vectors required')
    mean_r,means_e=fit_training_means(model,base_e,train_r,blocks,masks,raw_e,train,tr,te,args.output)
    old=torch.load(sources()['p2_validation'],map_location=device,weights_only=False)
    if old['reaction_ids']!=catalog['validation_reactions'] or old['enzyme_ids']!=catalog['validation_candidates']:raise ValueError('Frozen P2 validation order mismatch')
    r,e=old['reactions'],old['enzymes']
    ri=np.unique(np.linspace(0,len(vr)-1,17,dtype=int));ei=np.unique(np.linspace(0,len(ve)-1,17,dtype=int))
    if not torch.equal(model.encode_reactions(base_r[vr[ri]],{k:v[vr[ri]] for k,v in blocks.items()},{k:v[vr[ri]] for k,v in masks.items()}),r[ri]):raise ValueError('Fresh reaction endpoint does not replay P2 cache')
    if not torch.equal(model.encode_enzymes(base_e[ve[ei]],raw_e[ve[ei]],batch_size=512),e[ei]):raise ValueError('Fresh enzyme endpoint does not replay P2 cache')
    baseline=evaluate_scores(canonical_dot(r,e),truth);p2=baseline['summary'];save_result(args.output/'p2_baseline_ranks.npz',baseline)
    with np.load(sources()['p2_ranks']) as f:
        if not all(np.array_equal(f[d+'_'+k],v) for d,b in baseline['per_positive'].items() for k,v in b.items()):raise ValueError('Gamma0 does not exactly replay frozen P2 positive ranks')
    prior=json.loads(sources()['p2_selection'].read_text())['selected']
    if robust_value(p2)!=prior['robust_value']:raise ValueError('P2 balanced selector mismatch')
    f3=evaluate_scores(canonical_dot(row_unit(base_r[vr]),row_unit(base_e[ve])),truth)['summary']
    atomic_json(args.output/'baseline_replay.json',dict(gamma0_exact=True,positive_ranks_exact=True,subset_encoder_exact=True,p2=p2,f3_fp64=f3,balanced_p2=robust_value(p2),balanced_p4_reference=json.loads(sources()['p4_validation'].read_text())['selected']['robust_value']))
    directions={};direction_ranks={};rows=[]
    for direction in DIRECTIONS:
        for scheme in (['uniform_proteins'] if direction==DIRECTIONS[0] else SCHEMES):
            for gamma in GRID:
                if gamma==0:result=baseline
                else:
                    calibrator=MeanAffinityCalibration(mean_r,means_e[scheme],gamma if direction==DIRECTIONS[0] else 0.,gamma if direction==DIRECTIONS[1] else 0.)
                    scores=canonical_dot(calibrator.encode_reactions(r),calibrator.encode_enzymes(e))
                    result=evaluate_scores(scores,truth,directions=[direction])
                key=(direction,scheme,gamma);directions[key]=result['summary'][direction];direction_ranks[key]=result['per_positive'][direction]['rank']
                name=f'{direction}_{scheme}_gamma{gamma:g}'
                save_result(args.output/(name+'.npz'),result)
                print(json.dumps(dict(direction=direction,scheme=scheme,gamma=gamma,all_mrr=directions[key]['all']['reactzyme_mrr'],elapsed_seconds=time.monotonic()-started)),flush=True)
    for scheme,ge,gr in itertools.product(SCHEMES,GRID,GRID):
        summary={DIRECTIONS[0]:directions[(DIRECTIONS[0],'uniform_proteins',ge)],DIRECTIONS[1]:directions[(DIRECTIONS[1],scheme,gr)]}
        rows.append(dict(scheme=scheme,gamma_enzyme=ge,gamma_reaction=gr,summary=summary,balanced_value=robust_value(summary),aggregate_value=selection_value(summary),eligible=guard(summary,f3,p2)))
    valid=[x for x in rows if x['eligible']]
    selected=max(valid,key=lambda x:x['balanced_value']) if valid else None
    atomic_json(args.output/'directional_selection.json',dict(candidates=rows,selected=selected,baseline_balanced=robust_value(p2),test_used=False,external_used=False))
    if selected is None:raise ValueError('No eligible grid candidate, including P2, under both reference guard sets')
    calibrator=MeanAffinityCalibration(mean_r,means_e[selected['scheme']],selected['gamma_enzyme'],selected['gamma_reaction'])
    ra,ea=calibrator.encode_reactions(r),calibrator.encode_enzymes(e);scores=canonical_dot(ra,ea)
    combined=evaluate_scores(scores,truth);save_result(args.output/'selected_combined_ranks.npz',combined)
    ssr=torch.tensor(np.unique(np.linspace(0,len(r)-1,31,dtype=int)),device=device);sse=torch.tensor(np.unique(np.linspace(0,len(e)-1,193,dtype=int)),device=device)
    parity=dict(reaction_subset=torch.equal(ra[ssr],calibrator.encode_reactions(r[ssr])),enzyme_subset=torch.equal(ea[sse],calibrator.encode_enzymes(e[sse])),
        subset_scores=torch.equal(scores[ssr][:,sse],canonical_dot(calibrator.encode_reactions(r[ssr]),calibrator.encode_enzymes(e[sse]))),
        singleton_scores=torch.equal(scores[ssr[:1]][:,sse[:1]],canonical_dot(calibrator.encode_reactions(r[ssr[:1]]),calibrator.encode_enzymes(e[sse[:1]]))))
    rounding={}
    for direction in DIRECTIONS:
        key=(direction,'uniform_proteins',selected['gamma_enzyme']) if direction==DIRECTIONS[0] else (direction,selected['scheme'],selected['gamma_reaction'])
        before=direction_ranks[key];after=combined['per_positive'][direction]['rank'];delta={s:{k:combined['summary'][direction][s][k]-directions[key][s][k] for k in ['reactzyme_mrr','first_positive_mrr','top_1','top_5','top_10'] if combined['summary'][direction][s][k] is not None} for s in combined['summary'][direction]}
        rounding[direction]=dict(positive_ranks_changed=int(np.count_nonzero(before!=after)),total_positive_edges=len(after),maximum_rank_change=int(np.max(np.abs(before-after))),summary_delta_from_mathematically_query_constant_shift=delta)
    actual_balanced=robust_value(combined['summary']);strict_both=selected['balanced_value']>robust_value(p2) and actual_balanced>robust_value(p2)
    accepted=bool(strict_both and guard(combined['summary'],f3,p2) and all(parity.values()))
    torch.save(dict(schema='post_evaluation_p2_mean_calibration_v1',state_dict=calibrator.state_dict(),gamma_enzyme=selected['gamma_enzyme'],gamma_reaction=selected['gamma_reaction'],enzyme_mean_scheme=selected['scheme'],protocol=record(args.output/'protocol.json'),parent_bundle=record(sources()['p2_bundle'])),args.output/'selected_calibration.pt')
    final=dict(schema='post_evaluation_calibration_result_v1',selected_directional=selected,actual_combined_summary=combined['summary'],actual_balanced_value=actual_balanced,
        p2_balanced_reference=robust_value(p2),p4_balanced_reference=json.loads(sources()['p4_validation'].read_text())['selected']['robust_value'],
        qualifies_for_parent_review=accepted,selection_uses_no_external_data=True,exploratory_after_previous_external_evaluations=True,
        gamma0_replay=record(args.output/'baseline_replay.json'),noop_constant_shift_diagnostics=rounding,endpoint_and_score_parity=parity,
        actual_combined_guards=guard(combined['summary'],f3,p2),strict_directional_and_actual_gain=strict_both,
        elapsed_seconds=time.monotonic()-started,protocol=record(args.output/'protocol.json'),means=record(args.output/'train_means.pt'),selected_state=record(args.output/'selected_calibration.pt'),
        next_step='Parent reviews validation-only result before any new external evaluation; this screen changes no defaults or previous freezes.')
    atomic_json(args.output/'complete.json',final);print(json.dumps({k:v for k,v in final.items() if k not in ['selected_directional','actual_combined_summary','noop_constant_shift_diagnostics']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run']);p.add_argument('--output',type=Path,default=RUN/'post_evaluation_calibration');p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.action=='prepare':prepare(args)
    else:run(args)


if __name__=='__main__':main()
