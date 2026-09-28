#!/usr/bin/env python3
"""Apply the frozen training-only P2 mean calibration without activity labels."""
from __future__ import annotations
import argparse,json,shutil,sys,time
from pathlib import Path
import h5py,numpy as np,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from generalization_calibration_transfer import validate_freeze,record,same,RULE
from horizyn.generalization_retrieval import checked_artifact,canonical_dot
from horizyn.generalization_calibration import MeanAffinityCalibration
from horizyn.semantic_anchors import row_unit
from generalization_predict import validate_input_receipt
from generalization_phase4_predict import validate_features
from generalization_full_graph import atomic_json

@torch.inference_mode()
def run(args):
    started=time.monotonic();frozen_record=record(args.freeze);freeze=validate_freeze(frozen_record)
    if (args.output/'complete.json').exists():raise ValueError('Existing immutable prediction')
    approved=[r for r in freeze['approved_predictions'] if r['panel']==args.panel and r['seed']==args.seed]
    if len(approved)!=1:raise ValueError('Unapproved panel/seed')
    job=approved[0];models=[r for r in freeze['approved_models'] if r['split']==job['split'] and r['seed']==args.seed]
    if len(models)!=1:raise ValueError('Unapproved fitted model')
    model_record=models[0];state=torch.load(checked_artifact(model_record['calibration_state']),map_location='cpu',weights_only=False)
    if state.get('schema')!='p2_calibration_state_v1' or state['rule']!=RULE or state['split']!=job['split'] or state['seed']!=args.seed or not same(state['parent_bundle'],model_record['parent_bundle']):raise ValueError('Wrong calibration state lineage')
    for item in state['training_sources'].values():checked_artifact(item)
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;device=torch.device(args.device)
    from horizyn.generalization_phase2 import ComposedPhase2Encoder
    model,spec=ComposedPhase2Encoder.from_bundle(checked_artifact(model_record['parent_bundle']),device)
    input_path=checked_artifact(job['input_receipt']);input_=json.loads(input_path.read_text())
    for key in ['base','catalog','protein_means','reaction_features']:setattr(args,key,checked_artifact(input_['inputs'][key]))
    args.input_receipt=input_path;receipt=validate_input_receipt(args,spec)
    catalog=json.loads(args.catalog.read_text());reactions=catalog.get('reactions',catalog.get('query_ids'))
    with np.load(args.base) as source:be=torch.tensor(source['proteins'],device=device);br=torch.tensor(source[job['reaction_key']],device=device)
    with h5py.File(args.protein_means) as source:
        if source['ids'].asstr()[:].tolist()!=catalog['proteins'] or ('complete' in source and not source['complete'][:].all()):raise ValueError('Incomplete/reordered protein means')
        means=torch.tensor(source['vectors'][:],device=device)
    with np.load(args.reaction_features) as source:
        blocks={k:torch.tensor(source[k],device=device) for k in model.modalities};masks={k:torch.tensor(source[k+'_mask'],device=device) for k in model.modalities}
    validate_features(catalog,reactions,be,br,means,blocks,masks)
    e=model.encode_enzymes(be,means,args.batch_size);r=model.encode_reactions(br,blocks,masks,args.batch_size)
    parent=canonical_dot(r,e);baseline=canonical_dot(F.normalize(br,dim=1),F.normalize(be,dim=1));baseline64=canonical_dot(row_unit(br),row_unit(be))
    historical_path=checked_artifact(job['parent_scores']);historical_receipt=json.loads(checked_artifact(job['parent_receipt']).read_text())
    if historical_receipt.get('labels_used') is not False or historical_receipt['output_sha256']!=job['parent_scores']['sha256'] or not same(historical_receipt['bundle'],model_record['parent_bundle']) or not same(historical_receipt['input_receipt'],job['input_receipt']):raise ValueError('Parent control lineage mismatch')
    with np.load(historical_path) as source:old={k:source[k].copy() for k in ['selected','baseline','baseline_fp64']}
    checks={name:np.array_equal(value.cpu().numpy(),old[key]) for name,value,key in [('parent',parent,'selected'),('native',baseline,'baseline'),('fp64',baseline64,'baseline_fp64')]}
    if not all(checks.values()):raise ValueError('Frozen parent control scores failed exact replay')
    calibration=MeanAffinityCalibration(state['reaction_mean'].to(device),state['enzyme_mean'].to(device),0.,1.)
    ra,ea=calibration.encode_reactions(r),calibration.encode_enzymes(e);scores=canonical_dot(ra,ea)
    zero=MeanAffinityCalibration(state['reaction_mean'].to(device),state['enzyme_mean'].to(device),0.,0.)
    zero_exact=torch.equal(zero.encode_reactions(r),r) and torch.equal(zero.encode_enzymes(e),e)
    ir=torch.tensor(np.unique(np.linspace(0,len(r)-1,min(31,len(r)),dtype=int)),device=device);ie=torch.tensor(np.unique(np.linspace(0,len(e)-1,min(193,len(e)),dtype=int)),device=device)
    parity=dict(reaction_subset=torch.equal(ra[ir],calibration.encode_reactions(r[ir])),enzyme_subset=torch.equal(ea[ie],calibration.encode_enzymes(e[ie])),
        subset_scores=torch.equal(scores[ir][:,ie],canonical_dot(calibration.encode_reactions(r[ir]),calibration.encode_enzymes(e[ie]))),
        singleton_scores=torch.equal(scores[:1,:1],canonical_dot(calibration.encode_reactions(r[:1]),calibration.encode_enzymes(e[:1]))))
    if not zero_exact or not all(parity.values()) or not torch.isfinite(scores).all():raise ValueError('Calibrated numerical contract failed')
    # Parent controls are the original authenticated arrays, after exact replay.
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/'scores.npz',selected=scores.cpu().numpy(),parent=old['selected'],baseline=old['baseline'],baseline_fp64=old['baseline_fp64'])
    np.savez(args.output/'reaction_bias.npz',reaction_bias=ra[:,-2].cpu().numpy())
    shutil.copyfile(args.catalog,args.output/'catalog.json')
    manifest=dict(schema='generalization_predictions_v1',phase='post_evaluation_calibration',labels_used=False,development_exposed=True,
        calibration_freeze=frozen_record,phase2_frozen_recipe=spec['phase2_frozen_recipe'],frozen_recipe=spec['frozen_recipe'],bundle=model_record['parent_bundle'],
        calibration_state=model_record['calibration_state'],input_receipt=receipt,inputs={k:record(getattr(args,k)) for k in ['base','catalog','protein_means','reaction_features']},
        panel=args.panel,split=job['split'],seed=args.seed,rule=RULE,reaction_key=job['reaction_key'],shape=list(scores.shape),embedding_dimension=ea.shape[1],
        parent_reference=dict(scores=job['parent_scores'],receipt=job['parent_receipt']),parent_scores_exact=True,parent_control_checks=checks,zero_scores_exact=zero_exact,subset_parity=parity,
        score_contract=freeze['numeric_contract'],source=record(Path(__file__)),output_sha256=record(args.output/'scores.npz')['sha256'],reaction_bias=record(args.output/'reaction_bias.npz'),elapsed_seconds=time.monotonic()-started)
    atomic_json(args.output/'complete.json',manifest);print(json.dumps(dict(panel=args.panel,seed=args.seed,shape=list(scores.shape),seconds=manifest['elapsed_seconds'])),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--panel',required=True);p.add_argument('--seed',type=int,choices=[42,17,73],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=512);run(p.parse_args())

if __name__=='__main__':main()
