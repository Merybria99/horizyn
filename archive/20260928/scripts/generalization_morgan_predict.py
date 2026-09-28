#!/usr/bin/env python3
"""Predict the frozen Morgan recipe on one approved fixed candidate pool."""
import argparse,json,sys,time
from pathlib import Path
import h5py,numpy as np,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from generalization_morgan_transfer import validate_freeze,model_row,prediction_row,record,OUT,authenticate_input_receipt
from horizyn.generalization_retrieval import checked_artifact,canonical_dot,sha256
from horizyn.generalization_morgan_transfer import MorganComposedEncoder
from horizyn.semantic_anchors import row_unit
from generalization_predict import validate_input_receipt
from generalization_phase4_predict import validate_features
from generalization_full_graph import atomic_json

@torch.inference_mode()
def run(args):
    start=time.monotonic();freeze_record=record(args.freeze);freeze=validate_freeze(freeze_record)
    pred=prediction_row(freeze,args.panel,args.seed);row=model_row(freeze,pred['split'],args.seed)
    out=OUT/'predictions'/args.panel/f'seed{args.seed}'
    if (out/'complete.json').exists():raise ValueError('Existing immutable prediction')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;device=torch.device(args.device)
    model,spec=MorganComposedEncoder.from_artifacts(row['parent_bundle'],row['morgan_state'],device)
    native=authenticate_input_receipt(freeze,pred['input_receipt'])
    fields={k:checked_artifact(native['inputs'][k]) for k in ('catalog','base','protein_means','reaction_features')}
    validated=validate_input_receipt(argparse.Namespace(**fields,input_receipt=Path(pred['input_receipt']['path'])),spec)
    cat=json.loads(fields['catalog'].read_text());ids=cat.get('reactions',cat.get('query_ids'))
    query=json.loads(checked_artifact(pred['morgan_inputs']).read_text())
    if query['reaction_ids']!=ids or query['input_receipt']!=pred['input_receipt']:raise ValueError('Morgan query/catalog/source mismatch')
    for source in query['sources']:checked_artifact(source)
    with np.load(checked_artifact(query['vectors'])) as source:fp=torch.tensor(source['vectors'],device=device)
    with np.load(fields['base']) as source:be=torch.tensor(source['proteins'],device=device);br=torch.tensor(source[pred['reaction_key']],device=device)
    with h5py.File(fields['protein_means']) as source:
        if source['ids'].asstr()[:].tolist()!=cat['proteins'] or ('complete' in source and not source['complete'][:].all()):raise ValueError('Protein means order/completeness mismatch')
        means=torch.tensor(source['vectors'][:],device=device)
    with np.load(fields['reaction_features']) as source:
        blocks={k:torch.tensor(source[k],device=device) for k in model.modalities};masks={k:torch.tensor(source[k+'_mask'],device=device) for k in model.modalities}
    validate_features(cat,ids,be,br,means,blocks,masks)
    e=model.encode_enzymes(be,means,512);parent_r=model.parent.encode_reactions(br,blocks,masks,512)
    r=model.replace_reaction_semantics(parent_r,blocks,masks,fp)
    outputs=dict(selected=canonical_dot(r,e),parent=canonical_dot(parent_r,e),baseline=canonical_dot(F.normalize(br,dim=1),F.normalize(be,dim=1)),baseline_fp64=canonical_dot(row_unit(br),row_unit(be)))
    checked_artifact(pred['parent_reference']['receipt']);old_path=checked_artifact(pred['parent_reference']['scores'])
    with np.load(old_path) as source:
        exact=all(np.array_equal(outputs[k].cpu().numpy(),source[oldkey]) for k,oldkey in [('parent','selected'),('baseline','baseline'),('baseline_fp64','baseline_fp64')])
    if not exact:raise ValueError('Frozen P2/native control replay differs')
    ir=torch.tensor(np.unique(np.linspace(0,len(r)-1,min(31,len(r)),dtype=int)),device=device);ie=torch.tensor(np.unique(np.linspace(0,len(e)-1,min(193,len(e)),dtype=int)),device=device)
    small=model.replace_reaction_semantics(parent_r[ir],{k:v[ir] for k,v in blocks.items()},{k:v[ir] for k,v in masks.items()},fp[ir])
    single=model.replace_reaction_semantics(parent_r[:1],{k:v[:1] for k,v in blocks.items()},{k:v[:1] for k,v in masks.items()},fp[:1])
    checks=dict(reaction_subset=torch.equal(r[ir],small),reaction_singleton=torch.equal(r[:1],single),subset_scores=torch.equal(outputs['selected'][ir][:,ie],canonical_dot(small,e[ie])),sparse_dense=torch.equal(outputs['selected'],model.score_index(r,dict(dense=e[:,:512],anchors=e[:,512:].to_sparse_csr()))))
    if not all(checks.values()) or not all(torch.isfinite(v).all() for v in outputs.values()):raise ValueError('Morgan inference numerical contract failed')
    np.savez(out/'scores.npz',**{k:v.cpu().numpy() for k,v in outputs.items()})
    receipt=dict(schema='generalization_predictions_v1',phase='post_evaluation_morgan',labels_used=False,panel=args.panel,split=pred['split'],seed=args.seed,
        bundle=row['parent_bundle'],morgan_state=row['morgan_state'],morgan_inputs=pred['morgan_inputs'],morgan_freeze=freeze_record,
        frozen_recipe=spec['frozen_recipe'],phase2_frozen_recipe=spec['phase2_frozen_recipe'],input_receipt=validated,inputs={k:record(v) for k,v in fields.items()},
        parent_reference=pred['parent_reference'],parent_scores_exact=exact,subset_parity=checks,reaction_key=pred['reaction_key'],shape=list(outputs['selected'].shape),score_keys=list(outputs),
        output_sha256=sha256(out/'scores.npz'),elapsed_seconds=time.monotonic()-start,source=record(__file__),implementation_sources=freeze['implementation_sources'])
    atomic_json(out/'complete.json',receipt);print(json.dumps(dict(panel=args.panel,seed=args.seed,shape=receipt['shape'],seconds=receipt['elapsed_seconds'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--panel',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:2');run(p.parse_args())
if __name__=='__main__':main()
