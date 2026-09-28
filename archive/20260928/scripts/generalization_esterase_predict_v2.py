#!/usr/bin/env python3
"""Five fixed, independent encoders on the sequestered esterase inputs."""
from __future__ import annotations
import argparse,datetime,hashlib,json,sys,time
from pathlib import Path
import h5py,numpy as np,torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
METHODS=['F3_native','F3_fp64','phase2','phase4','morgan']
NUMERICAL_CHECKS={'p2_p4_enzyme_endpoints_exact','morgan_dense_reaction_unchanged',
    *(key+'_'+kind+'_'+what for key in METHODS for kind in ('subset','singleton') for what in ('endpoints','scores')),
    *(key+'_csr_dense' for key in ('phase2','phase4','morgan'))}

def record(path):
    path=Path(path).resolve();digest=hashlib.sha256()
    with path.open('rb') as source:
        for part in iter(lambda:source.read(8<<20),b''):digest.update(part)
    return dict(path=str(path),sha256=digest.hexdigest())

def same(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve() and a['sha256']==b['sha256']

def checked(item):
    actual=record(item['path'])
    if not same(actual,item):raise ValueError('Changed authenticated artifact: '+str(item['path']))
    return Path(actual['path'])

def contains_record(value,wanted):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value and same(value,wanted):return True
        return any(contains_record(item,wanted) for item in value.values())
    if isinstance(value,list):return any(contains_record(item,wanted) for item in value)
    return False

def check_numerical_receipt(checks):
    if set(checks)!=NUMERICAL_CHECKS or any(value is not True for value in checks.values()):
        raise ValueError('Missing, changed or failed fixed numerical acceptance check')

def checked_protocol(path):
    path=Path(path).resolve();plan=json.loads(path.read_text())
    if plan.get('schema')!='esterase_frozen_evaluation_protocol_v1' or plan.get('frozen_before_predictions') is not True:
        raise ValueError('The complete esterase protocol must be frozen before predictions')
    if plan.get('methods')!=METHODS or plan.get('expected_shape')!=[86,145]:raise ValueError('Fixed esterase methods/population changed')
    if (plan.get('primary'),plan.get('baseline'),plan.get('predecessor'))!=('morgan','F3_fp64','phase2'):raise ValueError('Fixed method roles changed')
    sources={str(checked(item)):item for item in plan['implementation_sources']}
    if str(Path(__file__).resolve()) not in sources:raise ValueError('Predictor source is not frozen')
    for key in ('eligibility_plan','feature_receipt','catalog','reaction_strings'):checked(plan[key])
    eligibility=json.loads(checked(plan['eligibility_plan']).read_text())
    if eligibility.get('schema')!='esterase_input_only_eligibility_v1':raise ValueError('Wrong input eligibility plan')
    recipes=plan['recipes']
    if set(recipes)!= {'phase2_bundle','phase4_bundle','morgan_freeze','morgan_state'}:raise ValueError('Unexpected model recipe set')
    for item in recipes.values():checked(item)
    p2=json.loads(checked(recipes['phase2_bundle']).read_text());p4=json.loads(checked(recipes['phase4_bundle']).read_text())
    for spec in (p2,p4):
        if spec.get('split')!='reaction_smi' or spec.get('seed')!=42 or spec.get('variant')!='primary':raise ValueError('Only Reaction-Sim seed42 primary models are fixed')
    if not same(p4['phase2_bundle'],recipes['phase2_bundle']):raise ValueError('P4 uses a different P2 parent')
    for item in (p2['phase2_frozen_recipe'],p4['phase4_frozen_recipe'],recipes['morgan_freeze']):
        frozen=json.loads(checked(item).read_text())
        for source in frozen['implementation_sources']:checked(source)
    # Import model guards only after authenticating their immutable closures.
    from horizyn.generalization_phase2 import validate_phase2_bundle
    from horizyn.generalization_phase4 import validate_phase4_bundle
    from generalization_morgan_transfer import validate_freeze,model_row
    validate_phase2_bundle(p2,Path(recipes['phase2_bundle']['path']).parent)
    validate_phase4_bundle(p4,Path(recipes['phase4_bundle']['path']).parent)
    frozen=validate_freeze(recipes['morgan_freeze']);row=model_row(frozen,'reaction_smi',42)
    if not same(row['parent_bundle'],recipes['phase2_bundle']) or not same(row['morgan_state'],recipes['morgan_state']):raise ValueError('Morgan model tuple differs from selected fixed model')
    native=json.loads(checked(plan['feature_receipt']).read_text())
    if native.get('schema')!='generalization_feature_bundle_receipt_v1':raise ValueError('Unknown feature provenance schema')
    for key in ('catalog','base','protein_means','reaction_features'):checked(native['inputs'][key])
    if not same(native['inputs']['catalog'],plan['catalog']):raise ValueError('Feature receipt and frozen catalog disagree')
    checkpoint=checked(native['checkpoint'])
    if checkpoint.resolve()!=Path(p2['base_checkpoint']['path']).resolve():raise ValueError('Features use a different F3 checkpoint')
    if p2['base_checkpoint'].get('sha256') and p2['base_checkpoint']['sha256']!=native['checkpoint']['sha256']:raise ValueError('Feature/checkpoint hashes differ')
    if native.get('freeze_sha256') and native['freeze_sha256']!=p2['frozen_recipe']['sha256']:raise ValueError('Parent feature freeze mismatch')
    source_records=native.get('source_receipts',[])
    if not source_records:raise ValueError('Native feature source receipts required')
    receipts=[json.loads(checked(item).read_text()) for item in source_records]
    if not contains_record(receipts,plan['reaction_strings']):raise ValueError('Morgan participant strings are not bound by the native feature source receipts')
    catalog=json.loads(checked(plan['catalog']).read_text());ids=catalog.get('reactions',catalog.get('query_ids'))
    if len(ids)!=86 or len(catalog['proteins'])!=145 or len(set(ids))!=86 or len(set(catalog['proteins']))!=145:raise ValueError('Catalog population/order IDs are invalid')
    if plan.get('reaction_key')!='reactions':raise ValueError('Frozen native reaction array key is reactions')
    return plan,native,catalog

def checked_predictions(protocol_path,score_path):
    plan,native,catalog=checked_protocol(protocol_path);score_path=Path(score_path)
    receipt=json.loads((score_path.parent/'complete.json').read_text())
    if receipt.get('schema')!='esterase_frozen_predictions_v1' or receipt.get('labels_used') is not False:raise ValueError('Wrong esterase prediction receipt')
    if not same(receipt['protocol'],record(protocol_path)) or receipt['output_sha256']!=record(score_path)['sha256']:raise ValueError('Protocol/score identity mismatch')
    for key in ('catalog','feature_receipt','reaction_strings'):
        if not same(receipt[key],plan[key]):raise ValueError('Input/order identity mismatch: '+key)
    if receipt['recipes']!=plan['recipes'] or receipt['methods']!=METHODS:raise ValueError('Method recipe binding changed')
    check_numerical_receipt(receipt['numerical_checks'])
    if receipt.get('native_subset_tolerance') != 1e-7 or set(receipt.get('native_subset_deltas',{})) != {'reaction_endpoint_max_abs','enzyme_endpoint_max_abs','score_max_abs'} or any(not np.isfinite(value) or value < 0 or value > 1e-7 for value in receipt['native_subset_deltas'].values()):
        raise ValueError('Native F3 subset precision exception is absent or exceeds its bound')
    if receipt.get('shape')!=[86,145] or receipt.get('model_fit_or_selection_performed') is not False:raise ValueError('Unexpected prediction shape or model adaptation')
    if not same(receipt['source'],record(__file__)):raise ValueError('Prediction producer identity changed')
    if receipt['implementation_sources']!=plan['implementation_sources']:raise ValueError('Prediction source closure differs')
    for key in ('catalog','base','protein_means','reaction_features'):
        if not same(receipt['inputs'][key],native['inputs'][key]):raise ValueError('Feature content disagreement: '+key)
    with np.load(score_path,allow_pickle=False) as source:
        if set(source.files)!=set(METHODS):raise ValueError('Exactly the five fixed score keys are required')
        scores={key:source[key] for key in METHODS}
    if any(value.shape!=(86,145) or value.dtype!=np.float32 or not np.isfinite(value).all() for value in scores.values()):raise ValueError('Nonfinite/malformed fixed score matrix')
    return scores,receipt

@torch.inference_mode()
def run(args):
    started=time.monotonic();plan,native,catalog=checked_protocol(args.protocol)
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a fresh immutable output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    from horizyn.generalization_morgan_transfer import MorganComposedEncoder
    from horizyn.generalization_phase4 import HybridPhase4Encoder
    from horizyn.generalization_morgan import participant_morgan,canonical_components
    from horizyn.generalization_retrieval import canonical_dot
    from horizyn.semantic_anchors import row_unit
    from generalization_phase4_predict import validate_features
    from generalization_morgan import reaction_table
    from generalization_full_graph import atomic_json
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;device=torch.device(args.device)
    model,_=MorganComposedEncoder.from_artifacts(plan['recipes']['phase2_bundle'],plan['recipes']['morgan_state'],device)
    phase4,_=HybridPhase4Encoder.from_bundle(checked(plan['recipes']['phase4_bundle']),device)
    with np.load(checked(native['inputs']['base'])) as source:be=torch.tensor(source['proteins'],device=device);br=torch.tensor(source['reactions'],device=device)
    with h5py.File(checked(native['inputs']['protein_means'])) as source:
        if source['ids'].asstr()[:].tolist()!=catalog['proteins'] or ('complete' in source and not source['complete'][:].all()):raise ValueError('Protein means/catalog mismatch')
        means=torch.tensor(source['vectors'][:],device=device)
    with np.load(checked(native['inputs']['reaction_features'])) as source:
        blocks={k:torch.tensor(source[k],device=device) for k in model.modalities};masks={k:torch.tensor(source[k+'_mask'],device=device) for k in model.modalities}
    ids=catalog.get('reactions',catalog.get('query_ids'));validate_features(catalog,ids,be,br,means,blocks,masks)
    table=reaction_table(checked(plan['reaction_strings']))
    if set(table)!=set(ids):raise ValueError('Feature reaction strings/catalog mismatch')
    for key in ids:
        if len(table[key].split('>'))!=3:raise ValueError('Require the exact matched participant self-reaction export')
        canonical_components(table[key])
    fp=torch.tensor(np.stack([participant_morgan(table[key],3) for key in ids]),device=device)
    e=model.encode_enzymes(be,means,512);e4=phase4.encode_enzymes(be,means,512)
    r2=model.parent.encode_reactions(br,blocks,masks,512);r4=phase4.encode_reactions(br,blocks,masks,512);rm=model.replace_reaction_semantics(r2,blocks,masks,fp)
    fn=torch.nn.functional.normalize
    endpoints={'F3_native':(fn(br,dim=1),fn(be,dim=1)),'F3_fp64':(row_unit(br),row_unit(be)),'phase2':(r2,e),'phase4':(r4,e4),'morgan':(rm,e)}
    scores={key:canonical_dot(*value) for key,value in endpoints.items()}
    ir=torch.arange(0,86,7,device=device);ie=torch.arange(0,145,11,device=device)
    checks={'p2_p4_enzyme_endpoints_exact':torch.equal(e,e4),'morgan_dense_reaction_unchanged':torch.equal(rm[:,:512],r2[:,:512])}
    for label,qr,ep in [('subset',ir,ie),('singleton',slice(0,1),slice(0,1))]:
        b={k:v[qr] for k,v in blocks.items()};m={k:v[qr] for k,v in masks.items()}
        es=model.encode_enzymes(be[ep],means[ep],7);e4s=phase4.encode_enzymes(be[ep],means[ep],7)
        p=model.parent.encode_reactions(br[qr],b,m,7);q4=phase4.encode_reactions(br[qr],b,m,7);qm=model.replace_reaction_semantics(p,b,m,fp[qr])
        sub={'F3_native':(fn(br[qr],dim=1),fn(be[ep],dim=1)),'F3_fp64':(row_unit(br[qr]),row_unit(be[ep])),'phase2':(p,es),'phase4':(q4,e4s),'morgan':(qm,es)}
        for key,(rr,ee) in sub.items():
            checks[key+'_'+label+'_endpoints']=(torch.equal(endpoints[key][0][qr],rr) and torch.equal(endpoints[key][1][ep],ee)) if not (key=='F3_native' and label=='subset') else (torch.max(torch.abs(endpoints[key][0][qr]-rr))<=1e-7 and torch.max(torch.abs(endpoints[key][1][ep]-ee))<=1e-7)
            checks[key+'_'+label+'_scores']=torch.equal(scores[key][qr][:,ep],canonical_dot(rr,ee)) if not (key=='F3_native' and label=='subset') else (torch.max(torch.abs(scores[key][qr][:,ep]-canonical_dot(rr,ee)))<=1e-7)
    index=model.encode_enzyme_index(be,means,512)
    for key,q in [('phase2',r2),('phase4',r4),('morgan',rm)]:checks[key+'_csr_dense']=torch.equal(scores[key],model.score_index(q,index))
    if not all(checks.values()):
        atomic_json(args.output/'numerical_failure.json',dict(protocol=record(args.protocol),checks=checks,labels_used=False));raise ValueError('Fixed cached-endpoint numerical contract failed')
    check_numerical_receipt(checks)
    native_subset_reaction=fn(br[ir],dim=1);native_subset_enzyme=fn(be[ie],dim=1)
    native_subset_deltas=dict(reaction_endpoint_max_abs=float(torch.max(torch.abs(endpoints['F3_native'][0][ir]-native_subset_reaction))),enzyme_endpoint_max_abs=float(torch.max(torch.abs(endpoints['F3_native'][1][ie]-native_subset_enzyme))),score_max_abs=float(torch.max(torch.abs(scores['F3_native'][ir][:,ie]-canonical_dot(native_subset_reaction,native_subset_enzyme)))))
    if any(v>1e-7 for v in native_subset_deltas.values()):raise ValueError('Native F3 batch-shape drift exceeds prespecified bound')
    if any(not torch.isfinite(value).all() for value in scores.values()):raise ValueError('Nonfinite esterase score')
    np.savez(args.output/'scores.npz',**{key:value.cpu().numpy() for key,value in scores.items()})
    receipt=dict(schema='esterase_frozen_predictions_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=record(args.protocol),catalog=plan['catalog'],feature_receipt=plan['feature_receipt'],reaction_strings=plan['reaction_strings'],recipes=plan['recipes'],methods=METHODS,
        inputs=native['inputs'],implementation_sources=plan['implementation_sources'],numerical_checks=checks,labels_used=False,model_fit_or_selection_performed=False,output_sha256=record(args.output/'scores.npz')['sha256'],source=record(__file__),native_subset_deltas=native_subset_deltas,native_subset_tolerance=1e-7,shape=[86,145],elapsed_seconds=time.monotonic()-started,
        baseline_contracts={'F3_native':'FP32 F.normalize of unchanged native F3 endpoints, canonical FP64 dot→FP32','F3_fp64':'FP64 norm→FP32 of unchanged native endpoints, canonical FP64 dot→FP32'},parent_replay='Same new input features, no prior esterase scores exist. Original fixed constructors used; P2/P4 enzyme equality, unchanged Morgan neural branch and independent subset/singleton/CSR replay checked.')
    atomic_json(args.output/'complete.json',receipt);checked_predictions(args.protocol,args.output/'scores.npz')
    print(json.dumps(dict(output=str(args.output),shape=[86,145],all_numerical_checks_pass=True,native_subset_deltas=native_subset_deltas,seconds=receipt['elapsed_seconds'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:2');run(p.parse_args())
if __name__=='__main__':main()
