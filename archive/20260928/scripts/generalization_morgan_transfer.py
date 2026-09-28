#!/usr/bin/env python3
"""Authenticated fixed Morgan transfer; input-only fitting and score guards."""
from __future__ import annotations
import argparse,datetime,json,sys,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from horizyn.generalization_retrieval import checked_artifact,sha256,canonical_dot
from horizyn.generalization_morgan import participant_morgan,canonical_components
from horizyn.generalization_morgan_transfer import RULE,MorganComposedEncoder
from generalization_full_graph import atomic_json,validation_data
from generalization_morgan import reaction_table

RUN=ROOT/'runs/generalization_20260919_2251';STUDY=RUN/'post_evaluation_morgan';OUT=STUDY/'transfer'
SPLITS=['reaction_smi','enzyme_smi','time'];SEEDS=[42,17,73]
PANELS=SPLITS+['case1','p450','nitrilase','aminotransferase']
def record(path):return dict(path=str(Path(path).resolve()),sha256=sha256(path))
def same(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve() and a['sha256']==b['sha256']
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def validate_freeze(item):
    frozen=json.loads(checked_artifact(item).read_text())
    if frozen.get('schema')!='post_evaluation_morgan_transfer_freeze_v1' or not frozen.get('frozen_before_new_predictions') or frozen.get('rule')!=RULE or frozen.get('development_exposed') is not True:
        raise ValueError('Require frozen, explicitly exploratory selected Morgan recipe')
    for source in frozen['implementation_sources']:checked_artifact(source)
    for key in ('phase2_frozen_recipe','phase4_frozen_recipe','original_frozen_recipe','validation_selection','transfer_authorization','external_evaluation_plan','large_case1_evaluation_plan'):checked_artifact(frozen[key])
    old=json.loads(checked_artifact(frozen['phase4_frozen_recipe']).read_text())
    expected=[old['feature_export_receipts'][panel] for panel in ('nitrilase','aminotransferase')]
    actual=frozen['optional_input_freeze_receipts']
    if len(actual)!=2 or not all(any(same(a,b) for b in expected) for a in actual):raise ValueError('Optional-field exception list differs from the two pinned exports')
    return frozen

def model_row(frozen,split,seed):
    rows=[x for x in frozen['approved_models'] if x['split']==split and x['seed']==seed]
    if len(rows)!=1:raise ValueError('Exactly one frozen split/seed model is required')
    for key in ('parent_bundle','morgan_state'):checked_artifact(rows[0][key])
    return rows[0]

def prediction_row(frozen,panel,seed):
    rows=[x for x in frozen['approved_predictions'] if x['panel']==panel and x['seed']==seed]
    if len(rows)!=1:raise ValueError('Exactly one frozen panel/seed prediction is required')
    return rows[0]

def authenticate_input_receipt(frozen,item):
    path=checked_artifact(item);inp=json.loads(path.read_text())
    if inp.get('schema')!='generalization_feature_bundle_receipt_v1':raise ValueError('Unknown input receipt')
    declared=inp.get('freeze_sha256')
    if declared:
        if declared!=frozen['original_frozen_recipe']['sha256']:raise ValueError('Input freeze mismatch')
    elif not any(same(item,allowed) for allowed in frozen['optional_input_freeze_receipts']):
        raise ValueError('Missing optional freeze field is permitted only for the two exact approved exports')
    for key in ('catalog','base','protein_means','reaction_features'):checked_artifact(inp['inputs'][key])
    checked_artifact(inp['checkpoint'])
    for source in inp['source_receipts']:checked_artifact(source)
    return inp

def checked_prediction(freeze_record,score_path,catalog_path,split,seed,score_key):
    frozen=validate_freeze(freeze_record);score_path=Path(score_path)
    receipt=json.loads((score_path.parent/'complete.json').read_text())
    if receipt.get('schema')!='generalization_predictions_v1' or receipt.get('phase')!='post_evaluation_morgan' or receipt.get('labels_used') is not False:raise ValueError('Wrong Morgan prediction phase/schema')
    if not same(receipt['morgan_freeze'],freeze_record) or receipt['output_sha256']!=sha256(score_path):raise ValueError('Morgan freeze/score mismatch')
    if score_key not in ('selected','parent','baseline','baseline_fp64'):raise ValueError('Unregistered score role')
    row=model_row(frozen,split,seed);pred=prediction_row(frozen,receipt['panel'],seed)
    if pred['split']!=split or not same(row['parent_bundle'],receipt['bundle']) or not same(row['morgan_state'],receipt['morgan_state']):raise ValueError('Mislabeled split/seed/state')
    for key in ('input_receipt','morgan_inputs'):
        if not same(pred[key],receipt[key]):raise ValueError('Unapproved feature identity: '+key)
        checked_artifact(receipt[key])
    for key in ('receipt','scores'):
        if not same(pred['parent_reference'][key],receipt['parent_reference'][key]):raise ValueError('Parent reference identity mismatch')
        checked_artifact(receipt['parent_reference'][key])
    inp=authenticate_input_receipt(frozen,receipt['input_receipt'])
    for key in ('catalog','base','protein_means','reaction_features'):
        if not same(inp['inputs'][key],receipt['inputs'][key]):raise ValueError('Feature receipt disagreement')
    if receipt['inputs']['catalog']['sha256']!=sha256(catalog_path):raise ValueError('Catalog order mismatch')
    if not receipt['parent_scores_exact'] or not all(receipt['subset_parity'].values()):raise ValueError('Numerical acceptance failed')
    parent=json.loads(checked_artifact(receipt['bundle']).read_text())
    from horizyn.generalization_phase2 import validate_phase2_bundle
    validate_phase2_bundle(parent,Path(receipt['bundle']['path']).parent)
    if parent['split']!=split or parent['seed']!=seed:raise ValueError('Parent split/seed mismatch')
    if Path(inp['checkpoint']['path']).resolve()!=Path(parent['base_checkpoint']['path']).resolve():raise ValueError('Input checkpoint differs from parent')
    if parent['base_checkpoint'].get('sha256') and inp['checkpoint']['sha256']!=parent['base_checkpoint']['sha256']:raise ValueError('Input checkpoint hash differs')
    for key,target in [('phase2_frozen_recipe','phase2_frozen_recipe'),('frozen_recipe','original_frozen_recipe')]:
        if not same(receipt[key],frozen[target]) or not same(parent[key],frozen[target]):raise ValueError('Parent frozen lineage mismatch')
    with np.load(score_path) as source:values=source[score_key]
    cat=json.loads(Path(catalog_path).read_text());shape=(len(cat.get('reactions',cat.get('query_ids'))),len(cat['proteins']))
    if values.shape!=shape or values.dtype!=np.float32 or not np.isfinite(values).all():raise ValueError('Malformed score array')
    return values,receipt

def checked_large_prediction(freeze_record,score_path,scan_dir):
    frozen=validate_freeze(freeze_record);scan_dir=Path(scan_dir);score_path=Path(score_path)
    receipt=json.loads((score_path.parent/'complete.json').read_text())
    if receipt.get('schema')!='morgan_large_case1_predictions_v1' or receipt.get('labels_used') is not False or receipt.get('original_five_scores_unchanged') is not True:raise ValueError('Wrong supplemental prediction schema')
    if not same(receipt['morgan_freeze'],freeze_record) or receipt['output_sha256']!=sha256(score_path):raise ValueError('Supplement freeze/score mismatch')
    for key,path in [('protocol',scan_dir/'protocol.json'),('catalog',scan_dir/'catalog.json'),('aliases',scan_dir/'aliases.json'),('scan_complete',scan_dir/'complete.json'),('numerical_audit',scan_dir/'completion_audit.json'),('primary_evaluation',RUN/'large_case1_evaluation/complete.json')]:
        if not same(receipt[key],record(path)):raise ValueError('Supplement scan lineage mismatch: '+key)
    for key,source in [('scan_complete','large_scan_complete'),('numerical_audit','large_numerical_audit'),('primary_evaluation','large_primary_evaluation')]:
        if not same(receipt[key],frozen[source]):raise ValueError('Supplement original gates differ from frozen completed artifacts')
    if not receipt.get('parent_scores_exact') or receipt.get('seed')!=42 or receipt.get('split')!='reaction_smi' or receipt.get('rule')!=RULE:raise ValueError('Wrong supplement model or parent replay')
    row=model_row(frozen,'reaction_smi',42)
    if not same(row['parent_bundle'],receipt['bundle']) or not same(row['morgan_state'],receipt['morgan_state']):raise ValueError('Supplement state mismatch')
    if not same(receipt['query_state'],frozen['large_query_state']):raise ValueError('Unapproved large query')
    checked_artifact(receipt['query_state'])
    scan=json.loads((scan_dir/'complete.json').read_text())
    expected=[]
    for item in scan['chunk_receipts']:
        chunk=json.loads(checked_artifact(item).read_text());expected.append(chunk['outputs']['enzyme_index'])
    if len(expected)!=len(receipt['index_sources']) or not all(same(a,b) for a,b in zip(expected,receipt['index_sources'])):raise ValueError('Index coverage/source order mismatch')
    cat=json.loads((scan_dir/'catalog.json').read_text())
    with np.load(score_path) as source:
        if source['ids'].tolist()!=cat['proteins']:raise ValueError('Supplement candidate order mismatch')
        scores=source['selected'];parent=source['parent']
    if scores.shape!=(len(cat['proteins']),) or scores.dtype!=np.float32 or not np.isfinite(scores).all():raise ValueError('Malformed supplement score')
    with np.load(scan_dir/'scores.npz') as source:
        if not np.array_equal(parent,source['phase2']):raise ValueError('Supplement parent score replay differs')
    return scores,receipt

def panel_strings(panel,input_receipt):
    receipt=json.loads(checked_artifact(input_receipt).read_text())
    cat=json.loads(checked_artifact(receipt['inputs']['catalog']).read_text());ids=cat.get('reactions',cat.get('query_ids'))
    records=[input_receipt,record(receipt['inputs']['catalog']['path'])]
    if panel in SPLITS:
        path=RUN/f'features_test_{panel}/manifest.json';manifest=json.loads(path.read_text())
        rec=manifest['inputs']['reactions'];records +=[record(path),rec];table=reaction_table(checked_artifact(rec))
    else:
        path=RUN/(panel+'_audit')/'features/f3_epoch29/complete.json';native=json.loads(path.read_text())
        if not any(same(record(path),x) for x in receipt['source_receipts']):raise ValueError('Native export receipt not bound by feature receipt')
        records.append(record(path))
        if panel=='case1':
            table={ids[0]:native['encoded_reaction_smiles']}
        else:
            policy=native['input_policy']
            if policy['policy']!='participant_self_reaction':raise ValueError('Actual model input policy is not participant collection')
            training=Path(policy['training_reactions']);checked_artifact(dict(path=str(training),sha256=policy['training_reactions_sha256']))
            rec=native.get('outputs',{}).get('feature_reactions.csv') or native['reaction_sources']['feature_reactions.csv']
            records +=[rec,record(training)];table=reaction_table(checked_artifact(rec))
    if set(ids)!=set(table):raise ValueError('Reaction source and catalog IDs differ')
    strings=[table[key] for key in ids]
    # No physical-to-participant conversion is performed here: use the exact
    # previously authenticated model-input strings and require self/collection.
    for value in strings:canonical_components(value)
    return ids,strings,records

@torch.inference_mode()
def prepare(args):
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'prepared.json').exists():raise ValueError('Existing immutable preparation')
    shutil.copyfile(__file__,OUT/'prepare_source.py');preparation_source=record(OUT/'prepare_source.py')
    for item in json.loads((RUN/'phase4/frozen_recipe.json').read_text())['implementation_sources']:checked_artifact(item)
    selection=json.loads((STUDY/'complete.json').read_text())
    if selection['selected']['label']!='radius3_eta1' or not selection['qualifies_for_parent_review']:raise ValueError('Selected fixed Morgan recipe not accepted')
    old=json.loads((RUN/'phase4/frozen_recipe.json').read_text());models=[];predictions=[]
    for split in SPLITS:
        directory=RUN/('features' if split=='reaction_smi' else 'features_'+split);catalog=json.loads((directory/'catalog.json').read_text());manifest=json.loads((directory/'manifest.json').read_text())
        with np.load(directory/'pairs.npz') as source:train=source['train']
        tr=np.unique(train[:,0]);ids=[catalog['reactions'][i] for i in tr]
        if ids!=catalog['train_reactions']:raise ValueError('Actual training reaction selector mismatch')
        rec=manifest['inputs']['train_reactions_path'];table=reaction_table(checked_artifact(rec))
        strings=[table[key] for key in ids]
        matrix=np.stack([participant_morgan(value,3) for value in strings])
        output=OUT/'models'/split;output.mkdir(parents=True,exist_ok=True)
        state=dict(schema='morgan_training_dictionary_v1',rule=RULE,split=split,train_ids=ids,train_morgan=torch.from_numpy(matrix),feature_manifest_sha256=sha256(directory/'manifest.json'),training_only=True,validation_used=False,training_sources={k:record(directory/k) for k in ('catalog.json','pairs.npz','manifest.json')},reaction_source=rec,source=preparation_source)
        torch.save(state,output/'morgan.pt');atomic_json(output/'reaction_inputs.json',dict(ids=ids,strings=strings,source=rec,parse_invalid_count=0,training_indices=tr.tolist()))
        for seed in SEEDS:
            parent=RUN/'phase2/models'/split/f'seed{seed}/bundle.json';spec=json.loads(parent.read_text())
            if spec['feature_manifest_sha256']!=state['feature_manifest_sha256'] or spec['seed']!=seed or spec['split']!=split:raise ValueError('Wrong fixed parent model')
            models.append(dict(split=split,seed=seed,parent_bundle=record(parent),morgan_state=record(output/'morgan.pt'),reaction_inputs=record(output/'reaction_inputs.json')))
    for panel in PANELS:
        first=json.loads(checked_artifact(old['previous_control_prediction_receipts'][panel]['seed42']).read_text());input_receipt=first['input_receipt']
        ids,strings,records=panel_strings(panel,input_receipt);directory=OUT/'inputs'/panel;directory.mkdir(parents=True,exist_ok=True)
        matrix=np.stack([participant_morgan(value,3) for value in strings]);np.savez(directory/'morgan.npz',vectors=matrix)
        inp=dict(schema='morgan_frozen_query_inputs_v1',panel=panel,rule=RULE,reaction_ids=ids,reaction_strings=strings,vectors=record(directory/'morgan.npz'),input_receipt=input_receipt,sources=records,invalid_count=0,conversion='Use exact authenticated parent model participant/self-reaction inputs; take one canonical self side only; no new physical conversion.',source=preparation_source)
        atomic_json(directory/'complete.json',inp)
        for seed in SEEDS:
            rec=old['previous_control_prediction_receipts'][panel][f'seed{seed}'];prior=json.loads(checked_artifact(rec).read_text())
            if not same(input_receipt,prior['input_receipt']):raise ValueError('Seed feature source disagreement')
            score=Path(rec['path']).parent/'scores.npz';score_rec=dict(path=str(score),sha256=prior['output_sha256'])
            # Pin old controls without reading their values during preparation.
            checked_artifact(score_rec)
            predictions.append(dict(panel=panel,split=panel if panel in SPLITS else 'reaction_smi',seed=seed,input_receipt=input_receipt,morgan_inputs=record(directory/'complete.json'),parent_reference=dict(receipt=rec,scores=score_rec),reaction_key=prior['reaction_key']))
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;device=torch.device(args.device)
    row=next(x for x in models if x['split']=='reaction_smi' and x['seed']==42);model,spec=MorganComposedEncoder.from_artifacts(row['parent_bundle'],row['morgan_state'],device)
    p=RUN/'features';cat=json.loads((p/'catalog.json').read_text())
    with np.load(p/'pairs.npz') as source:vr,ve,truth=validation_data(cat,{k:source[k] for k in source.files})
    oldval=torch.load(RUN/'phase2/composition_fp64/selected_validation_features.pt',map_location=device,weights_only=False)
    with np.load(p/'reaction_features.npz') as source:
        blocks={k:torch.tensor(source[k][vr],device=device) for k in model.modalities};masks={k:torch.tensor(source[k+'_mask'][vr],device=device) for k in model.modalities}
    with np.load(STUDY/'morgan_features.npz') as source:fp=torch.tensor(source['radius3'][vr],device=device);trainfp=torch.tensor(source['radius3'][np.unique(np.load(p/'pairs.npz')['train'][:,0])],device=device)
    if not torch.equal(trainfp,model.anchor.train_morgan):raise ValueError('Transfer dictionary differs from selected validation dictionary')
    rr=model.replace_reaction_semantics(oldval['reactions'],blocks,masks,fp);e=oldval['enzymes'];scores=canonical_dot(rr,e)
    from generalization_metrics import evaluate_scores
    result=evaluate_scores(scores,truth)
    with np.load(STUDY/'radius3_eta1_ranks.npz') as source:rank_exact=all(np.array_equal(source[d+'_'+k],v) for d,b in result['per_positive'].items() for k,v in b.items())
    def sub(ix):return model.replace_reaction_semantics(oldval['reactions'][ix],{k:v[ix] for k,v in blocks.items()},{k:v[ix] for k,v in masks.items()},fp[ix])
    checks=dict(validation_positive_ranks_exact=rank_exact,query_subset_exact=torch.equal(rr[::71],sub(slice(None,None,71))),query_singleton_exact=torch.equal(rr[:1],sub(slice(0,1))),sparse_dense_exact=torch.equal(scores,model.score_index(rr,dict(dense=e[:,:512],anchors=e[:,512:].to_sparse_csr()))))
    if not all(checks.values()):raise ValueError('Selected production inference does not replay validation')
    atomic_json(OUT/'production_check.json',dict(checks=checks,validation_only=True,model=row))
    case=next(x for x in predictions if x['panel']=='case1' and x['seed']==42);receipt=json.loads(checked_artifact(case['input_receipt']).read_text());inp=json.loads(checked_artifact(case['morgan_inputs']).read_text())
    with np.load(checked_artifact(receipt['inputs']['base'])) as source:base=torch.tensor(source[case['reaction_key']],device=device)
    with np.load(checked_artifact(receipt['inputs']['reaction_features'])) as source:
        b={k:torch.tensor(source[k],device=device) for k in model.modalities};m={k:torch.tensor(source[k+'_mask'],device=device) for k in model.modalities}
    with np.load(checked_artifact(inp['vectors'])) as source:fp=torch.tensor(source['vectors'],device=device)
    parentq=model.parent.encode_reactions(base,b,m);newq=model.replace_reaction_semantics(parentq,b,m,fp)
    large=STUDY/'large_case1_supplement';large.mkdir(exist_ok=True)
    torch.save(dict(selected=newq.cpu(),parent=parentq.cpu(),rule=RULE,model=row,query=case,labels_used=False),large/'query_state.pt')
    atomic_json(OUT/'prepared.json',dict(schema='morgan_transfer_input_preparation_v1',created_utc=utc(),rule=RULE,approved_models=models,approved_predictions=predictions,production_check=record(OUT/'production_check.json'),large_query_state=record(large/'query_state.pt'),source=preparation_source,no_new_external_scores_computed=True,no_large_pool_outcomes_read=True))
    print(json.dumps(dict(prepared=record(OUT/'prepared.json'),models=len(models),predictions=len(predictions),validation_checks=checks)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare']);p.add_argument('--device',default='cuda:2');args=p.parse_args();prepare(args)
if __name__=='__main__':main()
