#!/usr/bin/env python3
"""One frozen supplemental query over the unchanged completed enzyme index."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from generalization_morgan_transfer import validate_freeze,model_row,record,same,RUN,STUDY,RULE
from horizyn.generalization_retrieval import checked_artifact,GeneralizationDualEncoder,sha256
from generalization_full_graph import atomic_json

@torch.inference_mode()
def run(args):
    started=time.monotonic();freeze_record=record(args.freeze);freeze=validate_freeze(freeze_record)
    scan=RUN/'large_case1_scan';out=STUDY/'large_case1_supplement'
    if (out/'complete.json').exists():raise ValueError('Existing immutable supplementary predictions')
    protocol=json.loads((scan/'protocol.json').read_text());complete=json.loads((scan/'complete.json').read_text());audit=json.loads((scan/'completion_audit.json').read_text());primary=json.loads((RUN/'large_case1_evaluation/complete.json').read_text())
    for key,path in [('large_scan_complete',scan/'complete.json'),('large_numerical_audit',scan/'completion_audit.json'),('large_primary_evaluation',RUN/'large_case1_evaluation/complete.json')]:
        if not same(freeze[key],record(path)):raise ValueError('Previously completed original gate changed: '+key)
    if complete['schema']!='large_case1_scan_complete_v1' or not complete['complete_fixed_candidate_coverage'] or complete['labels_used']:raise ValueError('Original fixed scan is incomplete')
    if audit['schema']!='large_case1_completion_integrity_audit_v1' or not audit['all_exact'] or audit['labels_read'] or audit['ranks_inspected']:raise ValueError('Require completed input-only numerical integrity audit')
    if primary['schema']!='large_case1_evaluation_complete_v1' or not same(primary['scan_receipt'],record(scan/'complete.json')):raise ValueError('Original primary evaluation must finish first')
    if not same(audit['complete'],record(scan/'complete.json')) or not same(audit['protocol'],record(scan/'protocol.json')) or not same(primary['protocol'],record(scan/'protocol.json')):raise ValueError('Original completion lineage mismatch')
    for source in protocol['implementation_sources']:checked_artifact(source)
    for item in complete['outputs'].values():checked_artifact(item)
    for item in primary['outputs'].values():checked_artifact(item)
    if not same(record(scan/'protocol.json'),freeze['large_original_protocol']):raise ValueError('Supplement uses another original pool')
    row=model_row(freeze,'reaction_smi',42);state=torch.load(checked_artifact(freeze['large_query_state']),map_location=args.device,weights_only=False)
    if state['rule']!=RULE or state['model']!=row or state['labels_used']:raise ValueError('Supplement query/model recipe mismatch')
    cat=json.loads((scan/'catalog.json').read_text());chunks=[];index_sources=[];total=0
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    for item in complete['chunk_receipts']:
        chunk=json.loads(checked_artifact(item).read_text());source=chunk['outputs']['enzyme_index'];index=torch.load(checked_artifact(source),map_location=args.device,weights_only=False)
        scores=torch.cat([GeneralizationDualEncoder.score_index(state[key],index) for key in ('selected','parent')],0).cpu().numpy()
        if scores.shape!=(2,chunk['rows']) or not np.isfinite(scores).all():raise ValueError('Invalid supplemental chunk')
        chunks.append(scores);index_sources.append(source);total+=chunk['rows']
    if total!=len(cat['proteins']) or total!=complete['candidate_count']:raise ValueError('Incomplete index coverage')
    values=np.concatenate(chunks,axis=1)
    # Exact replay is a validation gate only; no old outcome is selected or shown.
    with np.load(scan/'scores.npz') as source:parent_exact=np.array_equal(values[1],source['phase2'])
    if not parent_exact:raise ValueError('Original phase2 scores do not replay exactly')
    restricted=STUDY/'transfer/predictions/case1/seed42';restricted_receipt=json.loads((restricted/'complete.json').read_text())
    if not same(restricted_receipt['morgan_freeze'],freeze_record) or restricted_receipt['output_sha256']!=sha256(restricted/'scores.npz'):raise ValueError('Restricted reference predictions not authenticated')
    aliases=json.loads((scan/'aliases.json').read_text())
    # The original scan protocol appends all123 canonical literature sequences;
    # this supplemental check makes no label-dependent candidate choice.
    litcat=json.loads(checked_artifact(restricted_receipt['inputs']['catalog']).read_text())
    if cat['proteins'][-len(litcat['proteins']):] != ['LIT_'+x for x in litcat['proteins']]:
        raise ValueError('Unexpected original literature append mapping; require explicit alias audit')
    with np.load(restricted/'scores.npz') as source:literature_exact=np.array_equal(values[0,-len(litcat['proteins']):],source['selected'][0])
    if not literature_exact:raise ValueError('Supplement and restricted Morgan reference scores differ')
    np.savez(out/'scores.npz',ids=np.asarray(cat['proteins']),selected=values[0],parent=values[1])
    receipt=dict(schema='morgan_large_case1_predictions_v1',morgan_freeze=freeze_record,bundle=row['parent_bundle'],morgan_state=row['morgan_state'],query_state=freeze['large_query_state'],rule=RULE,split='reaction_smi',seed=42,
        protocol=record(scan/'protocol.json'),catalog=record(scan/'catalog.json'),aliases=record(scan/'aliases.json'),scan_complete=record(scan/'complete.json'),numerical_audit=record(scan/'completion_audit.json'),primary_evaluation=record(RUN/'large_case1_evaluation/complete.json'),
        index_sources=index_sources,chunk_count=len(chunks),candidate_count=total,parent_scores_exact=parent_exact,restricted_reference_exact=literature_exact,restricted_prediction=record(restricted/'complete.json'),labels_used=False,original_five_scores_unchanged=True,
        output_sha256=sha256(out/'scores.npz'),source=record(__file__),elapsed_seconds=time.monotonic()-started)
    atomic_json(out/'complete.json',receipt);print(json.dumps(dict(candidate_count=total,parent_replay_exact=True,restricted_replay_exact=True,seconds=receipt['elapsed_seconds'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--device',default='cuda:2');run(p.parse_args())
if __name__=='__main__':main()
