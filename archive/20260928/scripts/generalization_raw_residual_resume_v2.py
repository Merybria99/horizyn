#!/usr/bin/env python3
"""Serializer-key guard correction; preserves the frozen raw-residual training math."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from generalization_raw_residual import *

def train_one(config,data,semantic_scores,device):
 output=OUT/config['label'];output.mkdir(parents=True,exist_ok=True)
 if (output/'registry.json').exists():raise ValueError('Immutable run exists')
 atomic_json(output/'registry.json',dict(protocol=record(OUT/'protocol.json'),prepared_receipt=record(OUT/'prepared_receipt.json'),config=config,seed=42,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 torch.manual_seed(42);model_config=dict(protein_raw_dimension=1024,reaction_raw_dimension=2409,hidden=1024,cap=config['cap'])
 model=RawFeatureResidual(**model_config).to(device);evaluator=RawFeatureResidual(**model_config).to(device).double().requires_grad_(False)
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-3)
 parent_e=F.normalize(data['train_e']['parent'],dim=1);parent_r=F.normalize(data['train_r']['parent'],dim=1)
 records=[];best=None;last={};started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
 evaluator.load_state_dict(model.state_dict())
 zero,zero_r,zero_e=evaluate(evaluator,data,False)
 normal_effect=dict(summary=zero['summary'],balanced_value=robust_value(zero['summary']),reaction_coordinates=compare(zero_r[:,:512],data['val_r']['parent']),enzyme_coordinates=compare(zero_e[:,:512],data['val_e']['parent']))
 atomic_json(output/'zero_weight_normalization_control.json',normal_effect);del zero,zero_r,zero_e
 for step in range(501):
  if step%25==0:
   evaluator.load_state_dict(model.state_dict());evaluator.eval()
   result,rr,ee=evaluate(evaluator,data,identity=step==0);summary=result['summary']
   row=dict(step=step,balanced_value=robust_value(summary),aggregate_value=selection_value(summary),guards=eligible(summary,data['p2_summary'],.005) and eligible(summary,data['f3_summary'],.005),summary=summary,training=last,elapsed_seconds=time.monotonic()-started,peak_vram_gib=torch.cuda.max_memory_allocated(device)/2**30)
   records.append(row)
   if row['guards'] and (best is None or row['balanced_value']>best['balanced_value']):
    best=row;torch.save(dict(schema='raw_feature_residual_selected_v1',state_dict=cpu(model.state_dict()),model_config=model_config,identity=step==0,selected=row,protocol=record(OUT/'protocol.json'),prepared_receipt=record(OUT/'prepared_receipt.json')),output/'selected.pt');save_result(output/'selected_validation_ranks.npz',result)
   atomic_json(output/'validation.json',dict(records=records,selected=best));print(json.dumps(dict(label=config['label'],step=step,balanced=row['balanced_value'],guards=row['guards'],seconds=row['elapsed_seconds'],training=last)),flush=True)
   del result,rr,ee
  if step==500:break
  model.train();optimizer.zero_grad(set_to_none=True)
  e=neural(model,data['train_e'],'enzymes');r=neural(model,data['train_r'],'reactions')
  logits=(r@e.T+semantic_scores)/.07
  contrastive,rloss,eloss=full_graph_contrastive_loss(logits,data['edge_r'],data['edge_e'],'uniform')
  identity=((1-(F.normalize(e,dim=1)*parent_e).sum(1)).mean()+(1-(F.normalize(r,dim=1)*parent_r).sum(1)).mean())/2
  loss=contrastive+config['identity_weight']*identity
  if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
  last=dict(total=float(loss.detach()),contrastive=float(contrastive.detach()),identity=float(identity.detach()),reaction_loss=float(rloss),enzyme_loss=float(eloss))
  del e,r,logits,loss,contrastive,identity
 selected=torch.load(output/'selected.pt',map_location=device,weights_only=False);evaluator.load_state_dict(selected['state_dict'])
 result,checks=final_checks(evaluator,data,selected['identity'],output)
 with np.load(output/'selected_validation_ranks.npz') as f:
  rank_exact=all(np.array_equal(f['positive__'+d+'__'+k],v) for d,b in result['per_positive'].items() for k,v in b.items())
 final=dict(config=config,selected=best,selected_checkpoint=record(output/'selected.pt'),selected_reload_ranks_exact=rank_exact,numerical_checks=checks,strict_p2_gain=best['balanced_value']>robust_value(data['p2_summary']),qualifies_for_parent_review=rank_exact and checks['all_exact'] and checks['all_caps_hold'] and best['balanced_value']>robust_value(data['p2_summary']),elapsed_seconds=time.monotonic()-started,test_or_external_used=False,large_pool_scores_used=False,training_steps=500)
 atomic_json(output/'complete.json',final);print(json.dumps(dict(label=config['label'],complete=True,best=best['balanced_value'],best_step=best['step'],qualifies=final['qualifies_for_parent_review'],seconds=final['elapsed_seconds'])),flush=True)

@torch.inference_mode()
def finalize_existing(config,data,device):
 output=OUT/config['label']
 if (output/'complete.json').exists():raise ValueError('Existing completion must not be overwritten')
 ledger=json.loads((output/'validation.json').read_text())
 if [row['step'] for row in ledger['records']]!=list(range(0,501,25)):raise ValueError('Only completed500-step histories may be finalized')
 selected=torch.load(output/'selected.pt',map_location=device,weights_only=False)
 if selected['selected']!=ledger['selected'] or selected['protocol']!=record(OUT/'protocol.json'):raise ValueError('Frozen selected state/ledger mismatch')
 model=RawFeatureResidual(**selected['model_config']).to(device).double().requires_grad_(False);model.load_state_dict(selected['state_dict'])
 correction=output/'guard_correction';correction.mkdir(exist_ok=True);started=time.monotonic()
 result,checks=final_checks(model,data,selected['identity'],correction)
 with np.load(output/'selected_validation_ranks.npz') as f:
  rank_exact=all(np.array_equal(f['positive__'+d+'__'+k],v) for d,b in result['per_positive'].items() for k,v in b.items())
 with np.load(output/'pointwise_caps.npz') as old,np.load(correction/'pointwise_caps.npz') as new:
  if set(old.files)!=set(new.files) or not all(np.array_equal(old[k],new[k]) for k in old.files):raise ValueError('Repeated pointwise cap diagnostic changed')
 best=ledger['selected'];final=dict(config=config,selected=best,selected_checkpoint=record(output/'selected.pt'),selected_reload_ranks_exact=rank_exact,numerical_checks=checks,strict_p2_gain=best['balanced_value']>robust_value(data['p2_summary']),qualifies_for_parent_review=rank_exact and checks['all_exact'] and checks['all_caps_hold'] and best['balanced_value']>robust_value(data['p2_summary']),elapsed_seconds=ledger['records'][-1]['elapsed_seconds']+time.monotonic()-started,test_or_external_used=False,large_pool_scores_used=False,training_steps=500,evaluation_only_guard_amendment=record(OUT/'evaluation_guard_amendment_v2.json'),original_failed_worker_log=record(OUT/('worker0.log' if config['identity_weight']==1 else 'worker1.log')))
 atomic_json(output/'complete.json',final)
 print(json.dumps(dict(finalized=config['label'],step=best['step'],balanced=best['balanced_value'],rank_replay=rank_exact,numerical=checks['all_exact'],caps=checks['all_caps_hold'])),flush=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='cuda:0');parser.add_argument('--worker',required=True,type=int,choices=[0,1]);args=parser.parse_args()
 amendment=json.loads((OUT/'evaluation_guard_amendment_v2.json').read_text())
 if amendment['original_protocol']!=record(OUT/'protocol.json'):raise ValueError('Amendment protocol mismatch')
 for source in amendment['new_sources']:checked_artifact(source)
 for artifact in amendment['completed_first_config_artifacts']:checked_artifact(artifact)
 device=torch.device(args.device);configure(device);data=load_prepared(device)
 first,remaining=GRID[args.worker::2]
 finalize_existing(first,data,device)
 with torch.inference_mode():
  anchors=data['train_e']['semantic'].double();reactions=data['train_r']['semantic'];semantic_scores=torch.empty((len(reactions),len(data['train_e']['native'])),device=device,dtype=torch.float32)
  for start in range(0,len(reactions),256):semantic_scores[start:start+256]=torch.sparse.mm(anchors,reactions[start:start+256].double().T).T.float()
 atomic_json(OUT/(remaining['label']+'_amended_launch.json'),dict(amendment=record(OUT/'evaluation_guard_amendment_v2.json'),protocol=record(OUT/'protocol.json'),config=remaining,training_math_unchanged=True))
 train_one(remaining,data,semantic_scores,device)
 authenticate()
 for source in amendment['new_sources']:checked_artifact(source)
 for artifact in amendment['completed_first_config_artifacts']:checked_artifact(artifact)
if __name__=='__main__':main()
