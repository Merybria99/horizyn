#!/usr/bin/env python3
"""Four fixed raw-feature residual controls, training/validation only."""
from __future__ import annotations
import argparse,datetime,json,math,sys,time
from pathlib import Path
import h5py,numpy as np,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from horizyn.generalization_raw_residual import RawFeatureResidual
from horizyn.generalization_residual import full_graph_contrastive_loss
from horizyn.generalization_retrieval import canonical_dot,checked_artifact,sha256
from horizyn.semantic_anchors import row_unit,centered_unit,reaction_features
from generalization_full_graph import atomic_json,validation_data,eligible,selection_value
from generalization_smooth_anchors import robust_value
from generalization_composed_calibration import save_result
from generalization_metrics import evaluate_scores
RUN=ROOT/'runs/generalization_20260919_2251';OUT=RUN/'post_evaluation_raw_residual'
GRID=[dict(label=f'cap{c:g}_identity{w:g}',cap=c,identity_weight=w) for c in [.1,.3] for w in [1.,5.]]

def record(path):return dict(path=str(Path(path).resolve()),sha256=sha256(path))
def paths():
 f=RUN/'features';v=RUN/'phase2/composition_fp64'
 return {**{k:f/k for k in ['catalog.json','pairs.npz','manifest.json','f3_features.npz','protein_mean.h5','reaction_features.npz']},
  'parent_bundle':RUN/'phase2/models/reaction_smi/seed42/bundle.json',
  'parent_validation':v/'selected_validation_features.pt','parent_ranks':v/'density_smooth_alpha0.25_ranks.npz',
  'authorization':RUN/'late_representation_authorization.json',
  'script':Path(__file__),'module':ROOT/'horizyn/generalization_raw_residual.py','tests':ROOT/'tests/unit/test_generalization_raw_residual.py'}
def configure(device):
 torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_device(device)
def authenticate():
 protocol=json.loads((OUT/'protocol.json').read_text())
 for item in protocol['implementation_sources']:checked_artifact(item)
 for k,p in paths().items():
  if record(p)!=protocol['inputs'][k]:raise ValueError('Changed input '+k)
 return protocol
def compare(a,b):return dict(exact=bool(torch.equal(a,b)),unequal_values=int((a!=b).sum()),max_abs=float((a.double()-b.double()).abs().max()) if a.numel() else 0.)
def cpu(value):
 if isinstance(value,torch.Tensor):return value.cpu()
 if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
 return value

@torch.inference_mode()
def prepare(args):
 if (OUT/'protocol.json').exists():raise ValueError('Immutable protocol already exists')
 OUT.mkdir(parents=True,exist_ok=True);p=paths();sources={k:record(v) for k,v in p.items()}
 previous=json.loads((RUN/'post_evaluation_semantic_scale/protocol.json').read_text())
 closure={r['path']:r for r in previous['implementation_sources']}
 for k in ['script','module','tests']:closure[sources[k]['path']]=sources[k]
 for r in closure.values():checked_artifact(r)
 protocol=dict(schema='late_raw_feature_residual_protocol_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=sources,implementation_sources=list(closure.values()),grid=GRID,
  seed=42,hidden=1024,steps=500,validation_every=25,optimizer=dict(name='AdamW',lr=1e-4,weight_decay=1e-3,clip_grad_norm=1.),temperature=.07,
  objective='Full training graph uniform known-positive bidirectional CE: uniform queries in each direction, equal direction weight. Logits include fixed weighted semantic contribution. Unknown pairs occur in softmax denominators and are not verified biological negatives. Identity penalty is equal mean neural cosine displacement across uniform training enzymes and reactions.',
  input='Normalized native F3 512 coordinates concatenated with training-centered normalized raw protein means1024 or all four reaction blocks. Centers/masks/normalization reused unchanged from authenticated P2 training dictionary; actual pairs[train] endpoint selector only.',
  tower='Independent LayerNorm(input),Linear(input,1024),GELU,Linear(1024,512); final layer zero initialized. No dropout. Seed42 for each of four configurations.',
  cap='Clip each correction norm to cap times the norm of the existing sqrt(.75)-weighted neural block. Normalize corrected block and restore sqrt(.75); equivalent relative cap on unweighted unit endpoint. Preserve existing sqrt(.25)-weighted semantic coordinates exactly. No additional OOD gate.',
  training_numeric='FP32 full graph forward/backward and fixed semantic scores, highest matmul precision, TF32 disabled. Fixed semantic scores use FP64 sparse dot then FP32 once. Full graph denominator always includes all actual training endpoints.',
  inference='Copy weights into FP64 tower; FP64 residual forward, pointwise cap and normalization, then store sqrt(.75)-weighted neural output FP32. Actual concatenated FP32 endpoints scored FP64 dot then FP32. Explicit identity candidate delegates original P2 cached coordinates exactly; zero-weight normalization-only effect separately recorded.',
  selection='For each config select largest equal-weight seen/unseen reaction all-positive MRR across both directions at steps0,25,...500; all/unseen eachdirection within.005 of BOTH F3fp64/P2. Step0 explicit P2 identity. Grid best fixed enumeration/earliest strict winner; strictP2 gain and all numerical checks required, no alternative after numerical failure.',
  checks='Every selected state reloaded; exact full/subset/singleton/chunked endpoint/score and sparse/dense parity; caps inspected for every training and validation endpoint. All configurations/checkpoints retained in metric history.',
  exposure='Late exploratory representation hypothesis; repeated validation consultation, no independent confirmation. Authorization pinned before large-background outcomes opened; this study does not read those outcomes.',
  external_or_test_used=False,large_pool_scores_used=False,residue_payload_used=False,stop='No external/test predictions until root review; no old frozen source/model/default changes.')
 atomic_json(OUT/'protocol.json',protocol);print(json.dumps(dict(protocol=record(OUT/'protocol.json'))),flush=True)
 device=torch.device(args.device);configure(device);started=time.monotonic()
 from horizyn.generalization_phase2 import ComposedPhase2Encoder
 parent,spec=ComposedPhase2Encoder.from_bundle(p['parent_bundle'],device)
 if spec['alpha']!=.25 or spec['feature_manifest_sha256']!=sources['manifest.json']['sha256']:raise ValueError('Wrong parent source')
 catalog=json.loads(p['catalog.json'].read_text())
 with np.load(p['pairs.npz']) as f:pairs={k:f[k] for k in f.files}
 train=pairs['train'];tr,te=np.unique(train[:,0]),np.unique(train[:,1])
 if len(np.unique(train,axis=0))!=len(train) or catalog['train_reactions']!=[catalog['reactions'][i] for i in tr] or catalog['train_proteins']!=[catalog['proteins'][i] for i in te]:raise ValueError('Actual train selector mismatch')
 vr,ve,truth=validation_data(catalog,pairs)
 with np.load(p['f3_features.npz']) as f:be=torch.tensor(f['proteins'],device=device);br=torch.tensor(f['reactions'],device=device);tr_native=torch.tensor(f['train_reactions'],device=device)
 with h5py.File(p['protein_mean.h5']) as f:
  if f['ids'].asstr()[:].tolist()!=catalog['proteins'] or not f['complete'][:].all():raise ValueError('Raw mean IDs/completeness mismatch')
  raw=torch.tensor(f['vectors'][:],device=device)
 with np.load(p['reaction_features.npz']) as f:
  blocks={k:torch.tensor(f[k],device=device) for k in ['t5v2','unimol2','chiro','chemistry']};masks={k:torch.tensor(f[k+'_mask'],device=device) for k in blocks}
 if set(parent.modalities)!=set(blocks):raise ValueError('Require all four native modalities')
 centers={k:getattr(parent.smooth,'reaction_center_'+k) for k in parent.modalities}
 raw_e=centered_unit(raw,parent.smooth.protein_center);raw_r=reaction_features(blocks,centers,masks,parent.modalities)
 if raw_e.shape[1]!=1024 or raw_r.shape[1]!=2409:raise ValueError('Unexpected raw geometry dimensions')
 index=parent.encode_enzyme_index(be[te],raw[te],batch_size=512)
 pr=parent.encode_reactions(tr_native,{k:v[tr] for k,v in blocks.items()},{k:v[tr] for k,v in masks.items()},batch_size=512)
 old=torch.load(p['parent_validation'],map_location=device,weights_only=False)
 if old['reaction_ids']!=catalog['validation_reactions'] or old['enzyme_ids']!=catalog['validation_candidates']:raise ValueError('Validation parent order mismatch')
 rmap={int(g):i for i,g in enumerate(tr)};emap={int(g):i for i,g in enumerate(te)}
 prepared=dict(schema='raw_feature_residual_prepared_v1',protocol=record(OUT/'protocol.json'),parent_bundle=sources['parent_bundle'],
  train_e=dict(native=row_unit(be[te]),raw=raw_e[te],parent=index['dense'],semantic=index['anchors']),
  train_r=dict(native=row_unit(tr_native),raw=raw_r[tr],parent=pr[:,:512],semantic=pr[:,512:]),
  val_e=dict(native=row_unit(be[ve]),raw=raw_e[ve],parent=old['enzymes'][:,:512],semantic=old['enzymes'][:,512:]),
  val_r=dict(native=row_unit(br[vr]),raw=raw_r[vr],parent=old['reactions'][:,:512],semantic=old['reactions'][:,512:]),
  edge_r=torch.tensor([rmap[int(x)] for x in train[:,0]],device=device),edge_e=torch.tensor([emap[int(x)] for x in train[:,1]],device=device),truth=truth,
  train_enzyme_ids=catalog['train_proteins'],train_reaction_ids=catalog['train_reactions'],validation_enzyme_ids=catalog['validation_candidates'],validation_reaction_ids=catalog['validation_reactions'],
  centers=dict(protein=parent.smooth.protein_center,reactions=centers),modalities=parent.modalities)
 # The unchanged cached validation parent is checked before any training.
 baseline=evaluate_scores(canonical_dot(old['reactions'],old['enzymes']),truth)
 with np.load(p['parent_ranks']) as f:
  if not all(np.array_equal(f[d+'_'+k],v) for d,b in baseline['per_positive'].items() for k,v in b.items()):raise ValueError('Parent rank replay failed')
 f3=evaluate_scores(canonical_dot(prepared['val_r']['native'],prepared['val_e']['native']),truth)
 prepared['p2_summary']=baseline['summary'];prepared['f3_summary']=f3['summary']
 torch.save(cpu(prepared),OUT/'prepared.pt')
 atomic_json(OUT/'prepared_receipt.json',dict(schema='raw_feature_residual_preparation_complete_v1',protocol=record(OUT/'protocol.json'),prepared=record(OUT/'prepared.pt'),train_enzymes=len(te),train_reactions=len(tr),train_edges=len(train),elapsed_seconds=time.monotonic()-started,parent_validation_ranks_exact=True,no_external_test_or_large_pool_used=True))
 print(json.dumps(dict(prepared=record(OUT/'prepared.pt'),seconds=time.monotonic()-started)),flush=True)

def load_prepared(device):
 authenticate();receipt=json.loads((OUT/'prepared_receipt.json').read_text())
 if receipt['protocol']!=record(OUT/'protocol.json'):raise ValueError('Preparation protocol changed')
 path=checked_artifact(receipt['prepared']);return torch.load(path,map_location=device,weights_only=False)

def neural(model,d,side,identity=False,diagnostics=False):
 return getattr(model,'encode_'+side)(d['native'],d['raw'],d['parent'],identity=identity,return_diagnostics=diagnostics)

@torch.inference_mode()
def evaluate(model,data,identity=False):
 r=neural(model,data['val_r'],'reactions',identity);e=neural(model,data['val_e'],'enzymes',identity)
 rr=model.compose(r,data['val_r']['semantic']);ee=model.compose(e,data['val_e']['semantic'])
 return evaluate_scores(canonical_dot(rr,ee),data['truth']),rr,ee

@torch.inference_mode()
def final_checks(model,data,identity,output):
 result,r,e=evaluate(model,data,identity)
 ir=torch.arange(0,len(r),max(1,len(r)//31),device=r.device)[:31];ie=torch.arange(0,len(e),max(1,len(e)//193),device=e.device)[:193]
 def subset(d,idx):return {k:v[idx] for k,v in d.items()}
 sr=subset(data['val_r'],ir);se=subset(data['val_e'],ie)
 rr=model.compose(neural(model,sr,'reactions',identity),sr['semantic']);ee=model.compose(neural(model,se,'enzymes',identity),se['semantic'])
 singleton_r=model.compose(neural(model,subset(data['val_r'],slice(0,1)),'reactions',identity),data['val_r']['semantic'][:1])
 singleton_e=model.compose(neural(model,subset(data['val_e'],slice(0,1)),'enzymes',identity),data['val_e']['semantic'][:1])
 chunks=torch.cat([neural(model,subset(data['val_r'],slice(i,i+7)),'reactions',identity) for i in range(0,len(r),7)])
 scores=canonical_dot(r,e);checks=dict(reaction_subset=compare(r[ir],rr),enzyme_subset=compare(e[ie],ee),reaction_singleton=compare(r[:1],singleton_r),enzyme_singleton=compare(e[:1],singleton_e),reaction_chunks=compare(r[:,:512],chunks),subset_scores=compare(scores[ir][:,ie],canonical_dot(rr,ee)),singleton_scores=compare(scores[:1,:1],canonical_dot(singleton_r,singleton_e)),sparse_dense=compare(scores,model.score_index(r,dict(dense=e[:,:512],anchors=e[:,512:].to_sparse_csr()))))
 cap_checks={};arrays={}
 for name in ['train_r','train_e','val_r','val_e']:
  d=data[name];side='reactions' if name.endswith('r') else 'enzymes';values=[]
  for start in range(0,len(d['native']),1024):
   # Semantic fields are irrelevant to the neural tower; do not slice sparse CSR.
   small={k:v[start:start+1024] for k,v in d.items() if k!='semantic'}
   _,diag=neural(model,small,side,identity,True);values.append(diag['relative_residual'])
  rel=torch.cat(values);cap_checks[name]=dict(rows=len(rel),maximum=float(rel.max()),above_cap=int((rel>model.cap+1e-12).sum()),finite=bool(torch.isfinite(rel).all()))
  arrays[name]=rel.cpu().numpy()
 np.savez_compressed(output/'pointwise_caps.npz',**arrays)
 return result,dict(checks=checks,all_exact=all(x['exact'] for x in checks.values()),pointwise_caps=cap_checks,all_caps_hold=all(x['finite'] and x['above_cap']==0 for x in cap_checks.values()))

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
  rank_exact=all(np.array_equal(f[d+'_'+k],v) for d,b in result['per_positive'].items() for k,v in b.items())
 final=dict(config=config,selected=best,selected_checkpoint=record(output/'selected.pt'),selected_reload_ranks_exact=rank_exact,numerical_checks=checks,strict_p2_gain=best['balanced_value']>robust_value(data['p2_summary']),qualifies_for_parent_review=rank_exact and checks['all_exact'] and checks['all_caps_hold'] and best['balanced_value']>robust_value(data['p2_summary']),elapsed_seconds=time.monotonic()-started,test_or_external_used=False,large_pool_scores_used=False,training_steps=500)
 atomic_json(output/'complete.json',final);print(json.dumps(dict(label=config['label'],complete=True,best=best['balanced_value'],best_step=best['step'],qualifies=final['qualifies_for_parent_review'],seconds=final['elapsed_seconds'])),flush=True)

def run(args):
 device=torch.device(args.device);configure(device);data=load_prepared(device)
 # Fixed semantic branch is precomputed from authenticated actual train endpoints.
 with torch.inference_mode():
  anchors=data['train_e']['semantic'].double();reactions=data['train_r']['semantic'];semantic_scores=torch.empty((len(reactions),len(data['train_e']['native'])),device=device,dtype=torch.float32)
  for start in range(0,len(reactions),256):semantic_scores[start:start+256]=torch.sparse.mm(anchors,reactions[start:start+256].double().T).T.float()
 for config in GRID[args.worker::2]:train_one(config,data,semantic_scores,device)
 authenticate()

def summarize(args):
 protocol=authenticate();rows=[]
 for config in GRID:
  p=OUT/config['label']/'complete.json';r=json.loads(p.read_text());rows.append(dict(**r,receipt=record(p)))
 best=max(rows,key=lambda r:r['selected']['balanced_value'])
 result=dict(schema='late_raw_feature_residual_result_v1',protocol=record(OUT/'protocol.json'),candidates=rows,selected=best['config']['label'],selected_step=best['selected']['step'],balanced_value=best['selected']['balanced_value'],qualifies_for_parent_review=best['qualifies_for_parent_review'],test_or_external_used=False,large_pool_scores_used=False,independent_confirmation=False,next_step='Stop for root review; no external prediction or default replacement authorized.')
 atomic_json(OUT/'complete.json',result);print(json.dumps(dict(selected=result['selected'],step=result['selected_step'],balanced=result['balanced_value'],qualifies=result['qualifies_for_parent_review'],receipt=record(OUT/'complete.json'))),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','summarize']);p.add_argument('--device',default='cuda:0');p.add_argument('--worker',type=int,choices=[0,1],default=0);args=p.parse_args();{'prepare':prepare,'run':run,'summarize':summarize}[args.action](args)
if __name__=='__main__':main()
