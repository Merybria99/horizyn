#!/usr/bin/env python3
"""Fixed 13-row late-exploratory semantic-scale train/validation study."""
from __future__ import annotations
import argparse,datetime,itertools,json,sys,time
from pathlib import Path
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from horizyn.generalization_semantic_scale import QuerySemanticScale
from horizyn.generalization_retrieval import canonical_dot,checked_artifact,sha256
from horizyn.semantic_anchors import row_unit
from generalization_full_graph import validation_data,eligible,selection_value,atomic_json
from generalization_smooth_anchors import robust_value
from generalization_composed_calibration import save_result
from generalization_metrics import evaluate_scores
RUN=ROOT/'runs/generalization_20260919_2251';OUT=RUN/'post_evaluation_semantic_scale'
BETAS=[.1,.25,.5,1.];CAPS=[10.,100.,1000.];BANK_SEED=20260920;BANK_SIZE=4096
GRID=[dict(label='identity',beta=None,cap=None)]+[dict(label=f'beta{b:g}_cap{c:g}',beta=b,cap=c) for b,c in itertools.product(BETAS,CAPS)]

def record(path):return dict(path=str(Path(path).resolve()),sha256=sha256(path))

def paths():
 f=RUN/'features';p=RUN/'phase2/composition_fp64'
 return {**{k:f/k for k in ['catalog.json','pairs.npz','manifest.json','f3_features.npz','protein_mean.h5','reaction_features.npz']},
  'parent_bundle':RUN/'phase2/models/reaction_smi/seed42/bundle.json','p2_freeze':RUN/'phase2/frozen_recipe.json','p4_freeze':RUN/'phase4/frozen_recipe.json',
  'means_freeze':RUN/'post_evaluation_calibration/transfer/frozen_recipe.json','mean_state':RUN/'post_evaluation_calibration/transfer/models/reaction_smi/seed42/calibration.pt',
  'validation':p/'selected_validation_features.pt','validation_ranks':p/'density_smooth_alpha0.25_ranks.npz','validation_selection':p/'selection.json',
  'script':Path(__file__),'module':ROOT/'horizyn/generalization_semantic_scale.py','tests':ROOT/'tests/unit/test_generalization_semantic_scale.py'}

def source_closure():
 frozen=json.loads(paths()['means_freeze'].read_text());items={x['path']:x for x in frozen['implementation_sources']}
 for name in ['script','module','tests']:
  item=record(paths()[name]);items[item['path']]=item
 for item in items.values():checked_artifact(item)
 return list(items.values())

def configure(device):
 torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_device(device)

def authenticate(protocol):
 for item in protocol['implementation_sources']:checked_artifact(item)
 for key,path in paths().items():
  if record(path)!=protocol['inputs'][key]:raise ValueError('Changed input/source: '+key)
 checked_artifact(protocol['bank'])

@torch.inference_mode()
def prepare(args):
 if (OUT/'protocol.json').exists() or (OUT/'bank.pt').exists():raise ValueError('Existing immutable preparation')
 OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic();implementation=source_closure();sources={k:record(v) for k,v in paths().items()}
 atomic_json(OUT/'bank_preparation.json',dict(schema='semantic_scale_train_bank_preparation_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,implementation_sources=implementation,training_only=True,bank_size=BANK_SIZE,bank_seed=BANK_SEED,rng='numpy Generator(PCG64(seed)), choice(sorted unique training global enzyme indices,size4096,replace=False), then sort selected global indices',model_seed=42,bank_encode_batch=512))
 p=paths();catalog=json.loads(p['catalog.json'].read_text())
 with np.load(p['pairs.npz']) as f:train=f['train']
 te=np.unique(train[:,1]);tr=np.unique(train[:,0])
 if len(te)<BANK_SIZE or len(np.unique(train,axis=0))!=len(train) or catalog['train_proteins']!=[catalog['proteins'][i] for i in te] or catalog['train_reactions']!=[catalog['reactions'][i] for i in tr]:raise ValueError('Actual training endpoint selector mismatch')
 chosen=np.sort(np.random.Generator(np.random.PCG64(BANK_SEED)).choice(te,size=BANK_SIZE,replace=False))
 frozen=json.loads(p['means_freeze'].read_text());approved=[x for x in frozen['approved_models'] if x['split']=='reaction_smi' and x['seed']==42 and x['calibration_state']==sources['mean_state'] and x['parent_bundle']==sources['parent_bundle']]
 if len(approved)!=1:raise ValueError('Uniform training mean state/parent not approved')
 state=torch.load(p['mean_state'],map_location='cpu',weights_only=False)
 if state['schema']!='p2_calibration_state_v1' or state['rule']!={'gamma_enzyme':0.,'gamma_reaction':1.,'enzyme_mean_scheme':'uniform_proteins'}:raise ValueError('Require exact fixed uniform actual-training mean')
 for key in ['catalog.json','pairs.npz','manifest.json','f3_features.npz','protein_mean.h5','reaction_features.npz']:
  if state['training_sources'][key]!=sources[key]:raise ValueError('Mean state and training bank feature sources disagree')
 device=torch.device(args.device);configure(device)
 from horizyn.generalization_phase2 import ComposedPhase2Encoder
 model,spec=ComposedPhase2Encoder.from_bundle(p['parent_bundle'],device)
 if spec['alpha']!=.25 or spec['feature_manifest_sha256']!=sources['manifest.json']['sha256']:raise ValueError('Wrong P2 weighted endpoint source')
 with np.load(p['f3_features.npz']) as f:base=torch.tensor(f['proteins'][chosen],device=device)
 with h5py.File(p['protein_mean.h5']) as f:
  if f['ids'].asstr()[:].tolist()!=catalog['proteins'] or not f['complete'][chosen].all():raise ValueError('Incomplete/reordered protein means')
  raw=torch.tensor(f['vectors'][chosen],device=device)
 bank=model.encode_enzymes(base,raw,batch_size=512)
 sample=np.unique(np.linspace(0,BANK_SIZE-1,31,dtype=int))
 parity=dict(parent_subset=torch.equal(bank[sample],model.encode_enzymes(base[sample],raw[sample],batch_size=7)),parent_singleton=torch.equal(bank[:1],model.encode_enzymes(base[:1],raw[:1],batch_size=1)))
 if not all(parity.values()) or bank.shape!=(BANK_SIZE,7489) or bank.dtype!=torch.float32 or not torch.isfinite(bank).all():raise ValueError('Training bank native/P2 numerical preparation failed')
 ids=[catalog['proteins'][i] for i in chosen]
 payload=dict(schema='fixed_p2_semantic_scale_training_bank_v1',bank=bank.cpu(),semantic_mean=state['enzyme_mean'][512:].double().cpu(),global_indices=chosen,protein_ids=ids,all_train_enzyme_count=len(te),training_edge_count=len(train),training_reaction_count=len(tr),bank_seed=BANK_SEED,parent_bundle=sources['parent_bundle'],mean_state=sources['mean_state'],preparation=record(OUT/'bank_preparation.json'),dense_width=512,actual_parent_encoder_parity=parity)
 torch.save(payload,OUT/'bank.pt');(OUT/'bank_protein_ids.txt').write_text('\n'.join(ids)+'\n')
 protocol=dict(schema='late_query_semantic_scale_validation_protocol_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=sources,implementation_sources=implementation,bank=record(OUT/'bank.pt'),bank_ids=record(OUT/'bank_protein_ids.txt'),preparation=record(OUT/'bank_preparation.json'),
  grid=GRID,selection='Equal-weight seen/unseen reaction all-positive MRR in both directions, each aggregate/unseen direction within0.005 of BOTH F3_fp64 and unchangedP2; strict objective improvement overP2 required. Fixed enumeration identity first and first tie winner. Selected point must pass every numerical check; no alternative chosen after numerical failure.',
  parent='Unchanged P2 seed42 Reaction-Sim weighted blocks Ed/Rd=sqrt(.75)*density, Es/Rs=sqrt(.25)*smooth.',
  bank_selection='Uniform4096 unique actual pairs[train] enzyme endpoints; numpyPCG64 seed20260920 choice withoutreplacement from sortedglobaltrainingindices, sortedoutput. Exact indices/IDs inbank.pt. No current retrieval candidate population used.',
  semantic_mean='Uniform mean of weighted Es over ALL147299 actual training enzymes, reused from authenticated train-only mean state; no bank-specific recentering.',
  formula='sdD=populationSD(FP64(bankEd) dot FP64(Rd)); sdS likewiseEs/Rs. m=max(1,min(cap,beta*sdD/max(sdS,1e-8))). Enew=[Ed,Es,1]; Rnew=[Rd,FP32(m*Rs),FP32(-(m-1)*(muS dot Rs))]. No renormalization.',
  numeric='Branch scores for SD remain pre-FP32-cast FP64, with fixed bankorder and FP64 population mean/variance(ddof0). m retainedFP64. Semantic scaling and centeringoffset computedFP64 thenstoredFP32. Actual augmented FP32 endpoints scored byFP64dot→FP32. Identity returns P2 endpoints/dimension directly.',
  identity_note='For scaled configurations, rowswithm1 retain oldcoordinates pluszero offset; implicit GEMMshapetie differences measured separately. Only explicit identity promisesparentbitwise score identity.',
  acceptance='Every grid point reports full/subset/singleton/chunk7 endpoint+score parity and full sparse/dense score comparisons. P2 query cache/positive ranks exactly replayed. Bank fitted before anyvalidation statistics. Selected numerical failure is rejection, not invitation toretune.',
  exploration='One root-authorized targeted test after branch-dispersion diagnostics and previously opened external outcomes. Repeated validation exploration, not independent confirmation.',
  external_test_or_large_pool_used=False,current_candidate_pool_used_in_multiplier=False,default_or_frozen_model_changed=False,external_predictions_authorized=False,budget_minutes=30,
  preparation_seconds=time.monotonic()-started,versions=dict(numpy=np.__version__,torch=torch.__version__))
 atomic_json(OUT/'protocol.json',protocol);print(json.dumps(dict(protocol=record(OUT/'protocol.json'),bank=record(OUT/'bank.pt'),seconds=protocol['preparation_seconds'])),flush=True)


def compare(actual,expected):
 return dict(exact=bool(torch.equal(actual,expected)),unequal_values=int(torch.count_nonzero(actual!=expected)),max_abs=float((actual.double()-expected.double()).abs().max()) if actual.numel() else 0.)

@torch.inference_mode()
def run(args):
 started=time.monotonic();protocol=json.loads((OUT/'protocol.json').read_text());authenticate(protocol)
 if (OUT/'registry.json').exists():raise ValueError('Existing grid outputs')
 atomic_json(OUT/'registry.json',dict(protocol=record(OUT/'protocol.json'),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 device=torch.device(args.device);configure(device);p=paths();bank=torch.load(OUT/'bank.pt',map_location=device,weights_only=False)
 catalog=json.loads(p['catalog.json'].read_text())
 with np.load(p['pairs.npz']) as f:pairs={k:f[k] for k in f.files}
 vr,ve,truth=validation_data(catalog,pairs)
 old=torch.load(p['validation'],map_location=device,weights_only=False)
 if old['reaction_ids']!=catalog['validation_reactions'] or old['enzyme_ids']!=catalog['validation_candidates']:raise ValueError('P2 validation ordering changed')
 r,e=old['reactions'],old['enzymes']
 baseline_scores=canonical_dot(r,e);baseline=evaluate_scores(baseline_scores,truth);p2=baseline['summary']
 with np.load(p['validation_ranks']) as f:
  if not all(np.array_equal(f[d+'_'+k],v) for d,b in baseline['per_positive'].items() for k,v in b.items()):raise ValueError('Identity P2 rank replay failed')
 if robust_value(p2)!=json.loads(p['validation_selection'].read_text())['selected']['robust_value']:raise ValueError('P2 objective mismatch')
 with np.load(p['f3_features.npz']) as f:br=torch.tensor(f['reactions'][vr],device=device);be=torch.tensor(f['proteins'][ve],device=device)
 f3=evaluate_scores(canonical_dot(row_unit(br),row_unit(be)),truth)['summary']
 atomic_json(OUT/'baseline_replay.json',dict(p2=p2,f3_fp64=f3,balanced_p2=robust_value(p2),identity_positive_ranks_exact=True))
 ir=torch.tensor(np.unique(np.linspace(0,len(r)-1,31,dtype=int)),device=device);ie=torch.tensor(np.unique(np.linspace(0,len(e)-1,193,dtype=int)),device=device)
 rows=[];selected=None
 for config in GRID:
  model=QuerySemanticScale(bank['bank'],bank['semantic_mean'],config['beta'],config['cap'])
  ra,diagnostics=model.encode_reactions(r,batch_size=64,return_diagnostics=True);ea=model.encode_enzymes(e)
  scores=canonical_dot(ra,ea);small_r=model.encode_reactions(r[ir],batch_size=7);small_e=model.encode_enzymes(e[ie]);single_r=model.encode_reactions(r[:1],1)
  sparse_scores=model.score_index(ra,model.encode_enzyme_index(e))
  checks=dict(reaction_subset=compare(ra[ir],small_r),reaction_singleton=compare(ra[:1],single_r),enzyme_subset=compare(ea[ie],small_e),subset_scores=compare(scores[ir][:,ie],canonical_dot(small_r,small_e)),singleton_scores=compare(scores[:1,:1],canonical_dot(single_r,model.encode_enzymes(e[:1]))),sparse_dense=compare(scores,sparse_scores),subset_index=compare(scores[ir][:,ie],model.score_index(small_r,model.encode_enzyme_index(e[ie]))))
  if config['beta'] is None:
   checks['identity_scores']=compare(scores,baseline_scores)
   if ra is not r or ea is not e or not checks['identity_scores']['exact']:raise ValueError('Explicit identity no longerdelegatesP2')
   result=baseline
  else:result=evaluate_scores(scores,truth)
  equal_mu=(diagnostics['multiplier']==1)
  unchanged=compare(scores[equal_mu],baseline_scores[equal_mu]) if bool(equal_mu.any()) else dict(exact=True,unequal_values=0,max_abs=0.)
  m=diagnostics['multiplier'];quantile=torch.quantile(m,torch.tensor([0.,.05,.25,.5,.75,.95,1.],device=device,dtype=torch.float64)).cpu().tolist()
  stats=dict(multiplier_quantiles=dict(zip(['min','p05','p25','median','p75','p95','max'],quantile)),mean_multiplier=float(m.mean()),unit_multiplier_queries=int(equal_mu.sum()),unit_multiplier_score_comparison=unchanged,
   capped_queries=int((m==config['cap']).sum()) if config['cap'] is not None else 0)
  if config['beta'] is not None:
   parent_mean=(r[:,512:].double()*bank['semantic_mean']).sum(1)
   transformed_mean=(ra[:,512:-1].double()*bank['semantic_mean']).sum(1)+ra[:,-1].double()
   stats['all_training_semantic_mean_preservation_max_abs']=float((transformed_mean-parent_mean).abs().max())
  row=dict(**config,summary=result['summary'],balanced_value=robust_value(result['summary']),aggregate_value=selection_value(result['summary']),metric_guards=eligible(result['summary'],p2,.005) and eligible(result['summary'],f3,.005),numeric_checks=checks,all_numerical_checks_exact=all(x['exact'] for x in checks.values()),diagnostics=stats)
  rows.append(row);save_result(OUT/(config['label']+'_ranks.npz'),result)
  np.savez_compressed(OUT/(config['label']+'_diagnostics.npz'),**{k:v.cpu().numpy() for k,v in diagnostics.items() if isinstance(v,torch.Tensor)})
  if row['metric_guards'] and (selected is None or row['balanced_value']>selected['balanced_value']):selected=row
  atomic_json(OUT/'progress.json',rows);print(json.dumps(dict(label=row['label'],balanced=row['balanced_value'],guards=row['metric_guards'],numeric=row['all_numerical_checks_exact'],m_median=quantile[3],elapsed=time.monotonic()-started)),flush=True)
  del sparse_scores,scores,ra,ea
 accepted=bool(selected is not None and selected['balanced_value']>robust_value(p2) and selected['all_numerical_checks_exact'])
 if selected is not None:atomic_json(OUT/'selected_state.json',dict(schema='query_semantic_scale_selected_v1',beta=selected['beta'],cap=selected['cap'],bank=record(OUT/'bank.pt'),protocol=record(OUT/'protocol.json'),parent_bundle=protocol['inputs']['parent_bundle'],dense_width=512,epsilon=1e-8))
 final=dict(schema='late_query_semantic_scale_validation_result_v1',protocol=record(OUT/'protocol.json'),selected=selected,candidates=rows,p2_balanced=robust_value(p2),qualifies_for_parent_review=accepted,external_test_used=False,external_prediction_performed=False,exploratory_after_previous_outcomes=True,independent_confirmation=False,elapsed_seconds=time.monotonic()-started,next_step='Stop for parent review. No external predictions or model/default replacement authorized by this result.')
 atomic_json(OUT/'complete.json',final);print(json.dumps(dict(selected=selected and selected['label'],balanced=selected and selected['balanced_value'],qualifies=accepted,seconds=final['elapsed_seconds'])),flush=True)
 for item in protocol['implementation_sources']:checked_artifact(item)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--device',default='cuda:0');args=parser.parse_args();prepare(args) if args.action=='prepare' else run(args)
if __name__=='__main__':main()
