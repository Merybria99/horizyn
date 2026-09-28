#!/usr/bin/env python3
"""One bounded validation-only study of centered, unsupported-query affinity."""
from __future__ import annotations
import argparse,datetime,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from horizyn.generalization_centered_affinity import CenteredAffinity
from horizyn.generalization_retrieval import canonical_dot,checked_artifact,sha256
from horizyn.generalization_density import nearest_support
from horizyn.semantic_anchors import row_unit,reaction_features
from generalization_full_graph import validation_data,eligible,selection_value,atomic_json
from generalization_smooth_anchors import robust_value
from generalization_composed_calibration import save_result
from generalization_metrics import evaluate_scores
RUN=ROOT/'runs/generalization_20260919_2251'
OUT=RUN/'post_evaluation_centered_affinity'
GAMMAS=[0.,.25,.5,1.,2.]
GATES=['one_minus_f3_gate','raw_support_ramp','ungated_control']
DIRECTIONS=['reaction_to_enzyme','enzyme_to_reaction']

def record(p):return dict(path=str(Path(p).resolve()),sha256=sha256(p))

def sources():
 f=RUN/'features';previous=RUN/'phase2/composition_fp64';density=RUN/'phase2/density_gate_fp64'
 paths={k:f/k for k in ['catalog.json','pairs.npz','manifest.json','f3_features.npz','reaction_features.npz']}
 paths.update(p2_bundle=RUN/'phase2/models/reaction_smi/seed42/bundle.json',p2_validation=previous/'selected_validation_features.pt',p2_ranks=previous/'density_smooth_alpha0.25_ranks.npz',p2_selection=previous/'selection.json',
  means=RUN/'post_evaluation_calibration/train_means.pt',means_protocol=RUN/'post_evaluation_calibration/protocol.json',raw_support=density/'validation_support.npz',raw_registry=density/'registry.json',raw_gate=density/'selected_gate.pt',
  script=Path(__file__),module=ROOT/'horizyn/generalization_centered_affinity.py',tests=ROOT/'tests/unit/test_generalization_centered_affinity.py')
 paths.update({f'old_global_{gamma:g}':RUN/'post_evaluation_calibration'/f'reaction_to_enzyme_uniform_proteins_gamma{gamma:g}.npz' for gamma in GAMMAS})
 return paths

def verify_protocol(protocol):
 for item in protocol['implementation_sources']:checked_artifact(item)
 for key,path in sources().items():
  if record(path)!=protocol['inputs'][key]:raise ValueError('Protocol source changed: '+key)
 for row in protocol['parent_freezes']:checked_artifact(row)


def prepare():
 if (OUT/'protocol.json').exists():raise ValueError('Existing immutable protocol')
 OUT.mkdir(parents=True,exist_ok=True)
 paths=sources();items={k:record(v) for k,v in paths.items()}
 old=json.loads((RUN/'post_evaluation_calibration/transfer/frozen_recipe.json').read_text());implementation={x['path']:x for x in old['implementation_sources']}
 for path in [paths['script'],paths['module'],paths['tests'],ROOT/'scripts/generalization_composed_calibration.py']:
  r=record(path);implementation[r['path']]=r
 for x in implementation.values():checked_artifact(x)
 with np.load(paths['raw_support']) as f:loo=torch.tensor(f['training_raw_reaction'])
 if loo.ndim!=1 or not torch.isfinite(loo).all():raise ValueError('Invalid frozen training LOO support')
 low,high=(float(torch.quantile(loo,q)) for q in (.05,.25))
 if not low<high:raise ValueError('Degenerate raw quantiles; no adaptive substitute allowed')
 p=dict(schema='late_exploratory_centered_affinity_validation_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  inputs=items,implementation_sources=list(implementation.values()),parent_freezes=[record(RUN/'phase2/frozen_recipe.json'),record(RUN/'phase4/frozen_recipe.json'),record(RUN/'post_evaluation_calibration/transfer/frozen_recipe.json')],
  exploratory_after_previous_external_results=True,independent_confirmation=False,external_or_test_scores_used=False,selection_scope='Original Reaction-Sim train/validation only; no external or official test input arrays opened by this script.',
  training_statistics='Reuse fixed uniform actual-training endpoint means. Raw q05/q25 computed on CPU by torch.quantile over already saved FP32 training-only LOO support vectors; no new encoders, centers, feature extraction or training.',
  raw_training_quantiles=dict(q05=low,q25=high,count=len(loo)),gates=GATES,gammas=GAMMAS,grid_size=15,
  formula='b(e)=FP32(sum_FP64((FP64(e)-mean_train_E)*mean_train_R)); Eaug=[e,b(e)], Raug=[r,FP32(-gamma*FP64(g(r)))]. No renormalization. Gamma0 returns original endpoints/dimension exactly.',
  gate_definitions={'one_minus_f3_gate':'1 minus exact frozen P2 density reaction gate; cap1 q25-to-q95 nativeF3 support','raw_support_ramp':'clip((trainRawLOOq25-nearestTrainRawCos)/(q25-q05),0,1), unchanged frozen raw reaction normalization/centers','ungated_control':'all ones'},
  numeric_contract='Actual augmented FP32 endpoint tensors; canonical FP64 dot then FP32 score rounding. Fixed rowwise FP64 bias reduction, independent source-only gates; no subtraction from rounded parent scores.',
  selection='Maximum equal-weight seen/unseen reaction all-positive MRR across both directions. Aggregate/all and unseen each direction within .005 of BOTH F3_fp64 and P2. Strict objective gain over P2 required. Fixed gate/gamma enumeration with zero first; ties keep first. No fallback search.',
  acceptance='Every candidate full/subset/singleton endpoint/score exactness; gamma0 exact P2 positive ranks; fresh P2 query encoding exact saved validation; recomputed raw support exact saved support; no external predictions pending parent review.',
  controls='Ungated E2R is a query-constant shift, so changed ranks are rounding effects. Ungated R2E compared with the old uncentered global gammaE validation ranks to expose centering-only tie shifts.',
  budget_minutes=25,stop_rule='Stop after this fixed validation grid regardless of outcome; no model/default/pool changes.')
 atomic_json(OUT/'protocol.json',p);verify_protocol(p);print(json.dumps(record(OUT/'protocol.json')),flush=True)

@torch.inference_mode()
def run(args):
 started=time.monotonic();p=json.loads((OUT/'protocol.json').read_text());verify_protocol(p)
 if (OUT/'registry.json').exists():raise ValueError('Existing screen output')
 atomic_json(OUT/'registry.json',dict(protocol=record(OUT/'protocol.json'),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 device=torch.device(args.device);torch.cuda.set_device(device)
 from horizyn.generalization_phase2 import ComposedPhase2Encoder
 paths=sources();catalog=json.loads(paths['catalog.json'].read_text())
 with np.load(paths['pairs.npz']) as f:pairs={k:f[k] for k in f.files}
 tr=np.unique(pairs['train'][:,0]);vr,ve,truth=validation_data(catalog,pairs)
 if catalog['train_reactions']!=[catalog['reactions'][i] for i in tr]:raise ValueError('Training reaction selector mismatch')
 model,spec=ComposedPhase2Encoder.from_bundle(paths['p2_bundle'],device)
 with np.load(paths['f3_features.npz']) as f:be=torch.tensor(f['proteins'][ve],device=device);br=torch.tensor(f['reactions'][vr],device=device)
 with np.load(paths['reaction_features.npz']) as f:
  trainblocks={k:torch.tensor(f[k][tr],device=device) for k in model.modalities};trainmasks={k:torch.tensor(f[k+'_mask'][tr],device=device) for k in model.modalities}
  blocks={k:torch.tensor(f[k][vr],device=device) for k in model.modalities};masks={k:torch.tensor(f[k+'_mask'][vr],device=device) for k in model.modalities}
 old=torch.load(paths['p2_validation'],map_location=device,weights_only=False)
 if old['reaction_ids']!=catalog['validation_reactions'] or old['enzyme_ids']!=catalog['validation_candidates']:raise ValueError('P2 validation order mismatch')
 r,e=old['reactions'],old['enzymes'];fresh,diag=model.encode_reactions(br,blocks,masks,512,return_diagnostics=True)
 if not torch.equal(fresh,r):raise ValueError('Fresh P2 reaction endpoints do not replay saved endpoints')
 if model.density.selected_gate!={'label':'f3_both_q25_cap1','space':'f3','endpoints':'both','lower_quantile':.25,'upper_quantile':.95,'cap':1.}:raise ValueError('Unexpected parent gate')
 raw= reaction_features(blocks,model.density.reaction_centers,masks,model.modalities)
 anchors=reaction_features(trainblocks,model.density.reaction_centers,trainmasks,model.modalities)
 raw_support=nearest_support(raw,anchors,512)
 with np.load(paths['raw_support']) as f:cached=torch.tensor(f['raw_reaction'],device=device)
 if not torch.equal(raw_support,cached):raise ValueError('Recomputed raw support differs from frozen validation support')
 bounds=p['raw_training_quantiles'];gates=dict(one_minus_f3_gate=1.-diag['gate_scale'],raw_support_ramp=((bounds['q25']-raw_support)/(bounds['q25']-bounds['q05'])).clamp(0,1),ungated_control=torch.ones_like(raw_support))
 ir=torch.tensor(np.unique(np.linspace(0,len(r)-1,31,dtype=int)),device=device);ie=torch.tensor(np.unique(np.linspace(0,len(e)-1,193,dtype=int)),device=device)
 _,small_diag=model.encode_reactions(br[ir],{k:v[ir] for k,v in blocks.items()},{k:v[ir] for k,v in masks.items()},7,return_diagnostics=True)
 support_parity=dict(f3=torch.equal(diag['gate_scale'][ir],small_diag['gate_scale']),raw=torch.equal(raw_support[ir],nearest_support(raw[ir],anchors,7)),raw_singleton=torch.equal(raw_support[:1],nearest_support(raw[:1],anchors,1)))
 if not all(support_parity.values()):raise ValueError('Support gate batch invariance failed')
 np.savez_compressed(OUT/'support.npz',**{k:v.cpu().numpy() for k,v in gates.items()},raw_support=raw_support.cpu().numpy(),f3_support=diag['support_cosine'].cpu().numpy(),reaction_seen=truth['reaction_seen'])
 means=torch.load(paths['means'],map_location=device,weights_only=False);mr,me=means['reaction_mean'],means['enzyme_means']['uniform_proteins']
 if not np.array_equal(means['train_reaction_indices'],tr) or not np.array_equal(means['train_enzyme_indices'],np.unique(pairs['train'][:,1])):raise ValueError('Training mean selector mismatch')
 baseline=evaluate_scores(canonical_dot(r,e),truth);p2=baseline['summary']
 with np.load(paths['p2_ranks']) as f:
  if not all(np.array_equal(f[d+'_'+k],v) for d,b in baseline['per_positive'].items() for k,v in b.items()):raise ValueError('P2 baseline rank mismatch')
 if robust_value(p2)!=json.loads(paths['p2_selection'].read_text())['selected']['robust_value']:raise ValueError('P2 objective mismatch')
 f3=evaluate_scores(canonical_dot(row_unit(br),row_unit(be)),truth)['summary']
 atomic_json(OUT/'baseline_replay.json',dict(p2=p2,f3_fp64=f3,balanced_p2=robust_value(p2),saved_parent_ranks_exact=True,fresh_parent_reactions_exact=True,support_parity=support_parity))
 rows=[];noops=[]
 for name in GATES:
  for gamma in GAMMAS:
   adapter=CenteredAffinity(mr,me,gamma);ra,ea=adapter.encode_reactions(r,gates[name]),adapter.encode_enzymes(e);scores=canonical_dot(ra,ea)
   parity=dict(reaction_subset=torch.equal(ra[ir],adapter.encode_reactions(r[ir],gates[name][ir])),enzyme_subset=torch.equal(ea[ie],adapter.encode_enzymes(e[ie])),subset_scores=torch.equal(scores[ir][:,ie],canonical_dot(adapter.encode_reactions(r[ir],gates[name][ir]),adapter.encode_enzymes(e[ie]))),singleton_scores=torch.equal(scores[:1,:1],canonical_dot(adapter.encode_reactions(r[:1],gates[name][:1]),adapter.encode_enzymes(e[:1]))))
   if not all(parity.values()):raise ValueError('Actual augmented numerical invariance failed')
   result=baseline if gamma==0 else evaluate_scores(scores,truth)
   if gamma==0 and (ra is not r or ea is not e or not torch.equal(scores,canonical_dot(r,e))):raise ValueError('Gamma0 changed baseline')
   row=dict(gate=name,gamma=gamma,summary=result['summary'],balanced_value=robust_value(result['summary']),aggregate_value=selection_value(result['summary']),eligible=eligible(result['summary'],f3,.005) and eligible(result['summary'],p2,.005),parity=parity)
   rows.append(row);save_result(OUT/f'{name}_gamma{gamma:g}_ranks.npz',result)
   if name=='ungated_control':
    with np.load(paths[f'old_global_{gamma:g}']) as f:oldrank=f['positive__reaction_to_enzyme__rank']
    changes={d:int(np.count_nonzero(result['per_positive'][d]['rank']!=baseline['per_positive'][d]['rank'])) for d in DIRECTIONS}
    noops.append(dict(gamma=gamma,positive_rank_changes_vs_p2=changes,r2e_changes_vs_uncentered_global=int(np.count_nonzero(oldrank!=result['per_positive']['reaction_to_enzyme']['rank'])),e2r_interpretation='Ungated shift is query-constant for E2R; rank differences are numerical only.'))
   atomic_json(OUT/'progress.json',rows);print(json.dumps(dict(gate=name,gamma=gamma,balanced=row['balanced_value'],eligible=row['eligible'],elapsed=time.monotonic()-started)),flush=True)
 eligible_rows=[x for x in rows if x['eligible']];selected=max(eligible_rows,key=lambda x:x['balanced_value']) if eligible_rows else None
 accepted=selected is not None and selected['balanced_value']>robust_value(p2)
 if selected is not None:
  torch.save(dict(schema='late_centered_affinity_selected_v1',gamma=selected['gamma'],gate=selected['gate'],reaction_mean=mr.cpu(),enzyme_mean=me.cpu(),protocol=record(OUT/'protocol.json'),parent_bundle=record(paths['p2_bundle']),training_quantiles=bounds),OUT/'selected.pt')
 final=dict(schema='late_centered_affinity_validation_result_v1',protocol=record(OUT/'protocol.json'),selected=selected,candidates=rows,qualifies_for_parent_review=accepted,p2_balanced=robust_value(p2),noop_controls=noops,support_parity=support_parity,external_or_test_used=False,exploratory_after_external_exposure=True,independent_confirmation=False,elapsed_seconds=time.monotonic()-started,next_step='Stop; parent review required before any external prediction. No automatic deployment or expanded search.')
 atomic_json(OUT/'complete.json',final);verify_protocol(p);print(json.dumps(dict(selected=selected and {k:selected[k] for k in ['gate','gamma','balanced_value']},qualifies=accepted,elapsed=final['elapsed_seconds'])),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run']);p.add_argument('--device',default='cuda:0');a=p.parse_args();prepare() if a.action=='prepare' else run(a)
if __name__=='__main__':main()
