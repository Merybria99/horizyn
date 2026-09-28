#!/usr/bin/env python3
"""Fixed phase-three positive-geometry objective control; train/validation only."""
from __future__ import annotations
import argparse,json,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from horizyn.generalization_positive_geometry import PositiveGeometryResidual,edge_marginals,positive_geometry_loss
from horizyn.generalization_retrieval import canonical_dot
from horizyn.semantic_anchors import row_unit
from generalization_full_graph import atomic_json,sha,validation_data,eligible,selection_value
from generalization_smooth_anchors import robust_value
from generalization_metrics import evaluate_scores


def identity(path):return dict(path=str(path.resolve()),sha256=sha(path))


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'registry.json').exists():raise ValueError('Use a fresh immutable run directory')
    started=time.monotonic();torch.manual_seed(args.seed);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    registry=dict(schema='phase3_positive_geometry_v1',exploratory_after_observed_phase2_failures=True,
        test_used=False,case1_used=False,p450_used=False,nitrilase_used=False,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        objective='Known-positive edge cosine loss +2*identity +moment_weight*moment; no unlisted pair explicitly negative',
        edge_weights='1/(R*degree(reaction)); enzyme marginal is sum of incident edge weights; repeated edges rejected',
        identity='Average of reaction-uniform and induced-enzyme-weighted cosine distance from native F3',
        moment='Average endpoints; each ||weighted_mean||^2 +d*||weighted_population_covariance-I/d||_F^2; collapsed=2,isotropic zero-mean=0',
        cap='Pointwise ||delta|| <=cap*||native_F3|| before normalization; caps .1/.3 imply cos displacement bound sqrt(1-cap^2)',
        inference='FP64 residual forward, norm cap and final normalization, FP32 stored endpoints; FP64 dot→FP32 scores',
        training='FP32, highest matmul precision, TF32 disabled; same full training graph each update',
        selection='Maximum equal-weight mean seen/unseen-reaction all-positive MRR across both directions; four aggregate/unseen drop guards .005 against F3_fp64; baseline eligible',
        optimizer=dict(name='AdamW',learning_rate=1e-4,weight_decay=.001,gradient_clip_norm=1.),
        sources={k:identity(v) for k,v in dict(feature_manifest=args.features/'manifest.json',catalog=args.features/'catalog.json',pairs=args.features/'pairs.npz',native_f3=args.features/'f3_features.npz',script=Path(__file__),module=ROOT/'horizyn/generalization_positive_geometry.py',parent_module=ROOT/'horizyn/generalization_residual.py',metrics=ROOT/'scripts/generalization_metrics.py',validation_helpers=ROOT/'scripts/generalization_full_graph.py',robust_selection=ROOT/'scripts/generalization_smooth_anchors.py',canonical_scoring=ROOT/'horizyn/generalization_retrieval.py',stable_normalization=ROOT/'horizyn/semantic_anchors.py',phase2_selection=args.phase2_selection).items()})
    atomic_json(args.output/'registry.json',registry)
    shutil.copyfile(__file__,args.output/'source.py');shutil.copyfile(ROOT/'horizyn/generalization_positive_geometry.py',args.output/'model_source.py')
    catalog=json.loads((args.features/'catalog.json').read_text())
    with np.load(args.features/'pairs.npz') as source:pairs={k:source[k] for k in source.files}
    if len(np.unique(pairs['train'],axis=0))!=len(pairs['train']):raise ValueError('Training graph must have unique known edges')
    tr,te=np.unique(pairs['train'][:,0]),np.unique(pairs['train'][:,1])
    train_reaction_ids=[catalog['reactions'][i] for i in tr]
    if train_reaction_ids!=catalog['train_reactions']:raise ValueError('Native training reaction order mismatch')
    vr,ve,truth=validation_data(catalog,pairs)
    with np.load(args.features/'f3_features.npz') as source:
        proteins=source['proteins'];reactions=source['reactions']
        train_e=torch.tensor(proteins[te],device=device);val_e=torch.tensor(proteins[ve],device=device)
        train_r=torch.tensor(source['train_reactions'],device=device);val_r=torch.tensor(reactions[vr],device=device)
    input_checks={}
    for name,tensor in dict(train_enzymes=train_e,train_reactions=train_r,validation_enzymes=val_e,validation_reactions=val_r).items():
        if tensor.ndim!=2 or tensor.shape[1]!=512 or tensor.dtype!=torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'{name}: expected finite native FP32 N by512 features')
        norms=tensor.double().norm(dim=1)
        if bool((norms<=1e-8).any()):raise ValueError(f'{name}: zero or numerically tiny native F3 endpoint')
        input_checks[name]=dict(shape=list(tensor.shape),min_norm=norms.min().item(),max_norm=norms.max().item())
    atomic_json(args.output/'input_checks.json',input_checks)
    rmap={int(v):i for i,v in enumerate(tr)};emap={int(v):i for i,v in enumerate(te)}
    r=torch.tensor([rmap[int(v)] for v in pairs['train'][:,0]],device=device)
    e=torch.tensor([emap[int(v)] for v in pairs['train'][:,1]],device=device)
    weights,enzyme_weight=edge_marginals(r,e,len(tr),len(te))
    base_r=F.normalize(train_r,dim=1);base_e=F.normalize(train_e,dim=1)
    baseline=evaluate_scores(canonical_dot(row_unit(val_r),row_unit(val_e)),truth)['summary']
    model_config=dict(dimension=train_e.shape[1],hidden=1024,scale=.2,cap=args.cap)
    model=PositiveGeometryResidual(**model_config).to(device)
    evaluator=PositiveGeometryResidual(**model_config).to(device).double().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
    records=[];best=None;events=[];last_terms={}
    phase2=json.loads(args.phase2_selection.read_text())['selected']
    for step in range(args.steps+1):
        if step%args.validate_every==0 or step==args.steps:
            evaluator.load_state_dict(model.state_dict());evaluator.eval()
            with torch.inference_mode():
                eq=evaluator.encode_reactions(val_r);ee=evaluator.encode_enzymes(val_e)
                if step==0 and (not torch.equal(eq,row_unit(val_r)) or not torch.equal(ee,row_unit(val_e))):
                    raise ValueError('Identity-initialized FP64 endpoints differ from F3_fp64 baseline')
                result=evaluate_scores(canonical_dot(eq,ee),truth)
            summary=result['summary'];row=dict(step=step,elapsed_seconds=time.monotonic()-started,
                robust_value=robust_value(summary),aggregate_value=selection_value(summary),
                eligible=eligible(summary,baseline,.005),validation=summary,
                train_losses={k:float(v) for k,v in last_terms.items()},
                peak_vram_gib=torch.cuda.max_memory_allocated(device)/2**30)
            records.append(row)
            if row['eligible'] and (best is None or row['robust_value']>best['robust_value']):
                best=row
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},model_config=model_config,
                    registry=registry,selected_validation=row),args.output/'selected.pt')
                np.savez(args.output/'selected_validation_ranks.npz',**{d+'_'+k:v for d,block in result['per_positive'].items() for k,v in block.items()})
            atomic_json(args.output/'validation.json',dict(baseline_fp64=baseline,records=records,selected=best))
            print(json.dumps({k:v for k,v in row.items() if k!='validation'}),flush=True)
            del result,eq,ee
        if step==args.steps:break
        model.train();optimizer.zero_grad(set_to_none=True)
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
        ze=model.encode_enzymes(train_e);zr=model.encode_reactions(train_r)
        loss,terms=positive_geometry_loss(zr,ze,r,e,weights,enzyme_weight,base_r,base_e,args.moment_weight,2.)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite positive geometry loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();end.record()
        events.append((begin,end));last_terms=dict(total=loss.detach(),**terms)
        del loss,terms,ze,zr
    torch.cuda.synchronize(device)
    training_seconds=sum(begin.elapsed_time(end) for begin,end in events)/1000
    report=dict(selected=best,baseline_fp64=baseline,phase2_reference=dict(robust_value=phase2['robust_value'],aggregate_value=phase2['aggregate_value']),
        dominated_by_frozen_phase2=best['robust_value']<=phase2['robust_value'] and best['aggregate_value']<=phase2['aggregate_value'],
        test_used=False,external_evaluation_performed=False,elapsed_seconds=time.monotonic()-started,
        gpu_training_seconds=training_seconds,training_steps_per_second=args.steps/training_seconds if training_seconds else None,
        known_edges_per_training_second=args.steps*len(r)/training_seconds if training_seconds else None,
        peak_vram_gib=torch.cuda.max_memory_allocated(device)/2**30,train_reactions=len(tr),train_enzymes=len(te),train_edges=len(r),
        selected_checkpoint=identity(args.output/'selected.pt'),torch_version=torch.__version__,gpu=torch.cuda.get_device_name(device))
    atomic_json(args.output/'complete.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected','baseline_fp64')}),flush=True)


def main():
    root=ROOT/'runs/generalization_20260919_2251';p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=root/'features');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase2-selection',type=Path,default=root/'phase2/composition_fp64/selection.json')
    p.add_argument('--cap',type=float,choices=[.1,.3],required=True);p.add_argument('--moment-weight',type=float,choices=[.1,1.,10.],required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--validate-every',type=int,default=50)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',required=True)
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),external_evaluation_performed=False))
        raise


if __name__=='__main__':main()
