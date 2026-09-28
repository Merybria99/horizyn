#!/usr/bin/env python3
"""Fixed nine-condition phase-four hybrid anchor geometry screen."""
from __future__ import annotations
import argparse,itertools,json,math,shutil,sys,time,traceback
from pathlib import Path
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from horizyn.semantic_hybrid import HybridAnchorDualEncoder
from horizyn.semantic_anchors import row_unit,enzyme_anchor_features
from horizyn.generalization_retrieval import canonical_dot
from generalization_full_graph import atomic_json,sha,validation_data,eligible,selection_value
from generalization_smooth_anchors import robust_value,support_diagnostic
from generalization_metrics import evaluate_scores


def identity(path):return dict(path=str(path.resolve()),sha256=sha(path))


def inputs(args):
    return dict(feature_manifest=args.features/'manifest.json',catalog=args.features/'catalog.json',pairs=args.features/'pairs.npz',
        native_f3=args.features/'f3_features.npz',protein_means=args.features/'protein_mean.h5',raw_reactions=args.features/'reaction_features.npz',
        smooth_state=args.smooth/'selected.pt',dictionary=args.dictionary,
        density_endpoints=args.density/'selected_validation_features.npz',density_catalog=args.density/'selected_validation_catalog.json',
        density_receipt=args.density/'selected_validation_receipt.json',phase2_selection=args.phase2/'selection.json',
        phase2_endpoints=args.phase2/'selected_validation_features.pt',phase2_ranks=args.phase2/'density_smooth_alpha0.25_ranks.npz',
        phase2_freeze=args.phase2.parent/'frozen_recipe.json',script=Path(__file__),module=ROOT/'horizyn/semantic_hybrid.py',
        smooth_module=ROOT/'horizyn/semantic_smooth.py',anchor_module=ROOT/'horizyn/semantic_anchors.py',
        canonical_scoring=ROOT/'horizyn/generalization_retrieval.py',metrics=ROOT/'scripts/generalization_metrics.py',
        validation_helpers=ROOT/'scripts/generalization_full_graph.py',selection_helpers=ROOT/'scripts/generalization_smooth_anchors.py')


def preregister(args):
    import datetime
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'protocol.json').exists():raise ValueError('Protocol already exists')
    protocol=dict(schema='phase4_hybrid_anchor_protocol_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        exploratory_after_observed_phase1_and_phase2_external_failures=True,external_inputs_used=False,aminotransferase_labels_or_scores_used=False,
        grid=[dict(eta_enzyme=e,eta_reaction=r) for e,r in itertools.product([0.,.5,1.],repeat=2)],
        geometry='g_eta(x)=concat(sqrt(1-eta)*raw_centered_unit(x),sqrt(eta)*native_F3_unit(x)); no additional norm; FP32 sqrt-weight blocks, actual concatenation with FP64 dot thenFP32 cosine',
        raw_centering='Unchanged phase2 train-only rawunit means, centered then unit; reaction blocks equal normalized concatenation and missing masks unchanged',
        native_geometry='Once row-normalize nativeF3 with FP64 norm thenFP32; no center; dictionary onlytrainuniqueenzymes andcompacttrainreactions',
        enzyme_evidence='Exact top32 trainingprotein neighbors at temperature.03; known-training-adjacency maxaggregation perreaction thenunit',
        reaction_evidence='Fullsupport exponential responses to all6977trainingreactions,temperature.03, unchanged positivefloor andunitnorm',
        final_endpoint='concat(sqrt(.75)*FROZENphase2density_endpoint,sqrt(.25)*hybrid_evidence)',
        zero_geometry='Delegate originalrawsemantic paths; require frozenphase2 exactendpoint andpositive-rank reconstruction beforeselection',
        selection='Max equal mean seen/unseenreaction allpositiveMRR bothdirections; four aggregate/unseen direction guards within.005 ofF3_fp64; retainfrozenphase2 without strict balancedgain',
        diagnostic_only='Puresemantic metrics/support andneighboroverlap; noalternative alpha,temperatures,k,centers selected',
        sources={k:identity(v) for k,v in inputs(args).items()},arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='preregister_only'})
    atomic_json(args.output/'protocol.json',protocol);print(json.dumps(dict(protocol=str(args.output/'protocol.json'),sha256=sha(args.output/'protocol.json'))),flush=True)


def run(args):
    start=time.monotonic();protocol=json.loads((args.output/'protocol.json').read_text())
    if (args.output/'registry.json').exists():raise ValueError('Choose fresh run; protocol and finished results are immutable')
    for name,path in inputs(args).items():
        if identity(path)!=protocol['sources'][name]:raise ValueError('Protocol/source mismatch: '+name)
    atomic_json(args.output/'registry.json',dict(protocol=identity(args.output/'protocol.json'),started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    shutil.copyfile(__file__,args.output/'source.py');shutil.copyfile(ROOT/'horizyn/semantic_hybrid.py',args.output/'model_source.py')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    catalog=json.loads((args.features/'catalog.json').read_text())
    with np.load(args.features/'pairs.npz') as f:pairs={k:f[k] for k in f.files}
    vr,ve,truth=validation_data(catalog,pairs);tr,te=np.unique(pairs['train'][:,0]),np.unique(pairs['train'][:,1])
    with np.load(args.features/'f3_features.npz') as f:
        native_train_e=row_unit(torch.tensor(f['proteins'][te],device=device));native_train_r=row_unit(torch.tensor(f['train_reactions'],device=device))
        native_val_e=torch.tensor(f['proteins'][ve],device=device);native_val_r=torch.tensor(f['reactions'][vr],device=device)
    if catalog['train_reactions']!=[catalog['reactions'][i] for i in tr]:raise ValueError('Compactnative trainreaction order mismatch')
    dictionary=torch.load(args.dictionary,map_location=device,weights_only=False);smooth=torch.load(args.smooth/'selected.pt',map_location=device,weights_only=False)
    if dictionary['feature_manifest_sha256']!=sha(args.features/'manifest.json') or smooth['feature_manifest_sha256']!=dictionary['feature_manifest_sha256']:
        raise ValueError('Dictionary/source manifest disagreement')
    if dictionary['train_protein_ids']!=[catalog['proteins'][i] for i in te] or dictionary['train_reaction_ids']!=catalog['train_reactions']:
        raise ValueError('Native/raw training dictionary ID mismatch')
    config=smooth['config']
    if (config['protein_neighbors'],config['enzyme_temperature'],config['reaction_temperature'],config['kernel'],config['reaction_neighbors'])!=(32,.03,.03,'exponential',None):
        raise ValueError('Unexpected frozen smooth geometry')
    with h5py.File(args.features/'protein_mean.h5','r') as f:
        if list(f['ids'].asstr()[:])!=catalog['proteins'] or not f['complete'][:].all():raise ValueError('Rawprotein IDs/incomplete cache')
        means=torch.tensor(f['vectors'][ve],device=device)
    with np.load(args.features/'reaction_features.npz') as f:
        blocks={k:torch.tensor(f[k][vr],device=device) for k in dictionary['modalities']}
        masks={k:torch.tensor(f[k+'_mask'][vr],device=device) for k in dictionary['modalities']}
    dc=json.loads((args.density/'selected_validation_catalog.json').read_text())
    if dc['proteins']!=catalog['validation_candidates'] or dc['reactions']!=catalog['validation_reactions']:raise ValueError('Density endpoint ID order mismatch')
    with np.load(args.density/'selected_validation_features.npz') as f:de=torch.tensor(f['proteins'],device=device);dr=torch.tensor(f['reactions'],device=device)
    phase2=json.loads((args.phase2/'selection.json').read_text())['selected']
    previous=torch.load(args.phase2/'selected_validation_features.pt',map_location=device,weights_only=False)
    if previous['enzyme_ids']!=catalog['validation_candidates'] or previous['reaction_ids']!=catalog['validation_reactions']:raise ValueError('Phase2 endpoint IDs mismatch')
    baseline=evaluate_scores(canonical_dot(row_unit(native_val_r),row_unit(native_val_e)),truth)['summary']
    efeatures={};rfeatures={};neighbors={};diagnostics=[]
    for eta in [0.,.5,1.]:
        model=HybridAnchorDualEncoder(dictionary,smooth['train_reactions'],config,native_train_e,native_train_r,eta,eta).to(device)
        values,indices=model.enzyme_neighbors(native_val_e,means,args.batch_size)
        efeatures[eta]=enzyme_anchor_features(values,indices,model.adjacency,len(tr),.03)
        rfeatures[eta]=model.hybrid_reaction_features(native_val_r,blocks,masks)
        neighbors[eta]=indices
        np.savez(args.output/f'neighbors_eta{eta:g}.npz',values=values.cpu().numpy(),indices=indices.cpu().numpy())
        overlap=(indices[:,:,None]==neighbors[0][:,None,:]).any(2).float().mean().item()
        diagnostics.append(dict(eta=eta,mean_top32_overlap_with_raw=overlap,nearest_protein_cosine_quantiles={str(q):torch.quantile(values[:,0],q).item() for q in [0.,.05,.25,.5,.75,.95,1.]}))
        print(json.dumps(dict(prepared_eta=eta,elapsed_seconds=time.monotonic()-start,neighbor_overlap_with_raw=overlap)),flush=True)
    rows=[];best=None
    for settings in protocol['grid']:
        ee,rr=settings['eta_enzyme'],settings['eta_reaction'];semantic_e,semantic_r=efeatures[ee],rfeatures[rr]
        e=torch.cat((math.sqrt(.75)*de,.5*semantic_e),1);r=torch.cat((math.sqrt(.75)*dr,.5*semantic_r),1)
        result=evaluate_scores(canonical_dot(r,e),truth);summary=result['summary'];name=f'etaE{ee:g}_etaR{rr:g}'
        if ee==rr==0:
            if not torch.equal(e,previous['enzymes']) or not torch.equal(r,previous['reactions']):raise ValueError('Rawonly endpoints do not reproduce frozenphase2')
            with np.load(args.phase2/'density_smooth_alpha0.25_ranks.npz') as old:
                if not all(np.array_equal(old[d+'_'+k],v) for d,b in result['per_positive'].items() for k,v in b.items()):raise ValueError('Rawonly ranks do not reproduce frozenphase2')
            atomic_json(args.output/'raw_only_reproduction.json',dict(endpoint_exact=True,positive_ranks_exact=True,phase2_selection=protocol['sources']['phase2_selection']))
        pure_scores=canonical_dot(semantic_r,semantic_e);pure=evaluate_scores(pure_scores,truth)
        row=dict(id=name,**settings,alpha=.25,robust_value=robust_value(summary),aggregate_value=selection_value(summary),eligible=eligible(summary,baseline,.005),summary=summary,
            puresemantic_diagnostic=pure['summary'],semantic_support=support_diagnostic(pure_scores,truth))
        rows.append(row);np.savez(args.output/(name+'_ranks.npz'),**{d+'_'+k:v for d,b in result['per_positive'].items() for k,v in b.items()})
        if row['eligible'] and (best is None or row['robust_value']>best['robust_value']):
            best=row
            state=dict(eta_enzyme=ee,eta_reaction=rr,alpha=.25,native_train_enzymes=native_train_e.cpu(),native_train_reactions=native_train_r.cpu(),
                dictionary=identity(args.dictionary),smooth_state=identity(args.smooth/'selected.pt'),feature_manifest_sha256=dictionary['feature_manifest_sha256'],
                protocol=identity(args.output/'protocol.json'),train_protein_ids=dictionary['train_protein_ids'],train_reaction_ids=dictionary['train_reaction_ids'],selected_validation=row)
            torch.save(state,args.output/'selected.pt')
            torch.save(dict(reactions=r.cpu(),enzymes=e.cpu(),reaction_ids=catalog['validation_reactions'],enzyme_ids=catalog['validation_candidates']),args.output/'selected_validation_features.pt')
        atomic_json(args.output/'validation.json',dict(baseline_fp64=baseline,records=rows,selected=best))
        print(json.dumps({k:v for k,v in row.items() if k not in ['summary','puresemantic_diagnostic','semantic_support']}),flush=True)
    strict_gain=best['robust_value']>rows[0]['robust_value']
    report=dict(selected=best,phase2_reference=phase2,raw_reproduction=rows[0],strict_gain_over_phase2=strict_gain,
        decision='Freeze new selected geometry before any external evaluation' if strict_gain else 'Retain frozen phase2; no external evaluation of alternativegeometry',
        protocol=identity(args.output/'protocol.json'),selected_checkpoint=identity(args.output/'selected.pt'),neighbor_diagnostics=diagnostics,
        elapsed_seconds=time.monotonic()-start,peak_vram_gib=torch.cuda.max_memory_allocated(device)/2**30,
        external_evaluation_performed=False,aminotransferase_labels_or_scores_used=False)
    atomic_json(args.output/'complete.json',report)
    print(json.dumps(dict(selected=best['id'],robust=best['robust_value'],aggregate=best['aggregate_value'],strict_gain=strict_gain,elapsed=report['elapsed_seconds'])),flush=True)


def main():
    root=ROOT/'runs/generalization_20260919_2251';p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=root/'features');p.add_argument('--dictionary',type=Path,default=root/'models/reaction_smi/seed42/anchors.pt')
    p.add_argument('--smooth',type=Path,default=root/'phase2/smooth_anchors');p.add_argument('--density',type=Path,default=root/'phase2/density_gate_fp64')
    p.add_argument('--phase2',type=Path,default=root/'phase2/composition_fp64');p.add_argument('--output',type=Path,default=root/'phase4/hybrid_anchors')
    p.add_argument('--device',default='cuda:2');p.add_argument('--batch-size',type=int,default=512);p.add_argument('--preregister-only',action='store_true')
    args=p.parse_args()
    try:
        if args.preregister_only:preregister(args)
        else:
            with torch.inference_mode():run(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),external_evaluation_performed=False))
        raise


if __name__=='__main__':main()
