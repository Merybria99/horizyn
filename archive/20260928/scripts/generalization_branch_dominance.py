#!/usr/bin/env python3
"""Posthoc score-branch dispersion audit; no labels, refitting or new predictions."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/generalization_20260919_2251'
OUT=RUN/'posthoc_branch_dominance'
PANELS=('case1','p450','nitrilase','aminotransferase','reaction_smi')
VARIANTS={'mixture':('seed42','primary',.25),'neural':('density_only','density_only',0.),'semantic':('smooth_only','smooth_only',1.)}
FREEZE_SHA='7353203fce81f1070649dbf535eac836602f6c4af576e7bbf8968b1e426fde2f'


def identity(path):
    path=Path(path).resolve();digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(4*1024**2),b''):digest.update(block)
    return dict(path=str(path),sha256=digest.hexdigest(),bytes=path.stat().st_size)


def checked(record,provenance):
    path=Path(record['path']).resolve()
    if str(path) not in provenance:
        provenance[str(path)]=identity(path)
    if provenance[str(path)]['sha256']!=record['sha256']:raise ValueError('Changed score/source artifact: '+str(path))
    return path


def same(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve() and a['sha256']==b['sha256']


def pearson(x,y):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    a=x-x.mean();b=y-y.mean();den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.clip(np.dot(a,b)/den,-1,1)) if den>0 else None


def ranking(x):
    """Average ranks for ties; stable order kept only as a sensitivity diagnostic."""
    x=np.asarray(x);order=np.argsort(-x,kind='stable');sorted_x=x[order]
    starts=np.r_[0,np.flatnonzero(sorted_x[1:]!=sorted_x[:-1])+1]
    stops=np.r_[starts[1:],len(x)];lengths=stops-starts
    avg=np.empty(len(x),np.float64);avg[order]=np.repeat((starts+stops+1)/2,lengths)
    return avg,order,float(lengths[lengths>1].sum()/len(x))


def top_probability(x,k):
    x=np.asarray(x);k=min(k,len(x));threshold=np.partition(x,len(x)-k)[len(x)-k]
    above=x>threshold;tied=x==threshold
    return above.astype(np.float64)+tied*(k-int(above.sum()))/int(tied.sum())


def summarize_values(values):
    finite=[float(x) for x in values if x is not None and np.isfinite(x)]
    if not finite:return dict(defined_queries=0,total_queries=len(values),mean=None,p05=None,p25=None,median=None,p75=None,p95=None)
    q=np.quantile(finite,[.05,.25,.5,.75,.95])
    return dict(defined_queries=len(finite),total_queries=len(values),mean=float(np.mean(finite)),**dict(zip(('p05','p25','median','p75','p95'),map(float,q))))


def row_stats(mixture,neural,semantic):
    m=np.asarray(mixture,np.float64);d=np.asarray(neural,np.float64);s=np.asarray(semantic,np.float64)
    if m.ndim!=1 or m.shape!=d.shape or m.shape!=s.shape or not len(m) or not np.isfinite(np.stack((m,d,s))).all():raise ValueError('Finite aligned branch rows required')
    std={k:float(x.std(ddof=0)) for k,x in [('mixture',m),('neural',d),('semantic',s)]}
    a,b=.75*std['neural'],.25*std['semantic']
    covariance=float(np.mean((d-d.mean())*(s-s.mean())))
    ideal=.75*d+.25*s
    ideal32=ideal.astype(np.float32)
    result=dict(candidate_count=len(m),**{k+'_std':v for k,v in std.items()},
        neural_weighted_std=a,semantic_weighted_std=b,
        weighted_neural_to_semantic_std_ratio=a/b if b>0 else None,
        neural_nonconstant_semantic_constant=bool(a>0 and b==0),both_branches_constant=bool(a==0 and b==0),
        neural_weighted_std_share=a/(a+b) if a+b>0 else None,
        neural_semantic_score_pearson=pearson(d,s),mixture_neural_score_pearson=pearson(m,d),mixture_semantic_score_pearson=pearson(m,s),
        neural_weighted_variance=a*a,semantic_weighted_variance=b*b,weighted_covariance_term=2*.75*.25*covariance,
        ideal_mixture_variance=float(ideal.var(ddof=0)),actual_mixture_variance=std['mixture']**2,
        ideal_score_residual_max_abs=float(np.abs(m-ideal).max()),ideal_score_residual_rms=float(np.sqrt(np.mean((m-ideal)**2))),
        ideal_once_rounded_score_unequal_fraction=float(np.mean(m.astype(np.float32)!=ideal32)))
    assert abs(result['ideal_mixture_variance']-(a*a+b*b+2*.75*.25*covariance))<1e-12
    ranks={};orders={}
    for key,values in [('mixture',m),('neural',d),('semantic',s),('ideal',ideal32)]:
        ranks[key],orders[key],tie_fraction=ranking(values)
        result[key+'_tied_candidate_fraction']=tie_fraction
        if key!='ideal':
            q=np.quantile(values,[0,.25,.5,.75,1])
            result[key+'_mean']=float(values.mean());result[key+'_iqr']=float(q[3]-q[1]);result[key+'_range']=float(q[4]-q[0])
    for key in ('neural','semantic','ideal'):
        result['mixture_'+key+'_spearman']=pearson(ranks['mixture'],ranks[key])
        result['mixture_'+key+'_stable_order_position_disagreement']=float(np.mean(orders['mixture']!=orders[key]))
    for cutoff in (5,25,100):
        k=min(cutoff,len(m));p=top_probability(m,k)
        result[f'top{cutoff}_effective_k']=k
        for key,values in [('neural',d),('semantic',s),('ideal',ideal32)]:
            result[f'mixture_{key}_top{cutoff}_uniform_tie_expected_overlap_fraction']=float(np.dot(p,top_probability(values,k))/k)
            result[f'mixture_{key}_top{cutoff}_stable_overlap_fraction']=len(set(orders['mixture'][:k])&set(orders[key][:k]))/k
    return result


def authenticate_panel(panel,freeze,provenance):
    scores={};reference=None;proof={}
    for name,(directory,variant,alpha) in VARIANTS.items():
        receipt_path=checked(freeze['previous_control_prediction_receipts'][panel][directory],provenance)
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('schema')!='generalization_predictions_v1' or receipt.get('phase')!='exploratory_phase2' or receipt.get('labels_used') is not False:raise ValueError('Wrong immutable prediction phase')
        if not same(receipt['phase2_frozen_recipe'],freeze['phase2_frozen_recipe']) or not same(receipt['frozen_recipe'],freeze['original_frozen_recipe']):raise ValueError('Prediction freeze mismatch')
        bundle_path=checked(receipt['bundle'],provenance);bundle=json.loads(bundle_path.read_text())
        if bundle.get('schema')!='generalization_phase2_bundle_v1' or bundle['variant']!=variant or bundle['alpha']!=alpha or bundle['seed']!=42 or bundle['split']!='reaction_smi':raise ValueError('Wrong branch/seed/split')
        parent=json.loads(checked(bundle['phase2_frozen_recipe'],provenance).read_text())
        matched=[r for r in parent['approved_models'] if r['split']==bundle['split'] and r['seed']==42 and r['feature_manifest_sha256']==bundle['feature_manifest_sha256']
                 and all(same(r[k],bundle[k]) for k in ('density_bundle','smooth_dictionary','parent_bundle','base_checkpoint'))]
        if len(matched)!=1 or bundle['smooth_config']!=parent['smooth']:raise ValueError('Branch state tuple not approved')
        density=json.loads(checked(bundle['density_bundle'],provenance).read_text())
        if density['seed']!=42 or density['feature_manifest_sha256']!=bundle['feature_manifest_sha256']:raise ValueError('Wrong density head lineage')
        checked(receipt['input_receipt'],provenance)
        inp=json.loads(Path(receipt['input_receipt']['path']).read_text())
        if not same(inp['checkpoint'],bundle['base_checkpoint']):raise ValueError('Feature checkpoint differs from branch checkpoint')
        for key in ('catalog','base','protein_means','reaction_features'):
            if not same(inp['inputs'][key],receipt['inputs'][key]):raise ValueError('Prediction feature source mismatch')
        catalog=json.loads(checked(receipt['inputs']['catalog'],provenance).read_text())
        diagnostics_path=receipt_path.parent/'support_diagnostics.npz'
        checked(dict(path=str(diagnostics_path),sha256=receipt['support_diagnostics_sha256']),provenance)
        with np.load(diagnostics_path,allow_pickle=False) as data:diagnostics={k:data[k] for k in data.files}
        if reference is None:reference=(receipt,bundle,catalog,diagnostics)
        else:
            previous,previous_bundle,previous_catalog,previous_diag=reference
            if catalog!=previous_catalog or receipt['inputs']!=previous['inputs'] or receipt['reaction_key']!=previous['reaction_key'] or receipt['batch_size']!=previous['batch_size'] or receipt['source_sha256']!=previous['source_sha256']:raise ValueError('Branch controls used different feature groups/order/implementation')
            for key in ('density_bundle','smooth_dictionary','base_checkpoint','parent_bundle'):
                if not same(bundle[key],previous_bundle[key]):raise ValueError('Branch control state differs: '+key)
            if bundle['feature_manifest_sha256']!=previous_bundle['feature_manifest_sha256'] or bundle['smooth_config']!=previous_bundle['smooth_config']:raise ValueError('Branch training/kernel mismatch')
            if diagnostics.keys()!=previous_diag.keys() or any(not np.array_equal(v,previous_diag[k]) for k,v in diagnostics.items()):raise ValueError('Branch support gates differ')
        score_path=checked(dict(path=str(receipt_path.parent/'scores.npz'),sha256=receipt['output_sha256']),provenance)
        with np.load(score_path,allow_pickle=False) as source:scores[name]=source['selected']
        queries=catalog.get('reactions',catalog.get('query_ids'));shape=(len(queries),len(catalog['proteins']))
        if scores[name].shape!=shape or scores[name].dtype!=np.float32 or not np.isfinite(scores[name]).all():raise ValueError('Invalid saved branch matrix')
        proof[name]=dict(receipt=identity(receipt_path),bundle=receipt['bundle'],density_bundle=bundle['density_bundle'],smooth_dictionary=bundle['smooth_dictionary'],base_checkpoint=bundle['base_checkpoint'],alpha=alpha,score_shape=list(shape),score_sha256=receipt['output_sha256'])
    return catalog,scores,proof


def main():
    if OUT.exists():raise ValueError('Use the fixed fresh descriptive audit directory')
    OUT.mkdir(parents=True);started=time.monotonic();provenance={}
    freeze_path=checked(dict(path=str(RUN/'phase4/frozen_recipe.json'),sha256=FREEZE_SHA),provenance)
    freeze=json.loads(freeze_path.read_text())
    for record in freeze['implementation_sources']:checked(record,provenance)
    protocol=dict(study='Posthoc branch score dispersion of already evaluated frozen controls',panels=list(PANELS),
        additional_official_scope='Reaction-Sim is the smallest official score matrix; larger Time/Enzyme panels omitted to limit shared-storage I/O.',
        no_activity_labels_read=True,no_candidate_masking=True,no_predictions_or_training=True,no_weight_selection=True,no_refseq_access=True,
        branch_contract='Same seed42 gated neural heads, smooth dictionaries, kernels, base checkpoint, native/input catalogs and support diagnostics. Alpha0,.25,1 controls pinned by phase4 freeze.',
        formulas={'ideal_score':'.75*D+.25*S before floating-point endpoint scaling and final score rounding',
                  'weighted_std_ratio':'.75*populationSD(D)/(.25*populationSD(S)); undefined when semanticSD=0',
                  'variance':'Var(.75D+.25S)=.75^2Var(D)+.25^2Var(S)+2*.75*.25Cov(D,S)',
                  'weighted_std_share':'.75SD(D)/(.75SD(D)+.25SD(S)); an amplitude description, not a variance or causal attribution',
                  'spearman':'Pearson correlation of average descending ranks; undefined for a constant candidate row',
                  'uniform_tie_overlap':'Expected topK set intersection/k under independent uniform boundary-tie resolution in the two rankings; stable common-index sensitivity also shown'},
        numeric='Population statistics computed inFP64 from savedFP32 scores. Actual mixed scores retained; compare both idealFP64 sum and once-rounded approximate sum, never replace actual rankings.',
        summaries='Per-query statistics for both directions; equal-query summaries with explicit defined-query counts. No labels, chemistry strata or favorable-query filter.',
        freeze=identity(freeze_path),source=identity(__file__))
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    all_rows=[];summaries={};proofs={}
    for panel in PANELS:
        catalog,scores,proof=authenticate_panel(panel,freeze,provenance);proofs[panel]=proof;summaries[panel]={}
        for direction,ids,transpose in [('reaction_to_enzyme',catalog.get('reactions',catalog.get('query_ids')),False),('enzyme_to_reaction',catalog['proteins'],True)]:
            matrices={k:v.T if transpose else v for k,v in scores.items()};rows=[]
            for index,query_id in enumerate(ids):
                row=dict(panel=panel,direction=direction,query_index=index,query_id=query_id,**row_stats(*(matrices[k][index] for k in ('mixture','neural','semantic'))))
                rows.append(row);all_rows.append(row)
            keys=[k for k in rows[0] if k not in ('panel','direction','query_index','query_id')]
            summaries[panel][direction]={key:summarize_values([r[key] for r in rows]) for key in keys}
            summaries[panel][direction]['query_count']=len(rows)
            print(json.dumps(dict(panel=panel,direction=direction,queries=len(rows),elapsed_seconds=time.monotonic()-started)),flush=True)
        del scores
    with (OUT/'per_query.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    result=dict(schema='frozen_p2_branch_dominance_descriptive_v1',panels=summaries,branch_identity_proofs=proofs,
        caveats=['Score dispersion measures contribution to ranking geometry, not catalytic truth or causal performance.',
                 'Covariance can reinforce or cancel branch variance; nominalweights and SDshare are not performance attribution.',
                 'A semantic score can carry useful activity conditioning while a fixed mixture follows the higher-dispersion branch; this audit does not show another weight would generalize.',
                 'Average-rank correlations are undefined forconstant rows. TopK overlap becomes trivial whenK includes allcandidates; effectiveK retained.',
                 'Case1 reverse retrieval has one reaction candidate and is degenerate; its zero dispersion is reported without a correlation claim.',
                 'All panels were already evaluated. This descriptive audit supplies no new independent test, model selection or experimental biology.'],
        no_activity_labels_read=True,no_candidate_pool_changed=True,no_new_predictions=True,no_refseq_access=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    receipt=dict(schema='frozen_p2_branch_dominance_complete_v1',source=identity(__file__),protocol=identity(OUT/'protocol.json'),
        inputs=list(provenance.values()),outputs={p.name:identity(p) for p in (OUT/'per_query.csv',OUT/'summary.json')},
        elapsed_seconds=time.monotonic()-started,query_rows=len(all_rows),scores_reencoded=False,labels_read=False)
    (OUT/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(identity(OUT/'complete.json')))


if __name__=='__main__':main()
