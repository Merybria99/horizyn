#!/usr/bin/env python3
"""Independently check key measurements and write the checkpoint comparison."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# Keep reporting independent of the training environment. Run with system
# Python / Matplotlib 3.6, compatible with the workspace's SciencePlots 2.2.
ROOT=Path(__file__).resolve().parents[1]
CROSS=ROOT/'runs/generalization_20260919_2251/cross_paper_retraining'
OUT=ROOT/'runs/clipzyme_embedding_comparison_20260924'
PREVIOUS=ROOT/'runs/cersei_embedding_organization_20260923_v1'
OLD=PREVIOUS/'enzymemap'
CURRENT=ROOT/'runs/cersei_horizyn_challenge_20260923/no_dictionary_validation_20260923'
RELEASE=CROSS/'clipzyme_released_screen_v1'
MODELS=['CLIPZyme','CERSEI']


def read(path): return json.loads(Path(path).read_text())
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def norm(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def prefix(labels,level):
    return sorted({'.'.join(s.split('.')[:level]) for s in labels if len(s.split('.'))>=level and '-' not in s.split('.')[:level]})
def shared():
    return read(OLD/'metadata.json'),read(OLD/'subsets.json'),np.load(OLD/'truth.npy'),np.load(OLD/'expanded.npy')


def macro(values, labels):
    classes=defaultdict(list)
    for value,labs in zip(values,labels):
        if np.isfinite(value):
            for label in set(labs):classes[label].append(float(value))
    return float(np.mean([np.mean(v) for v in classes.values()]))


def check():
    meta,sub,truth,expanded=shared();bank=np.asarray(meta['neighborhood_indices']);labels=[prefix(meta['enzyme_ec'][i],3) for i in bank]
    fam=read(PREVIOUS/'homology/families.json');family=np.array([fam[meta['protein_ids'][i]][1] for i in bank])
    summary=read(OUT/'neighborhoods_summary.json');receipts=read(OUT/'prepared.json')
    checks=[]
    for name in MODELS:
        file=OUT/f'{name.lower()}_embeddings.npz';assert sha(file)==receipts['export_sha256'][file.name]
        z=np.load(file);e=z['enzyme'];assert np.array_equal(z['bank'],bank)
        for key in ['enzyme','reaction']:
            assert np.isfinite(z[key]).all() and np.max(abs(np.linalg.norm(z[key],axis=1)-1))<2e-6
        nn=np.load(OUT/f'neighbors_{name}_ec3_exclude1.npy')
        available=np.array([bool(x) for x in labels]);ids=np.flatnonzero(available)
        for i in ids[:30]:
            target=nn[i,:10];assert i not in target and all(family[j]!=family[i] for j in target)
            scores=norm(e)@norm(e[i:i+1])[0];valid=available&(family!=family[i])
            assert scores[target].min()>=np.sort(scores[valid])[-10]-3e-6
        agreement=np.array([np.mean([bool(set(labels[i])&set(labels[j])) for j in nn[i,:10]]) for i in ids])
        expected=next(r['macro'] for r in summary if r['model']==name and r['ec_level']==3 and r['k']==10 and r['exclude_homologs'])
        assert abs(macro(agreement,[labels[i] for i in ids])-expected)<1e-12
        checks.append(dict(model=name,neighbor_spot_checks=30,neighborhood_macro_recomputed=True))
    positives=defaultdict(list)
    for r,e in truth:positives[int(r)].append(int(e))
    for direction in ['R2E','E2R']:
        data=rows(OUT/f'alignment_per_query_{direction}.csv')
        for name,path in [('CLIPZyme',RELEASE),('CERSEI',CURRENT/'reference_shared_test/enzymemap')]:
            scores=np.load(path/'scores.npy',mmap_mode='r')
            part=[r for r in data if r['stage']==name]
            for row in part[:15]:
                qi=int(row['query_index']);v=np.asarray(scores[qi] if direction=='R2E' else scores[:,qi])
                pos=positives[qi] if direction=='R2E' else truth[truth[:,1]==qi,0].tolist()
                for k in [10,50]:
                    coverage=np.mean([max(0,min(1,(k-int(np.sum(v>v[j])))/int(np.sum(v==v[j])))) for j in pos])
                    assert abs(coverage-float(row[f'coverage{k}']))<1e-12
            checks.append(dict(model=name,direction=direction,independent_tie_rank_checks=30))
    matching=read(OUT/'reaction_matching_coverage.json');assert matching['matched_queries']==427 and matching['comparisons']==55704
    assert matching['max_tanimoto_gap']<=.02000001
    rr=read(OUT/'reaction_summary.json');labels_by_id=dict(zip(meta['reaction_ids'],meta['reaction_rules']))
    data=rows(OUT/'reaction_per_query.csv')
    for name in MODELS:
        part=[r for r in data if r['model']==name and r['metric']=='quality_matched_gap']
        measured=macro([float(r['value']) for r in part],[labels_by_id[r['query_id']] for r in part])
        expected=next(r['macro'] for r in rr if r['model']==name and r['metric']=='quality_matched_gap')
        assert abs(measured-expected)<1e-12
    catalog=CROSS/'clipzyme_f3_catalog_v1';prep=read(catalog/'preparation.json')
    manifest=CROSS/'clipzyme_manifests_v2/manifest.json'
    assert sha(manifest)==prep['source_manifest_sha256']
    assert sha(manifest)==receipts['receipts']['CLIPZyme']['manifest_sha256']
    assert sha(catalog/'train_pairs.csv')==prep['split_files']['train']['pairs_sha256']
    validation=dict(passed=True,checks=checks,quality_matching_macro_recomputed=True,
        same_downstream_manifest_verified=True,original_training_associations=34427,
        cersei_unique_sequence_training_pairs=prep['split_files']['train']['sequence_level_pairs'],
        training_data_note='Same official source associations; CERSEI collapses accession aliases with identical sequences to 34180 unique training pairs. Different pretrained encoders and structures preclude an architecture-only causal attribution.',
        inference_only=True,completed_utc=datetime.now(timezone.utc).isoformat())
    dump(OUT/'validation.json',validation)


def pick(data, **conditions):
    matches=[r for r in data if all(r.get(k)==v for k,v in conditions.items())]
    assert len(matches)==1,conditions
    return matches[0]


def table_row(label, a, b, delta, factor=100, digits=2):
    def value(r):return f"{factor*r['macro']:.{digits}f}"
    lo,hi=np.array(delta['ci95'])*factor
    return f"| {label} | {value(a)} | {value(b)} | {factor*delta['macro']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] |"


def figures(n,r,a):
    dump(OUT/'chart_contract.json',dict(renderer='Matplotlib + SciencePlots',surface='Scientific PDF/SVG/PNG and Markdown',
        question='Which aspects of held-out embedding organization differ between CERSEI and released CLIPZyme?',
        variant='Four small panels: EC hierarchy, reaction-rule neighbors, positive coverage, and matched transformation gap',
        colors={'CERSEI':'#0072B2','CLIPZyme':'#009E73'},non_color='Filled circle versus open square',
        intervals='95% family-bootstrap intervals for each model; paired differences belong in the table',
        scale='Percent panels start at zero; cosine-gap panel starts at zero',
        annotations='No overall titles, subtitles, or footnotes; axes, legend, and panel letters only'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(ROOT/'.deps/ablation-figures'))
    import scienceplots
    colors={'CERSEI':'#0072B2','CLIPZyme':'#009E73'}
    with plt.style.context(['science','no-latex']):
        plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':8,
            'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(2,2,figsize=(5.5,4.3))
        for i,model in enumerate(['CERSEI','CLIPZyme']):
            offset=(i-.5)*.12
            specifications=[
                ([pick(n,model=model,ec_level=j,k=10,exclude_homologs=True) for j in [1,2,3,4]],np.arange(4),'EC depth',['1','2','3','4'],'EC agreement@10 (%)',100),
                ([pick(r,model=model,metric='rule_agreement10',exclude_similar_participants=v) for v in [False,True]],np.arange(2),'Participant similarity',['All','Tanimoto < 0.5'],'Rule agreement@10 (%)',100),
                ([pick(a,stage=model,direction=d,degree_group='all',metric='coverage50') for d in ['R2E','E2R']],np.arange(2),'Retrieval direction',[r'$R\to E$',r'$E\to R$'],'Positive coverage@50 (%)',100),
                ([pick(r,model=model,metric='quality_matched_gap')],np.array([0]),'Participant similarity',['Matched within 0.02'],'Same-rule cosine gap',1),
            ]
            for ax,(data,x,xlabel,labels,ylabel,scale) in zip(axes.flat,specifications):
                y=np.array([row['macro'] for row in data])*scale; ci=np.array([row['ci95'] for row in data]).T*scale
                ax.errorbar(x+offset,y,yerr=np.array([y-ci[0],ci[1]-y]),fmt='o' if model=='CERSEI' else 's',
                    color=colors[model],markerfacecolor=colors[model] if model=='CERSEI' else 'white',
                    markersize=4,linewidth=.9,capsize=2,label=model)
                ax.set_xticks(x,labels);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
                ax.set_xlim(-.4,float(x[-1])+.4);ax.set_ylim(0,100 if scale==100 else .4)
                ax.yaxis.grid(True,color='#e4e4e4',linewidth=.4);ax.set_axisbelow(True)
        for letter,ax in zip('abcd',axes.flat):ax.text(-.23,1.04,f'({letter})',transform=ax.transAxes,fontsize=8)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.54,1.005))
        fig.subplots_adjust(left=.13,right=.985,bottom=.12,top=.89,hspace=.65,wspace=.48)
        for ext in ['pdf','svg','png']:fig.savefig(OUT/f'comparison.{ext}',dpi=350)
        plt.close(fig)


def report():
    n=read(OUT/'neighborhoods_summary.json');r=read(OUT/'reaction_summary.json');a=read(OUT/'alignment_summary.json')
    selected=[('EC3 agreement@10; close sequence components excluded',n,dict(ec_level=3,k=10,exclude_homologs=True),'model',100,2),
      ('Rule agreement@10; participant Tanimoto < 0.5',r,dict(metric='rule_agreement10',exclude_similar_participants=True),'model',100,2),
      ('R→E positive coverage@50',a,dict(direction='R2E',degree_group='all',metric='coverage50'),'stage',100,2),
      ('E→R positive coverage@50',a,dict(direction='E2R',degree_group='all',metric='coverage50'),'stage',100,2),
      ('Chemistry-matched same-rule cosine gap',r,dict(metric='quality_matched_gap'),'model',1,3)]
    lines=['| Diagnostic | CLIPZyme | CERSEI | CERSEI − CLIPZyme [95% interval] |','|---|---:|---:|---:|']
    for label,data,conditions,key,factor,digits in selected:
        lines.append(table_row(label,pick(data,**conditions,**{key:'CLIPZyme'}),pick(data,**conditions,**{key:'CERSEI'}),
                              pick(data,**conditions,**{key:'CERSEI-CLIPZyme'}),factor,digits))
    text='''# Released CLIPZyme versus current CERSEI: embedding organization

Completed on 24 September 2026 using frozen checkpoints. **CERSEI has higher functional and transformation-neighborhood agreement under the selected exclusions, but does not uniformly outperform CLIPZyme on alignment diagnostics.** This is an exploratory comparison, not a claim of universal geometric or architectural superiority.

The [official CLIPZyme release](https://github.com/pgmikhael/clipzyme) provides the trained checkpoint and screening embeddings. Both methods are evaluated on the same official EnzymeMap rule split and candidate axes. CERSEI is refreshed from the **current manuscript** checkpoint, with fusion multiplier 2, residual cap 0.5, and dictionary weight 0. The older September 23 CERSEI geometry results are not substituted for this model.

## Main comparison

The first four rows are percentages; their paired differences are percentage points. The last row is a cosine-similarity difference. These class-balanced diagnostics are not the benchmark BEDROC/EF or MRR.

'''+ '\n'.join(lines)+'''

![Checkpoint comparison](comparison.png)

[Vector PDF](comparison.pdf) · [Editable SVG](comparison.svg)

Panels show (a) protein EC agreement after close sequence-component exclusion, (b) reaction-rule agreement with and without highly similar participants, (c) the fraction of all known positives retrieved in the first 50 candidates, and (d) same-rule versus different-rule cosine separation after chemistry matching. Error bars are model-wise 95% family-bootstrap intervals. The table uses paired intervals for model differences; overlap of individual error bars is not the paired comparison.

## Conclusions

1. **There is evidence of improved local functional organization.** EC3 agreement at ten neighbors is 28.71% for CERSEI and 25.91% for CLIPZyme after removing the query's detected 50%-identity sequence component. The paired difference is +2.80 pp [0.33, 4.29]. Frozen ProtT5 is 19.38% under the same evaluation. This supports function-related local organization beyond the frozen encoder and gives a modest advantage over this released competitor on this diagnostic.
2. **Reaction neighborhoods retain more rule agreement when very similar participants are excluded.** At participant Tanimoto < 0.5, agreement@10 is 19.26% versus 16.94%, with a paired difference of +2.32 pp [0.19, 4.21]. Unrestricted agreement is 44.96% versus 42.97%; its paired interval crosses zero. The exclusion analysis supports a limited local transformation-related advantage rather than complete independence from substrate composition.
3. **The more stringent matching analysis does not establish superiority.** Matching participant similarity within 0.02 gives cosine gaps of 0.279 and 0.247; the paired interval for their difference includes zero. Cosine scales also depend on each model's representation geometry. In the stricter comparison of dissimilar same-rule pairs against highly similar different-rule pairs, CLIPZyme has the less negative point estimate (−0.007 versus −0.110), and the paired interval crosses zero. Complete substrate invariance is not demonstrated.
4. **Better local neighborhoods do not imply uniformly better retrieval.** R→E positive coverage@50 is 27.96% versus 27.58%; E→R is 56.91% versus 59.19%. Both paired intervals cross zero. Coverage@10 and positive-degree strata are also retained in the full alignment summaries. This measures recovery of all known partners, not only whether one correct partner appears.
5. **Hard distractors remain a limitation in both models.** Mean-positive minus hardest-unannotated cosine margins are negative in both directions. CERSEI has larger separation from random distractors but more negative hardest-distractor margins. Raw margin magnitudes are descriptive: the two spaces have different dimensions and score distributions, so a larger raw cosine margin is not automatically a better ranking model. Unannotated pairs may include undiscovered positives.

These diagnostics add comparative evidence about the learned representations. They do not establish wet-lab activity, a universal advantage over public methods, or which architectural component causes the observed differences. Family-bootstrap intervals are unadjusted and conditional on one fitted checkpoint per model; they do not include training-seed variability or correct for the many diagnostics examined.

## Data and comparability audit

- Both methods use the same released EnzymeMap source split: 34,427 training associations, 7,287 validation associations, and 4,642 test associations. The source-manifest hash is shared between the CLIPZyme replay and CERSEI data preparation. CERSEI collapses identical-sequence accession aliases to **34,180 unique training pairs**. This is the same source-data protocol, not identical example weighting or a pure architecture-only experiment.
- The reaction subset is the original label-independent SHA256 sample of **500 queries**. R→E retains all **261,907 candidate accessions**; E→R retains all **1,521 test reactions**, including for difficult or unannotated candidates. Candidate IDs and order agree exactly between both saved score matrices and the original diagnostics.
- Protein neighborhoods contain the same **1,357 unique test-positive sequences**. Native EC3 annotations cover **720** of them. CLIPZyme is structure-dependent and has accession-specific embeddings; its representative for each sequence is the lexicographically first test-positive accession, chosen without looking at scores. No embeddings are averaged. Retrieval keeps all accessions, including aliases: there are **1,377** positive accession queries, of which **730** have EC3 labels for macro summaries.
- Neighborhood self matches and the same detected 50%-identity sequence component are excluded. This is an operational MMseqs component exclusion, not proof of absent distant homology. Confidence intervals resample the same whole query families as the earlier analysis.
- Reaction matching retains unambiguous single-rule reactions with minimum native mapping quality at least 0.5. It yields **427 matched queries and 55,704 comparisons**, with mean absolute participant-Tanimoto mismatch 0.00535. Matching uses reaction annotations and chemistry, not model outcomes.
- CLIPZyme uses its released 1,280-dimensional space; CERSEI uses its 512-dimensional space. No projection, alignment between models, or per-model dimensionality reduction is fitted for these quantitative comparisons.
- The CLIPZyme structure/ESM encoder and CERSEI's pretrained features differ. Matching downstream data does not match pretraining, input resources, objective, or compute. This is a method comparison under a shared downstream protocol.
- No checkpoints were updated. No test results were used to select a model in this analysis. These benchmarks were inspected during development, so this remains exploratory evidence.

## Artifacts and reproduction

Run from the `horizyn` repository with the existing environment:

```bash
../.capability-run-py/bin/python scripts/cersei_clipzyme_embedding_comparison.py prepare --device cuda:0
../.capability-run-py/bin/python scripts/cersei_clipzyme_embedding_comparison.py neighbors --device cuda:1
../.capability-run-py/bin/python scripts/cersei_clipzyme_embedding_comparison.py alignment
../.capability-run-py/bin/python scripts/cersei_clipzyme_embedding_comparison.py reactions
python3 scripts/cersei_clipzyme_embedding_report.py
```

The final three analysis stages can run independently after preparation. The report script verifies rank-based tie handling, nearest neighbors, class-balanced means, source hashes, and training-data provenance before rendering figures.

[Protocol](protocol.json) · [Preparation and checkpoint hashes](prepared.json) · [Validation](validation.json) · [Companion notebook](comparison.ipynb)

Per-query CSVs and complete JSON summaries include all EC depths, both neighborhood sizes, both retrieval directions, positive-degree strata, participant-similarity bins, and margin diagnostics. Original analysis artifacts are left intact. Learned-view suppression is not applied to CLIPZyme because it has no matching CERSEI view architecture.
'''
    (OUT/'README.md').write_text(text)
    figures(n,r,a)
    cells=[dict(cell_type='markdown',metadata={},source=['# Released-checkpoint embedding comparison\n','The saved report and query-level files are the canonical results. This notebook inspects them without refitting a model.\n']),
           dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=['from pathlib import Path\n','import json\n','from IPython.display import display, Markdown, Image\n','root = Path.cwd()\n',"assert (root / 'validation.json').is_file(), 'Open this notebook from its artifact directory'\n","display(Markdown((root / 'README.md').read_text()))\n"]),
           dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=["validation = json.loads((root / 'validation.json').read_text())\n","assert validation['passed']\n","validation\n"]),
           dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=["import csv\n","with (root / 'alignment_per_query_R2E.csv').open() as stream:\n","    query_rows = list(csv.DictReader(stream))\n","{name: sum(row['stage'] == name for row in query_rows) for name in ['CERSEI', 'CLIPZyme']}\n"])]
    dump(OUT/'comparison.ipynb',dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5))
    dump(OUT/'complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),evaluation_only=True,
         analyses_completed=['functional_neighborhoods','cross_modal_alignment','reaction_organization'],
         artifact_sha256={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.json','.csv','.md','.pdf','.svg','.png'] and p.name!='complete.json'}))
    print(OUT/'README.md',flush=True)


if __name__=='__main__':
    check()
    report()
