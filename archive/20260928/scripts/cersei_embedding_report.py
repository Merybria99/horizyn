#!/usr/bin/env python3
"""Standalone scientific figures and a source-linked evaluation report."""
from datetime import datetime,timezone
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cersei_embedding_organization import ROOT,OUT,SPLITS,read,dump,sha
from cersei_embedding_metrics import write_table

NAMES={'reaction_smi':'Reaction-Sim','enzyme_smi':'Enzyme-Sim','time':'Time','enzymemap':'EnzymeMap'}
BLUE='#245B91';GOLD='#B57A18';GRAY='#7B838C';PINK='#BA668E';OLIVE='#77883A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
    'axes.spines.right':False,'axes.edgecolor':'#68717A','text.color':'#24313B',
    'axes.labelcolor':'#24313B','xtick.color':'#48545E','ytick.color':'#48545E',
    'savefig.facecolor':'white'})


def select(records,**criteria):
    rr=[r for r in records if all(r.get(k)==v for k,v in criteria.items())]
    assert len(rr)==1,(criteria,len(rr))
    return rr[0]


def number(row,scale=1,digits=3):
    if row['macro'] is None:return 'not estimable'
    value=row['macro']*scale
    return f'{value:.{digits}f}' + (f" [{row['ci95'][0]*scale:.{digits}f}, {row['ci95'][1]*scale:.{digits}f}]" if row['ci95'] else ' [CI unavailable]')


def interval(ax,row,y,color,marker='o',scale=1):
    if row['macro'] is None:return
    if row['ci95']:
        lo,hi=np.array(row['ci95'])*scale;ax.hlines(y,lo,hi,color=color,lw=1.5);ax.plot([lo,hi],[y,y],'|',color=color,ms=5)
    ax.plot(row['macro']*scale,y,marker,color=color,ms=6,mfc='white' if marker=='s' else color,zorder=3)


def finish(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.png',bbox_inches='tight',dpi=200)
    plt.close(fig)


def main_figures(data):
    fig,axes=plt.subplots(1,3,figsize=(15.5,4.4),gridspec_kw={'width_ratios':[1.05,1.05,1.]})
    ax=axes[0]
    for i,s in enumerate(SPLITS):
        for j,(stage,color,marker) in enumerate([('prott5',GRAY,'s'),('refined',BLUE,'o')]):
            r=select(data[s]['a1'],stage=stage,ec_level=3,k=10,exclude_homologs=True)
            interval(ax,r,3-i+(j-.5)*.22,color,marker)
    ax.set(yticks=range(4),yticklabels=[NAMES[s] for s in SPLITS[::-1]],xlim=(0,1),xlabel='EC3 agreement among 10 neighbors',title='A  Functional neighborhoods')
    ax.text(0,1.025,'Macro EC3; same 50% sequence\ncomponent excluded',transform=ax.transAxes,fontsize=8.5,va='bottom')
    ax.plot([],[],'s',color=GRAY,mfc='white',label='Frozen ProtT5');ax.plot([],[],'o',color=BLUE,label='CERSEI neural')
    ax.legend(frameon=False,loc='lower right',fontsize=9)
    ax=axes[1]
    for i,s in enumerate(SPLITS):
        for j,(stage,color,marker) in enumerate([('phase1',GRAY,'s'),('refined',BLUE,'o'),('final_score',GOLD,'^')]):
            r=select(data[s]['a2'],stage=stage,direction='R2E',degree_group='all',metric='coverage50')
            interval(ax,r,3-i+(j-1)*.2,color,marker)
    ax.set(yticks=range(4),yticklabels=[NAMES[s] for s in SPLITS[::-1]],xlim=(0,1.02),xlabel='Fraction of known positives retrieved at 50',title='B  Many-to-many coverage')
    ax.text(0,1.025,'Reaction → enzyme\nFull candidate banks',transform=ax.transAxes,fontsize=8.5,va='bottom')
    for stage,color,marker,label in [('phase1',GRAY,'s','Phase 1'),('refined',BLUE,'o','Refined neural'),('final_score',GOLD,'^','Final score')]:
        ax.plot([],[],marker,color=color,label=label,mfc='white' if marker=='s' else color)
    ax.legend(frameon=False,loc='lower right',fontsize=9)
    ax=axes[2];rr=read(OUT/'enzymemap/analysis3/quality_matching_summary.json')
    for i,(stage,color,marker) in enumerate([('ReactionT5v2',GRAY,'s'),('phase1',GOLD,'^'),('refined',BLUE,'o')]):
        interval(ax,select(rr,stage=stage),2-i,color,marker)
    ax.set(yticks=range(3),yticklabels=['Refined neural','Phase 1','Frozen ReactionT5v2'],xlim=(-.02,.39),xlabel='Same-rule minus different-rule cosine',title='C  Transformation-sensitive geometry')
    ax.text(0,1.025,'EnzymeMap; participant Tanimoto\nmatched within 0.02',transform=ax.transAxes,fontsize=8.5,va='bottom')
    ax.axvline(0,color=GRAY,lw=1,ls='--');ax.text(.98,.1,'427 matched queries\nNative mapping quality ≥ 0.5',ha='right',transform=ax.transAxes,fontsize=9)
    for ax in axes:
        title=ax.get_title();ax.set_title('');ax.set_title(title,loc='left',pad=36)
        ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
    fig.subplots_adjust(wspace=.62,bottom=.22,top=.8)
    fig.text(.02,.015,'Points: class-balanced estimates. Lines: 95% intervals from 1,000 query-family bootstrap draws. Fixed subsets; existing checkpoints; no retraining.\nEnzymeMap EC annotations cover 720/1,357 proteins. Panel B balances EC3 classes in ReactZyme and reaction rules in EnzymeMap; these are diagnostic summaries, not benchmark MRR.',fontsize=8.5)
    finish(fig,'embedding_organization')


def view_figures(data):
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.0))
    ax=axes[0,0]
    for i,s in enumerate(SPLITS):
        for j,(metric,color,marker) in enumerate([('top10pct_jaccard',BLUE,'o'),('centered_attention_cosine',GOLD,'s')]):
            r=select(data[s]['views'][4],variant='attention',metric=metric)
            interval(ax,r,3-i+(j-.5)*.2,color,marker)
    ax.set(yticks=range(4),yticklabels=[NAMES[s] for s in SPLITS[::-1]],xlim=(0,1),xlabel='Pairwise attention similarity',title='A  Learned-view overlap (K = 4)')
    ax.plot([],[],'o',color=BLUE,label='Top-10% residue Jaccard');ax.plot([],[],'s',color=GOLD,mfc='white',label='Centered attention cosine');ax.legend(frameon=False,fontsize=8.5,loc='center right')
    ax=axes[0,1]
    for i,s in enumerate(SPLITS):
        r=select(data[s]['views'][4],variant='without_all_slots-intact',metric='coverage50')
        interval(ax,r,3-i,BLUE,scale=100)
    ax.axvline(0,color=GRAY,ls='--',lw=1)
    ax.set(yticks=range(4),yticklabels=[NAMES[s] for s in SPLITS[::-1]],xlabel='Change in known-positive coverage@50 (pp)',title='B  Suppress all learned views (K = 4)')
    for i,s in enumerate(SPLITS):
        color=[BLUE,GOLD,PINK,OLIVE][i];marker=['o','s','^','D'][i]
        xx=[1,2,4,8]
        r=[select(data[s]['views'][k],variant='intact',metric='coverage50') for k in xx]
        ax=axes[1,0];ax.plot(xx,[x['macro'] for x in r],marker=marker,color=color,label=NAMES[s],lw=1.25)
        for x,rec in zip(xx,r):
            if rec['ci95']:ax.vlines(x,*rec['ci95'],color=color,lw=.8,alpha=.65)
        xx=[2,4,8];r=[select(data[s]['views'][k],variant='attention',metric='top10pct_jaccard') for k in xx]
        ax=axes[1,1];ax.plot(xx,[x['macro'] for x in r],marker=marker,color=color,label=NAMES[s],lw=1.25)
        for x,rec in zip(xx,r):
            if rec['ci95']:ax.vlines(x,*rec['ci95'],color=color,lw=.8,alpha=.65)
    axes[1,0].set(xticks=[1,2,4,8],ylim=(0,1.05),xlabel='Number of learned views K',ylabel='Known-positive coverage@50',title='C  Matched-seed neural retrieval')
    axes[1,0].legend(frameon=False,fontsize=8.5,loc='lower right')
    axes[1,1].set(xticks=[2,4,8],ylim=(0,1),xlabel='Number of learned views K',ylabel='Top-10% residue Jaccard',title='D  Attention overlap across K')
    for ax in axes.ravel():ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.12,right=.98,wspace=.55,hspace=.53,bottom=.17,top=.94)
    fig.text(.02,.02,'Macro EC3 summaries; 95% intervals resample whole protein families. Neural enzyme → reaction retrieval, full test reaction banks.\nSuppression freezes all parameters and renormalizes remaining feature-wise gates. Each K is a separate seed-42 fit. K = 1 has no view-pair overlap.\nThese diagnostics measure complementarity and reliance; they do not assign biochemical roles to the learned queries.',fontsize=9)
    finish(fig,'learned_views')


def report(data):
    text=['# CERSEI embedding organization — evaluation-only first pass',
        f'Completed {datetime.now(timezone.utc).isoformat()}. Existing primary checkpoints plus all 16 matched-seed K fits. No retraining.',
        '**The clearest evidence is improved functional neighborhoods and transformation-sensitive reaction geometry. Residual refinement has mixed benefits against hard distractors. Learned views overlap partially and help on some settings, but these analyses do not establish that K = 4 is uniformly optimal.**',
        '## Scope and annotation coverage',
        'Query subsets follow the requested cap of 2,000 proteins and 500 reactions. IDs were selected by a fixed SHA256 ordering, independently of annotation labels and results. Candidate banks remain complete. The same query identities are used across stages and K. Reaction-Sim has 386 test reactions; EnzymeMap has 1,357 unique test-positive protein sequences, so those populations are used in full.',
        '| Setting | Protein queries | EC3-annotated protein queries | Protein families among annotated queries | EC3 classes | Reaction queries |',
        '|---|---:|---:|---:|---:|---:|']
    for s in SPLITS:
        sub=read(OUT/s/'subsets.json');r=select(data[s]['a1'],stage='refined',ec_level=3,k=10,exclude_homologs=True)
        text.append(f"| {NAMES[s]} | {len(sub['protein_indices']):,} | {r['evaluated_queries']:,} | {r['families']} | {r['classes']} | {len(sub['reaction_indices'])} |")
    text+=['EnzymeMap protein EC coverage is 720/1,357 (53.1%) using the native `ec2uniprot` source; no missing EC labels were imputed. Its accession-based alignment test has 1,377 enzyme queries, with 730 EC3-annotated queries. Reaction rule annotations cover all 1,521 EnzymeMap test reactions (63 rule labels; 61 connected rule families); the 500-query subset covers 51 labels and 49 families. For ReactZyme, reaction EC classes are coarse proxies obtained from native EC annotations on known associated proteins. Directed transformation analysis is not applicable to its unordered participant sets.',
        '## 1. Functional neighborhoods beyond close sequence similarity',
        'For each protein, measure the fraction of its 10 or 50 nearest annotated neighbors sharing at least one EC prefix at depths 1–4. Self matches are removed. The principal table below additionally removes all proteins in the query’s 50%-identity MMseqs connected component. This is a conservative operational family exclusion, not proof that every remaining pair has identity below 50%. Each comparison uses identical queries and candidate pools.',
        '| Setting | Frozen ProtT5 | CERSEI refined neural | Paired improvement (percentage points) |',
        '|---|---:|---:|---:|']
    for s in SPLITS:
        r=[select(data[s]['a1'],stage=x,ec_level=3,k=10,exclude_homologs=True) for x in ['prott5','refined','refined-prott5']]
        text.append(f"| {NAMES[s]} | {number(r[0],100,1)} | {number(r[1],100,1)} | {number(r[2],100,1)} |")
    text+=['Values are macro EC3 agreement percentages, with 95% family-bootstrap intervals. All four paired intervals are above zero. Most of this organization is already present after phase 1; the residual head makes comparatively small changes. This supports functional organization beyond the frozen protein representation, without assigning catalytic-site meaning to the embeddings.',
        '## 2. Many-to-many cross-modal alignment',
        'Known-positive coverage@k averages the fraction of every query’s annotated positives present in its top k; it is not hit rate or benchmark MRR. Exact ties use uniform random tie resolution. CSVs also report conservative coverage bounds in a 1e-6 score band, all-positive coverage, coarse-EC distractors, chemistry-similar reaction distractors, and degree groups 1, 2–5 and 6+.',
        '| Setting | Phase 1 R→E coverage@50 | Refined neural | Final combined score |',
        '|---|---:|---:|---:|']
    for s in SPLITS:
        r=[select(data[s]['a2'],stage=x,direction='R2E',degree_group='all',metric='coverage50') for x in ['phase1','refined','final_score']]
        text.append(f"| {NAMES[s]} | {number(r[0],100,1)} | {number(r[1],100,1)} | {number(r[2],100,1)} |")
    text+=['These are class-balanced diagnostics: EC3 for ReactZyme and reaction rule for EnzymeMap. They should not replace or be numerically compared with the paper’s standard benchmark metrics. The final combined score includes the existing training-reaction dictionary term and is kept separate from the 512-dimensional neural geometry.',
        'Refinement increases mean-positive separation from fixed random unannotated distractors in every setting and both directions. It does **not** consistently increase the margin over the highest-scoring unannotated distractor. In particular, Reaction-Sim R→E and EnzymeMap retain negative mean-positive versus hardest-distractor margins. Thus “better than random” does not establish that difficult neighborhoods are cleanly separated. Unannotated pairs are not experimentally established negatives.',
        '## 3. Reaction organization by transformation',
        'The primary analysis compares native EnzymeMap reaction rules within participant-Morgan-fingerprint similarity bins and evaluates rule neighborhoods with and without high participant similarity. A subsequent quality sensitivity retains unambiguous single-rule reactions with minimum native mapping quality ≥0.5, and matches each same-rule comparison to a different-rule comparison within 0.02 Tanimoto. This sensitivity was added after the mapping-quality audit, not presented as preregistered.',
        'The tight-matching check includes 427 of the 500 sampled reaction queries, 55,704 comparisons, and a mean absolute Tanimoto gap of 0.00535. Comparisons are averaged within query before balancing rule classes and resampling rule families.',
        '| Representation | Same-rule minus different-rule cosine, matched chemistry |',
        '|---|---:|']
    rr=read(OUT/'enzymemap/analysis3/quality_matching_summary.json')
    for stage,label in [('participant_fingerprint','Participant fingerprint (matching diagnostic)'),('ReactionT5v2','Frozen ReactionT5v2'),('phase1','CERSEI phase 1'),('refined','CERSEI refined neural'),('refined-ReactionT5v2','Paired increase over frozen ReactionT5v2')]:
        text.append(f'| {label} | {number(select(rr,stage=stage))} |')
    text+=['This supports transformation-related organization beyond participant similarity. It does not show complete substrate invariance: in the stricter cross-regime comparison, same-rule pairs with Tanimoto ≤0.3 remain less similar than different-rule pairs with Tanimoto ≥0.7. That contrast has only 95 eligible queries spanning nine rule labels, and is reported with its limited coverage.',
        '![Embedding organization](figures/embedding_organization.png)',
        '## 4. Learned-view complementarity and reliance',
        'All K=1,2,4,8 analyses use the completed matched seed-42 fits. Their K=4 checkpoints are refits, distinct from the primary checkpoint set above. At fixed weights, each suppression sets the selected learned-view gates to zero, renormalizes the remaining gates per feature, and recomputes the frozen projection and residual head. Global and SLEEC branches remain active. Only the query is perturbed; candidate banks remain fixed.',
        '| Setting, K=4 | Top-10% attention Jaccard | Learned views’ total gate mass | Removing all views: Δ neural E→R coverage@50 (pp) |',
        '|---|---:|---:|---:|']
    for s in SPLITS:
        r=data[s]['views'][4]
        text.append(f"| {NAMES[s]} | {number(select(r,variant='attention',metric='top10pct_jaccard'))} | {number(select(r,variant='attention',metric='learned_gate_mass'))} | {number(select(r,variant='without_all_slots-intact',metric='coverage50'),100,2)} |")
    text+=['The views attend to partly different residue subsets, with substantial overlap rather than fully independent specialization. Time shows a measurable loss under all-view suppression; the other coverage@50 intervals include zero. EC-neighborhood agreement also declines on Time and Enzyme-Sim, while changes on Reaction-Sim and EnzymeMap are uncertain. Gate mass and distinct attention alone are not proof of performance benefit or distinct biochemical roles.',
        'K is not uniformly ordered: on these neural coverage diagnostics, K=4 is strongest on Time, K=1 is strongest on Reaction-Sim, and other counts can perform better on EnzymeMap. These single-seed results support describing K=4 as a pragmatic trade-off, not a proven universal optimum. Per-view suppression, all K metrics, projected-view cosine, and descriptive EC3-weighted linear CKA matrices are saved; CKA matrices are not given independent-pair confidence intervals.',
        '![Learned view analysis](figures/learned_views.png)',
        '## Statistical definitions and limits',
        'A class-balanced summary is the average of class means: θ = (1/|C|) Σ_c (1/|Q_c|) Σ_{q∈Q_c} m_q. A protein with multiple labels contributes to each corresponding class. Missing annotations are excluded from annotated summaries and counted explicitly. Unannotated-query micro means remain in JSON for context.',
        'Intervals use 1,000 draws of whole query families, with replacement, recalculating the macro statistic in each draw. Protein families are connected components of retrieved MMseqs alignments with ≥30% identity, ≥80% coverage of both sequences, and E≤1e-3; the search maximum was 675 hits, below its 10,000-hit cap. ReactZyme reaction families are participant-fingerprint components at Tanimoto ≥0.7; EnzymeMap uses shared-rule components. Candidate banks remain fixed. Absent annotation classes in bootstrap draws are omitted; intervals for sparse classes are exploratory. These intervals do not capture variability across training seeds, candidate-bank sampling, annotation errors, or undetected homology.',
        'Both benchmarks have been inspected during earlier method development. These analyses are exploratory characterization of frozen fits, not an untouched confirmatory evaluation. The EnzymeMap protein-neighborhood bank is the 1,357 test-positive sequences, whereas its R→E alignment bank retains all 261,907 official candidate accessions.',
        '## Artifacts and reproduction',
        '- [Base protocol](protocol.json), [user-requested subset amendment](subset_amendment.json), and per-setting `subsets.json` fix the analyzed IDs.',
        '- `analysis1/per_protein.csv`, `analysis2/per_query_R2E.csv`, `analysis2/per_query_E2R.csv`, and `analysis3/*per_query*.csv` retain the query-level measurements. Their corresponding `summary.json` files include class counts, family counts, coverage and intervals.',
        '- `learned_views/k*/export_receipt.json` records checkpoint and residual-head hashes plus maximum errors when reproducing saved embeddings. No model parameters were optimized.',
        '- [Validation record](validation.json) documents independent rank/tie checks, annotation joins, nearest-neighbor spot checks, class-mean recomputation, bootstrap checks and unchanged checkpoint hashes.',
        '- [Standalone PDF: main analyses](figures/embedding_organization.pdf); [standalone PDF: learned views](figures/learned_views.pdf). Figures are drawn from the same saved summary files.',
        '- [Reproduction commands](reproduce.sh) and [source snapshot](source/) are provided for the current workspace.',
        'Method references: [MMseqs2 user guide](https://www.mmseqs.com/latest/userguide.pdf) specifies alignment coverage and identity controls; [RDKit fingerprint documentation](https://rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html) documents the radius-2, 2,048-bit Morgan representation used here. The comparison designs were motivated by [CLIPZyme](https://proceedings.mlr.press/v235/mikhael24a.html), [FGW-CLIP](https://arxiv.org/html/2512.08508v2), and [TIGER](https://arxiv.org/html/2605.24489v1); no competitor geometry is claimed to have been evaluated.']
    formatted=[]
    for i,line in enumerate(text):
        if i:formatted.append('\n' if line.startswith('|') and text[i-1].startswith('|') else '\n\n')
        formatted.append(line)
    (OUT/'README.md').write_text(''.join(formatted)+'\n')


def main():
    validation=read(OUT/'validation.json')
    assert validation['passed'] and validation['include_views'] and not validation['incomplete']
    data={}
    for s in SPLITS:
        data[s]={'a1':read(OUT/s/'analysis1/summary.json'),'a2':read(OUT/s/'analysis2/summary.json'),
                 'views':{k:read(OUT/s/f'learned_views/k{k}/summary.json') for k in (1,2,4,8)}}
    (OUT/'figures').mkdir(exist_ok=True)
    dump(OUT/'figures/chart_contract.json',dict(surface='Standalone scientific PDF and PNG',
        main=dict(question='Does neural geometry reflect function, many-to-many associations, and transformation?',
            type='Faceted point-and-interval comparison',sources=['analysis1/summary.json','analysis2/summary.json','analysis3/quality_matching_summary.json'],
            palette='Blue and gold plus gray, distinct marker shapes',intervals='Whole-query-family percentile bootstrap'),
        views=dict(question='Do views overlap and does suppression alter retrieval?',type='Point-and-interval panels and discrete K comparisons',
            sources=['learned_views/k*/summary.json'],palette='Blue and gold for focused comparisons; four fixed roots and marker shapes for benchmark series'),
        caveats='Exploratory frozen-fit diagnostics, not benchmark MRR or causal biochemical assignments'))
    main_figures(data);view_figures(data);report(data)
    dest=OUT/'source';dest.mkdir(exist_ok=True)
    paths=[ROOT/'scripts'/f for f in ['cersei_embedding_organization.py','cersei_embedding_metrics.py','cersei_learned_view_analysis.py',
        'cersei_reaction_matching.py','cersei_embedding_validate.py','cersei_embedding_report.py']]
    for path in paths:shutil.copy2(path,dest/path.name)
    dump(dest/'manifest.json',{p.name:sha(p) for p in paths})
    (OUT/'reproduce.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd /datastor2/deep-proteins/EnzymeDiscovery
CERSEI_PY=.capability-run-py/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/tmp/enzymediscovery_v4_residue_view_count_20260922_seed42/python_packages:horizyn:horizyn/scripts
"$CERSEI_PY" horizyn/scripts/cersei_embedding_organization.py prepare
"$CERSEI_PY" horizyn/scripts/cersei_embedding_organization.py homology
"$CERSEI_PY" horizyn/scripts/cersei_embedding_organization.py subsets
for split in reaction_smi enzyme_smi time enzymemap; do
  for analysis in analysis1 analysis2; do
    "$CERSEI_PY" horizyn/scripts/cersei_embedding_metrics.py "$analysis" --split "$split" --device cuda:0
  done
  for k in 1 2 4 8; do
    "$CERSEI_PY" horizyn/scripts/cersei_learned_view_analysis.py --split "$split" --k "$k" --device cuda:0
  done
done
"$CERSEI_PY" horizyn/scripts/cersei_embedding_metrics.py analysis3
"$CERSEI_PY" horizyn/scripts/cersei_reaction_matching.py
"$CERSEI_PY" horizyn/scripts/cersei_embedding_validate.py --include-views
"$CERSEI_PY" horizyn/scripts/cersei_embedding_report.py
''')
    print(OUT/'README.md')


if __name__=='__main__':main()
