"""SciencePlots figures and source-backed Markdown for the public-model audit."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/public_embedding_comparison_20260924'
sys.path.insert(0,str(ROOT/'.deps/ablation-figures'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
SPLITS=['time','enzyme_smi','reaction_smi','enzymemap'];LABELS=['Time','Enzyme split','Reaction split','EnzymeMap']
NAMES={'cersei':'CERSEI','horizyn':'Horizyn','creep':'CREEP','clipzyme':'CLIPZyme','prott5':'ProtT5','clean':'CLEAN','esm1b':'ESM-1b'}
COLORS={'cersei':'#0072B2','horizyn':'#E69F00','creep':'#CC79A7','clipzyme':'#009E73','prott5':'#666666','clean':'#CC79A7','esm1b':'#009E73'}

def read(p):return json.loads(p.read_text())
def pick(rows,**filters):
    a=[x for x in rows if all(x.get(k)==v for k,v in filters.items())];assert len(a)==1,(filters,len(a));return a[0]
def pct(x):return '—' if x is None else f'{100*x:.2f}'
def interval(row):return '—' if row['ci95'] is None else f"[{100*row['ci95'][0]:+.2f}, {100*row['ci95'][1]:+.2f}]"
def models(split):return ['cersei','horizyn']+(['clipzyme'] if split=='enzymemap' else ['creep'])
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def point(ax,x,row,color,marker='o',offset=0):
    y=row['macro']
    if y is None:return
    ax.plot(x+offset,100*y,marker=marker,color=color,ms=6,zorder=3,linestyle='None')
    if row['ci95'] is not None:
        lo,hi=np.array(row['ci95'])*100;ax.vlines(x+offset,lo,hi,color=color,lw=1.1);ax.hlines([lo,hi],x+offset-.045,x+offset+.045,color=color,lw=1.1)
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=230,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def figures():
    plt.style.use(['science','no-latex'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'xtick.top':False,'ytick.right':False,'axes.labelsize':10})
    fig,axes=plt.subplots(2,2,figsize=(8.1,5.6),sharey=True)
    for ax,s,label in zip(axes.flat,SPLITS,LABELS):
        rows=read(OUT/s/'neighborhoods_summary.json');names=models(s)+['prott5']
        for i,m in enumerate(names):point(ax,i,pick(rows,model=m,ec_level=3,k=10,exclude_homologs=True,exclude_training_overlap=False),COLORS[m])
        ax.set_xticks(range(len(names)),[NAMES[m] for m in names]);ax.set_xlabel(label);ax.set_ylim(0,75);ax.set_xlim(-.5,len(names)-.5);ax.grid(axis='y',alpha=.15)
    for ax in axes[:,0]:ax.set_ylabel('EC3 agreement@10 (%)')
    fig.tight_layout();save(fig,'01_matched_functional_neighborhoods')
    fig,axes=plt.subplots(2,2,figsize=(8.1,5.6),sharey=True)
    for ax,s,label in zip(axes.flat,SPLITS,LABELS):
        rows=read(OUT/s/'alignment_summary.json');names=models(s)
        for i,m in enumerate(names):
            for d,mk,offset in [('R2E','o',-.1),('E2R','^',.1)]:point(ax,i,pick(rows,model=m,direction=d,metric='coverage10'),COLORS[m],mk,offset)
        ax.set_xticks(range(len(names)),[NAMES[m] for m in names]);ax.set_xlabel(label);ax.set_ylim(0,105);ax.set_xlim(-.5,len(names)-.5);ax.grid(axis='y',alpha=.15)
    for ax in axes[:,0]:ax.set_ylabel('Positive coverage@10 (%)')
    axes[0,0].plot([],[],marker='o',color='#333333',ls='None',label='R → E');axes[0,0].plot([],[],marker='^',color='#333333',ls='None',label='E → R');axes[0,0].legend(frameon=False,loc='lower left',ncol=2)
    fig.tight_layout();save(fig,'02_matched_alignment')
    fig,axes=plt.subplots(2,2,figsize=(8.1,5.6),sharey=True)
    for ax,s,label in zip(axes.flat,SPLITS,LABELS):
        rows=read(OUT/s/'clean_neighborhoods_summary.json');names=['cersei','clean','esm1b']
        for i,m in enumerate(names):point(ax,i,pick(rows,model=m,ec_level=3,k=10,exclude_homologs=True,exclude_training_overlap=True),COLORS[m])
        ax.set_xticks(range(len(names)),[NAMES[m] for m in names]);ax.set_xlabel(label);ax.set_ylim(0,40);ax.set_xlim(-.5,2.5);ax.grid(axis='y',alpha=.15)
    for ax in axes[:,0]:ax.set_ylabel('EC3 agreement@10 (%)')
    fig.tight_layout();save(fig,'03_clean_transfer_control')
    dest=OUT/'enzymecage_p450';rows=read(dest/'neighborhoods_summary.json');align=read(dest/'alignment_summary.json')
    fig,axes=plt.subplots(1,2,figsize=(8.1,3.3))
    for i,(m,c) in enumerate([('EnzymeCAGE','#E69F00'),('ESMC pocket','#009E73')]):
        for ex,mk,offset in [(False,'o',-.1),(True,'^',.1)]:point(axes[0],i,pick(rows,model=m,k=10,exclude_homologs=ex),c,mk,offset)
    axes[0].set_xticks([0,1],['EnzymeCAGE\npocket branch','ESM-C\npocket mean']);axes[0].set_ylabel('Shared-reaction agreement@10 (%)');axes[0].set_ylim(-.3,10);axes[0].set_xlim(-.5,1.5)
    axes[0].plot([],[],marker='o',color='#333333',ls='None',label='All families');axes[0].plot([],[],marker='^',color='#333333',ls='None',label='Exclude close families');axes[0].legend(frameon=False,loc='upper left',fontsize=8)
    for i,d in enumerate(['R2E','E2R']):
        for k,mk,offset,c in [(10,'o',-.1,'#E69F00'),(50,'s',.1,'#CC79A7')]:point(axes[1],i,pick(align,direction=d,metric=f'coverage{k}'),c,mk,offset)
    axes[1].set_xticks([0,1],['R → E','E → R']);axes[1].set_ylabel('Positive coverage (%)');axes[1].set_ylim(0,65);axes[1].set_xlim(-.5,1.5)
    axes[1].plot([],[],marker='o',color='#E69F00',ls='None',label='Top 10');axes[1].plot([],[],marker='s',color='#CC79A7',ls='None',label='Top 50');axes[1].legend(frameon=False,loc='upper left',fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    fig.tight_layout();save(fig,'04_enzymecage_native_panel')

def report():
    lines=['# Public checkpoint embedding analysis',
        'Completed 24 September 2026. This extends the released CLIPZyme analysis to original Horizyn, CREEP, CARE-released CLEAN, and released EnzymeCAGE. No model was fitted or selected in this analysis. The comparator is the current dictionary-free manuscript CERSEI (fusion multiplier 2, residual cap 0.5, dictionary weight 0; no added biological-label supervision).',
        '## What the results support',
        '**CERSEI has better local functional organization than the matched-data dual encoders on the ReactZyme enzyme and reaction splits, but this does not establish better exact-pair retrieval.** The time split is close. Original Horizyn still exceeds current CERSEI in all six full-test ReactZyme MRR cells. The advantage in functional neighborhoods and the disadvantage in exact-pair ranking coexist: grouping enzymes by EC3 is a coarser requirement than assigning each observed reaction–enzyme edge its correct rank.',
        'CLEAN provides a useful EC-supervised protein control. Its downstream training data differ, and exact training-sequence overlap must be removed before interpreting its transfer behavior. CERSEI has higher EC3 agreement on the common overlap-filtered banks; this is descriptive transfer evidence, not an isolated architectural comparison.',
        'The available EnzymeCAGE pockets do not support a full common-pool comparison. Its native P450 panel supports a separate branch-level and pair-score analysis. A pooled pocket branch is not its complete pair-conditioned model, so weak cosine neighborhoods in that branch cannot establish that EnzymeCAGE lacks biological information.',
        '## Scope and checkpoint provenance',
        table(['Model','Weights used','Evaluation scope','Interpretation'],[
            ['CERSEI','Current validation-frozen manuscript checkpoints','All three ReactZyme splits and EnzymeMap','Independent cosine dual encoder'],
            ['Horizyn','Our validation-selected refits of the original public architecture','All three ReactZyme splits and EnzymeMap','Same downstream data; different representations/objective/budget'],
            ['CREEP','Our validation-selected two-modality ReactZyme refits','All three ReactZyme splits','Same association graph; disclosed exact-edge/participant-set adaptation'],
            ['CLIPZyme','Official released checkpoint and screening embeddings','EnzymeMap; reused verified export','Same source split; sequence alias handling differs'],
            ['CLEAN','CARE authors’ released protein_train50.pth','Fixed protein subsets across four partitions','EC-supervised transfer; protein only'],
            ['EnzymeCAGE','Official pretrained seed42 epoch19','Covered common proteins plus complete native P450 panel','Transfer/branch control; pair-conditioned native scoring']]),
        'Sources: [Horizyn](https://github.com/dayhofflabs/horizyn), [CARE/CREEP code](https://github.com/jsunn-y/CARE), [CARE/CLEAN weights](https://huggingface.co/jsunn-y/CARE_pretrained/tree/main), [CLEAN encoder](https://github.com/tttianhao/CLEAN/blob/main/app/src/CLEAN/model.py), [EnzymeCAGE](https://github.com/GENTEL-lab/EnzymeCAGE). Local refits are not described as author-released weights. Each export has a checkpoint hash and selection/protocol receipt.',
        '## Matched-data functional neighborhoods',
        'EC3 agreement@10 is the fraction of ten annotated nearest neighbors sharing at least one EC3 prefix with the query, macro-averaged across observed EC3 classes. Queries in the same detected 50%-identity component are excluded from one another’s neighbor candidates. The full original protein bank is retained: 12,277 / 8,734 / 14,688 / 1,357 proteins for Time / enzyme / reaction / EnzymeMap. The ReactZyme query subsets contain 2,000 proteins each; EnzymeMap uses all 1,357 test-positive unique sequences. Annotation eligibility further reduces evaluated query counts.',
        '![Functional neighborhoods](01_matched_functional_neighborhoods.png)',
        '**Figure 1.** Functional-neighborhood agreement after excluding close sequence components. Points are macro means; intervals use the family-weighted procedure described below. Frozen ProtT5 is an input-representation control. All models within a panel use identical queries, annotations, and candidate banks.']
    rows=[];deltas=[]
    for s,l in zip(SPLITS,LABELS):
        n=read(OUT/s/'neighborhoods_summary.json')
        for m in models(s)+['prott5']:
            x=pick(n,model=m,ec_level=3,k=10,exclude_homologs=True,exclude_training_overlap=False);rows.append([l,NAMES[m],pct(x['macro']),x['evaluated_queries']])
        for m in models(s)[1:]:
            x=pick(n,model='cersei-'+m,ec_level=3,k=10,exclude_homologs=True,exclude_training_overlap=False);deltas.append([l,'CERSEI − '+NAMES[m],pct(x['macro']),interval(x),str([round(100*v,2) for v in x['legacy_family_resampling_ci95']])])
    lines += [table(['Partition','Model','EC3 agreement@10 (%)','Annotated queries'],rows),table(['Partition','Paired difference','Percentage points','Weighted interval','Legacy family-resampling interval'],deltas),
        'The strongest consistent evidence is the enzyme and reaction splits. On EnzymeMap, the CERSEI–Horizyn advantage is only 1.18 percentage points; its legacy family-resampling interval crosses zero, so its certainty depends on the uncertainty convention. Both interval conventions are retained rather than selecting whichever is favorable.',
        '## Alignment and retrieval',
        'Known-positive coverage@k averages the fraction of a query’s listed positives among its first k candidates, with expected uniform resolution of exact score ties. It is not MRR or hit@k. Reaction queries use fixed label-independent subsets (500, 500, 386, 500 respectively); enzyme queries use 2,000 proteins per ReactZyme split and 1,377 positive accessions on EnzymeMap. Full candidate pools remain intact, including all 261,907 screening accessions. Summary means are class-balanced over EC3 or EnzymeMap rule labels, so they are diagnostic summaries rather than the published benchmark aggregation.',
        '![Alignment](02_matched_alignment.png)',
        '**Figure 2.** Positive coverage@10 in the two retrieval directions. Circles denote reaction-to-enzyme retrieval and triangles enzyme-to-reaction retrieval. Complete candidate pools are used. This shows exact association retrieval rather than same-modality functional neighborhoods.']
    rows=[]
    for s,l in zip(SPLITS,LABELS):
        a=read(OUT/s/'alignment_summary.json')
        for m in models(s):rows.append([l,NAMES[m]]+[pct(pick(a,model=m,direction=d,metric=f'coverage{k}')['macro']) for d,k in [('R2E',10),('R2E',50),('E2R',10),('E2R',50)]])
    lines += [table(['Partition','Model','R→E @10 (%)','R→E @50 (%)','E→R @10 (%)','E→R @50 (%)'],rows),
        'Per-query tables additionally record positive scores, random-distractor margins, hardest-unannotated margins, same-EC1 distractors, and chemically similar reaction distractors. An unannotated pair is not a measured inactive pair. Cosine margin scales differ across learned spaces and should not be used as a calibrated cross-model ranking of quality.',
        '### Full ReactZyme benchmark MRR (existing complete evaluations)',
        'These are the complete all-positive benchmark MRR values, not the class-balanced subset diagnostics above.']
    rows=[]
    for s,l in zip(SPLITS[:3],LABELS[:3]):
        for m in ['cersei','horizyn','creep']:
            if m=='cersei':x=read(ROOT/f'runs/cersei_horizyn_challenge_20260923/no_dictionary_validation_20260923/reference_shared_test/{s}/summary.json')['summary']
            else:
                prefix='horizyn_participant_set' if m=='horizyn' else 'creep_two_modality';x=read(ROOT/f'runs/reactzyme_public_baselines_20260921/models/{prefix}_{s}_seed42/complete.json')['test']
            rows.append([l,NAMES[m],f"{x['reaction_to_enzyme']['all']['reactzyme_mrr']:.4f}",f"{x['enzyme_to_reaction']['all']['reactzyme_mrr']:.4f}"])
    lines += [table(['Partition','Model','R→E MRR','E→R MRR'],rows),
        '### Reaction-space organization',
        'Reaction-neighborhood diagnostics use native rule IDs on EnzymeMap and EC3 inherited from held-out associations on ReactZyme. The latter is a descriptive functional association label, not an independently measured reaction mechanism. The chemically restricted diagnostic excludes participants with Morgan-fingerprint Tanimoto ≥0.5. Per-query and summary files preserve both restricted and unrestricted results.']
    rows=[]
    for s,l in zip(SPLITS,LABELS):
        a=read(OUT/s/'reaction_summary.json')
        for m in models(s):rows.append([l,NAMES[m],pct(pick(a,model=m,k=10,exclude_similar_participants=False)['macro']),pct(pick(a,model=m,k=10,exclude_similar_participants=True)['macro'])])
    lines += [table(['Partition','Model','Agreement@10 (%)','After chemistry exclusion (%)'],rows),
        'On EnzymeMap we also repeat the tighter chemistry-matched analysis: single-rule reactions with native mapping quality ≥0.5, matching same-rule and different-rule comparisons within 0.02 participant Tanimoto. All models use the same 427 queries and 55,704 matched comparisons. A positive gap supports rule separation within that model; its magnitude is not a calibrated cross-model performance metric.',
        table(['Model','Matched same-rule minus different-rule cosine'],[[NAMES[m],f"{pick(read(OUT/'enzymemap/reaction_summary.json'),model=m,metric='quality_matched_gap')['macro']:.4f}"] for m in models('enzymemap')]),
        '## CLEAN: protein-only transfer control',
        'We loaded the released CARE CLEAN head with its exact 1280→512→512→128 LayerNorm architecture and extracted frozen ESM-1b layer-33 residue means. Sequences use the native first-1022-residue truncation. No retrieval head or reaction encoder was invented. The analysis uses the original 2,000-protein subsets (1,357 for EnzymeMap) as common banks for every displayed model; these smaller banks differ from Figure 1. Exact sequences present in CARE’s protein_train50 table are removed from both queries and candidates for the primary transfer control. Close sequence components among the retained evaluation proteins are also excluded. This removes exact training overlap, not all homologous pretraining/training exposure.',
        '![CLEAN control](03_clean_transfer_control.png)',
        '**Figure 3.** Functional-neighborhood agreement on common banks after removing CLEAN training-sequence matches and close query–candidate sequence components. Frozen ESM-1b is the CLEAN backbone control. The training objectives, supervised data, and backbones remain different, so this is a transfer diagnostic.']
    rows=[]
    for s,l in zip(SPLITS,LABELS):
        rec=read(OUT/s/'clean_receipt.json');a=read(OUT/s/'clean_neighborhoods_summary.json')
        vals=[pick(a,model=m,ec_level=3,k=10,exclude_homologs=True,exclude_training_overlap=True) for m in ['cersei','clean','clean-native-l2','esm1b']]
        rows.append([l,rec['proteins'],rec['exact_training_sequence_overlap'],vals[0]['evaluated_queries']]+[pct(x['macro']) for x in vals])
    lines += [table(['Partition','Initial bank','Training matches','Annotated retained queries','CERSEI cosine (%)','CLEAN cosine (%)','CLEAN native L2 (%)','ESM-1b cosine (%)'],rows),
        'CLEAN’s native unnormalized Euclidean geometry is included as a sensitivity analysis. This avoids judging CLEAN solely through an imposed cosine metric. Its values are close to the cosine control here. Neither table is a same-training-data architectural superiority test. Three EnzymeMap entries had empty sequence strings in the legacy analysis metadata; their original screening-catalog sequences were recovered and verified against their SHA256-derived protein IDs before the final export. All three lack EC labels and are ineligible for the functional-neighborhood metrics. The correction is recorded in sequence_metadata_repairs.json; the earlier metadata artifact is unchanged.',
        '## EnzymeCAGE: available coverage and native panel',
        'The common-pool audit matches complete protein sequences to existing released case-study pockets. It finds 37 Time proteins, 10 enzyme-split proteins, 13 reaction-split proteins, and zero EnzymeMap proteins; 59 sequences are unique across those overlaps. Of these per-split proteins, 32, 8, and 12 occur in the EnzymeCAGE training table. These pools cannot support a broad held-out competitor comparison. Counts and all computable diagnostics are retained, with undefined metrics left missing rather than filled with zero.',
        'The independent native-panel analysis uses the released pretrained seed42 checkpoint on the complete P450 panel: 490 enzymes × 191 reactions, 93,590 candidate pairs, and 318 listed positives involving 168 enzymes. We export the mean valid-residue output of the learned pocket self-attention branch and compare it with the input ESM-C pocket mean. This branch precedes enzyme–molecule interaction and is not EnzymeCAGE’s complete pair representation. We separately analyse its verified native sigmoid pair scores. We use neither the domain-fine-tuned model nor the external similarity prior; these raw-score diagnostics do not reproduce the paper’s prior-filtered screening table.',
        '![EnzymeCAGE native panel](04_enzymecage_native_panel.png)',
        '**Figure 4.** Left: protein-neighbor agreement defined by sharing at least one listed positive reaction, macro-averaged over those reaction labels. Only the 168 proteins with listed activity enter the functional neighbor bank. Circles allow all other proteins; triangles exclude the same detected 50%-identity component. Right: native pair-score positive coverage on the full 490-enzyme / 191-reaction candidate pools. Intervals are family-weighted. No CERSEI comparison is implied by this native-panel figure.']
    a=read(OUT/'enzymecage_p450/neighborhoods_summary.json');rows=[]
    for ex in [False,True]:
        for m in ['EnzymeCAGE','ESMC pocket']:rows.append([m,ex,pct(pick(a,model=m,k=10,exclude_homologs=ex)['macro'])])
    lines += [table(['Representation','Close families excluded','Shared-reaction agreement@10 (%)'],rows)]
    a=read(OUT/'enzymecage_p450/alignment_summary.json');rows=[[d]+[pct(pick(a,direction=d,metric=f'coverage{k}')['macro']) for k in [10,50]] for d in ['R2E','E2R']]
    cov=read(OUT/'enzymecage_p450/coverage.json')
    lines += [table(['Native direction','Positive coverage@10 (%)','Positive coverage@50 (%)'],rows),
        f"The P450 pool has {cov['exact_training_sequence_overlap']}/490 exact sequence matches in EnzymeCAGE’s training table. Among the 318 listed positive test pairs, {cov['exact_test_positive_pairs_seen_as_training_positive']} match a training-positive pair and {cov['exact_test_positive_pairs_seen_as_training_negative']} match a training-negative pair by exact sequence and canonical reaction string; chemically equivalent aliases are not excluded by that test. Sequence novelty is therefore not established by this native panel. Raw pocket means retain more of this sparse substrate-profile neighborhood signal than the learned pocket branch; that result cannot be extended to the full cross-attention scorer. Near-zero agreement after homology exclusion also reflects sparse known-positive profiles, not verified lack of shared activity.",
        '## Uncertainty, fairness, and checks',
        'Main intervals use 1,000 shared family-level exponential-weight draws (a family-weighted/Bayesian bootstrap), with the observed annotation classes and candidate bank fixed. All members of a family receive the same weight. We recompute class means and their macro-average in each draw; paired differences use identical weights. These are conditional uncertainty intervals, not multi-seed confidence intervals or a correction for model selection. All comparisons are exploratory and unadjusted for multiplicity.',
        'The previous analysis resampled whole families with replacement and omitted any annotation class missing from a replicate. With sparse classes this changes the class mixture and can noticeably shift the bootstrap distribution, occasionally placing a percentile interval away from its observed point estimate. The new weighted intervals keep every observed class. The original family-resampling intervals remain in every summary under legacy_family_resampling_ci95, and the paired table above reports both conventions. This amendment was made for a consistent estimand, not to select a favorable result; conclusions sensitive to it are qualified.',
        'Protein families use previously audited MMseqs hit components at ≥30% identity with ≥80% coverage of both sequences and E≤1e−3; 50% components define the close-homology exclusion. These are operational search-derived groups, not guarantees of no undetected homology. Reaction families are participant-fingerprint components on ReactZyme and native rule components on EnzymeMap. The native P450 panel has its own recorded all-vs-all MMseqs search.',
        'Current CERSEI exports reproduce stored score coordinates within 6e−7. Horizyn and CREEP exports reproduce original scores within 3e−5. CREEP’s original inference batch size, input order, and per-split attention kernel are replayed; its original evaluated matrices are used for retrieval diagnostics. The older, dictionary-augmented CERSEI embedding exports are not substituted. CLEAN and EnzymeCAGE checkpoint loads are strict. Download/source hashes, candidate order, exact sequence identity, coverage, finite vectors, and native score receipts are retained.',
        'Same downstream training sources control one major confound, but these complete methods still differ in pretrained backbones, inputs, losses, capacity, batch size, and training budget. CREEP and Horizyn require the documented participant-set adaptation on ReactZyme. Neither local geometry, native transfer performance, nor comparisons among one-seed fits prove wet-lab generalization or isolate the effect of a single architectural component.',
        '## Reproducibility and files',
        'The analysis is on branch `research/embedding-public-checkpoints-20260924`. Existing fitted checkpoints and the manuscript were left unchanged. See `protocol.json`, `validation.json`, `complete.json`, `analysis.ipynb`, and per-split `*_receipt.json`, `*_summary.json`, and `*_per_query.csv` / `*_per_protein.csv` files. Vector PDF, SVG, and PNG copies of all figures accompany this document. The plot source uses SciencePlots and the requested blue/pink/orange/green palette without figure titles or footnotes.',
        'Run `scripts/cersei_public_embedding_exports.py` for matched baseline/current CERSEI exports, `scripts/cersei_clean_embedding_control.py` for CLEAN, `scripts/cersei_enzymecage_embedding_control.py` for covered EnzymeCAGE pockets, and `scripts/cersei_enzymecage_p450_geometry.py` for its native panel. Then run `scripts/cersei_public_embedding_analysis.py` for each split/control, `scripts/cersei_enzymecage_p450_analysis.py`, and `python3 scripts/cersei_public_embedding_report.py`. Runtime environments and exact commands are in the notebook/protocol.']
    rendered = '\n\n'.join(lines)+'\n'
    if (OUT/'neighborhood_curves/protocol_explanation.md').exists() and (OUT/'ec_levels/README.md').exists():
        (OUT/'report_sources').mkdir(exist_ok=True)
        (OUT/'report_sources/base_comparison.md').write_text(rendered)
        from consolidate_public_embedding_report import build
        build()
    else:
        (OUT/'README.md').write_text(rendered)

if __name__=='__main__':figures();report();print('REPORT AND FIGURES COMPLETE')
