#!/usr/bin/env python3
"""Render and report the complete frozen-checkpoint functional-recovery analysis."""
from pathlib import Path
import csv
import json
import math
import shutil
import sys
import numpy as np
from cersei_crossmodal_functional_recovery import ROOT,OUT,SPLITS,KS,read,dump,sha,csvwrite

PAPER=ROOT.parent/'-ICLR2027---Enzyme-Reaction-Retrieval'
PUBLIC=ROOT/'runs/public_embedding_comparison_20260924'
NAMES={'time':'Time','enzyme_smi':'Enzyme-Sim','reaction_smi':'Reaction-Sim','enzymemap':'EnzymeMap',
 'cersei':'CERSEI','horizyn':'Horizyn','creep':'CREEP','clipzyme':'CLIPZyme','random':'Random',
 'phase1_native':'Phase 1, native fusion','phase2_native':'Phase 2, native fusion',
 'phase1_fusion2':'Phase 1, reference fusion','phase2_fusion2':'Phase 2, reference fusion'}
MODELS=['cersei','horizyn','creep','clipzyme','random']
COLORS={'cersei':'#0072B2','horizyn':'#E69F00','creep':'#CC79A7','clipzyme':'#009E73','random':'#666666'}
STYLES={'cersei':('-', 'o'),'horizyn':('--','s'),'creep':('-.','^'),'clipzyme':(':','D'),'random':('--',None)}
FILTERS=['partners','components']

def mdtable(headers,rr):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in rr])

def main():
    allrows=[];scopes=[]
    for split in SPLITS:
        for exclusion in FILTERS:
            root=OUT/split/exclusion;receipt=read(root/'complete.json')
            assert sha(root/'curves.csv')==receipt['curves_sha256']
            scopes.append(read(root/'scope.json'))
            with (root/'curves.csv').open() as f:allrows.extend(csv.DictReader(f))
    def curve(split,exclusion,model):
        rr=[r for r in allrows if (r['split'],r['exclusion'],r['model'])==(split,exclusion,model)]
        rr.sort(key=lambda r:int(r['k']));assert [int(r['k']) for r in rr]==list(KS)
        return rr
    def row(split,exclusion,model,k=10):return curve(split,exclusion,model)[k-1]
    def percent(r,key='macro'):return 100*float(r[key])
    def value(split,exclusion,model,k=10):return percent(row(split,exclusion,model,k))
    def available(split):return ['cersei','horizyn','creep']+(['clipzyme'] if split=='enzymemap' else [])
    csvwrite(OUT/'all_curves.csv',allrows)
    csvwrite(OUT/'summary_at10.csv',[r for r in allrows if int(r['k'])==10])
    csvwrite(OUT/'scope.csv',scopes)
    sys.path.insert(0,str(ROOT/'.deps/ablation-figures'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import scienceplots  # noqa: F401
    plt.style.use(['science','no-latex'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.3,'axes.labelsize':7.3,
        'axes.spines.top':False,'axes.spines.right':False,'xtick.top':False,'ytick.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
    ymax=10*math.ceil(max(value(s,e,m,k) for s in SPLITS for e in FILTERS for m in available(s) for k in KS)/10)
    fig,axs=plt.subplots(2,4,figsize=(7.3,3.7),sharex=True,sharey=True)
    for i,exclusion in enumerate(FILTERS):
        for j,split in enumerate(SPLITS):
            ax=axs[i,j]
            for model in available(split)+['random']:
                y=[percent(r) for r in curve(split,exclusion,model)];ls,mk=STYLES[model]
                ax.plot(KS,y,color=COLORS[model],ls=ls,lw=1.35 if model=='cersei' else 1,
                    marker=mk,markevery=[9],markersize=3.4,markerfacecolor=COLORS[model] if model=='cersei' else 'white',markeredgewidth=.8)
            ax.axvline(10,color='.78',lw=.5,zorder=0)
            ax.set(xlim=(1,50),ylim=(0,ymax),xticks=[1,10,25,50])
            if i==1:ax.set_xlabel(NAMES[split]+r': $k$')
            if j==0:ax.set_ylabel('EC3 recovery (%)')
    handles=[Line2D([0],[0],color=COLORS[m],ls=STYLES[m][0],marker=STYLES[m][1],markersize=3,lw=1.2,label=NAMES[m]) for m in MODELS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1.005),ncol=5,frameon=False,handlelength=2)
    fig.subplots_adjust(left=.075,right=.99,bottom=.14,top=.91,hspace=.18,wspace=.16)
    for suffix in ['pdf','svg','png']:fig.savefig(OUT/f'functional_recovery_curves.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)

    contrasts=[('phase2-native-minus-phase1-native','Native fusion: phase 2 − phase 1','#0072B2'),
        ('cersei-minus-phase1_fusion2','Reference fusion: final − phase 1','#E69F00')]
    fig,axs=plt.subplots(1,2,figsize=(7.3,2.35),sharey=True)
    limits=[]
    for ax,exclusion in zip(axs,FILTERS):
        for j,(model,label,color) in enumerate(contrasts):
            rr=[row(s,exclusion,model) for s in SPLITS]
            y=np.arange(4)+(-.12 if j==0 else .12)
            x=np.asarray([percent(r) for r in rr]);lo=np.asarray([percent(r,'ci_low') for r in rr]);hi=np.asarray([percent(r,'ci_high') for r in rr])
            ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt='o' if j==0 else 's',color=color,markersize=3.5,lw=1,capsize=2,label=label)
            limits.extend(lo);limits.extend(hi)
        ax.axvline(0,color='.4',lw=.7,ls='--');ax.set_yticks(range(4),[NAMES[s] for s in SPLITS])
        ax.set_xlabel('Change in recovery@10 (pp)')
    low,high=min(limits),max(limits);margin=.08*(high-low)
    for ax in axs:ax.set_xlim(low-margin,high+margin)
    axs[0].invert_yaxis();handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.56,1.03),ncol=1,frameon=False)
    fig.subplots_adjust(left=.12,right=.99,bottom=.2,top=.78,wspace=.15)
    for suffix in ['pdf','svg','png']:fig.savefig(OUT/f'refinement_effects_at10.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)

    comparison=[];stage=[];paired=[]
    for split in SPLITS:
        for exclusion in FILTERS:
            label='Partners' if exclusion=='partners' else 'Components'
            comparison.append([NAMES[split],label]+[f'{value(split,exclusion,m):.2f}' if m in available(split) else '—' for m in MODELS[:-1]]+
                [f'{value(split,exclusion,"random"):.2f}',f'{value(split,exclusion,"ceiling"):.2f}'])
            stage.append([NAMES[split],label]+[f'{value(split,exclusion,m):.2f}' for m in ['phase1_native','phase2_native','phase1_fusion2','phase2_fusion2','cersei']])
            for m in available(split)[1:]:
                r=row(split,exclusion,'cersei-minus-'+m)
                paired.append([NAMES[split],label,NAMES[m],f'{percent(r):+.2f}',f'[{percent(r,"ci_low"):+.2f}, {percent(r,"ci_high"):+.2f}]'])
    scope_rows=[[NAMES[s['split']],'Partners' if s['exclusion']=='partners' else 'Components',s['queries'],s['classes'],s['families'],s['annotated_candidates'],f'{s["eligible_candidate_min"]}–{s["eligible_candidate_max"]}',s['queries_without_alternative']] for s in scopes]
    leaders=[]
    for e in FILTERS:
        for s in SPLITS:
            best=max(available(s),key=lambda m:value(s,e,m));comp=max(available(s)[1:],key=lambda m:value(s,e,m))
            leaders.append(dict(split=s,exclusion=e,observed_best=best,strongest_comparator_at10=comp,
                cersei_minus_strongest_pp=value(s,e,'cersei')-value(s,e,comp)))
    curve_order=[];exclusion_effect=[]
    for s in SPLITS:
        for e in FILTERS:
            margins=np.asarray([value(s,e,'cersei',k)-max(value(s,e,m,k) for m in available(s)[1:]) for k in KS])
            curve_order.append(dict(split=s,exclusion=e,k_with_cersei_above_every_comparator=int((margins>0).sum()),
                minimum_margin_pp=float(margins.min()),maximum_margin_pp=float(margins.max()),
                nonpositive_margin_k=KS[margins<=0].tolist()))
        first={int(x['query_index']):int(x['eligible_candidates']) for x in csv.DictReader((OUT/s/'partners/query_scope.csv').open())}
        second={int(x['query_index']):int(x['eligible_candidates']) for x in csv.DictReader((OUT/s/'components/query_scope.csv').open())}
        extra=np.asarray([first[q]-n for q,n in second.items()]);assert (extra>=0).all()
        exclusion_effect.append(dict(split=s,common_queries=len(second),queries_with_additional_removals=int((extra>0).sum()),
            mean_additional_removals=float(extra.mean()),median_additional_removals=float(np.median(extra)),
            maximum_additional_removals=int(extra.max()),queries_omitted_for_sequence_coverage=len(first)-len(second)))
    dump(OUT/'curve_ordering.json',curve_order);dump(OUT/'component_exclusion_effect.json',exclusion_effect)
    report=[
        '# Cross-modal functional recovery beyond recorded partners',
        '25 September 2026. Fixed-checkpoint, evaluation-only diagnostic on all eligible test reaction queries. Training jobs and retrieval parameters are unchanged.',
        '## What this measures',
        'Can a reaction embedding retrieve other EC3-compatible enzymes after all locally recorded partners have been removed? This probes reaction-to-enzyme geometry directly. EC3 agreement is a coarse functional compatibility signal, not evidence that an unrecorded pair catalyzes the particular reaction.',
        '**Main conclusion:** CERSEI has the highest observed recovery@10 in every partition under both exclusion policies. The clearest comparative evidence is on Enzyme-Sim and EnzymeMap, where the paired intervals against every evaluated comparator remain positive. Time is nearly tied with CREEP and Reaction-Sim differences are uncertain. This supports functional organization beyond recorded partner identities, not universal retrieval superiority or validated new catalytic activity.',
        '## Results at ten candidates',
        mdtable(['Partition','Exclusion','CERSEI','Horizyn','CREEP','CLIPZyme','Random','Attainable ceiling'],comparison),
        'All values are EC3 class-macro agreement percentages. “Partners” removes recorded partner identifiers and every exact-sequence alias. “Components” additionally removes detected close-sequence components of those partners. The ceiling depends only on how many compatible candidates remain; queries with no alternative are retained at zero. All methods share each panel’s queries and candidate exclusions.',
        'Observed rankings: '+ '; '.join(f'{NAMES[x["split"]]} / {x["exclusion"]}: {NAMES[x["observed_best"]]}; CERSEI − {NAMES[x["strongest_comparator_at10"]]} = {x["cersei_minus_strongest_pp"]:+.2f} pp' for x in leaders)+'.',
        '![Cross-modal EC3 recovery](functional_recovery_curves.png)',
        '**Figure 1.** EC3 recovery for every k=1–50. Columns: Time, Enzyme-Sim, Reaction-Sim and EnzymeMap. Top: recorded partners and exact-sequence aliases excluded. Bottom: detected 50%-identity partner components additionally excluded. Markers highlight k=10. Curves use exact-tie expectations; the gray line is exact expected random retrieval from the same eligible bank. These correlated curve points are not independent replications. Scope differs from the earlier 500-query geometry sample: this study uses every eligible test reaction.',
        mdtable(['Partition','Exclusion','k values above every comparator (of 50)','Smallest margin (pp)','Largest margin (pp)'],
            [[NAMES[x['split']],x['exclusion'],x['k_with_cersei_above_every_comparator'],f'{x["minimum_margin_pp"]:+.2f}',f'{x["maximum_margin_pp"]:+.2f}'] for x in curve_order]),
        'EnzymeMap leads persist across all 50 neighborhood sizes under both exclusions. Enzyme-Sim also leads throughout the stronger control, but not at k=2 under partner exclusion. Time does not show a persistent advantage: CREEP overtakes CERSEI from k=18 onward under partner exclusion. The k=10 observation must therefore not be described as superiority across the entire curve.',
        '## Paired comparison uncertainty',
        mdtable(['Partition','Exclusion','Comparator','CERSEI − comparator (pp)','95% family-multiplier interval'],paired),
        'Intervals use the same 1,000 exponential weights for every member of a reaction family and for every method. Query weights are renormalized within each observed EC3 class before class averaging, so no class disappears from a draw. ReactZyme families use the existing participant-Tanimoto ≥0.7 components; EnzymeMap families connect reactions sharing native rule labels. These intervals condition on the checkpoints, candidate banks and annotation set. They do not measure training-seed variation or correct exploratory multiple comparisons.',
        '## Stage comparison with the exact manuscript checkpoints',
        mdtable(['Partition','Exclusion','Phase 1 (1,0)','Phase 2 (1,0.2)','Phase 1 (2,0)','Phase 2 (2,0.2)','Final (2,0.1)'],stage),
        'Column parentheses are (inference fusion multiplier, residual coefficient). Native phase 1 → native phase 2 isolates application of the trained residual heads at their training coefficient. Phase 1 at fusion 2 → final isolates application of the heads at the deployed coefficient. Native phase 1 → final changes both fusion and refinement and cannot attribute the difference solely to phase 2. None of these interventions retrains the encoders or heads.',
        'Most of the coarse functional organization is already present in phase 1. Under partner exclusion, native refinement changes recovery by −0.50, −0.63, −0.83 and +0.59 percentage points on Time, Enzyme-Sim, Reaction-Sim and EnzymeMap. At reference fusion, applying the deployed heads changes it by −0.40, −0.26, −0.46 and +1.03 points. These fixed-fusion contrasts show a modest ReactZyme decrease and an EnzymeMap increase; they do not establish overfitting or prove a causal trade-off with exact-pair retrieval.',
        '![Refinement effects](refinement_effects_at10.png)',
        '**Figure 2.** Paired changes in class-macro EC3 recovery@10. Left: partner exclusion. Right: additional component exclusion. Blue compares native phase 2 with native phase 1; orange compares the final model with phase 1 at the reference fusion multiplier. Whiskers are paired 95% family-multiplier intervals. Positive values indicate higher functional recovery. The stage effects need not agree with changes in exact association retrieval.',
        '## Cohort and denominators',
        mdtable(['Partition','Exclusion','Queries','EC3 classes','Query families','Annotated bank','Eligible bank range','No alternative'],scope_rows),
        'Only fully specified three-level prefixes are eligible (for example 1.1.1.- is sufficient, 1.1.-.- is not). EC3 labels are never filled by homology or model predictions. Candidate units follow the benchmark: unique sequence IDs on ReactZyme and UniProt accessions on EnzymeMap. Accessions with identical sequences can remain separate alternatives when their sequence is not a recorded partner; this metric therefore counts identifiers, not necessarily k distinct sequences.',
        'EnzymeMap has 13 locally recorded partner associations whose full sequences are unavailable. All listed partner identifiers are excluded in the primary analysis; exact-sequence aliases of those unavailable sequences cannot be audited. Reactions with any unavailable partner sequence are omitted from the stricter component analysis. The scope table records the resulting query counts. No query is removed because the model performs poorly or because no EC3-compatible alternative remains.',
        mdtable(['Partition','Shared queries','Queries with extra removals','Mean extra candidates removed','Median extra','Maximum extra','Queries omitted for sequence coverage'],
            [[NAMES[x['split']],x['common_queries'],x['queries_with_additional_removals'],f'{x["mean_additional_removals"]:.2f}',f'{x["median_additional_removals"]:.1f}',x['maximum_additional_removals'],x['queries_omitted_for_sequence_coverage']] for x in exclusion_effect]),
        'The stronger sequence filter is most informative on EnzymeMap, where it changes the candidate bank for 894 of 1,067 queries. It changes only 28 of 381 Reaction-Sim banks: unchanged performance on that split alone is therefore weak additional evidence against a sequence-neighbor explanation.',
        '## Exact protocol and estimand',
        '1. Freeze the primary manuscript checkpoints and validation-selected competitors. CERSEI uses the exact dictionary-free main-paper model: fusion multiplier 2 and residual coefficient 0.1. CREEP is the two-modality local refit on the corresponding benchmark associations; its EnzymeMap checkpoint was selected by validation BEDROC85. Horizyn uses the corresponding local refits; CLIPZyme uses its released EnzymeMap checkpoint. Downstream training data match their benchmark, but pretrained resources, objectives and budgets differ.',
        '2. Construct the known-partner graph from the union of all local train, validation and test records. ReactZyme uses exact benchmark reaction IDs (with a representational suffix removed if present); EnzymeMap uses the native directed reaction strings. Chemical aliases outside these identity conventions are not merged. “All recorded partners” refers to these source datasets, not every association in the literature.',
        '3. Assign each reaction the union of native EC3 labels carried by its recorded partners, including partners outside its candidate bank. ReactZyme labels come from cleaned_uniprot_rhea.tsv; EnzymeMap labels come from the released ec2uniprot mapping, with annotation unions for identical sequences. These are association-derived reaction labels, not independent measurements of reaction specificity.',
        '4. Restrict candidate banks to enzymes with specified EC3 labels, then exclude all partner IDs and exact-sequence aliases. For the stronger control, also exclude all candidates in any partner’s detected sequence component. Require at least 50 remaining candidates, keeping query identities fixed over k and methods within each filter.',
        '5. Compute the fraction of top-k candidates sharing any query EC3 label. Average within each represented EC3 class, then equally over classes. A multilabel query enters each class it carries; its per-query success still means sharing any one of its labels, not necessarily the averaging class. The CSV also retains the ordinary query mean.',
        '6. Resolve exact score ties by expected agreement over uniform permutations of the entire tie block, including members beyond k. Retain candidate-index tie resolution as a sensitivity. The random reference is the compatible fraction of each eligible bank, averaged by the identical class weights. It requires no Monte Carlo sampling.',
        '7. Compare every predefined stage and method at every k. No model, coefficient or query is selected using the new metric. Full benchmark score matrices of final CERSEI were reproduced exactly on all four partitions before exclusions were applied.',
        'The correct aggregate is F_macro(k) = (1/|C|) Σ_c (1/|Q_c|) Σ_{r∈Q_c} F_k(r), with Q_c containing eligible reactions carrying c. A simple average over reactions is the query mean and is not the class-macro metric.',
        '## Training-query exposure',
        mdtable(['Partition','Eligible queries','Exact training reaction seen','Exact training reaction unseen'],
            [[NAMES[s],d['eligible_primary_queries'],d['exact_training_reaction_seen'],d['exact_training_reaction_unseen']]
             for s,d in read(OUT/'training_query_exposure.json').items()]),
        'Exposure is defined by the benchmark reaction ID or native directed reaction string, not by an exhaustive chemical-equivalence test. Most Time and Enzyme-Sim reaction queries were present in their training graphs with other partners; Reaction-Sim and EnzymeMap queries were not. Removing recorded partners does not make every query or candidate unseen during training.',
        '## Sequence exclusion and interpretation',
        'MMseqs2 searches all 184,423 distinct EC3-annotated candidate or recorded-partner sequences used by the four targets against one another. Accepted alignments require identity ≥50%, coverage ≥80% of both sequences and E≤1e−3, with sensitivity 7.5 and a 10,000-hit cap. Undirected transitive components of the detected hits define the exclusion. This is stronger than removing only direct high-identity hits, but search heuristics and the specified sequence universe prevent a claim that every homolog has been excluded.',
        'Higher values support cross-modal EC3 organization beyond the identities of locally recorded partners. They do not establish new catalytic activity, exact substrate specificity, absence of training exposure, or freedom from memorization. EC3 was chosen after the preceding hierarchy analysis; this is exploratory evidence on already inspected benchmarks. The exact-association coverage results remain in the previous report and manuscript; this analysis complements them rather than replacing an unfavorable result.',
        '## Reproducibility',
        'Sources: [frozen protocol](protocol.json), [cohort/source audit](prepared.json), [all curves](all_curves.csv), [k=10 values](summary_at10.csv), [scope](scope.csv), [sequence-search receipt](homology/complete.json), and per-partition per-query/per-class arrays. The analysis and report scripts are `scripts/cersei_crossmodal_functional_recovery.py` and `scripts/cersei_crossmodal_functional_report.py`. Figures use SciencePlots and the requested blue/orange/pink/green palette, with explanations outside the plots.',
    ]
    checked = read(OUT/'validation.json')
    assert checked['all_eight_panels_complete'] and len(checked['records']) == 58
    max_tie_effect = max(read(OUT/s/e/'complete.json')['sources'][m]['max_stable_vs_expected_difference']
                        for s in SPLITS for e in FILTERS for m in available(s))
    report.extend([
        '## Validation and numerical sensitivity',
        f'All eight panels and 58 method/stage evaluations passed the [independent checker](validation.json). It recomputes class-macro aggregation from per-query values; verifies annotations, partner/alias and component exclusions for every saved top-50 list; and independently fully sorts seven hash-selected queries per method/panel to check ranks and tie expectations at k=1, 10 and 50. Algorithm checks include a tied block extending beyond rank 50 and a multilabel macro-weighting example. These are numerical/protocol checks, not independent training replications.',
        f'Across all displayed final methods, policies and k=1–50, replacing exact-tie expectations with stable candidate-index tie resolution changes a macro score by at most {100*max_tie_effect:.5f} percentage points. The reported conclusions do not depend on a favorable tie ordering. All source score hashes and cohort identities are retained.'
    ])
    (OUT/'README.md').write_text('\n\n'.join(report)+'\n')
    dump(OUT/'observed_rankings.json',leaders)
    # Standalone LaTeX tables; prose and inclusion are reviewed separately.
    tex=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Partition & Exclusion & \method{} & Horizyn & CREEP & CLIPZyme & Random \\',r'\midrule']
    for s in SPLITS:
        for e in FILTERS:
            vals=[f'{value(s,e,m):.2f}' if m in available(s) else '--' for m in MODELS[:-1]]+[f'{value(s,e,"random"):.2f}']
            tex.append(NAMES[s]+' & '+('Partners' if e=='partners' else 'Components')+' & '+' & '.join(vals)+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}']);(OUT/'comparison_table.tex').write_text('\n'.join(tex)+'\n')
    tex=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Partition & Exclusion & $(1,0)$ & $(1,0.2)$ & $(2,0)$ & $(2,0.2)$ & $(2,0.1)$ \\',r'\midrule']
    for s in SPLITS:
        for e in FILTERS:
            vals=[f'{value(s,e,m):.2f}' for m in ['phase1_native','phase2_native','phase1_fusion2','phase2_fusion2','cersei']]
            tex.append(NAMES[s]+' & '+('Partners' if e=='partners' else 'Components')+' & '+' & '.join(vals)+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}']);(OUT/'stage_table.tex').write_text('\n'.join(tex)+'\n')
    assets=PAPER/'assets/representation_analysis';assets.mkdir(exist_ok=True)
    for name in ['functional_recovery_curves','refinement_effects_at10']:shutil.copy2(OUT/(name+'.pdf'),assets/(name+'.pdf'))
    for name in ['comparison','stage']:shutil.copy2(OUT/(name+'_table.tex'),PAPER/'sections/appendix/tables'/('crossmodal_functional_'+name+'.tex'))
    tex=[r'\begin{table}[htbp]',r'\centering\footnotesize',
        r'\caption{Functional-recovery cohorts. The eligible-bank range is after query-specific exclusions. Queries without a compatible alternative remain in the metric. All methods share the cohort within each row.}',
        r'\label{tab:crossmodal-functional-scope}',r'\resizebox{\linewidth}{!}{\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',
        r'Partition & Exclusion & Queries & EC3 classes & Families & Eligible-bank range & No alternative \\',r'\midrule']
    for s in scopes:
        tex.append(f'{NAMES[s["split"]]} & '+('Partners' if s['exclusion']=='partners' else 'Components')+
            f' & {s["queries"]:,} & {s["classes"]} & {s["families"]} & {s["eligible_candidate_min"]:,}--{s["eligible_candidate_max"]:,} & {s["queries_without_alternative"]}'+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}}',r'\end{table}'])
    (PAPER/'sections/appendix/crossmodal_functional_scope.tex').write_text('\n'.join(tex)+'\n')
    tex=[r'\begin{table}[htbp]',r'\centering\small',
        r'\caption{Paired differences in functional recovery@10: \method{} minus each comparator, in percentage points. Brackets give 95\% reaction-family multiplier intervals, conditional on the fixed checkpoints and candidate banks.}',
        r'\label{tab:crossmodal-functional-uncertainty}',r'\begin{tabular}{@{}lllrr@{}}',r'\toprule',
        r'Partition & Exclusion & Comparator & Difference & 95\% interval \\',r'\midrule']
    for r in paired:tex.append(' & '.join(r[:3]+['$'+r[3]+'$','$'+r[4]+'$'])+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (PAPER/'sections/appendix/crossmodal_functional_uncertainty.tex').write_text('\n'.join(tex)+'\n')
    dump(OUT/'report_receipt.json',dict(created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        all_curves_sha256=sha(OUT/'all_curves.csv'),report_sha256=sha(OUT/'README.md'),scienceplots=True,
        figures={name:sha(OUT/name) for name in ['functional_recovery_curves.pdf','refinement_effects_at10.pdf']},script_sha256=sha(__file__)))
    print('Report and plots exported',OUT,flush=True)

if __name__=='__main__':main()
