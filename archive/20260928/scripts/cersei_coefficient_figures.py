#!/usr/bin/env python3
"""Validate and plot the completed fixed-checkpoint coefficient campaign."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D

BLUE='#3B6FB6'; ORANGE='#CF792F'; INK='#303030'; GRAY='#777777'
TASKS=['reaction_smi','enzyme_smi','time','enzymemap']
NAMES=dict(reaction_smi='Reaction-Sim',enzyme_smi='Enzyme-Sim',time='Time',enzymemap='EnzymeMap')
KEYS=['alpha','cap','nu']; LABELS=[r'Dictionary weight $\alpha$',r'Residual multiplier $c$',r'Protein-fusion multiplier $\nu$']
DEFAULT=dict(alpha=.4,cap=.5,nu=3.)

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tag(r):return 'a%03d_c%03d_n%03d'%tuple(round(r[k]*100) for k in KEYS)
def value(result,metric):
    if metric in ('reaction_to_enzyme','enzyme_to_reaction'): return result['summary'][metric]['all']['reactzyme_mrr']
    table,m=metric.split('/');return result['summary'][table][m]
def save(fig,path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=200,bbox_inches='tight');plt.close(fig)
def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#E5E5E5',lw=.5);ax.set_axisbelow(True)
    ax.tick_params(length=3,pad=3)

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--paper',type=Path)
    args=p.parse_args();root=args.campaign;plan=read(root/'protocol.json');arms=plan['arms'];allrows=[];data={};audit=[]
    figs=root/'figures';figs.mkdir(exist_ok=True)
    contract=dict(question='How do the three inference coefficients affect fixed-checkpoint retrieval, and what happens when score components are removed?',
        audience='CERSEI paper readers',surface='Standalone vector PDF and raster preview, embedded in LaTeX appendix',
        data_sufficiency='21 dictionary weights, 7 residual multipliers, 10 fusion multipliers, 6x4 dictionary/residual grid; complete official pools and all arms, seed42.',
        charts=[dict(name='reactzyme_coefficient_curves',form='3x3 line/point small multiples',x=LABELS,y='Change in all-positive query-macro MRR from default, percentage points; panel-specific ranges',series='E to R solid filled circles; R to E dashed open squares'),
                dict(name='enzymemap_coefficient_curves',form='4x3 line/point small multiples',x=LABELS,y='Change from default: BEDROC85/20 percentage points, EF5/10 fold enrichment; panel-specific ranges',series='full solid filled circles; ID-excluded dashed open squares'),
                dict(name='coefficient_interactions',form='4x2 annotated matrix',x='dictionary alpha',y='residual c',units='percentage-point metric difference from fixed default'),
                dict(name='coefficient_tie_diagnostic',form='3x3 line/point small multiples',x=LABELS,y='Absolute E to R MRR',series='Reported candidate-index tie break versus expected reciprocal rank over uniformly random exact-score ties')],
        titles='Neutral descriptive titles; subtitle states fixed checkpoints and complete pools.',
        renderer='Matplotlib',palette='Hard two-root cap: blue #3B6FB6 and orange #CF792F plus neutrals',
        grayscale='Distinct markers, open fills and line styles; matrix values explicitly printed',
        reference='Vertical gray dashed default coefficient and larger open circle at default outcome',
        uncertainty='No seed-variance CIs from one trained fit. No smoothing. Scores are exploratory repeated-test evaluations; no default reselection.',
        export_paths=[str(figs/(n+'.pdf')) for n in ['reactzyme_coefficient_curves','enzymemap_coefficient_curves','coefficient_interactions','coefficient_tie_diagnostic']])
    (root/'chart_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    for task in TASKS:
        done=read(root/task/'complete.json')
        assert done['arms']==len(arms) and done['protocol_sha256']==sha(root/'protocol.json')
        data[task]={}
        for a in arms:
            path=root/task/(a['id']+'.json');r=read(path);assert r['arm']==a and r['task']==task
            assert r['protocol_sha256']==sha(root/'protocol.json');data[task][a['id']]=r
            if task=='enzymemap':
                query_rows=[json.loads(line) for line in (root/task/(a['id']+'_per_query.jsonl')).read_text().splitlines()]
                assert len(query_rows)==1521
                for t in ['table1','table2']:
                    valid=[q[t] for q in query_rows if q[t] is not None]
                    assert len(valid)==r['summary'][t]['queries']
                    for m in ['bedroc85','bedroc20','ef0.05','ef0.1']:
                        assert abs(np.mean([q[m] for q in valid])-r['summary'][t][m])<1e-12
                        x=r['summary'][t];allrows.append(dict(task=task,arm=a['id'],alpha=a['alpha'],cap=a['cap'],nu=a['nu'],groups=';'.join(a['groups']),
                            metric=m,pool=t,value=x[m],queries=x['queries'],candidate_count=x['candidate_ids'],tie_expected_mrr='',source=str(path.resolve()),source_sha256=sha(path)))
            else:
                query_arrays=np.load(root/task/(a['id']+'_per_query.npz'))
                for d in ['enzyme_to_reaction','reaction_to_enzyme']:
                    x=r['summary'][d]['all']; expected_count={'reaction_smi':(14688,386),'enzyme_smi':(8734,1573),'time':(12277,2634)}[task]
                    n=expected_count[0] if d=='enzyme_to_reaction' else expected_count[1]
                    assert x['num_queries']==n
                    assert len(query_arrays[d+'_reactzyme_mrr'])==n
                    assert abs(float(np.mean(query_arrays[d+'_reactzyme_mrr']))-x['reactzyme_mrr'])<1e-8
                    assert abs(float(np.mean(query_arrays[d+'_tie_expected_mrr']))-r['tie_diagnostics'][d]['expected_tie_mrr'])<1e-12
                    assert np.array_equal(query_arrays[d+'_query_index'],query_arrays[d+'_tie_query_index'])
                    allrows.append(dict(task=task,arm=a['id'],alpha=a['alpha'],cap=a['cap'],nu=a['nu'],groups=';'.join(a['groups']),metric='reactzyme_mrr',pool=d,
                        value=x['reactzyme_mrr'],queries=x['num_queries'],candidate_count=expected_count[1] if d=='enzyme_to_reaction' else expected_count[0],
                        tie_expected_mrr=r['tie_diagnostics'][d]['expected_tie_mrr'],source=str(path.resolve()),source_sha256=sha(path)))
            if 'default' in a['groups']:
                assert r['default_max_metric_error']<=1e-6
                audit.append(dict(task=task,default_max_metric_error=r['default_max_metric_error'],default_max_score_error=r.get('default_max_score_error')))
        dictionary_only=[value(data[task][a['id']], 'table1/bedroc85' if task=='enzymemap' else 'enzyme_to_reaction') for a in arms if a['alpha']==1]
        spread=max(dictionary_only)-min(dictionary_only)
        # Repeated FP32 GPU sparse products can perturb near-equal screening
        # scores. Record their scale; never reinterpret it as a c effect.
        assert spread<(2e-6 if task=='enzymemap' else 1e-8), 'Dictionary-only repeatability exceeded tolerance'
        audit[-1]['dictionary_only_metric_spread']=spread
    with (root/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                         'axes.edgecolor':INK,'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42})
    handles=[Line2D([],[],color=BLUE,marker='o',lw=1.4,ms=3,label=r'E$\to$R'),
             Line2D([],[],color=ORANGE,ls='--',marker='s',mfc='white',lw=1.4,ms=3,label=r'R$\to$E')]
    fig,axes=plt.subplots(3,3,figsize=(7.0,6.1),layout='constrained')
    fig.suptitle('ReactZyme coefficient sensitivity\nChange from default; fixed checkpoints, seed 42; complete candidate sets',fontsize=10)
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
    for i,task in enumerate(TASKS[:3]):
        for j,k in enumerate(KEYS):
            ax=axes[i,j];style(ax)
            subset=sorted((a for a in arms if k in a['groups']),key=lambda a:a[k]);x=[a[k] for a in subset]
            for d,color,mark,ls in [('enzyme_to_reaction',BLUE,'o','-'),('reaction_to_enzyme',ORANGE,'s','--')]:
                baseline=value(data[task][tag(DEFAULT)],d)
                y=[100*(value(data[task][a['id']],d)-baseline) for a in subset]
                ax.plot(x,y,color=color,marker=mark,ls=ls,lw=1.25,ms=2.6,mfc=color if mark=='o' else 'white')
                ax.plot(DEFAULT[k],0,marker='o',mfc='none',mec=INK,ms=7,lw=0)
            ax.axvline(DEFAULT[k],color=GRAY,ls=':',lw=.8)
            ax.axhline(0,color=GRAY,lw=.6)
            ax.set_xlabel(LABELS[j]);ax.set_ylabel(r'$\Delta$MRR (pp)' if j==0 else '')
            ax.set_title(NAMES[task]);ax.margins(y=.15)
            if k=='alpha':ax.set_xticks([0,.2,.4,.6,.8,1])
            if k=='cap':ax.set_xticks([0,.5,1,1.5])
            if k=='nu':ax.set_xticks([0,1,2,3,4,5,6])
    save(fig,figs/'reactzyme_coefficient_curves')
    fig,axes=plt.subplots(3,3,figsize=(7.0,6.1),layout='constrained')
    fig.suptitle(r'ReactZyme E$\to$R exact-tie diagnostic'+'\nFixed checkpoints, seed 42; complete candidate sets; panel-specific ranges',fontsize=10)
    handles=[Line2D([],[],color=BLUE,marker='o',lw=1.4,ms=3,label='Candidate-index ties'),
             Line2D([],[],color=ORANGE,ls='--',marker='s',mfc='white',lw=1.4,ms=3,label='Uniform-tie expectation')]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
    for i,task in enumerate(TASKS[:3]):
        for j,k in enumerate(KEYS):
            ax=axes[i,j];style(ax);subset=sorted((a for a in arms if k in a['groups']),key=lambda a:a[k]);x=[a[k] for a in subset]
            official=[value(data[task][a['id']],'enzyme_to_reaction') for a in subset]
            expected=[data[task][a['id']]['tie_diagnostics']['enzyme_to_reaction']['expected_tie_mrr'] for a in subset]
            ax.plot(x,official,color=BLUE,marker='o',lw=1.25,ms=2.6)
            ax.plot(x,expected,color=ORANGE,marker='s',ls='--',mfc='white',lw=1.25,ms=2.6)
            ax.axvline(DEFAULT[k],color=GRAY,ls=':',lw=.8)
            ax.set_xlabel(LABELS[j]);ax.set_ylabel('MRR' if j==0 else '');ax.set_title(NAMES[task]);ax.margins(y=.15)
            if k=='alpha':ax.set_xticks([0,.2,.4,.6,.8,1])
            if k=='cap':ax.set_xticks([0,.5,1,1.5])
            if k=='nu':ax.set_xticks([0,1,2,3,4,5,6])
    save(fig,figs/'coefficient_tie_diagnostic')
    fig,axes=plt.subplots(4,3,figsize=(7.0,7.5),layout='constrained')
    fig.suptitle('EnzymeMap coefficient sensitivity\nChange from default; fixed checkpoint, seed 42; complete libraries',fontsize=10)
    handles=[Line2D([],[],color=BLUE,marker='o',lw=1.4,ms=3,label='Full library'),Line2D([],[],color=ORANGE,ls='--',marker='s',mfc='white',lw=1.4,ms=3,label='ID-excluded')]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
    for i,(metric,label) in enumerate([('bedroc85',r'BEDROC$_{85}$'),('bedroc20',r'BEDROC$_{20}$'),('ef0.05','EF5 (fold)'),('ef0.1','EF10 (fold)')]):
        for j,k in enumerate(KEYS):
            ax=axes[i,j];style(ax);subset=sorted((a for a in arms if k in a['groups']),key=lambda a:a[k]);x=[a[k] for a in subset]
            for table,color,mark,ls in [('table1',BLUE,'o','-'),('table2',ORANGE,'s','--')]:
                factor=100 if metric.startswith('bedroc') else 1
                baseline=value(data['enzymemap'][tag(DEFAULT)],table+'/'+metric)
                y=[factor*(value(data['enzymemap'][a['id']],table+'/'+metric)-baseline) for a in subset]
                ax.plot(x,y,color=color,marker=mark,ls=ls,lw=1.25,ms=2.6,mfc=color if mark=='o' else 'white')
                ax.plot(DEFAULT[k],0,marker='o',mfc='none',mec=INK,ms=7,lw=0)
            ax.axvline(DEFAULT[k],color=GRAY,ls=':',lw=.8);ax.axhline(0,color=GRAY,lw=.6)
            ax.set_xlabel(LABELS[j]);ax.set_ylabel(r'$\Delta$'+label+(' (pp)' if metric.startswith('bedroc') else '') if j==0 else '')
            ax.margins(y=.15)
            if k=='alpha':ax.set_xticks([0,.2,.4,.6,.8,1])
            if k=='cap':ax.set_xticks([0,.5,1,1.5])
            if k=='nu':ax.set_xticks([0,1,2,3,4,5,6])
    save(fig,figs/'enzymemap_coefficient_curves')
    matrices=[]
    for task in TASKS:
        for metric in (['table1/bedroc85','table1/bedroc20'] if task=='enzymemap' else ['enzyme_to_reaction','reaction_to_enzyme']):
            base=value(data[task][tag(DEFAULT)],metric)
            mat=np.array([[100*(value(data[task][tag(dict(alpha=a,cap=c,nu=3.))],metric)-base)
                           for a in plan['grids']['interaction_alpha']] for c in plan['grids']['interaction_cap']])
            matrices.append((task,metric,mat))
    limit=max(abs(m).max() for _,_,m in matrices)
    cmap=LinearSegmentedColormap.from_list('orange_white_blue',[ORANGE,'#FFFFFF',BLUE]);norm=TwoSlopeNorm(0,vmin=-limit,vmax=limit)
    fig,axes=plt.subplots(4,2,figsize=(7.0,8.0),layout='constrained')
    fig.suptitle('Dictionary–residual interactions\nMetric change from the default (percentage points); protein fusion fixed at 3',fontsize=11)
    for ax,(task,metric,mat) in zip(axes.flat,matrices):
        im=ax.imshow(mat,cmap=cmap,norm=norm,aspect='auto')
        for (i,j),v in np.ndenumerate(mat):ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=7,color='white' if abs(v)>.7*limit else INK)
        ax.set_xticks(range(6),plan['grids']['interaction_alpha']);ax.set_yticks(range(4),plan['grids']['interaction_cap'])
        ax.set_xlabel(r'Dictionary weight $\alpha$');ax.set_ylabel(r'Residual $c$')
        label={ 'enzyme_to_reaction':r'E$\to$R MRR','reaction_to_enzyme':r'R$\to$E MRR','table1/bedroc85':r'full BEDROC$_{85}$','table1/bedroc20':r'full BEDROC$_{20}$'}[metric]
        ax.set_title(NAMES[task]+' · '+label)
        for s in ax.spines.values():s.set_visible(False)
        ax.scatter([2],[1],s=260,facecolors='none',edgecolors=INK,linewidths=1)
    fig.colorbar(im,ax=axes,shrink=.8,label='Difference from default (percentage points)',pad=.02)
    save(fig,figs/'coefficient_interactions')
    controls=[('Default',DEFAULT),('No dictionary',dict(alpha=0.,cap=.5,nu=3.)),('Dictionary only',dict(alpha=1.,cap=.5,nu=3.)),
              ('No residual',dict(alpha=.4,cap=0.,nu=3.)),('No fused branch',dict(alpha=.4,cap=.5,nu=0.)),
              ('No dictionary or residual',dict(alpha=0.,cap=0.,nu=3.)),('Native neural score',dict(alpha=0.,cap=0.,nu=1.))]
    controls_rows=[]
    for label,a in controls:
        values=[]
        for t in TASKS[:3]:
            for metric in ['enzyme_to_reaction','reaction_to_enzyme']:values.append(value(data[t][tag(a)],metric))
        for metric in ['table1/bedroc85','table2/bedroc85']:values.append(value(data['enzymemap'][tag(a)],metric))
        controls_rows.append(dict(label=label,**a,values=values))
    table=['% Generated by cersei_coefficient_figures.py from the frozen full-pool evaluation.',r'\begin{tabular}{@{}lrrrrrrrr@{}}',r'\toprule',
           r'& \multicolumn{2}{c}{Reaction-Sim} & \multicolumn{2}{c}{Enzyme-Sim} & \multicolumn{2}{c}{Time} & \multicolumn{2}{c}{EnzymeMap}\\',
           r'Setting & E$\to$R & R$\to$E & E$\to$R & R$\to$E & E$\to$R & R$\to$E & Full & Excl.\\',r'\midrule']
    for row in controls_rows:table.append(row['label']+' & '+' & '.join(f'{v:.4f}' for v in row['values'])+r'\\')
    table.extend([r'\bottomrule',r'\end{tabular}'])
    (root/'coefficient_controls.tex').write_text('\n'.join(table)+'\n')
    (root/'controls.json').write_text(json.dumps(controls_rows,indent=2)+'\n')
    # Full curves remain the primary output; extrema are descriptive, never selectors.
    extrema=[]
    for task in TASKS:
        metrics=['table1/bedroc85','table1/bedroc20','table2/bedroc85'] if task=='enzymemap' else ['enzyme_to_reaction','reaction_to_enzyme']
        for k in KEYS:
            for metric in metrics:
                subset=[a for a in arms if k in a['groups']];best=max(subset,key=lambda a:value(data[task][a['id']],metric))
                extrema.append(dict(task=task,coefficient=k,metric=metric,descriptive_maximum=best[k],value=value(data[task][best['id']],metric),default=value(data[task][tag(DEFAULT)],metric)))
    (root/'descriptive_extrema.json').write_text(json.dumps(extrema,indent=2)+'\n')
    (root/'validation.json').write_text(json.dumps(dict(passed=True,settings_per_task=len(arms),evaluations=len(arms)*4,csv_rows=len(allrows),default_parity=audit,
        protocol_sha256=sha(root/'protocol.json'),results_sha256=sha(root/'results.csv'),source_sha256=sha(__file__)),indent=2)+'\n')
    if args.paper:
        target=args.paper/'assets';target.mkdir(exist_ok=True)
        for p in figs.glob('*.pdf'):shutil.copy2(p,target/('CERSEI-'+p.name))
        notes=args.paper/'notes/coefficient_sweep_20260923';notes.mkdir(parents=True,exist_ok=True)
        for name in ['protocol.json','results.csv','controls.json','descriptive_extrema.json','validation.json','chart_contract.json','cersei_coefficient_sweep.py','restored_residue_path.json']:
            shutil.copy2(root/name,notes/name)
        shutil.copy2(__file__,notes/Path(__file__).name)
        shutil.copy2(root/'coefficient_controls.tex',args.paper/'sections/appendix/tables/coefficient_controls.tex')
    print(json.dumps(dict(evaluations=len(arms)*4,figure_dir=str(figs),controls=controls_rows,parity=audit),indent=2))

if __name__=='__main__':main()
