#!/usr/bin/env python3
"""Verify and publish the complete dictionary-free coefficient study."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps/ablation-figures'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style.core
from matplotlib.lines import Line2D
import scienceplots

TASKS = ['reaction_smi', 'enzyme_smi', 'time', 'enzymemap']
SPLITS = ['Reaction-Sim', 'Enzyme-Sim', 'Time']
DIRS = ['reaction_to_enzyme', 'enzyme_to_reaction']
METRICS = ['bedroc85', 'bedroc20', 'ef0.05', 'ef0.1']
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, x): Path(p).write_text(json.dumps(x, indent=2) + '\n')


def main(campaign, paper):
    plan = read(campaign / 'protocol.json')
    notes = paper / 'notes/dictionary_free_coefficients_20260924'
    assets = paper / 'assets/coefficient_sensitivity'
    tables = paper / 'sections/experiments/tables'
    for p in [notes, assets]: p.mkdir(exist_ok=True, parents=True)
    data, rows, checks = {}, [], {}
    for task in TASKS:
        assert read(campaign / task / 'complete.json')['arms'] == len(plan['arms'])
        data[task] = {}
        for arm in plan['arms']:
            p = campaign / task / (arm['id'] + '.json'); x = read(p)
            assert x['arm'] == arm and x['protocol_sha256'] == sha(campaign / 'protocol.json')
            data[task][arm['id']] = x
            if task == 'enzymemap':
                per = [json.loads(line) for line in (p.parent / (arm['id']+'_per_query.jsonl')).read_text().splitlines()]
                for pool in ['table1','table2']:
                    selected = [v[pool] for v in per if v[pool] is not None]
                    assert len(selected) == x['summary'][pool]['queries']
                    for metric in METRICS:
                        value = x['summary'][pool][metric]
                        assert abs(np.mean([v[metric] for v in selected])-value) < 1e-12
                        rows.append(dict(task=task, arm=arm['id'], alpha_inf=arm['fusion_multiplier'],
                            kappa_inf=arm['kappa_inf'], metric=metric, pool=pool, value=value,
                            expected_tie_mrr='', queries=len(selected), candidates=x['summary'][pool]['candidate_ids'],
                            source=str(p), source_sha256=sha(p)))
            else:
                per = np.load(p.parent / (arm['id']+'_per_query.npz'))
                for d in DIRS:
                    summary = x['summary'][d]['all']; value = summary['reactzyme_mrr']
                    # Float32 mean reductions can differ by one ULP across NumPy builds.
                    assert abs(float(np.mean(per[d+'_reactzyme_mrr'], dtype=np.float64))-value) < 2e-7
                    ties = x['tie_diagnostics'][d]['expected_tie_mrr']
                    assert abs(np.mean(per[d+'_tie_expected_mrr'])-ties) < 1e-12
                    rows.append(dict(task=task, arm=arm['id'], alpha_inf=arm['fusion_multiplier'],
                        kappa_inf=arm['kappa_inf'], metric='MRR', pool=d, value=value,
                        expected_tie_mrr=ties, queries=summary['num_queries'],
                        candidates=x['summary'][DIRS[1-DIRS.index(d)]]['all']['num_queries'],
                        source=str(p), source_sha256=sha(p)))
        ref = data[task]['a2_k0.1']
        checks[task] = {k:v for k,v in ref.items() if k.startswith('reference_')}
        assert checks[task]['reference_max_score_error'] < 1e-6
        assert checks[task]['reference_max_metric_error'] < 1e-6
    assert len(rows) == len(plan['arms']) * 14
    with (notes / 'results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    shutil.copy2(campaign / 'protocol.json', notes / 'protocol.json')
    shutil.copy2(__file__, notes / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('cersei_dictionary_free_sensitivity.py'), notes / 'cersei_dictionary_free_sensitivity.py')
    # Keep the immutable source summaries and validation-selection evidence reviewable.
    for task in TASKS:
        p = notes / 'sources' / task; p.mkdir(parents=True, exist_ok=True)
        for arm in plan['arms']:
            shutil.copy2(campaign / task / (arm['id'] + '.json'), p / (arm['id'] + '.json'))
        shutil.copy2(plan['tasks'][task]['validation'], p / 'validation.json')
    dump(notes / 'validation.json', dict(complete=True, tasks=4, arms_per_task=len(plan['arms']),
        metric_rows=len(rows), default_reproduction=checks, all_per_query_means_verified=True,
        per_query_mean_tolerance=dict(reactzyme=2e-7, enzymemap=1e-12),
        protocol_sha256=sha(campaign / 'protocol.json'), results_sha256=sha(notes / 'results.csv')))
    contract = dict(question='How do inference fusion and residual coefficients affect the current dictionary-free checkpoints?',
        family='Comparison; fixed-checkpoint sensitivity lines', source='results.csv, all 238 metric rows retained',
        plots=dict(reactzyme='2x2: retrieval direction by coefficient, delta MRR in percentage points from (2,0.1)',
                   enzymemap='2x4: coefficient by metric; both complete screening libraries',
                   ties='E-to-R MRR sensitivity, candidate-index versus uniform-exact-tie ordering'),
        palette=dict(blue=COLORS[0], orange=COLORS[1], green=COLORS[2], pink=COLORS[3]),
        redundant_encoding='ReactZyme split-specific markers; screening pool markers/line style; ties dashed',
        reference='Grey dotted vertical lines: alpha_inf=2 or kappa_inf=0.1; no best-test selection',
        exclusions='No titles, subtitles, plot footnotes, confidence intervals, or hidden unfavorable arms',
        limits='Retrospective test sensitivity, one fit per split; no interpolation claims; exact ties do not cover near ties')
    dump(notes / 'chart_contract.json', contract)
    plt.style.use(['science', 'no-latex'])
    plt.rcParams.update({'font.size':9, 'axes.labelsize':9, 'legend.fontsize':8,
                         'xtick.labelsize':8, 'ytick.labelsize':8, 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False, 'savefig.dpi':220})
    def select(key):
        fixed = 'kappa_inf' if key == 'fusion_multiplier' else 'fusion_multiplier'
        val = .1 if fixed == 'kappa_inf' else 2.
        return sorted([a for a in plan['arms'] if a[fixed] == val], key=lambda a:a[key])
    def save(fig, name):
        for ext in ['pdf','png','svg']:
            target = assets / (name+'.'+ext)
            fig.savefig(target, bbox_inches='tight', pad_inches=.035)
            if ext == 'svg':
                target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
        plt.close(fig)
    reference = Line2D([], [], color='0.5', ls=':', label='Reference coefficient')
    keys = ['fusion_multiplier','kappa_inf']
    labels = [r'Fusion multiplier $\alpha_{\mathrm{inf}}$', r'Residual coefficient $\kappa_{\mathrm{inf}}$']
    fig, axs = plt.subplots(2,2,figsize=(6.6,4.0))
    for j,key in enumerate(keys):
        arms = select(key); xx = [a[key] for a in arms]
        for i,d in enumerate(DIRS):
            ax = axs[i,j]; ax.axhline(0,color='.8',lw=.7,zorder=0)
            ax.axvline([2.,.1][j],color='.5',ls=':',lw=.9,zorder=0)
            for t,(task,name) in enumerate(zip(TASKS[:3],SPLITS)):
                ref = data[task]['a2_k0.1']['summary'][d]['all']['reactzyme_mrr']
                yy = [100*(data[task][a['id']]['summary'][d]['all']['reactzyme_mrr']-ref) for a in arms]
                ax.plot(xx,yy,color=COLORS[t],marker=['o','s','^'][t],ms=3,lw=1.2,label=name)
            ax.set_ylabel([r'$R\!\to\!E$ $\Delta$MRR (pp)', r'$E\!\to\!R$ $\Delta$MRR (pp)'][i])
            ax.set_xlabel(labels[j]); ax.set_xticks([0,1,2,3,4] if j==0 else [0,.1,.2,.3])
    handles,_ = axs[0,0].get_legend_handles_labels()
    fig.legend(handles=handles+[reference],loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.51,1.015))
    fig.subplots_adjust(wspace=.34,hspace=.42,top=.90)
    save(fig,'reactzyme_dictionary_free_coefficients')

    fig,axs = plt.subplots(2,4,figsize=(6.8,3.9))
    for i,key in enumerate(keys):
        arms=select(key);xx=[a[key] for a in arms]
        for j,metric in enumerate(METRICS):
            ax=axs[i,j];ax.axvline([2.,.1][i],color='.5',ls=':',lw=.9,zorder=0)
            factor=100 if metric.startswith('bedroc') else 1
            for k,(pool,label) in enumerate([('table1','Full library'),('table2','Training-ID-excluded')]):
                yy=[factor*data['enzymemap'][a['id']]['summary'][pool][metric] for a in arms]
                ax.plot(xx,yy,color=COLORS[k],marker=['o','s'][k],ls=['-','--'][k],ms=2.8,lw=1.1,label=label)
            ax.set_xlabel([r'$\alpha_{\mathrm{inf}}$',r'$\kappa_{\mathrm{inf}}$'][i])
            ax.set_ylabel([r'BEDROC$_{85}$ (%)',r'BEDROC$_{20}$ (%)','EF5 (fold)','EF10 (fold)'][j])
            ax.set_xticks([0,2,4] if i==0 else [0,.1,.2,.3])
    handles,_=axs[0,0].get_legend_handles_labels()
    fig.legend(handles=handles+[reference],loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.51,1.02))
    fig.subplots_adjust(wspace=.63,hspace=.48,top=.89)
    save(fig,'enzymemap_dictionary_free_coefficients')

    fig,axs=plt.subplots(1,2,figsize=(6.6,2.3))
    for j,key in enumerate(keys):
        arms=select(key);xx=[a[key] for a in arms];ax=axs[j]
        ax.axvline([2.,.1][j],color='.5',ls=':',lw=.9)
        for t,task in enumerate(['enzyme_smi','time']):
            for mode,ls in [('indexed','-'),('expected','--')]:
                yy=[data[task][a['id']]['summary']['enzyme_to_reaction']['all']['reactzyme_mrr'] if mode=='indexed'
                    else data[task][a['id']]['tie_diagnostics']['enzyme_to_reaction']['expected_tie_mrr'] for a in arms]
                ax.plot(xx,yy,color=COLORS[t+1],ls=ls,lw=1.2,marker=['s','^'][t],ms=2.5)
        ax.set_xlabel(labels[j]);ax.set_ylabel(r'$E\!\to\!R$ MRR')
    fig.legend(handles=[Line2D([],[],color=COLORS[1],marker='s',label='Enzyme-Sim'),
                        Line2D([],[],color=COLORS[2],marker='^',label='Time'),
                        Line2D([],[],color='.3',ls='-',label='Candidate order'),
                        Line2D([],[],color='.3',ls='--',label='Expected within ties')],
               loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.51,1.09))
    fig.subplots_adjust(wspace=.34,top=.80)
    save(fig,'dictionary_free_tie_diagnostic')
    arms=sorted(plan['arms'],key=lambda a:(a['fusion_multiplier'],a['kappa_inf']))
    def fmt(a): return f"{a['fusion_multiplier']:g} & {a['kappa_inf']:g}"
    rt=[r'\begin{tabular}{ccrrrrrr}',r'\toprule',
        r' & & \multicolumn{2}{c}{Reaction-Sim} & \multicolumn{2}{c}{Enzyme-Sim} & \multicolumn{2}{c}{Time} \\',
        r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
        r'$\alpha_{\mathrm{inf}}$ & $\kappa_{\mathrm{inf}}$ & $R\!\to\!E$ & $E\!\to\!R$ & $R\!\to\!E$ & $E\!\to\!R$ & $R\!\to\!E$ & $E\!\to\!R$ \\',r'\midrule']
    mt=[r'\begin{tabular}{ccrrrrrrrr}',r'\toprule',
        r' & & \multicolumn{4}{c}{Full library} & \multicolumn{4}{c}{Training-ID-excluded} \\',
        r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}',
        r'$\alpha_{\mathrm{inf}}$ & $\kappa_{\mathrm{inf}}$ & B85 & B20 & EF5 & EF10 & B85 & B20 & EF5 & EF10 \\',r'\midrule']
    for arm in arms:
        prefix=r'\rowcolor{blue!10}'+'\n' if arm['id']=='a2_k0.1' else ''
        rv=[data[t][arm['id']]['summary'][d]['all']['reactzyme_mrr'] for t in TASKS[:3] for d in DIRS]
        rt.append(prefix+fmt(arm)+' & '+' & '.join(f'{v:.4f}' for v in rv)+r' \\')
        mv=[data['enzymemap'][arm['id']]['summary'][pool][m]*(100 if m.startswith('bedroc') else 1) for pool in ['table1','table2'] for m in METRICS]
        mt.append(prefix+fmt(arm)+' & '+' & '.join(f'{v:.2f}' for v in mv)+r' \\')
    for name, lines in [('coefficient_reactzyme',rt),('coefficient_enzymemap',mt)]:
        (tables/(name+'.tex')).write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n')
    print(json.dumps(read(notes / 'validation.json'),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--paper',type=Path,required=True)
    a=p.parse_args();main(a.campaign.resolve(),a.paper.resolve())
