"""Reproducible scientific figures for the existing ablation audit.

Run: python3 documents/ablation_study_20260921/plot_study.py
Reads saved evidence only; writes PNG/SVG/PDF figures and adds them to report.md.
"""
from __future__ import annotations

import csv
import hashlib
from importlib.metadata import version
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGURES = HERE / 'figures'
try:
    import scienceplots  # Registers the actual SciencePlots styles.
    from colorspacious import cspace_convert
except ImportError:
    # Optional isolated installation, without changing the training environment.
    sys.path.insert(0, str(ROOT/'.deps/ablation-figures'))
    import scienceplots
    from colorspacious import cspace_convert

# User-selected blue–orange palette; explicit anchors, checked with CVD simulations.
BLUE, BLUE_LIGHT = '#28679C', '#ADC8E2'
ORANGE, ORANGE_LIGHT = '#B86B2D', '#E8C3A3'
INK, GREY, NEUTRAL = '#243342', '#697787', '#FAFAFA'
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    'orange_neutral_blue', [(0., ORANGE), (.25, ORANGE_LIGHT), (.5, NEUTRAL),
                              (.75, BLUE_LIGHT), (1., BLUE)], N=513)
METRICS = [('bedroc85', 'BEDROC85'), ('bedroc20', 'BEDROC20'),
           ('ef0.05', 'EF5'), ('ef0.1', 'EF10')]
SPLITS = [('reaction_smi', 'Reaction-Sim'), ('enzyme_smi', 'Enzyme-Sim'), ('time', 'Time')]
CONTRASTS = [
    ('a025_c1', 'a040_c1', 'Dictionary α: .25 → .40\nresidual = 1'),
    ('a040_c1', 'a050_c1', 'Dictionary α: .40 → .50\nresidual = 1'),
    ('a040_c1', 'a040_c05', 'Residual: 1 → .5\nα = .40 (V4)'),
]
BIOLOGY = [('all_1', 'All labels'), ('shuffled_1', 'Shuffled labels'),
           ('without_ec', 'Without EC'), ('without_cofactor', 'Without cofactor'),
           ('without_mechanism', 'Without mechanism')]


def style():
    plt.style.use(['science', 'no-latex'])
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11,
        'mathtext.fontset': 'dejavuserif', 'text.usetex': False,
        'axes.titlesize': 12, 'axes.labelsize': 10.5,
        'axes.edgecolor': GREY, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.linewidth': .65, 'grid.color': '#E1E5EA', 'grid.linewidth': .5,
        'axes.prop_cycle': plt.cycler(color=[BLUE, ORANGE, BLUE_LIGHT, ORANGE_LIGHT]),
        'ytick.minor.visible': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'axes.unicode_minus': True,
    })


def relative_luminance(rgb):
    rgb=np.asarray(rgb)[..., :3]
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    return linear @ np.array([.2126,.7152,.0722])


def annotation_color(rgb):
    luminance=float(relative_luminance(rgb))
    white_ratio=1.05/(luminance+.05)
    dark_ratio=(luminance+.05)/(float(relative_luminance(to_rgb(INK)))+.05)
    if white_ratio >= dark_ratio and white_ratio >= 4.5:
        return 'white'
    return INK if dark_ratio >= 4.5 else 'black'


def audit_palette():
    ramp=SIGNED_CMAP(np.linspace(0,1,513))[:,:3]
    views=[('Standard colour vision',ramp)]
    for cvd,label in [('protanomaly','Protanopia simulation'),('deuteranomaly','Deuteranopia simulation'),('tritanomaly','Tritanopia simulation')]:
        transformed=cspace_convert(ramp,{'name':'sRGB1+CVD','cvd_type':cvd,'severity':100},'sRGB1')
        views.append((label,np.clip(transformed,0,1)))
    qa=[]
    fig,axs=plt.subplots(4,1,figsize=(10,3.4))
    fig.subplots_adjust(left=.26,right=.96,top=.98,bottom=.12,hspace=.7)
    for ax,(label,rgb) in zip(axs,views):
        lum=relative_luminance(rgb)
        left=bool(np.all(np.diff(lum[:257])>=-1e-6))
        right=bool(np.all(np.diff(lum[256:])<=1e-6))
        assert left and right, f'Non-monotonic gradient luminance: {label}'
        lab=cspace_convert(rgb[[0,-1]],'sRGB1','CAM02-UCS')
        qa.append(dict(view=label,endpoint_distance_CAM02_UCS=float(np.linalg.norm(lab[0]-lab[1])),monotonic_luminance_toward_neutral=left and right))
        ax.imshow(rgb[np.newaxis,:,:],aspect='auto',extent=(-1,1,0,1))
        ax.set_yticks([]); ax.set_ylabel(label,rotation=0,ha='right',va='center',fontsize=9)
        ax.tick_params(which='both',bottom=False,top=False,right=False,left=False)
        ax.set_xticks([-1,0,1],['Negative / orange','Zero / neutral','Positive / blue'] if ax is axs[-1] else [])
    fig.savefig(FIGURES/'palette_accessibility_preview.png',dpi=150,bbox_inches='tight')
    plt.close(fig)
    receipt=dict(style=['science','no-latex'],scienceplots_version=version('SciencePlots'),
                 palette_source='Custom blue–orange anchors declared in plot_study.py; selected by the user from palette_previews.',
                 anchors=[ORANGE,ORANGE_LIGHT,NEUTRAL,BLUE_LIGHT,BLUE],
                 interpolation='Custom signed orange–neutral–blue ramp; each side is luminance-monotonic. Not claimed perceptually uniform.',
                 simulations=qa,
                 accessibility='Signed cell labels, circular versus square markers, open versus filled markers, zero lines, and hatching convey values without relying on hue. Simulations are diagnostics, not universal accessibility certification.')
    (FIGURES/'palette_accessibility.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def clean_delta_axis(ax):
    ax.axvline(0, color=INK, linewidth=1, zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis='x')
    ax.tick_params(axis='y', length=0)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 4))
    fmt.set_useOffset(False)
    ax.xaxis.set_major_formatter(fmt)


def delta_limits(values, pad=.17):
    lo, hi = min(0., min(values)), max(0., max(values))
    width = max(hi-lo, 1e-8)
    return lo-pad*width, hi+pad*width


def extract():
    evidence = json.loads((HERE/'evidence.json').read_text())
    # Detect stale source records before drawing conclusions from cached values.
    for rel, expected in evidence['source_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel
    values = {}
    for row in evidence['records']:
        key = (row['study'], row['variant'], row['seed'], row['setting'], row['metric'])
        assert key not in values, f'Duplicate result: {key}'
        assert np.isfinite(row['value'])
        values[key] = row['value']
    assert len(values) == 260
    return evidence, values


def render_study():
    style()
    evidence, values = extract()
    FIGURES.mkdir(exist_ok=True)
    palette_receipt=audit_palette()
    manifest, plotted = [], []

    def value(study, variant, setting, metric, seed=42):
        return values[study, variant, seed, setting, metric]

    def point(figure, setting, metric, variant, reference, seed, target, control):
        plotted.append(dict(figure=figure, setting=setting, metric=metric,
                            variant=variant, reference=reference, seed=seed,
                            target=target, control=control, delta=target-control))
        return target-control

    def save(fig, name, question, contract):
        # Manuscript captions carry titles, interpretation and scope notes.
        assert fig._suptitle is None and not fig.texts, name
        assert all(not ax.get_title(loc=loc) for ax in fig.axes
                   for loc in ('left', 'center', 'right')), name
        # Draw first, so invalid text/math/layout is caught before the handoff.
        fig.canvas.draw()
        files = {}
        for extension in ('png', 'svg', 'pdf'):
            path = FIGURES/f'{name}.{extension}'
            fig.savefig(path, dpi=190, bbox_inches='tight', pad_inches=.14)
            files[extension] = dict(path=str(path.relative_to(HERE)),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        plt.close(fig)
        manifest.append(dict(id=name, question=question, contract=contract, files=files))

    def heatmap(name, columns, study, *, symlog=False):
        labels, matrix = [], []
        for setting, label in SPLITS:
            for metric, direction in [('reaction_to_enzyme', 'R→E'), ('enzyme_to_reaction', 'E→R')]:
                labels.append(f'{label}  {direction}')
                row = []
                for ref, var, _ in columns:
                    a, b = value(study, var, setting, metric), value(study, ref, setting, metric)
                    row.append(100*point(name, setting, metric, var, ref, 42, a, b))
                matrix.append(row)
        data = np.array(matrix)
        fig, ax = plt.subplots(figsize=(12.8, 5.2))
        fig.subplots_adjust(left=.235, right=.88, top=.98, bottom=.20)
        limit = float(np.max(np.abs(data)))
        cmap = SIGNED_CMAP
        if symlog:
            # One shared scale: retain cross-cell comparisons without allowing
            # the largest Time E→R effects to wash out every smaller difference.
            norm = SymLogNorm(linthresh=.001, linscale=1., base=10,
                              vmin=-limit, vmax=limit)
            scale_contract = ('Symmetric logarithmic colors, linear for |ΔMRR×100|≤0.001; '
                              'exact cell annotations remain untransformed; no clipping or row normalization.')
        else:
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
            scale_contract = 'Linear colors.'
        im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm)
        ax.set_xticks(range(len(columns)), [c[2] for c in columns], fontsize=10)
        ax.set_yticks(range(6), labels, fontsize=11)
        ax.minorticks_off()
        ax.tick_params(length=0, pad=10)
        for i in range(6):
            for j in range(len(columns)):
                ax.text(j, i, f'{data[i,j]:+.4f}', ha='center', va='center',
                        fontsize=11, color=annotation_color(cmap(im.norm(data[i,j]))))
        for y in (1.5, 3.5):
            ax.axhline(y, color='white', linewidth=2)
        cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.035)
        cb.set_label('ΔMRR × 100')
        if symlog:
            ticks = [-1, -.1, -.01, -.001, 0, .001, .01, .1, 1]
            cb.set_ticks(ticks)
            cb.set_ticklabels(['−1', '−0.1', '−0.01', '−0.001', '0',
                               '0.001', '0.01', '0.1', '1'])
            cb.ax.minorticks_off()
        save(fig, name, 'How do controlled changes affect both ReactZyme retrieval directions?',
             'Annotated signed heatmap; all six split/direction cells; one centered, symmetric color scale per figure; exact differences in MRR×100; raw MRR retained in CSV. '+scale_contract)

    heatmap('01_reactzyme_inference', CONTRASTS, 'inference')

    name = '02_enzymemap_inference'
    fig, axs = plt.subplots(2, 2, figsize=(13.8, 6.4))
    fig.subplots_adjust(left=.24, right=.97, top=.90, bottom=.12, hspace=.62, wspace=.58)
    handles = [Line2D([], [], marker='o', color=BLUE, linestyle='none', label='Table 1'),
               Line2D([], [], marker='s', color=ORANGE, linestyle='none', markerfacecolor='white', label='Table 2')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.60,1.0), ncol=2, frameon=False)
    for ax, (metric, label) in zip(axs.flat, METRICS):
        differences = []
        for j, (ref, var, _) in enumerate(CONTRASTS):
            for setting, offset, color, marker in [('table1', .13, BLUE, 'o'), ('table2', -.13, ORANGE, 's')]:
                d = point(name, setting, metric, var, ref, 42, value('inference',var,setting,metric), value('inference',ref,setting,metric))
                differences.append(d)
                ax.plot(d, 2-j+offset, marker=marker, color=color, markerfacecolor=color if marker=='o' else 'white', markersize=7, linestyle='none')
        ax.set_yticks([2,1,0], [c[2] for c in CONTRASTS], fontsize=9.5)
        ax.set_ylim(-.55,2.55)
        ax.set_xlim(*delta_limits(differences, .15))
        ax.set_xlabel(f'Δ {label}')
        clean_delta_axis(ax)
    save(fig,name,'Do dictionary and residual choices improve every screening metric?',
         'Four metric-specific signed dot plots, two screening pools distinguished by color and marker shape. Zero line included. No mixing BEDROC and EF units.')

    name = '03_temperature_paired_seeds'
    fig, axs = plt.subplots(2,4,figsize=(15.8,6.2))
    fig.subplots_adjust(left=.10,right=.98,top=.98,bottom=.11,wspace=.43,hspace=.50)
    for col,(metric,label) in enumerate(METRICS):
        group = []
        for row,setting in enumerate(('table1','table2')):
            ax = axs[row,col]
            ds = []
            for y,seed in zip([3,2,1],(17,42,73)):
                d = point(name,setting,metric,'beta5','beta10',seed,value('temperature_18epochs','beta5',setting,metric,seed),value('temperature_18epochs','beta10',setting,metric,seed))
                ds.append(d)
                ax.plot(d,y,'o',color=BLUE,markersize=7)
            avg = float(np.mean(ds)); group.extend(ds+[avg])
            ax.plot(avg,0,'D',color=INK,markersize=7)
            ax.axhline(.5,color='#CDD4DC',linewidth=.7)
            ax.set_yticks([3,2,1,0],['Seed 17','Seed 42','Seed 73','Mean'],fontsize=10)
            ax.set_ylim(-.65,3.5)
            if col == 0: ax.set_ylabel(f'Table {row+1}', labelpad=10)
            ax.set_xlabel(f'Δ {label}')
            clean_delta_axis(ax)
        for ax in axs[:,col]: ax.set_xlim(*delta_limits(group,.20))
    save(fig,name,'Is the temperature benefit consistent across paired seeds and screening protocols?',
         'Eight panels of individual paired differences plus means. Same metric scale across the two screening settings. No confidence or significance claims; the 24 metric cells are correlated.')

    heatmap('04_reactzyme_phase2_biology',
            [('control',var,label.replace(' ','\n',1)) for var,label in BIOLOGY],
            'phase2_biology', symlog=True)

    name = '05_enzymemap_phase2_biology'
    fig,axs = plt.subplots(2,4,figsize=(16.2,6.4))
    fig.subplots_adjust(left=.13,right=.98,top=.98,bottom=.12,wspace=.64,hspace=.55)
    for col,(metric,label) in enumerate(METRICS):
        group = []
        for row,setting in enumerate(('table1','table2')):
            ax = axs[row,col]
            for i,(var,_) in enumerate(BIOLOGY):
                d = point(name,setting,metric,var,'control',42,value('phase2_biology',var,setting,metric),value('phase2_biology','control',setting,metric))
                group.append(d)
                color = BLUE if i<2 else GREY
                marker = 's' if var=='shuffled_1' else 'o'
                ax.plot(d,4-i,marker=marker,color=color,markerfacecolor='white' if var=='shuffled_1' else color,markersize=6.5,linestyle='none')
            ax.set_yticks(range(4,-1,-1),[x[1] for x in BIOLOGY],fontsize=9)
            ax.set_ylim(-.5,4.5)
            if col == 0: ax.set_ylabel(f'Table {row+1}', labelpad=10)
            ax.set_xlabel(f'Δ {label}')
            clean_delta_axis(ax)
        for ax in axs[:,col]: ax.set_xlim(*delta_limits(group,.17))
    save(fig,name,'Does correctly assigned biological supervision improve screening over its controls?',
         'Metric-specific dot plots with zero reference; fixed same-metric scale across settings; shuffled uses open square and family removals neutral circles; all five contrasts shown.')

    name = '06_reactzyme_near_tie_sensitivity'
    tie_path = ROOT/'runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_phase2_20260921_v1/tie_sensitivity.json'
    ties = json.loads(tie_path.read_text())
    fig,axs = plt.subplots(1,3,figsize=(14.8,3.0))
    fig.subplots_adjust(left=.105,right=.98,top=.98,bottom=.25,wspace=.65)
    bounds_rows=[]
    for ax,(setting,label) in zip(axs,SPLITS):
        extremes=[]
        for variant,y,color,marker in [('control',1,GREY,'o'),('all_1',0,BLUE,'s')]:
            r=next(r for r in ties['records'] if r['target']==setting and r['variant']==variant)
            bound=next(b for b in r['bounds'] if b['tolerance']==1e-6)
            lo,hi=bound['pessimistic_mrr'],bound['optimistic_mrr']
            dot=value('phase2_biology',variant,setting,'enzyme_to_reaction')
            assert lo-1e-6 <= dot <= hi+1e-6
            ax.hlines(y,lo,hi,color=color,lw=1.5)
            ax.vlines([lo,hi],y-.09,y+.09,color=color,lw=1)
            ax.plot(dot,y,marker=marker,color=color,markersize=7,linestyle='none')
            extremes += [lo,hi]
            bounds_rows.append(dict(setting=setting,variant=variant,mrr=dot,lower=lo,upper=hi,score_tolerance=1e-6,positive_edges=bound['positive_edges'],near_competitor_edges=bound['positive_edges_with_near_competitor']))
        pad=(max(extremes)-min(extremes))*.12
        ax.set_xlim(min(extremes)-pad,max(extremes)+pad)
        ax.set_ylim(-.45,1.55)
        ax.set_yticks([1,0],['No added\nbiology','All labels\n(weight 1)'],fontsize=10)
        ax.set_xlabel(f'{label} E→R MRR')
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.grid(axis='x'); ax.set_axisbelow(True); ax.tick_params(axis='y',length=0)
    (FIGURES/'near_tie_bounds.json').write_text(json.dumps(bounds_rows,indent=2)+'\n')
    save(fig,name,'Can rank ordering of near-equal scores explain the large ReactZyme MRR changes?',
         'Intervals of possible near-tie orderings, not error bars or uncertainty estimates; individual split-specific zoomed scales explicitly labeled; exact source bounds retained in near_tie_bounds.json.')

    name='07_f3_biology_paired_seeds'
    fig,axs=plt.subplots(2,4,figsize=(15.8,6.5))
    fig.subplots_adjust(left=.10,right=.98,top=.90,bottom=.12,wspace=.43,hspace=.50)
    handles=[Line2D([],[],marker='o',color=BLUE,linestyle='none',label='Weight 0.1'),Line2D([],[],marker='s',color=ORANGE,markerfacecolor='white',linestyle='none',label='Weight 1')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.0),ncol=2,frameon=False)
    for col,(metric,label) in enumerate(METRICS):
        group=[]
        for row,setting in enumerate(('table1','table2')):
            ax=axs[row,col]
            for variant,offset,color,marker in [('relative_0p1',.13,BLUE,'o'),('relative_1',-.13,ORANGE,'s')]:
                ds=[]
                for y,seed in zip([3,2,1],(42,43,44)):
                    d=point(name,setting,metric,variant,'control',seed,value('f3_biology_seeds',variant,setting,metric,seed),value('f3_biology_seeds','control',setting,metric,seed))
                    ds.append(d)
                    ax.plot(d,y+offset,marker=marker,color=color,markerfacecolor=color if marker=='o' else 'white',markersize=6.5,linestyle='none')
                avg=float(np.mean(ds)); group.extend(ds+[avg])
                ax.plot(avg,offset,marker=marker,color=color,markerfacecolor=color if marker=='o' else 'white',markersize=7.5,linestyle='none')
            ax.axhline(.5,color='#CDD4DC',linewidth=.7)
            ax.set_yticks([3,2,1,0],['Seed 42','Seed 43','Seed 44','Mean'],fontsize=10)
            ax.set_ylim(-.55,3.55)
            if col == 0: ax.set_ylabel(f'Table {row+1}', labelpad=10)
            ax.set_xlabel(f'Δ {label}')
            clean_delta_axis(ax)
        for ax in axs[:,col]: ax.set_xlim(*delta_limits(group,.18))
    save(fig,name,'Is the F3 biology benefit stable across seeds and early-retrieval metrics?',
         'All paired per-seed effects plus arithmetic means, two weights encoded with color and open/filled marker shapes. No independent-test or confidence claims. Eight panels keep all screening metrics visible.')

    name='08_case1_controls'
    rows={r['variant']:r for r in evidence['datasets']['case1']}
    raw_root=ROOT/'runs/generalization_20260919_2251/cross_paper_retraining'
    control_case=json.loads((raw_root/'shared_recipe_alpha04_cap05_v1/case1/summary.json').read_text())
    exact_cases={'control':control_case['methods']['reaction_smi/selected']}
    for record in json.loads((raw_root/'v4_relative_biology_phase2_case1_controls_20260921_v1/comparison.json').read_text())['records']:
        if record['weight']==1:
            exact_cases[record['variant']]=record['methods']['reaction_smi/selected']
    order=[('control','No added biology'),('all','All labels'),('shuffled_1','Shuffled labels'),('without_ec','Without EC'),('without_cofactor','Without cofactor'),('without_mechanism','Without mechanism')]
    fig,axs=plt.subplots(1,3,figsize=(14.8,4.8))
    fig.subplots_adjust(left=.17,right=.98,top=.98,bottom=.15,wspace=.62)
    for i,(variant,label) in enumerate(order):
        row=rows[variant]; y=5-i
        u=int(row['unique'].split('/')[0]); e=int(row['entries'].split('/')[0])
        assert u==exact_cases[variant]['primary_papers']['recovered_at_25']
        assert e==exact_cases[variant]['entry_level_144']['primary_papers']['recovered_at_25']
        axs[0].barh(y,u,height=.58,color=BLUE,edgecolor=BLUE,linewidth=.8)
        axs[0].text(u-.22,y,f'{u}/12',ha='right',va='center',color='white',fontsize=11)
        axs[1].barh(y,e,height=.58,color='white',edgecolor=ORANGE,linewidth=1.2,hatch='//')
        axs[1].text(e-.22,y,f'{e}/15',ha='right',va='center',color=INK,fontsize=11,bbox=dict(facecolor='white',edgecolor='none',pad=.8))
        auc=exact_cases[variant]['broad_assay_conditional_discrimination']['auc']
        assert f'{auc:.6f}'==row['auroc']
        d=auc-exact_cases['control']['broad_assay_conditional_discrimination']['auc']
        axs[2].plot(d,y,'o',color=BLUE,markersize=6.5)
        axs[2].annotate(f'{d:+.6f}',(d,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
    axs[0].set_xlim(0,12.6); axs[0].set_xticks([0,3,6,9,12]); axs[0].set_xlabel('Unique catalysts @25')
    axs[1].set_xlim(0,15.6); axs[1].set_xticks([0,5,10,15]); axs[1].set_xlabel('Paper entries @25')
    axs[2].set_xlabel('Δ conditional AUROC')
    axs[2].set_xlim(-.0023,.0036); clean_delta_axis(axs[2])
    axs[2].ticklabel_format(axis='x',style='plain',useOffset=False)
    for ax in axs:
        ax.set_yticks(range(5,-1,-1),[label for _,label in order] if ax is axs[0] else [])
        ax.set_ylim(-.6,5.6); ax.grid(axis='x'); ax.set_axisbelow(True); ax.tick_params(axis='y',length=0)
    save(fig,name,'Does the Case1 improvement survive changing the evaluation unit and biological controls?',
         'Two zero-based count-bar panels with explicit positive/candidate denominators, plus a signed AUROC-difference dot panel. AUROC differences calculated from unrounded raw summaries; counts exact; no pooling checkpoints or labels.')
    (FIGURES/'case1_plot_values.json').write_text(json.dumps([dict(variant=variant,unique_recovery=exact_cases[variant]['primary_papers']['recovered_at_25'],entry_recovery=exact_cases[variant]['entry_level_144']['primary_papers']['recovered_at_25'],conditional_auroc=exact_cases[variant]['broad_assay_conditional_discrimination']['auc'],auroc_difference=exact_cases[variant]['broad_assay_conditional_discrimination']['auc']-exact_cases['control']['broad_assay_conditional_discrimination']['auc']) for variant,_ in order],indent=2)+'\n')

    with (FIGURES/'plotted_contrasts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(plotted[0])); writer.writeheader(); writer.writerows(plotted)
    assert len(plotted)==184
    receipt=dict(source_evidence_sha256=hashlib.sha256((HERE/'evidence.json').read_bytes()).hexdigest(),
                 source_records_verified=len(evidence['source_sha256']), metric_records=260,
                 signed_contrasts=len(plotted), figures=manifest,
                 palette=dict(blue=BLUE,blue_light=BLUE_LIGHT,orange=ORANGE,orange_light=ORANGE_LIGHT,neutral=NEUTRAL),
                 scienceplots=palette_receipt,
                 layout='Minimal manuscript figures: no titles, subtitles, panel titles or footnotes; essential axes, legends and data labels only. Explanations and scope are in report.md captions.',
                 scope='Scientific figures for the user’s research, without OpenAI branding. No new model runs or recomputed predictions.',
                 validation='Source hashes, unique metric keys, finite values, paired-seed joins and near-tie interval membership checked; visual inspection recorded separately.')
    (FIGURES/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    annotate_report()
    notes_path=HERE/'source_notes.json'
    notes=json.loads(notes_path.read_text())
    notes.update(delivery='Markdown with embedded scientific plots (explicit user request)',
                 figure_manifest='figures/manifest.json',
                 visual_contract='Eight minimal scientific Matplotlib figures using actual SciencePlots science/no-latex styles, serif fonts and orange–neutral–blue gradients using the user-selected custom anchors. No titles, subtitles, panel titles or footnotes in the figures; explanations in report.md captions. CVD simulations checked; paired seeds and all metrics retained; PNG plus SVG/PDF exports.',
                 delivery_status='Markdown report and 24 figure exports generated; legacy HTML failure is unrelated to this requested surface.')
    notes_path.write_text(json.dumps(notes,indent=2)+'\n')
    print(f'Saved {len(manifest)} figures as PNG, SVG and PDF; {len(plotted)} signed contrasts; inserted figures and commentary into report.md.')


INSERTIONS = [
    ('inference','**Inference sensitivity: all recorded test cells**', '''![ReactZyme inference effects across both retrieval directions](figures/01_reactzyme_inference.png)

**Figure 1 — ReactZyme inference settings.** Each column is a separate controlled change; the first two adjust dictionary weight and the last adjusts residual strength. Checkpoints and candidate pools remain fixed; results use one seed and were examined during repeated test-set exploration. Cell values are ΔMRR ×100, so +1.3945 corresponds to +0.013945 raw MRR; positive values favor the stated change. Increasing α beyond 0.40 lowers R→E on every split. The preferred residual multiplier 0.5 mainly raises the Enzyme-Sim E→R point estimate; it is not a uniform improvement. Enzyme-Sim and Time E→R require the near-tie sensitivity check in Figure 6.

![EnzymeMap inference effects for all four screening metrics](figures/02_enzymemap_inference.png)

**Figure 2 — EnzymeMap inference settings.** Blue circles show Table 1 (1,521 queries against 261,907 candidates); open orange squares show Table 2 (1,337 queries against 252,113 candidates after removing training enzyme IDs). Each point is a test difference at a fixed checkpoint, using one seed; right of zero favors the change. Each metric has its own scale. Increasing α from 0.40 to 0.50 improves EF5 and EF10 in both pools while lowering BEDROC85. Reducing the residual to 0.5 also lowers BEDROC85 in both pools. Thus stronger recovery within broad top fractions can coexist with worse very-early ranking; the preferred configuration is a compromise rather than a winner on every screening metric.'''),
    ('temperature','**Paired inverse-temperature comparison**', '''![Paired temperature effects for all seeds and screening metrics](figures/03_temperature_paired_seeds.png)

**Figure 3 — Temperature, with every seed visible.** Values are β=5 minus β=10 at 18 fixed epochs. Blue circles show paired differences for seeds 17, 42 and 73; dark diamonds show their arithmetic mean. The upper row is Table 1 and the lower row is Table 2, with pools defined in Figure 2 and matching scales within each metric. All BEDROC85, BEDROC20 and EF5 seed differences favor β=5; seed 73 accounts for both negative EF10 cells. The mean gains therefore have broad within-study support, although their sizes vary by seed. These are 24 correlated metric comparisons from three paired seeds, not 24 independent trials or confidence intervals. This experiment should remain separate from the final 10-epoch V4 study.'''),
    ('phase2','**Phase-2 biological-supervision ablation**', '''![ReactZyme effects of correct shuffled and removed biological labels](figures/04_reactzyme_phase2_biology.png)

**Figure 4 — ReactZyme biological-loss controls.** Values are differences from V4 without added annotation supervision, expressed as ΔMRR ×100. Colors use one shared symmetric logarithmic scale, with a linear region within ±0.001 displayed units (±0.00001 raw MRR), so the largest Time E→R effects do not obscure smaller differences elsewhere. Blue indicates positive differences and orange negative differences. Cell annotations retain the exact displayed metric differences, without a logarithmic transformation; color intensity is nonlinear in magnitude and is not directly comparable to the linear color scale in Figure 1. Correct and shuffled labels use loss weight 1, with the F3 checkpoint, head architecture, inference settings and candidate pools fixed. Removing a label family retains the other loss coefficients but reduces total regularization. These single-seed comparisons are exploratory. Correct labels do not dominate the shuffled or removed-family variants. The largest metric changes occur in Time E→R, precisely where the rank-sensitivity diagnostic below shows broad near-tie bounds; visible color in a small-change cell does not establish a substantial effect.

![EnzymeMap effects of phase-2 biological labels for every screening metric](figures/05_enzymemap_phase2_biology.png)

**Figure 5 — EnzymeMap biological-loss controls.** Each point is the stated phase-2 variant minus V4 without added supervision, using the same controls and weight 1 as Figure 4. The upper row is Table 1 and the lower row is Table 2, as defined in Figure 2; scales match within each metric. Filled blue circles show all labels, open blue squares shuffled labels, and gray circles removed-family controls. Every displayed variant lowers BEDROC85 and BEDROC20 relative to the unannotated control; EF5 and EF10 show mixed small changes. Correct and shuffled labels nearly coincide on BEDROC85. The scientific-notation axes are essential: these BEDROC changes are on the order of 10⁻⁴, not large performance shifts. This is evidence against attributing the benchmark wins to the added phase-2 biological labels.

![Recorded ReactZyme MRR and bounds under near-tie score orderings](figures/06_reactzyme_near_tie_sensitivity.png)

**Figure 6 — Numerical sensitivity of E→R MRR.** Dots show the recorded metric; segments show the pessimistic and optimistic ranks permitted by a 10⁻⁶ score band. They are not confidence intervals. Reaction-Sim has a small, numerically distinct gain; Enzyme-Sim and Time have strongly overlapping bounds despite larger point-estimate changes. Each panel uses its own labeled, zoomed scale, so compare positions within a panel rather than segment lengths across panels.'''),
    ('f3','**EnzymeMap F3 biological-loss seed sensitivity**', '''![Effects of F3 biological loss at two weights across paired seeds](figures/07_f3_biology_paired_seeds.png)

**Figure 7 — F3 biology, with every paired seed retained.** Each point is a biological-loss variant minus its same-seed unannotated control; the mean row shows arithmetic means. Blue circles indicate loss weight 0.1 and open orange squares weight 1. Nine runs cover seeds 42, 43 and 44, with the archived control used for seed 42; phase 2 remains unannotated. The upper row is Table 1 and the lower row is Table 2, as defined in Figure 2, with matching scales within each metric. Seed 42 supplies much of the positive BEDROC effect; seed 44 generally favors the unannotated control. Weight 1 lowers Table 2 EF5 in all three seeds, even though its average BEDROC85 increases. Weight 0.1 consistently improves Table 2 EF10, but does not consistently improve the other metrics. This is a mixed loss trade-off, not yet a reliable improvement across early-screening objectives.'''),
    ('case1','**Case1 phase-2 controls: Reaction-Sim model**', '''![Case1 recovery by unique sequence original entry and conditional AUROC](figures/08_case1_controls.png)

**Figure 8 — Case1 under different evaluation units.** All variants use the same Reaction-Sim model lineage and phase-2 loss weight 1. The left panel measures recovery at rank 25 among 123 unique candidate sequences, including 12 paper-supported catalysts. The middle panel measures recovery at rank 25 among the original 144 entries, including 15 paper-supported entries. The right panel shows conditional AUROC differences from the unannotated control (AUROC 0.612875), using workbook-reported activity and non-detection labels. Correct labels and the cofactor-removal control recover one additional unique paper-supported catalyst at rank 25. All six methods still recover 10 of the 15 paper-supported entries when the original 144-row panel is ranked. Conditional AUROC changes by less than 0.002 in either direction. The observation therefore concerns one sequence crossing a cutoff in a retrospective literature panel; it does not establish a prospective wet-lab activity improvement.'''),
]


def annotate_report():
    path=HERE/'report.md'
    text=path.read_text()
    for key,anchor,body in INSERTIONS:
        pattern=rf'\n<!-- ablation-plots:{key}:start -->.*?<!-- ablation-plots:{key}:end -->\n'
        text=re.sub(pattern,'',text,flags=re.S)
        assert text.count(anchor)==1,anchor
        insertion=f'\n<!-- ablation-plots:{key}:start -->\n\n{body}\n\n<!-- ablation-plots:{key}:end -->\n\n'
        text=text.replace(anchor,insertion+anchor,1)
    anchor='Audit companions:'
    pattern=r'\n<!-- ablation-plots:reproduce:start -->.*?<!-- ablation-plots:reproduce:end -->\n'
    text=re.sub(pattern,'',text,flags=re.S)
    note='''
<!-- ablation-plots:reproduce:start -->

All figures are available as embedded PNGs and editable [SVG / PDF exports](figures/README.md). Rebuild them with `python3 documents/ablation_study_20260921/plot_study.py`. The [plotting code](plot_study.py), [plotted differences](figures/plotted_contrasts.csv) and [figure/source manifest](figures/manifest.json) preserve the exact comparisons. No training or new test evaluation was performed to make these plots.

Figure styling uses [SciencePlots](https://github.com/garrettj403/SciencePlots) (`science`, `no-latex`) and the selected blue–orange palette, with a neutral midpoint for signed differences. Figures contain only plotted data, essential axes and legends; titles, interpretation and scope notes are in the manuscript captions above. Marker shapes, hatching and signed labels retain information without relying on color. The [palette diagnostics](figures/palette_accessibility.json) include protanopia, deuteranopia and tritanopia simulations.

<!-- ablation-plots:reproduce:end -->

'''
    assert text.count(anchor)==1
    text=text.replace(anchor,note+anchor,1)
    path.write_text(text)
    links=['# Ablation figure exports','', 'PNG files are embedded in the report. SVG preserves editable vector text; PDF is suitable for manuscript inclusion. Each plot uses the audited experiment records, with source hashes in `manifest.json`. The images have no titles, subtitles, panel titles or footnotes. Explanations and scope notes appear in the [manuscript captions](../report.md).','',
           'Figure 04 uses one shared symmetric logarithmic color scale (linear within ±0.001 ΔMRR×100); its printed cell values are unchanged. Figure 01 retains a linear color scale. See the manuscript captions for interpretation.','',
           'Style: [SciencePlots 2.2.1](https://github.com/garrettj403/SciencePlots), using `science` and `no-latex`, serif fonts, inward ticks and thin axes. The selected custom gradient runs from orange `#B86B2D` through pale orange `#E8C3A3` and neutral `#FAFAFA` to pale blue `#ADC8E2` and blue `#28679C`. Signed labels, marker shapes and hatching provide redundant cues. See [color-vision diagnostics](palette_accessibility.json) and the [simulation preview](palette_accessibility_preview.png).','',
           'Install the small plotting-only dependencies without changing the training environment:','',
           '```bash\npython3 -m pip install --no-deps --target .deps/ablation-figures -r documents/ablation_study_20260921/requirements-figures.txt\npython3 documents/ablation_study_20260921/plot_study.py\n```','',
           '| Figure | PNG | SVG | PDF |', '| --- | --- | --- | --- |']
    for p in sorted(FIGURES.glob('*.png')):
        if re.match(r'^\d\d_',p.stem):
            name=p.stem
            links.append(f'| {name.replace("_"," ")} | [PNG]({name}.png) | [SVG]({name}.svg) | [PDF]({name}.pdf) |')
    (FIGURES/'README.md').write_text('\n'.join(links)+'\n')


if __name__=='__main__':
    render_study()
