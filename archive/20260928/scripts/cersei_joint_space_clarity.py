"""Readable SciencePlots exports of the already frozen representation analysis."""
from pathlib import Path
import csv
import json
import shutil
import sys
import zipfile
import numpy as np
from cersei_joint_space_figure import (
    OUT, ROOT, PAPER, RECOVERY, MODELS, NAMES, COLORS, EC_COLORS,
    read, dump, sha, recovery_values,
)


def export_data():
    cohort=read(OUT/'cohort.json');protocol=read(OUT/'protocol.json')
    seeds=[protocol['projection']['primary_seed']]+protocol['projection']['sensitivity_seeds']
    with (OUT/'plot_coordinates.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['model','seed','primary_projection','protein_id','representative_accession',
                         'ec1_labels','display_group','tsne_x','tsne_y'])
        for seed in seeds:
            z=np.load(OUT/f'projections_{seed}.npz')
            for model in MODELS:
                for i,pid in enumerate(cohort['protein_ids']):
                    writer.writerow([NAMES[model],seed,seed==seeds[0],pid,cohort['representative_accessions'][i],
                                     ';'.join(cohort['annotation_ec1'][i]),cohort['color_group'][i],
                                     repr(float(z[model][i,0])),repr(float(z[model][i,1]))])
    values=recovery_values()
    with (OUT/'recovery_at10.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['model','recovery_percent','ci_low_percent','ci_high_percent',
                         'reaction_queries','annotated_bank_before_query_exclusions'])
        for model in MODELS+['random']:
            r=values[model]
            writer.writerow([NAMES[model],100*float(r['macro']),
                100*float(r['ci_low']) if model!='random' else '',
                100*float(r['ci_high']) if model!='random' else '',1067,131409])
    with (RECOVERY/'enzymemap/components/query_scope.csv').open() as f:
        scope={int(r['query_index']):r for r in csv.DictReader(f)}
    with (OUT/'recovery_per_query.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['model','query_index','reaction_id','reaction_ec3_labels',
                         'eligible_candidates','compatible_candidates','agreement_at10_fraction',
                         'class_macro_weight','contribution_to_macro_percent'])
        for model in MODELS+['random']:
            z=np.load(RECOVERY/'enzymemap/components'/
                      (f'{model}_per_query.npz' if model!='random' else 'controls_per_query.npz'))
            y=z['agreement' if model!='random' else 'random'][:,9]
            for i,q in enumerate(z['query_indices']):
                r=scope[int(q)];weight=float(z['macro_weights'][i])
                writer.writerow([NAMES[model],int(q),r['query_id'],r['ec3'],r['eligible_candidates'],
                                 r['compatible_candidates'],repr(float(y[i])),repr(weight),
                                 repr(100*weight*float(y[i]))])
    with (OUT/'plot_coordinates.csv').open() as f: coordinates=list(csv.DictReader(f))
    assert len(coordinates)==720*4*3
    for seed in seeds:
        z=np.load(OUT/f'projections_{seed}.npz')
        for model in MODELS:
            rr=[r for r in coordinates if r['model']==NAMES[model] and int(r['seed'])==seed]
            assert [r['protein_id'] for r in rr]==cohort['protein_ids']
            xy=np.asarray([[float(r['tsne_x']),float(r['tsne_y'])] for r in rr])
            assert np.array_equal(xy,z[model])
    with (OUT/'recovery_per_query.csv').open() as f: queries=list(csv.DictReader(f))
    assert len(queries)==1067*5
    errors={}
    for model in MODELS+['random']:
        total=sum(float(r['contribution_to_macro_percent']) for r in queries if r['model']==NAMES[model])
        errors[model]=abs(total-100*float(values[model]['macro']))
        assert errors[model]<1e-10
    (OUT/'DATA_README.md').write_text(
        '# Figure source data\n\n'
        '- plot_coordinates.csv: 8,640 rows = 720 identical proteins × four methods × three fixed seeds. '
        'Filter primary_projection=True for the main figure (2,880 rows). Each row includes the sequence ID, '
        'representative accession, native EC1 labels, display category and exact saved t-SNE coordinates.\n'
        '- recovery_at10.csv: the five displayed values and saved 95% model-specific intervals, in percent. '
        'Random is an exact expected reference, so its interval fields are blank.\n'
        '- recovery_per_query.csv: 5,335 rows = 1,067 queries × four methods plus random. '
        'Sum contribution_to_macro_percent within a method to reproduce the displayed percentage. '
        'The contribution is 100 × class_macro_weight × agreement_at10_fraction. '
        'An unweighted query mean is not the class-macro result.\n'
        '- embeddings.npz: the original normalized enzyme embeddings, keyed by model; rows follow cohort.csv. '
        'Dimensions are 512/256/1280/512 for Horizyn/CREEP/CLIPZyme/CERSEI.\n'
        '- projections_*.npz: the exact saved projection arrays for all three seeds.\n\n'
        'The visualization and retrieval populations differ. Maps use 720 unique EC-annotated, test-associated '
        'sequences. Retrieval uses 1,067 reactions and an initial annotated bank of 131,409 accessions, then '
        'excludes recorded partners, aliases and detected close-sequence components. EC3 compatibility '
        'does not establish experimentally verified new catalytic activity.\n')
    dump(OUT/'data_export_validation.json',dict(passed=True,coordinate_rows=len(coordinates),
         primary_coordinate_rows=2880,per_query_rows=len(queries),coordinate_roundtrip_max_error=0.,
         class_macro_percent_roundtrip_errors=errors,
         files={n:sha(OUT/n) for n in ['plot_coordinates.csv','recovery_at10.csv','recovery_per_query.csv']}))
    names=['DATA_README.md','plot_coordinates.csv','recovery_at10.csv','recovery_per_query.csv',
           'cohort.csv','cohort.json','embeddings.npz','protocol.json','prepared.json',
           'projection_diagnostics.json','data_export_validation.json']
    names += [f'projections_{s}.npz' for s in seeds]
    with zipfile.ZipFile(OUT/'figure_source_data.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(OUT/name,name)
    print('Exported and validated coordinates and per-query CSVs',flush=True)


def render(models=('creep', 'clipzyme', 'cersei'), output_dir=None):
    models = list(models)
    assert len(models) in (3, 4) and len(set(models)) == len(models)
    assert set(models).issubset(MODELS)
    out = Path(output_dir) if output_dir else OUT / 'manuscript_comparison'
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0,str(ROOT/'.deps/ablation-figures'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import importlib.metadata
    try:
        import scienceplots
        plt.style.use(['science','no-latex'])
        style_loader='import scienceplots; plt.style.use(["science", "no-latex"])'
    except AttributeError:
        import importlib.util
        styles=Path(importlib.util.find_spec('scienceplots').origin).parent/'styles'
        plt.style.use([str(next(styles.rglob('science.mplstyle'))),str(next(styles.rglob('no-latex.mplstyle')))])
        style_loader='SciencePlots package style files (Matplotlib compatibility fallback)'
    # Retain SciencePlots' serif typography; no sans-serif override.
    plt.rcParams.update({'font.size':9.5,'axes.labelsize':9.5,'legend.fontsize':8.5,
        'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.top':False,'ytick.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    protocol=read(OUT/'protocol.json');cohort=read(OUT/'cohort.json')
    groups=np.asarray(cohort['color_group'])
    seeds=[protocol['projection']['primary_seed']]+protocol['projection']['sensitivity_seeds']
    values=recovery_values();primary=np.load(OUT/f'projections_{seeds[0]}.npz')
    bounds={}
    def map_panel(ax,data,label,small=False):
        colors=np.asarray([EC_COLORS[x] for x in groups]);single=groups!='Multiple'
        ax.scatter(data[single,0],data[single,1],c=colors[single],s=6.8 if not small else 5.5,
                   alpha=.86,linewidths=.15,edgecolors='white')
        ax.scatter(data[~single,0],data[~single,1],c=EC_COLORS['Multiple'],
                   s=10 if not small else 8,marker='x',alpha=.8,linewidths=.55)
        # Tight, centered square windows remove empty margins without cropping,
        # rotating, stretching, or modifying any saved coordinates.
        low,high=data.min(0),data.max(0);center=(low+high)/2
        half=float((high-low).max())*.535
        limits=np.c_[center-half,center+half]
        assert ((data>=limits[:,0])&(data<=limits[:,1])).all()
        bounds[label]=limits.tolist()
        ax.set(xlim=limits[0],ylim=limits[1],xticks=[],yticks=[],xlabel=label,aspect='equal')
        ax.xaxis.label.set_size(9.5 if not small else 8.5)
        ax.xaxis.labelpad=5
        for spine in ax.spines.values():
            spine.set_visible(True);spine.set_color('.80');spine.set_linewidth(.55)
        ax.minorticks_off()
    def recovery_panel(ax,small=False,display_names=None):
        display_names = NAMES if display_names is None else display_names
        order=models+['random']
        markers={'horizyn':'s','creep':'^','clipzyme':'D','cersei':'o','random':'|'}
        for i,model in enumerate(order):
            marker=markers[model]
            r=values[model];value=100*float(r['macro'])
            if model!='random':
                lo,hi=100*float(r['ci_low']),100*float(r['ci_high'])
                ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt=marker,color=COLORS[model],
                            markersize=5.1,capsize=3,capthick=.9,elinewidth=1.1,zorder=3)
            else:ax.plot(value,i,marker=marker,color=COLORS[model],markersize=7,linestyle='',zorder=3)
            ax.text(43.2,i,f'{value:.2f}',ha='right',va='center',
                    fontsize=8.5 if small else 10,
                    fontweight='bold' if model=='cersei' else 'normal')
        ax.set(xlim=(0,44),ylim=(len(order)-.4,-.6),xticks=[0,10,20,30],yticks=range(len(order)),
               yticklabels=[display_names[m] for m in order])
        ax.tick_params(axis='y',length=0,pad=3,labelsize=8.8 if small else 10)
        ax.tick_params(axis='x',labelsize=8.2 if small else 9)
        ax.spines['left'].set_visible(False);ax.minorticks_off()
        ax.grid(axis='x',color='.90',linewidth=.45,zorder=0)
    legend=[Line2D([0],[0],marker='o' if c!='Multiple' else 'x',color=EC_COLORS[c],
            linestyle='',markersize=4.4,label=f'EC {c}' if c!='Multiple' else 'Multiple')
            for c in EC_COLORS]
    # More horizontal room also makes the equal-aspect square maps taller;
    # increasing figure height alone would only add blank space around them.
    fig=plt.figure(figsize=(6.8,2.45))
    main_names = dict(NAMES, cersei='CIRCE')
    map_width=(.68-.017*(len(models)-1))/len(models)
    for j,model in enumerate(models):
        ax=fig.add_axes([.015+j*(map_width+.017),.235,map_width,.62])
        map_panel(ax,primary[model],f'({chr(97+j)}) {main_names[model]}')
    ax=fig.add_axes([.80,.235,.193,.62]);recovery_panel(ax,True,display_names=main_names)
    fig.text(.855,.915,f'({chr(97+len(models))}) EC3 Recovery@10 (%)',ha='center',va='center',fontsize=9)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.355,.97),ncol=7,
               frameon=False,handletextpad=.22,columnspacing=.72,fontsize=8.4)
    for suffix in ['pdf','svg','png']:
        fig.savefig(out/f'joint_space_composite.{suffix}',dpi=450,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    # Larger standalone panels make the data convenient to inspect and reuse.
    fig,axes=plt.subplots(1,len(models),figsize=(7.2,2.8))
    for j,model in enumerate(models):
        map_panel(axes[j],primary[model],f'({chr(97+j)}) {NAMES[model]}')
    fig.legend(handles=legend,loc='upper center',ncol=7,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.025,right=.985,bottom=.15,top=.84,wspace=.15)
    for suffix in ['pdf','svg','png']:
        fig.savefig(out/f'enzyme_maps.{suffix}',dpi=400,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.8,2.35));recovery_panel(ax)
    ax.set_xlabel('EC3 Recovery@10 (%)')
    fig.subplots_adjust(left=.25,right=.98,bottom=.23,top=.97)
    for suffix in ['pdf','svg','png']:
        fig.savefig(out/f'functional_recovery_at10.{suffix}',dpi=400,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    fig,axes=plt.subplots(3,len(models),figsize=(7.3,7.0))
    for i,seed in enumerate(seeds):
        z=np.load(OUT/f'projections_{seed}.npz')
        for j,model in enumerate(models):
            map_panel(axes[i,j],z[model],NAMES[model] if i==2 else '',True)
    fig.legend(handles=legend,loc='upper center',ncol=7,frameon=False,fontsize=9)
    fig.subplots_adjust(top=.93,left=.03,right=.975,bottom=.075,hspace=.12,wspace=.12)
    for suffix in ['pdf','svg','png']:
        fig.savefig(out/f'projection_seed_sensitivity.{suffix}',dpi=300,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    assets=PAPER/'assets/representation_analysis'
    for name in ['joint_space_composite','projection_seed_sensitivity']:
        shutil.copy2(out/f'{name}.pdf',assets/f'{name}.pdf')
    dump(out/'export_receipt.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
        script_sha256=sha(__file__),figure_sha256=sha(out/'joint_space_composite.pdf'),
        metric_source_sha256=sha(RECOVERY/'enzymemap/components/curves.csv'),
        scienceplots=True,scienceplots_version=importlib.metadata.version('SciencePlots'),
        matplotlib_version=matplotlib.__version__,style_loader=style_loader,
        font_family=plt.rcParams['font.family'],selected_layout='single horizontal row',
        main_figure_inches=[6.8,2.45],main_map_span=.68,main_map_aspect='equal',
        metric_label='EC3 Recovery@10 (%); upright, panel (d) label above final chart',
        coordinate_hashes={str(s):sha(OUT/f'projections_{s}.npz') for s in seeds},
        coordinates_changed=False,all_points_inside_windows=True,
        main_point_count_per_model=720,main_panel_count=len(models)+1,displayed_models=models,
        main_display_names={m:main_names[m] for m in models},
        selection_reason='User requested removal of Horizyn from the manuscript comparison.',
        full_diagnostic_archive=str(OUT),
        outputs=[str(out/f'{n}.{s}') for n in ['joint_space_composite','enzyme_maps',
                 'functional_recovery_at10','projection_seed_sensitivity'] for s in ['pdf','svg','png']]))
    # Export the exact displayed rows; the complete diagnostic remains intact.
    allowed={NAMES[m] for m in models}|{'Random'}
    counts={}
    for name in ['plot_coordinates.csv','recovery_at10.csv','recovery_per_query.csv']:
        with (OUT/name).open() as f:
            reader=csv.DictReader(f);fields=reader.fieldnames
            rows=[r for r in reader if r['model'] in allowed]
        with (out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        counts[name]=len(rows)
    assert counts['plot_coordinates.csv']==720*len(models)*3
    assert counts['recovery_at10.csv']==len(models)+1
    assert counts['recovery_per_query.csv']==1067*(len(models)+1)
    dump(out/'data_validation.json',dict(passed=True,exported_rows=counts,
        displayed_models=models,source_values_unchanged=True))
    (out/'README.md').write_text(
        '# Manuscript comparison without Horizyn\n\n'
        'At the user\'s request, the paper compares CREEP, CLIPZyme and CERSEI. '
        'The main figure contains three enzyme maps and one functional-recovery panel in a horizontal row. '
        'All three prespecified seeds are retained in the companion projection figure. '
        'The selected 720 sequence identities, projection coordinates and recovery values are unchanged.\n\n'
        'The CSV files contain exactly the displayed methods (plus Random for recovery). '
        'The [data guide](../DATA_README.md) defines their columns and class-macro weights. '
        'The full four-model diagnostic and original source arrays remain in the parent directory. '
        'This is a requested change to the manuscript comparison, not a new experiment.\n\n'
        'Reproduce from horizyn with: python3 scripts/cersei_joint_space_clarity.py\n')
    print('Rendered SciencePlots horizontal composite and readable standalone panels',flush=True)


if __name__=='__main__':
    export_data()
    render()
