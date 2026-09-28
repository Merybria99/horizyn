"""Build ICLR-width tables and selected figures from the frozen evidence only.

Requires numpy, matplotlib and SciencePlots for figures; no GPU or model access.
The four figures adapt the existing ablation figures 01, 03, 04 and 08.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
try:
    import scienceplots
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'.deps/ablation-figures'))
    import scienceplots

HERE=Path(__file__).resolve().parent
D=json.loads((HERE/'evidence_snapshot.json').read_text())
V={(r['study'],r['variant'],r['seed'],r['setting'],r['metric']):r['value'] for r in D['ablation']['records']}
SPLITS=[('reaction_smi','Reaction-Sim'),('enzyme_smi','Enzyme-Sim'),('time','Time')]
DIRS=['reaction_to_enzyme','enzyme_to_reaction']
METRICS=['bedroc85','bedroc20','ef0.05','ef0.1']
BLUE,ORANGE,INK='#28679C','#B86B2D','#243342'
CMAP=LinearSegmentedColormap.from_list('orange_white_blue',['#B86B2D','#E8C3A3','#FAFAFA','#ADC8E2','#28679C'],N=513)
LEDGER=[]
def value(study,var,setting,metric,seed=42):return V[study,var,seed,setting,metric]
def cell(table,row,col,x,fmt='.4f',bold=False):
    LEDGER.append(dict(table=table,row=row,column=col,value=x))
    s=format(x,fmt)
    return r'\textbf{'+s+'}' if bold else s
def line(cells):return ' & '.join(cells)+r' \\'
def write_table(name,spec,lines):
    (HERE/'tables'/f'{name}.tex').write_text('% Generated from bundled frozen evidence; stages uses stage_comparison.json.\n'+
        r'\begingroup\setlength{\tabcolsep}{2.8pt}'+'\n'+r'\begin{tabular}{@{}'+spec+r'@{}}'+'\n'+
        '\n'.join([r'\toprule',*lines,r'\bottomrule',r'\end{tabular}',r'\endgroup'])+'\n')

def tables():
    rows=[line(['Method',r'\multicolumn{4}{c}{Full library}',r'\multicolumn{4}{c}{Training-ID-excluded}']),
          r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
          line(['',r'B$_{85}$',r'B$_{20}$','EF5','EF10',r'B$_{85}$',r'B$_{20}$','EF5','EF10']),r'\midrule']
    papers=[('CLIPZyme (ESM)',[36.91,53.04,11.93,6.84],[None]*4),('CLIPZyme (CGR)',[38.91,57.58,13.16,7.73],[None]*4),
            ('CLIPZyme',[44.69,62.98,14.09,8.06],[39.13,58.86,13.40,7.81]),('FGW-CLIP',[48.66,66.69,14.91,8.18],[45.14,61.43,13.57,7.61])]
    for label,full,exc in papers:
        rows.append(line([label+r'$^\dagger$',*['--' if x is None else cell('screening',label,str(i),x,'.2f') for i,x in enumerate(full+exc)]]))
    rows.append(r'\midrule')
    local=[]
    for label,var in [('CLIPZyme (released)',None),(r'\methodname','control')]:
        vals=[]
        for setting in ['table1','table2']:
            for i,m in enumerate(METRICS):
                x=D['clipzyme']['summary'][setting][m] if var is None else value('phase2_biology',var,setting,m)
                vals.append(x*(100 if i<2 else 1))
        local.append((label,vals))
    maxima=np.max([x[1] for x in local],axis=0)
    for label,vals in local: rows.append(line([label,*[cell('screening',label,str(i),x,'.2f',x==maxima[i]) for i,x in enumerate(vals)]]))
    write_table('screening','lrrrrrrrr',rows)
    rows=[line(['Split','Train','Val.','Test',r'$|R_{\rm test}|$',r'$|E_{\rm test}|$']),r'\midrule']
    for split,label in SPLITS:
        c=D['data']['counts'][split]
        vals=[c[k]['pairs'] for k in ['train','validation','test']]+[c['test']['reactions'],c['test']['proteins']]
        rows.append(line([label,*[cell('data',split,str(i),x,',d') for i,x in enumerate(vals)]]))
    write_table('data','lrrrrr',rows)
    rows=[line(['Method',r'\multicolumn{2}{c}{Reaction-Sim}',r'\multicolumn{2}{c}{Enzyme-Sim}',r'\multicolumn{2}{c}{Time}']),
          r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
          line(['',r'R$\to$E',r'E$\to$R',r'R$\to$E',r'E$\to$R',r'R$\to$E',r'E$\to$R']),r'\midrule']
    for label,vals in [('FGW-CLIP (EC Mode)',[.3181,.3722,.5253,.8683,.3394,.5222]),('FGW-CLIP (EC Max)',[.3113,.3804,.5300,.8581,.3392,.5229]),
                       ('TIGER (ESM2Text)',[.319,.518,.592,.956,.366,.690]),('TIGER (ProtT3)',[.337,.472,.579,.940,.372,.683])]:
        rows.append(line([label+r'$^\dagger$',*[cell('reactzyme',label,str(i),x) for i,x in enumerate(vals)]]))
    rows.append(r'\midrule')
    local=[]
    for label,prefix in [('MLP','mlp_esm2_mat_2d'),('Contrastive MLP','contrastive_esm2_mat_2d'),('Transformer','transformer_esm2_mat_2d'),
                         ('Bi-RNN','birnn_esm2_mat_2d'),('Corrected contrastive','contrastive_corrected_esm2_mat_2d'),
                         ('EnzGFM control','transformer_enzgfm650_mat_2d'),('Horizyn (adapted)','horizyn_participant_set')]:
        vals=[]
        for split,_ in SPLITS:
            r=next(r for r in D['public_runs'] if r['_run']==f'{prefix}_{split}_seed42')
            vals += [r['test'][di]['all']['reactzyme_mrr'] for di in DIRS]
        local.append((label,vals))
    for label,var in [(r'\methodname','control')]:
        local.append((label,[value('phase2_biology',var,s,m) for s,_ in SPLITS for m in DIRS]))
    maxima=np.max([x[1] for x in local],axis=0)
    for label,vals in local:
        if label==r'\methodname':rows.append(r'\midrule')
        rows.append(line([label,*[cell('reactzyme',label,str(i),x,bold=x==maxima[i]) for i,x in enumerate(vals)]]))
    write_table('reactzyme','lrrrrrr',rows)
    cases=[('Native F3',D['case']['methods']['reaction_smi/native_before_phase2']),
           (r'\methodname',D['case']['methods']['reaction_smi/selected']),
           ('V4 (EnzymeMap)',D['case']['methods']['enzymemap/selected'])]
    rows=[line(['Model','Unique @25','Entries @25','AUROC']),r'\midrule']
    for label,m in cases:
        u=m['primary_papers']['recovered_at_25'];e=m['entry_level_144']['primary_papers']['recovered_at_25'];a=m['broad_assay_conditional_discrimination']['auc']
        rows.append(line([label,cell('case1',label,'unique',u,'d')+'/12',cell('case1',label,'entries',e,'d')+'/15',cell('case1',label,'auroc',a)]))
    write_table('case1','lrrr',rows)
    # This comparison is kept in the ablation section, separate from primary results.
    stages=json.loads((HERE/'stage_comparison.json').read_text())
    labels={'reaction_smi':'Reaction-Sim','enzyme_smi':'Enzyme-Sim','time':'Time',
        'table1':'Full library','table2':'Training-ID-excluded',
        'enzyme_to_reaction':r'E$\to$R MRR','reaction_to_enzyme':r'R$\to$E MRR',
        'bedroc85':r'BEDROC$_{85}$','bedroc20':r'BEDROC$_{20}$','ef0.05':'EF5','ef0.1':'EF10'}
    rows=[line(['Dataset / setting','Metric','Phase 1','Phase 2','Phase 2 + Bio']),r'\midrule']
    for r in stages['rows']:
        if r['dataset']=='Case1':continue
        setting=labels[r['setting']]; metric=labels[r['metric']]
        row_name=r['dataset']+' / '+setting
        rows.append(line([row_name,metric,*[cell('stages',row_name+' / '+metric,k,r[k],'.6f')
            for k in ['phase1','phase2','phase2_bio']]]))
    write_table('stages','llrrr',rows)
    with (HERE/'table_values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(LEDGER[0]));w.writeheader();w.writerows(LEDGER)

def figures():
    plt.style.use(['science','no-latex'])
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8,
        'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'text.usetex':False,
        'axes.edgecolor':'#697787','axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
        'axes.linewidth':.5,'grid.color':'#E1E5EA','grid.linewidth':.4,'ytick.minor.visible':False,
        'pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})
    receipt=[];contrasts=[]
    def delta(name,study,var,ref,setting,m,seed=42):
        x=value(study,var,setting,m,seed)-value(study,ref,setting,m,seed)
        contrasts.append(dict(figure=name,study=study,variant=var,reference=ref,setting=setting,metric=m,seed=seed,delta=x));return x
    def save(fig,name):
        assert fig._suptitle is None and not fig.texts
        assert all(not ax.get_title() for ax in fig.axes)
        fig.canvas.draw()
        for ext in ['pdf','png']:
            f=HERE/'figures'/f'{name}.{ext}';fig.savefig(f,dpi=220)
            receipt.append(dict(file=str(f.relative_to(HERE)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        plt.close(fig)
    def heatmap(name,study,cols,symlog=False):
        a=np.array([[100*delta(name,study,var,ref,s,m) for ref,var,label in cols] for s,_ in SPLITS for m in DIRS])
        fig,ax=plt.subplots(figsize=(5.5,2.5));fig.subplots_adjust(left=.23,right=.85,top=.97,bottom=.24)
        lim=np.abs(a).max();norm=SymLogNorm(.001,linscale=1,vmin=-lim,vmax=lim) if symlog else TwoSlopeNorm(0,-lim,lim)
        im=ax.imshow(a,aspect='auto',cmap=CMAP,norm=norm)
        ax.set_xticks(range(len(cols)),[c[2] for c in cols],fontsize=7.3)
        ax.set_yticks(range(6),[f'{label}  {di}' for s,label in SPLITS for di in ['R→E','E→R']],fontsize=7.3)
        ax.minorticks_off();ax.tick_params(length=0,pad=5)
        for i in range(6):
            for j in range(len(cols)):
                rgb=np.asarray(CMAP(norm(a[i,j]))[:3])
                luminance=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)@np.array([.2126,.7152,.0722])
                color='white' if 1.05/(luminance+.05)>=4.5 else 'black'
                ax.text(j,i,f'{a[i,j]:+.4f}',ha='center',va='center',fontsize=7.0,color=color)
        for y in [1.5,3.5]:ax.axhline(y,color='white',lw=1)
        ca=fig.add_axes([.88,.24,.025,.73]);cb=fig.colorbar(im,cax=ca);cb.set_label('ΔMRR × 100',fontsize=7.5,labelpad=4)
        cb.ax.tick_params(labelsize=7)
        if symlog:
            cb.set_ticks([-1,-.01,0,.01,1]);cb.set_ticklabels(['−1','−.01','0','.01','1']);cb.ax.minorticks_off()
        else:cb.locator=MaxNLocator(5);cb.update_ticks()
        save(fig,name)
    heatmap('inference','inference',[
        ('a025_c1','a040_c1','α: .25 → .40\nc = 1'),('a040_c1','a050_c1','α: .40 → .50\nc = 1'),('a040_c1','a040_c05','c: 1 → .5\nα = .40')])
    heatmap('biology','phase2_biology',[
        ('control','all_1','All\nlabels'),('control','shuffled_1','Shuffled\nlabels'),('control','without_ec','Without\nEC'),
        ('control','without_cofactor','Without\ncofactor'),('control','without_mechanism','Without\ntransform.')],True)
    fig,axs=plt.subplots(2,2,figsize=(5.5,3.35));fig.subplots_adjust(left=.14,right=.98,bottom=.12,top=.88,wspace=.48,hspace=.51)
    handles=[Line2D([],[],marker='o',color=BLUE,ls='none',label='Full library'),Line2D([],[],marker='s',color=ORANGE,mfc='white',ls='none',label='Training-ID-excluded')]
    fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False,fontsize=8)
    for ax,m,label in zip(axs.flat,METRICS,['BEDROC85','BEDROC20','EF5','EF10']):
        ds=[]
        for setting,offset,col,marker in [('table1',.12,BLUE,'o'),('table2',-.12,ORANGE,'s')]:
            xs=[delta('temperature','temperature_18epochs','beta5','beta10',setting,m,s) for s in [17,42,73]]
            xs=xs+[float(np.mean(xs))];ds.extend(xs)
            ax.plot(xs,np.array([3,2,1,0])+offset,marker=marker,ms=4.2,mfc=col if marker=='o' else 'white',color=col,ls='none')
        lo,hi=min(0,min(ds)),max(0,max(ds));pad=(hi-lo)*.17;ax.set_xlim(lo-pad,hi+pad)
        ax.axvline(0,color=INK,lw=.7);ax.axhline(.5,color='#CDD4DC',lw=.5)
        ax.set_yticks([3,2,1,0],['Seed 17','Seed 42','Seed 73','Mean']);ax.set_ylim(-.45,3.45)
        ax.set_xlabel('Δ '+label);ax.xaxis.set_major_locator(MaxNLocator(3));ax.grid(axis='x');ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    save(fig,'temperature')
    cases={'control':D['case']['methods']['reaction_smi/selected']}
    cases.update({r['variant']:r['methods']['reaction_smi/selected'] for r in D['case_controls']['records'] if r['weight']==1})
    order=[('control','No added biology'),('all','All labels'),('shuffled_1','Shuffled labels'),('without_ec','Without EC'),('without_cofactor','Without cofactor'),('without_mechanism','Without transform.')]
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.1));fig.subplots_adjust(left=.24,right=.98,top=.97,bottom=.23,wspace=.27)
    case_points=[]
    for i,(variant,label) in enumerate(order):
        u=cases[variant]['primary_papers']['recovered_at_25'];e=cases[variant]['entry_level_144']['primary_papers']['recovered_at_25'];y=5-i
        case_points.append(dict(variant=variant,unique=u,entries=e))
        axs[0].barh(y,u,height=.62,color=BLUE);axs[0].text(u-.3,y,f'{u}/12',ha='right',va='center',color='white',fontsize=8)
        axs[1].barh(y,e,height=.62,color='white',ec=ORANGE,lw=.6,hatch='///');axs[1].text(e-.3,y,f'{e}/15',ha='right',va='center',fontsize=8,bbox=dict(fc='white',ec='none',pad=.1))
    for i,ax in enumerate(axs):
        ax.set_yticks(range(5,-1,-1),[x[1] for x in order] if i==0 else []);ax.set_ylim(-.6,5.6)
        ax.set_xlim(0,12.7 if i==0 else 15.7);ax.set_xticks([0,3,6,9,12] if i==0 else [0,5,10,15]);ax.set_axisbelow(True);ax.grid(axis='x');ax.tick_params(axis='y',length=0)
        ax.set_xlabel('Unique catalysts @25' if i==0 else 'Paper entries @25')
    save(fig,'case1')
    (HERE/'figure_values.json').write_text(json.dumps(dict(contrasts=contrasts,case1=case_points),indent=2)+'\n')
    assert len(contrasts)==72
    (HERE/'figures/manifest.json').write_text(json.dumps(dict(files=receipt,style=['science','no-latex'],source_sha256=hashlib.sha256((HERE/'evidence_snapshot.json').read_bytes()).hexdigest(),contrast_count=72,case1_counts=12),indent=2)+'\n')

if __name__=='__main__':
    tables();figures();print(f'Prepared five tables ({len(LEDGER)} numeric cells) and four ICLR-width figures.')
