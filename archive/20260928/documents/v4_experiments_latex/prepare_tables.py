"""Freeze recorded evidence and build editable LaTeX tables; no model execution.

Run from any directory. The first invocation captures a snapshot; later runs
reuse it. Use --refresh only to intentionally incorporate newly completed runs.
The prose is hand-edited in experiments.tex and supplementary.tex.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT/'runs/generalization_20260919_2251/cross_paper_retraining'
AUDIT = ROOT/'documents/ablation_study_20260921'
PUBLIC = ROOT/'runs/reactzyme_public_baselines_20260921'
SPLITS = [('reaction_smi', 'Reaction similarity'), ('enzyme_smi', 'Enzyme similarity'), ('time', 'Time')]
DIRS = ['reaction_to_enzyme', 'enzyme_to_reaction']
METRICS = ['bedroc85', 'bedroc20', 'ef0.05', 'ef0.1']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    hashes = {}
    def read(path):
        hashes[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())
    ablation = read(AUDIT/'evidence.json')
    for rel, expected in ablation['source_sha256'].items():
        assert digest(ROOT/rel) == expected, rel
    hashes.update(ablation['source_sha256'])
    v4 = read(RUNS/'shared_recipe_alpha04_cap05_v1/qualification.json')
    bio = read(RUNS/'v4_relative_biology_phase2_20260921_v1/comparison.json')
    case = read(RUNS/'shared_recipe_alpha04_cap05_v1/case1/summary.json')
    case_controls = read(RUNS/'v4_relative_biology_phase2_case1_controls_20260921_v1/comparison.json')
    records = []
    for path in sorted((PUBLIC/'models').glob('*/complete.json')):
        d = read(path)
        assert 'test' in d and set(DIRS) <= set(d['test']), path
        d['_run'] = path.parent.name
        d['_source'] = str(path.relative_to(ROOT))
        selection = path.parent/'selection.json'
        if selection.exists():
            d['_selection'] = read(selection)
        records.append(d)
    v4_test, bio_test = {}, {}
    for split, _ in SPLITS:
        v4_test[split] = read(RUNS/f'shared_semantic_strength_variants_v1/{split}/alpha04_cap05/test_summary.json')['summary']
        bio_test[split] = read(RUNS/f'v4_relative_biology_phase2_20260921_v1/{split}/all_1/test_summary.json')['summary']
    entry_path = ROOT/'wet_lab/Case1/sequence_pool/final_entry_sequences.csv'
    hashes[str(entry_path.relative_to(ROOT))] = digest(entry_path)
    with entry_path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r['sheet'] == 'Homologs']
    eligible = [r for r in rows if r['status'].startswith('resolved') and r['sequence']]
    assert len(rows) == 145 and len(eligible) == 144
    assert len({r['sequence'] for r in eligible}) == 123
    figures = read(AUDIT/'figures/manifest.json')
    figure_dir = HERE/'figures'; figure_dir.mkdir(exist_ok=True)
    for figure in figures['figures']:
        file = figure['files']['pdf']
        source = AUDIT/file['path']
        assert digest(source) == file['sha256'], source
        shutil.copy2(source, figure_dir/source.name)
        hashes[str(source.relative_to(ROOT))] = digest(source)
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
                ablation=ablation, v4=v4, bio=bio, case=case, case_controls=case_controls,
                public_runs=records, v4_test=v4_test, bio_test=bio_test,
                clipzyme=read(RUNS/'clipzyme_released_screen_evaluation_v1/summary.json'),
                data=read(PUBLIC/'features/data_audit.json'),
                parity=read(PUBLIC/'features/candidate_parity_audit.json'),
                official_status=read(PUBLIC/'official_baselines_status.json'),
                extended_status=read(PUBLIC/'extended_status.json'),
                case_sources=read(ROOT/'runs/generalization_20260919_2251/case1_audit/source_verification.json'),
                case_pool=dict(catalogue_entries=len(rows), ranked_entries=len(eligible), unique_sequences=123,
                               excluded=[r['entry_id'] for r in rows if r not in eligible]),
                figure_manifest=figures)


def escape(s):
    return str(s).replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')


def build(d):
    tables = HERE/'tables'; tables.mkdir(exist_ok=True)
    ledger = []
    def number(table, row, column, value, source, fmt='.6f'):
        ledger.append(dict(table=table, row=row, column=column, value=value, source=source))
        return format(value, fmt)
    def write(name, lines):
        (tables/f'{name}.tex').write_text('% Generated from evidence_snapshot.json by prepare_tables.py.\n'+'\n'.join(lines)+'\n')
    def line(cells):
        return ' & '.join(map(str, cells)) + r' \\'
    def small_table(name, specification, header, rows):
        write(name, [r'\begin{tabular}{'+specification+'}',r'\toprule',line(header),r'\midrule',*rows,r'\bottomrule',r'\end{tabular}'])
    values = {(r['study'],r['variant'],r['seed'],r['setting'],r['metric']):r['value'] for r in d['ablation']['records']}
    def v(variant, setting, metric):
        return values['phase2_biology',variant,42,setting,metric]
    rows=[]
    for split,label in SPLITS:
        c=d['data']['counts'][split]
        cells=[number('reactzyme_data',split,s,c[s]['pairs'],'data_audit/counts',',d') for s in ['train','validation','test']]
        cells += [number('reactzyme_data',split,k,c['test'][k],'data_audit/counts',',d') for k in ['reactions','proteins']]
        rows.append(line([label,*cells]))
    small_table('reactzyme_data','lrrrrr',['Split','Train','Validation','Test','Reactions','Enzymes'],rows)

    for setting in ['table1','table2']:
        rows=[]
        published = ([('CLIPZyme (ESM), reported',[36.91,53.04,11.93,6.84]),
                      ('CLIPZyme (CGR), reported',[38.91,57.58,13.16,7.73]),
                      ('CLIPZyme, reported',[44.69,62.98,14.09,8.06]),
                      ('FGW-CLIP, reported',[48.66,66.69,14.91,8.18])] if setting=='table1' else
                     [('CLIPZyme, reported',[39.13,58.86,13.40,7.81]),
                      ('FGW-CLIP, reported',[45.14,61.43,13.57,7.61])])
        for label, metrics in published:
            rows.append(line([label,*[number(setting,label,m,x,'FGW-CLIP v2, Tables 1--2','.2f') for m,x in zip(METRICS,metrics)]]))
        rows.append(r'\midrule')
        for label, variant in [('CLIPZyme, released checkpoint',None),(r'\methodname','control'),(r'\vfourBio','all_1')]:
            vals=[d['clipzyme']['summary'][setting][m] if variant is None else v(variant,setting,m) for m in METRICS]
            vals=[x*100 if i<2 else x for i,x in enumerate(vals)]
            rows.append(line([label,*[number(setting,label,m,x,'clipzyme released summary' if variant is None else f'phase2_biology/{variant}','.4f') for m,x in zip(METRICS,vals)]]))
        small_table('screening_'+setting,'lrrrr',['Method',r'BEDROC$_{85}$ (\%)',r'BEDROC$_{20}$ (\%)','EF5','EF10'],rows)

    published = [
        ('FGW-CLIP, EC Mode',[.3181,.3722,.5253,.8683,.3394,.5222],'FGW-CLIP v2, Tables 3,7,8'),
        ('FGW-CLIP, EC Max',[.3113,.3804,.5300,.8581,.3392,.5229],'FGW-CLIP v2, Tables 3,7,8'),
        ('TIGER (ESM2Text)',[.319,.518,.592,.956,.366,.690],'TIGER v1, Table 1'),
        ('TIGER (ProtT3)',[.337,.472,.579,.940,.372,.683],'TIGER v1, Table 1')]
    header=['Method',r'\multicolumn{2}{c}{Reaction similarity}',r'\multicolumn{2}{c}{Enzyme similarity}',r'\multicolumn{2}{c}{Time}']
    direction_header=line(['',r'R$\to$E',r'E$\to$R',r'R$\to$E',r'E$\to$R',r'R$\to$E',r'E$\to$R'])
    rows=[direction_header,r'\midrule']
    for label,vals,source in published:
        rows.append(line([label,*[number('literature_mrr',label,str(i),x,source,'.4f') for i,x in enumerate(vals)]]))
    rows.append(r'\midrule')
    for variant,label in [('control',r'\methodname'),('all_1',r'\vfourBio')]:
        rows.append(line([label,*[number('literature_mrr',label,split+'/'+direction,v(variant,split,direction),'phase2_biology/'+variant) for split,_ in SPLITS for direction in DIRS]]))
    small_table('reactzyme_literature','lrrrrrr',header,rows)

    rows=[direction_header,r'\midrule']
    representatives=[('MLP','mlp_esm2_mat_2d'),('Contrastive MLP','contrastive_esm2_mat_2d'),
                     ('Transformer','transformer_esm2_mat_2d'),('Bi-RNN','birnn_esm2_mat_2d'),
                     ('Corrected contrastive MLP','contrastive_corrected_esm2_mat_2d'),
                     ('EnzGFM backbone control','transformer_enzgfm650_mat_2d'),
                     ('Horizyn participant-set adapter','horizyn_participant_set')]
    for label,prefix in representatives:
        cells=[]
        for split,_ in SPLITS:
            r=next((r for r in d['public_runs'] if r['_run']==f'{prefix}_{split}_seed42'),None)
            cells += ['--','--'] if r is None else [number('retrained_mrr',label,split+'/'+di,r['test'][di]['all']['reactzyme_mrr'],r['_source']) for di in DIRS]
        rows.append(line([label,*cells]))
    rows.append(r'\midrule')
    for variant,label in [('control',r'\methodname'),('all_1',r'\vfourBio')]:
        rows.append(line([label,*[number('retrained_mrr',label,split+'/'+di,v(variant,split,di),'phase2_biology/'+variant) for split,_ in SPLITS for di in DIRS]]))
    small_table('reactzyme_retrained','lrrrrrr',header,rows)

    rows=[]
    cases=[('Native F3',d['case']['methods']['reaction_smi/native_before_phase2']),
           (r'\methodname',d['case']['methods']['reaction_smi/selected'])]
    for variant,label in [('all',r'\vfourBio'),('shuffled_1','Shuffled labels'),('without_ec','Without EC'),('without_cofactor','Without cofactor'),('without_mechanism','Without mechanism')]:
        r=next(r for r in d['case_controls']['records'] if r['weight']==1 and r['variant']==variant)
        cases.append((label,r['methods']['reaction_smi/selected']))
    for label,m in cases:
        nums=[m['primary_papers'][f'recovered_at_{k}'] for k in (5,10,25)]
        nums.append(m['entry_level_144']['primary_papers']['recovered_at_25'])
        rows.append(line([label,*[number('case1',label,str(i),x,'Case1 saved summaries','d') for i,x in enumerate(nums)],number('case1',label,'AUROC',m['broad_assay_conditional_discrimination']['auc'],'Case1 saved summaries')]))
    small_table('case1','lrrrrr',['Model / labels','Unique @5','Unique @10','Unique @25','Entries @25','AUROC'],rows)
    rows=[]
    for split,label in [*SPLITS,('enzymemap','EnzymeMap')]:
        m=d['case']['methods'][split+'/selected']
        b=next(r for r in d['case_controls']['records'] if r['weight']==1 and r['variant']=='all')['methods'][split+'/selected']
        cells=[]
        for variant,result in [('control',m),('all_1',b)]:
            for metric,x,fmt in [('unique',result['primary_papers']['recovered_at_25'],'d'),('entries',result['entry_level_144']['primary_papers']['recovered_at_25'],'d'),('AUROC',result['broad_assay_conditional_discrimination']['auc'],'.6f')]:
                cells.append(number('case1_targets',split,variant+'/'+metric,x,'Case1 saved summaries',fmt))
        rows.append(line([label,*cells]))
    small_table('case1_targets','lrrrrrr',['Training split',r'\multicolumn{3}{c}{\methodname}',r'\multicolumn{3}{c}{\vfourBio}'],[line(['','Unique','Entries','AUROC','Unique','Entries','AUROC']),r'\midrule',*rows])

    rows=[]
    for label,field in [(r'\methodname','v4_test'),(r'\vfourBio','bio_test')]:
        for split,split_label in SPLITS:
            for di in DIRS:
                m=d[field][split][di]['all']
                rows.append(line([label,split_label,r'R$\to$E' if di==DIRS[0] else r'E$\to$R',*[number('retrieval_diagnostics',label,split+'/'+di+'/'+key,m[key],field) for key in ['reactzyme_mrr','first_positive_mrr','top_1','top_5','top_10']]]))
    small_table('retrieval_diagnostics','lllrrrrr',['Model','Split','Direction','MRR','First MRR','Hit@1','Hit@5','Hit@10'],rows)

    for split,label in SPLITS:
        rows=[]
        for r in sorted(d['public_runs'],key=lambda r:r['_run']):
            if r.get('split')!=split: continue
            family=r.get('family',r['_run'].removesuffix('_'+split+'_seed42'))
            protein=r.get('protein','--'); reaction=r.get('reaction','--')
            rows.append(line([escape(family),escape(protein),escape(reaction),number('all_public_'+split,r['_run'],'selected_epoch',r['selected_epoch'],r['_source'],'d'),*[number('all_public_'+split,r['_run'],di,r['test'][di]['all']['reactzyme_mrr'],r['_source']) for di in DIRS]]))
        write('all_public_'+split,[r'\begin{longtable}{p{.32\linewidth}llrrr}',
             r'\caption{Completed matched-data runs: '+label+r'. Every recorded configuration is retained; rows are not selected by test performance.}\label{tab:exp-all-'+split.replace('_','-')+r'}\\',
             r'\toprule',line(['Family / adapter','Protein','Reaction','Epoch',r'R$\to$E',r'E$\to$R']),r'\midrule\endfirsthead',
             r'\toprule',line(['Family / adapter','Protein','Reaction','Epoch',r'R$\to$E',r'E$\to$R']),r'\midrule\endhead',
             r'\bottomrule\endfoot',*rows,r'\end{longtable}'])
    counts={}
    for r in d['public_runs']:
        key=r.get('family','external');counts[key]=counts.get(key,0)+1
    write('snapshot_macros',[r'\providecommand{\experimentSnapshot}{'+escape(d['captured_utc'][:19].replace('T',' '))+r' UTC}',
          r'\providecommand{\completedOfficialRuns}{'+str(sum(counts.get(k,0) for k in ['mlp','contrastive','transformer','birnn'])-sum(r['_run'].startswith('transformer_enzgfm') and r.get('family')=='transformer' for r in d['public_runs']))+'}',
          r'\providecommand{\completedPublicRuns}{'+str(len(d['public_runs']))+'}'])
    with (HERE/'table_values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    print(f'Built tables from {len(d["public_runs"])} completed public runs; {len(ledger)} numerical cells logged.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--refresh',action='store_true'); args=parser.parse_args()
    path=HERE/'evidence_snapshot.json'
    if args.refresh or not path.exists():
        data=snapshot(); path.write_text(json.dumps(data,indent=2)+'\n')
    else:
        data=json.loads(path.read_text())
    build(data)
