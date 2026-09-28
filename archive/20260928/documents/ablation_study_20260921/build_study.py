"""Audit existing ablations and package their evidence; never starts model runs."""
from pathlib import Path
import contextlib
import csv
import hashlib
import inspect
import io
import json
import statistics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'


def audit():
    sources, checks, records, datasets = {}, [], [], {}

    def read(path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = BASE / path
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected:
            assert digest == expected, f'Source hash mismatch: {path}'
        sources[str(path.relative_to(ROOT))] = digest
        return json.loads(raw)

    def verify_row(row):
        raw = read(row['source'], row.get('source_sha256'))
        summary = raw['summary']
        if row['benchmark'] == 'ReactZyme':
            value = summary[row['metric']]['all']['reactzyme_mrr']
        else:
            value = summary[row['setting']][row['metric']]
            assert summary['table1']['queries'] == 1521
            assert summary['table1']['candidate_ids'] == 261907
            assert summary['table2']['queries'] == 1337
            assert summary['table2']['candidate_ids'] == 252113
        assert abs(value - row['value']) < 1e-12
        return raw

    metric_names = {'bedroc85': 'BEDROC85', 'bedroc20': 'BEDROC20',
                    'ef0.05': 'EF5', 'ef0.1': 'EF10',
                    'reaction_to_enzyme': 'R→E MRR', 'enzyme_to_reaction': 'E→R MRR'}
    recipes = {
        'a025_c1': 'shared_recipe_beta5_b512_e10_fusion3_v2',
        'a040_c1': 'shared_recipe_alpha04_cap1_v1',
        'a050_c1': 'shared_recipe_alpha05_cap1_v1',
        'a040_c05': 'shared_recipe_alpha04_cap05_v1',
    }
    protocols = {k: read(v + '/protocol.json') for k, v in recipes.items()}
    reference = protocols['a025_c1']
    for key, p in protocols.items():
        assert {k: v for k, v in p['recipe'].items() if k not in ('semantic_alpha', 'residual_cap')} == {
            k: v for k, v in reference['recipe'].items() if k not in ('semantic_alpha', 'residual_cap')}
        assert p['enzymemap'] == reference['enzymemap']
        for a, b in zip(p['reactzyme'], reference['reactzyme']):
            assert a['split'] == b['split']
            assert a['checkpoint_sha256'] == b['checkpoint_sha256']
            assert a['train_config_sha256'] == b['train_config_sha256']
            assert a['task'] == b['task']
    matrices, axes = {}, {}
    for variant, dirname in recipes.items():
        rows = read(dirname + '/qualification.json')['rows']
        assert len(rows) == 14
        for row in rows:
            raw = verify_row(row)
            key = (row['benchmark'], row['setting'], row['metric'])
            if row['benchmark'] == 'ReactZyme':
                axis = {k: v['sha256'] for k, v in raw['truth_provenance'].items()}
            else:
                axis = {k: raw[k] for k in ('protocol_receipt_sha256', 'query_ids_sha256', 'candidate_ids_sha256', 'notebook_sha256')}
            old = axes.setdefault((row['benchmark'], row['setting']), axis)
            assert old == axis, 'Inference variants changed evaluation axes'
            matrices.setdefault(key, {})[variant] = row['value']
            records.append(dict(study='inference', variant=variant, seed=42, **row))
    checks.append('Four inference recipes: identical recorded F3 lineage/configuration and EnzymeMap run; only alpha/cap change; evaluation axes and raw metrics match.')
    datasets['inference'] = [dict(order=i, setting=f'{k[0]} / {k[1]}', metric=metric_names[k[2]], **{a: f'{b:.6f}' for a, b in vals.items()}) for i, (k, vals) in enumerate(matrices.items())]

    bio_dir = 'v4_relative_biology_phase2_weight1_controls_20260921_v1'
    biology = read(bio_dir + '/comparison.json')['methods']
    reproduction = read(bio_dir + '/control_reproduction.json')
    assert len(reproduction) == 4
    assert all(r['identical_state'] and r['architecture_identical'] and r['max_parameter_difference'] == 0 for r in reproduction)
    read(bio_dir + '/protocol.json')
    bm = {}
    for method in biology:
        variant = method['variant']['name']
        for row in method['rows']:
            raw = verify_row(row)
            key = (row['benchmark'], row['setting'], row['metric'])
            bm.setdefault(key, {})[variant] = row['value']
            if variant == 'control':
                assert abs(row['value'] - matrices[key]['a040_c05']) < 1e-12
            records.append(dict(study='phase2_biology', variant=variant, seed=42, **row))
    checks.append('Phase-2 biology: all 84 values verified against hashed raw results; unannotated control equals V4 in all 14 cells; archived exact-state reproduction passes for four tasks.')
    datasets['biology'] = [dict(order=i, setting=f'{k[0]} / {k[1]}', metric=metric_names[k[2]], **{a: f'{b:.6f}' for a, b in vals.items()}) for i, (k, vals) in enumerate(bm.items())]
    read('v4_relative_biology_phase2_20260921_v1/tie_sensitivity.json')

    temp = read('sleec_beta10_matched18_replication_v1/paired_summary.json')
    conf = read('sleec_beta10_matched18_replication_v1/paired_config_audit.json')
    assert conf['same_model_data_optimizer_epoch_budget_and_phase2'] and conf['epochs'] == 18
    for entry in conf['configs']:
        assert {x['field'] for x in entry['config_differences']} <= {'.training.loss.beta', '.ablation.variant', '.logging.checkpoint_dir', '.logging.log_dir'}
        for beta, dn in [(5, 'sleec_beta5_fresh_replication_v1'), (10, 'sleec_beta10_matched18_replication_v1')]:
            f = BASE / dn / f"seed{entry['seed']}/configs/train.yaml"
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            assert digest == entry[f'sha256_beta{beta}']
            sources[str(f.relative_to(ROOT))] = digest
    tm = {}
    for r in temp['sources']:
        raw = read(r['path'], r['sha256'])
        tm[r['beta'], r['seed']] = raw['summary']
        for setting in ('table1', 'table2'):
            for metric in ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1'):
                records.append(dict(study='temperature_18epochs', variant=f"beta{r['beta']}", seed=r['seed'], benchmark='EnzymeMap', setting=setting, metric=metric, value=raw['summary'][setting][metric], source=r['path']))
    datasets['temperature'] = []
    datasets['temperature_seed_deltas'] = []
    for setting in ('table1', 'table2'):
        for metric in ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1'):
            v5 = [tm[5, seed][setting][metric] for seed in (17, 42, 73)]
            v10 = [tm[10, seed][setting][metric] for seed in (17, 42, 73)]
            delta = [a-b for a, b in zip(v5, v10)]
            if metric == 'bedroc85':
                for seed, a, b, change in zip((17, 42, 73), v5, v10, delta):
                    datasets['temperature_seed_deltas'].append(dict(label=f'Seed {seed} / {setting}', seed=seed, setting=setting, beta5=a, beta10=b, gain=change))
            assert abs(statistics.mean(v5) - temp['three_seed_summary']['5'][setting][metric]['mean']) < 1e-12
            datasets['temperature'].append(dict(order=len(datasets['temperature']), setting=setting, metric=metric_names[metric], beta5=f'{statistics.mean(v5):.6f} ± {statistics.stdev(v5):.6f}', beta10=f'{statistics.mean(v10):.6f} ± {statistics.stdev(v10):.6f}', delta=f'{statistics.mean(delta):+.6f}', favorable=f'{sum(d>0 for d in delta)}/3'))
    checks.append('Temperature: six current YAML hashes match archived config audit; six raw test hashes verified; paired means recomputed over all three seeds.')

    sd = read('v4_relative_biology_seed_replication_20260921_v1/seed_comparison.json')
    read('v4_relative_biology_seed_replication_20260921_v1/protocol.json')
    seed_rows = {}
    for row in sd['records']:
        raw = read(row['source'], row['source_sha256'])
        assert raw['summary'] == row['summary']
        seed_rows[row['method'], row['seed']] = raw['summary']
        for setting in ('table1', 'table2'):
            for metric in ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1'):
                records.append(dict(study='f3_biology_seeds', variant=row['method'], seed=row['seed'], benchmark='EnzymeMap', setting=setting, metric=metric, value=raw['summary'][setting][metric], source=row['source']))
    assert len(seed_rows) == 9
    datasets['seeds'] = []
    for setting in ('table1', 'table2'):
        for metric in ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1'):
            control = [seed_rows['control', s][setting][metric] for s in (42, 43, 44)]
            row = dict(order=len(datasets['seeds']), setting=setting, metric=metric_names[metric], control=f'{statistics.mean(control):.6f} ± {statistics.stdev(control):.6f}')
            for method, label in [('relative_0p1', 'w01'), ('relative_1', 'w1')]:
                values = [seed_rows[method, s][setting][metric] for s in (42, 43, 44)]
                delta = [a-b for a, b in zip(values, control)]
                row[label] = f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}'
                row[label+'_delta'] = f'{statistics.mean(delta):+.6f} ({sum(d>0 for d in delta)}/3)'
            datasets['seeds'].append(row)
    checks.append('F3 biology: all nine hashed raw summaries verified; paired effects include seeds 42, 43, 44 without selecting a seed; phase-2 objective remains unannotated in this study.')

    cases = [('control', read('shared_recipe_alpha04_cap05_v1/case1/summary.json'))]
    for row in read('v4_relative_biology_phase2_case1_controls_20260921_v1/comparison.json')['records']:
        if row['weight'] == 1:
            raw = read(row['source'], row['source_sha256'])
            assert raw['methods'] == row['methods']
            cases.append((row['variant'], raw))
    datasets['case1'] = []
    for variant, case in cases:
        r = case['methods']['reaction_smi/selected']
        entry = r['entry_level_144']
        datasets['case1'].append(dict(order=len(datasets['case1']), variant=variant, unique=f"{r['primary_papers']['recovered_at_25']}/12", entries=f"{entry['primary_papers']['recovered_at_25']}/15", auroc=f"{r['broad_assay_conditional_discrimination']['auc']:.6f}"))
    checks.append('Case1: raw frozen follow-up summaries checked; unique-sequence recovery and original-entry recovery kept separate; only Reaction-Sim checkpoint displayed, other checkpoints not pooled.')

    datasets['coverage'] = [
        dict(order=0, choice='Semantic dictionary weight', status='Controlled sensitivity', coverage='All 3 ReactZyme splits + EnzymeMap T1/T2', contrast='α=0.25/0.40/0.50, cap=1; identical weights'),
        dict(order=1, choice='Phase-2 residual strength', status='Controlled sensitivity', coverage='All 3 ReactZyme splits + EnzymeMap T1/T2', contrast='cap=0.5 vs 1 at α=0.40; identical weights'),
        dict(order=2, choice='Phase-2 biological labels', status='Controlled loss ablation', coverage='Both benchmarks + retrospective Case1', contrast='None / correct / shuffled / remove EC, cofactor, mechanism'),
        dict(order=3, choice='Inverse temperature β', status='Controlled component study', coverage='EnzymeMap; seeds 17, 42, 73; 18 epochs', contrast='β=5 vs 10; not exact final 10-epoch V4'),
        dict(order=4, choice='F3 biological supervision', status='Paired seed sensitivity', coverage='EnzymeMap; seeds 42, 43, 44', contrast='Weights 0/0.1/1, fixed unannotated phase 2'),
        dict(order=5, choice='Fusion multiplier', status='Partial; validation study', coverage='Both benchmarks, older recipe variants', contrast='Saved calibration/transfer; no complete fixed V4 test grid established'),
        dict(order=6, choice='Directional encoding / anchor-balanced loss / learned queries', status='Historical pilots; needs matched V4 controls', coverage='Scattered earlier runs', contrast='Do not attribute the combined recipe improvement to one component'),
        dict(order=7, choice='Remove dictionary / remove phase 2', status='Needs exact V4 comparison grid', coverage='Earlier compositions exist', contrast='Use α=0 and residual multiplier=0 with all other V4 settings fixed'),
    ]
    for file in ['dual_encoder_architecture_pilots_v1/completed_test_results.json', 'reactzyme_multiview_fusion_calibration_v1/protocol.json', 'enzymemap_multiview_fusion_transfer_v1/protocol.json', 'enzymemap_multiview_fusion_transfer_v1/seed42/selection.json']:
        read(file)
    read('v4_relative_biology_phase2_20260921_v1/case1_rank_change_audit.json')
    result = dict(generated_utc=datetime.now(timezone.utc).isoformat(), checks=checks, source_sha256=sources, datasets=datasets, records=records)
    print(f'Validated {len(records)} benchmark measurements from {len(sources)} source/config files.')
    for check in checks:
        print('PASS:', check)
    return result


def build_report(evidence):
    now = evidence['generated_utc']
    blocks, tables, sources = [], [], []
    title = 'What the existing V4 runs can establish'

    def prose(key, heading, body, source=None):
        block = dict(id=key, type='markdown', layout='full', body=heading+'\n\n'+body)
        if source:
            block['sourceId'] = source
        blocks.append(block)

    def table(key, title, columns):
        source = dict(id=key, label=title+' — audited local experiment records', path='documents/ablation_study_20260921/evidence.json', query=dict(engine='Python standard library', language='python', description='Extract existing experiment summaries, verify hashes and metric values, calculate paired differences. No new training or evaluation.', tables_used=sorted(evidence['source_sha256']), filters=['Recorded experiments audited on September 21, 2026; no best-seed selection.'], metric_definitions=['MRR is ReactZyme all-positive per-query MRR, not first-positive MRR.', 'EnzymeMap rows preserve each screening protocol; no averaging across Table 1 and Table 2.', 'Reported ± values are sample standard deviations across three seeds, not confidence intervals.']))
        sources.append(source)
        tables.append(dict(id=key, title=title, dataset=key, sourceId=key, layout='full', density='dense', defaultSort=dict(field='order', direction='asc'), columns=[dict(field='order', label='#', type='number')]+[dict(field=f, label=l, type='text') for f, l in columns]))
        blocks.append(dict(id=key+'_table', type='table', tableId=key, layout='full'))

    prose('title', '# '+title, '')
    prose('summary', '## Technical summary', '**Yes: the recorded runs support a substantial exploratory ablation study.** The strongest reusable evidence isolates inference weights, added phase-2 biological supervision, and temperature. Three-seed EnzymeMap controls also quantify the instability of adding biological supervision to F3. These results do not yet provide a complete component-removal study of the final V4 architecture.\n\nThe study should explain which choices help, hurt, or trade one metric for another. Winning literature comparator cells is not evidence that a particular component caused the win. No new model runs were launched for this audit.')
    prose('definitions', '## Keep the two benchmark protocols separate', '**ReactZyme:** report all-positive per-query mean reciprocal rank (MRR) in both R→E and E→R directions, separately for Reaction-Sim, Enzyme-Sim, and Time. Do not substitute first-positive MRR. Each contrast must retain the same training split, candidate order, positives, and rank/tie implementation.\n\n**EnzymeMap:** retain the original 34,427/7,287/4,642 train/validation/test association rows. Table 1 evaluates 1,521 eligible unique test reactions against 261,907 candidate IDs. Table 2 removes training enzyme IDs: 1,337 eligible queries and 252,113 candidates. Report BEDROC85, BEDROC20, EF5 and EF10 separately for each setting. BEDROC measures early ranking; EF measures enrichment in the top 5% or 10%. All reported values are raw metric units.\n\n**Case1:** keep the 144 original entries separate from 123 unique protein sequences. The primary-paper positive denominators are 15 entries and 12 unique sequences. This is a retrospective literature panel, not a new wet-lab experiment.')
    prose('design', '## Five experimental choices already have usable controls', 'The table classifies evidence by what actually changes, not by the run name. A hyperparameter sensitivity study is useful, but does not prove that the corresponding component is necessary. Missing matched controls below mean a complete final-V4 comparison was not established by this audit; they do not imply that no historical run exists.')
    table('coverage', 'Existing ablation coverage', [('choice','Choice'),('status','Evidence'),('coverage','Coverage'),('contrast','Contrast')])
    prose('inference_text', '## Inference weights produce measurable trade-offs', '**Same trained weights, different inference coefficients:** at residual multiplier 1, raising dictionary weight α from 0.25 to 0.40 improves Reaction-Sim E→R MRR from 0.523804 to 0.537749. EnzymeMap Table 1 BEDROC85 changes little (0.577817→0.577945), while Table 2 decreases (0.536819→0.534129). Raising α further to 0.50 reduces R→E MRR on all three ReactZyme splits.\n\nAt α=0.40, reducing the residual multiplier from 1 to 0.5 gives the preferred V4 configuration, but lowers EnzymeMap BEDROC85 in both settings. V4 therefore represents a trade-off; the current evidence does not make the smaller residual universally better. The parameter called “cap” in the code multiplies the residual update; it is not a norm clamp. These runs are single-seed and exploratory. Apparent Enzyme-Sim/Time E→R changes also require the near-tie sensitivity caveat below.', 'inference')
    table('inference', 'Inference sensitivity: all recorded test cells', [('setting','Benchmark / setting'),('metric','Metric'),('a025_c1','α .25 / cap 1'),('a040_c1','α .40 / cap 1'),('a050_c1','α .50 / cap 1'),('a040_c05','α .40 / cap .5 (V4)')])
    prose('temperature_text', '## Lower inverse temperature has the clearest replicated benefit', '**β=5 outperforms β=10 in 22 of 24 paired seed-by-metric comparisons.** Table 1 mean BEDROC85 increases from 0.494040 to 0.521656; mean EF5 increases from 14.685863 to 15.651865. Both improve in all three seeds. EF10 is the exception: it worsens for seed 73 in both screening settings.\n\nThe configurations hold architecture, target data, optimizer, 18-epoch budget and phase 2 fixed; current YAML hashes match the archived difference audit. Report this as an EnzymeMap component study. It uses an earlier 18-epoch recipe, so it is not a three-seed ablation of final 10-epoch V4 and supplies no matched ReactZyme temperature result. “±” below is sample standard deviation, not a confidence interval.', 'temperature')
    table('temperature', 'Paired inverse-temperature comparison', [('setting','Setting'),('metric','Metric'),('beta5','β=5 mean ± SD'),('beta10','β=10 mean ± SD'),('delta','Mean Δ (5−10)'),('favorable','Favorable seeds')])
    blocks.insert(len(blocks)-1, dict(id='temperature_seed_chart', type='chart', chartId='temperature_seed_chart', layout='full'))
    prose('biology_text', '## Phase-2 biology does not explain the benchmark wins', '**This is a clean loss ablation with a mostly negative result for early screening.** Keep F3, existing heads, dictionary and inference fixed; compare no added supervision, correct labels, shuffled labels, and removal of EC, cofactor or mechanism terms at weight 1. The archived unannotated controls reproduce the original head parameters exactly, and this audit confirms identical V4 test values in all 14 cells.\n\nCorrect labels give Table 1 BEDROC85 0.572096 versus 0.572337 without them; shuffled labels give 0.572093. The difference between correct and shuffled labels is only about 0.000003. Retaining all three annotation families does not improve every metric. Family-removal contrasts preserve remaining coefficients and the fixed divisor of three; removing a family also reduces total regularization, so shuffled labels are an essential additional control.\n\nThe Time E→R point estimate rises from 0.780669 to 0.800614 with biology. However, the archived ±10⁻⁶ score-band diagnostic gives strongly overlapping rank bounds (control approximately 0.77158–0.81982; biology 0.77161–0.81988). These are numerical-sensitivity bounds, not statistical intervals. Do not describe that MRR jump as robust biological generalization.', 'biology')
    table('biology', 'Phase-2 biological-supervision ablation', [('setting','Benchmark / setting'),('metric','Metric'),('control','None'),('all_1','All, weight 1'),('shuffled_1','Shuffled'),('without_ec','−EC'),('without_cofactor','−Cofactor'),('without_mechanism','−Mechanism')])
    prose('seeds_text', '## F3 biology improves some averages but remains seed-sensitive', '**The F3 study is a different experiment from the phase-2-only study above.** It retrains F3 with added relative biological loss at weight 0.1 or 1 and retains the unannotated phase-2 objective. Within each seed, compare to its corresponding unannotated control.\n\nAt weight 1, the mean Table 1 BEDROC85 gain is +0.010256, favorable in two of three seeds; mean EF5 falls by 0.151444. Table 2 EF5 decreases in all three seeds. At weight 0.1, Table 1 BEDROC85 improves in only one of three seeds. Thus seed 42 alone would overstate reliability. The nine raw results and paired differences are preserved in the companion data. The archived seed-42 control is reused; a separately recorded fresh seed-42 audit supports similar, but not bit-identical, behavior. Three seeds remain too few to claim a robust generalization mechanism.', 'seeds')
    table('seeds', 'EnzymeMap F3 biological-loss seed sensitivity', [('setting','Setting'),('metric','Metric'),('control','No added loss, mean ± SD'),('w01','Weight .1, mean ± SD'),('w01_delta','Δ .1 (positive seeds)'),('w1','Weight 1, mean ± SD'),('w1_delta','Δ 1 (positive seeds)')])
    prose('case_text', '## Case1 supports a small retrospective observation', '**For the Reaction-Sim model, phase-2 biology moves one literature-supported catalyst from rank 26 to rank 25 in the unique-sequence analysis.** Recovery therefore changes from 10/12 to 11/12 at 25. The original 144-entry analysis remains 10/15. Removing the cofactor term retains the unique-sequence change and slightly increases conditional AUROC.\n\nThe table uses the same Reaction-Sim checkpoint lineage for every row; it does not choose the best benchmark-trained checkpoint per metric. AUROC distinguishes workbook-reported activity from conditional non-detection, not universal biochemical negatives. The panel was inspected earlier in development, so it cannot serve as untouched or prospective validation.', 'case1')
    table('case1', 'Case1 phase-2 controls: Reaction-Sim model', [('variant','Added biology'),('unique','Unique paper catalysts @25'),('entries','Paper entries @25 / 144'),('auroc','Conditional AUROC')])
    prose('limits', '## A full architectural explanation still needs matched removals', 'Historical pilots changed combinations of reaction encoding, model size, losses, training duration, validation rules and downstream composition. Their scores are useful development evidence, but juxtaposing their best checkpoints would not isolate directional F3, anchor balancing, learned residue queries, or phase 2. The saved fusion experiments also include validation-selected transfers rather than a complete final-V4 test grid.\n\nRepeated test inspection makes this entire retrospective study exploratory. Keep all arms, use validation for future selection, report paired seeds, and preserve failed or negative variants. Added EC/cofactor/mechanism annotations are additional supervision even when association training data and architecture are matched. Do not describe the biology arms as identical-information comparisons. Literature thresholds and currently running public competitors are external comparisons, not V4 component ablations.')
    prose('next', '## Complete the paper study with a small fixed matrix', '1. **First, finish the inference removals using existing weights:** V4; α=0; residual multiplier=0; both zero; fusion multiplier=1 with other V4 settings fixed. This isolates dictionary, residual and fusion contributions without retraining. Compare all four target-specific fits, not just the most favorable split.\n2. **Then retrain only architecture/loss controls that lack matched runs:** original MLNCE versus anchor-balanced loss; ordered versus direction-invariant chemistry on EnzymeMap; four learned residue queries versus one (or no learned-query view). Change one choice at a time and retain the same frozen inputs, 10-epoch budget, phase 2, inference settings and paired seeds. ReactZyme participant sets do not provide physical reaction sides, so do not claim that split alone tests chemical directionality.\n3. **Use one reference throughout:** unannotated V4 for architecture and inference ablations; V4 plus weight-1 phase-2 biology as a separately labeled extension with correct/shuffled/family-removal controls. Do not combine best cells from different models.\n4. **Report both benchmarks:** six ReactZyme MRR cells plus eight EnzymeMap screening cells; show per-seed points and mean ± sample SD where replicated. Revisit Case1 only after configurations are fixed. No new runs have been scheduled by this audit.')
    prose('questions', '## Questions the existing evidence cannot yet settle', 'Does the dictionary account for most of the final score gain? Are learned residue queries necessary once frozen protein features and phase 2 are held fixed? Does anchor balancing improve generalization independently of temperature and directional encoding? Can any added biological supervision produce consistent benefit across seeds and an independently collected reaction panel? These are the next defensible ablation questions; the existing winning scores alone do not answer them.')
    charts = [dict(id='temperature_seed_chart', title='BEDROC85 differences by paired seed', subtitle='β=5 minus β=10; fixed 18 epochs; each bar uses the same seed and screening setting', showDescription=True, type='bar', dataset='temperature_seed_deltas', sourceId='temperature', encodings=dict(x=dict(field='label', type='nominal', label='Seed / screening setting'), y=dict(field='gain', type='quantitative', label='BEDROC85 difference'), tooltip=[dict(field='beta5', type='quantitative', label='β=5'), dict(field='beta10', type='quantitative', label='β=10')]))]
    artifact = dict(surface='report', manifest=dict(version=1, surface='report', title=title, generatedAt=now, blocks=blocks, tables=tables, charts=charts, sources=sources), snapshot=dict(version=1, generatedAt=now, status='ready', datasets=evidence['datasets']), sources=sources)
    (OUT/'artifact.json').write_text(json.dumps(artifact, indent=2, ensure_ascii=False)+'\n')
    # Preserve the complete report when the packaged HTML reader cannot accept
    # honest local-file/Python provenance. Do not fabricate a SQL source.
    markdown = []
    for block in blocks:
        if block['type'] == 'markdown':
            markdown.append(block['body'])
        elif block['type'] == 'table':
            spec = next(t for t in tables if t['id'] == block['tableId'])
            cols = spec['columns'][1:]
            rows = evidence['datasets'][spec['dataset']]
            text = ['**'+spec['title']+'**', '', '| '+' | '.join(c['label'] for c in cols)+' |', '| '+' | '.join('---' for c in cols)+' |']
            text += ['| '+' | '.join(str(r.get(c['field'], '')).replace('|', '/') for c in cols)+' |' for r in rows]
            markdown.append('\n'.join(text))
    markdown.append('Audit companions: [executed notebook](audit.ipynb), [260 measurements](measurements.csv), [raw-source hashes and paired tables](evidence.json), [rebuild script](build_study.py). These checks verify saved summaries and provenance; predictions were not recomputed. The HTML export is blocked by the bundled report validator requiring SQL provenance for local-file/Python evidence; the full report is preserved here without inventing a SQL source.')
    (OUT/'report.md').write_text('\n\n'.join(markdown)+'\n')


if __name__ == '__main__':
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        evidence = audit()
    print(capture.getvalue(), end='')
    (OUT/'evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+'\n')
    cols = ['study', 'variant', 'seed', 'benchmark', 'setting', 'metric', 'value', 'source']
    with (OUT/'measurements.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(evidence['records'])
    intro = 'This notebook re-reads local experiment artifacts, verifies provenance and raw metric values, and reconstructs paired ablation tables. It launches no training and performs no new test selection. Run from anywhere within the repository. Source paths and hashes are retained in evidence.json.'
    setup = "from pathlib import Path\nimport hashlib, json, statistics\nfrom datetime import datetime, timezone\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'runs/generalization_20260919_2251').exists())\nBASE = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'\n"
    code = setup+'\n'+inspect.getsource(audit)+'\nevidence = audit()\n'
    nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python', version='3')), cells=[dict(id='purpose', cell_type='markdown', metadata={}, source=intro.splitlines(True)), dict(id='audit', cell_type='code', metadata={}, execution_count=1, source=code.splitlines(True), outputs=[dict(output_type='stream', name='stdout', text=capture.getvalue().splitlines(True))]), dict(id='inspect-paired-effects', cell_type='code', metadata={}, execution_count=2, source=["for row in evidence['datasets']['temperature']:\n", "    print(row)\n"], outputs=[dict(output_type='stream', name='stdout', text=[str(r)+'\n' for r in evidence['datasets']['temperature']])])])
    (OUT/'audit.ipynb').write_text(json.dumps(nb, indent=2, ensure_ascii=False)+'\n')
    build_report(evidence)
    notes = {'audience':'technical', 'delivery':'portable HTML', 'structure':'Summary; definitions moved before evidence for interpretability; experimental-design coverage; inference, temperature, biology, seed and Case1 evidence; limitations; next steps; further questions.', 'visual_contract':'Full-width exact-value audit tables. Multiple metrics with distinct scales and small near-tie effects make exact lookup more useful than bars. Neutral text; signs explicit; no cross-metric aggregation. Table order is declared.', 'scope':'Retrospective audit only. No model runs launched, stopped, or changed. Raw source/config hashes checked; large model/score arrays not rehashed or predictions recomputed.', 'sources':evidence['source_sha256']}
    (OUT/'source_notes.json').write_text(json.dumps(notes, indent=2)+'\n')
    (OUT/'html_export_blocker.json').write_text(json.dumps({'status':'blocked', 'attempted_surface':'portable HTML', 'reason':'Canonical packaged validator requires SQL query/file provenance for every chart/table; reviewed sources are local JSON/YAML and Python audit code. Python source provenance is rejected. No fabricated SQL source supplied.', 'fallback':'Complete report.md and executed audit.ipynb', 'browser_qa':'Not run: no report.html produced.'}, indent=2)+'\n')
    # The requested Markdown report includes its scientific figures on rebuild.
    from plot_study import render_study
    render_study()
