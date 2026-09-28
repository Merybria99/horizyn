#!/usr/bin/env python3
"""Report fresh biological F3 results and freeze qualified Case1 follow-ups."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generalization_full_graph import atomic_json, sha
from generalization_biological_geometry_evaluate import CROSS

VARIANTS = ('f3_biology', 'f3_phase2_biology')
NAMES = ('reaction_smi', 'enzyme_smi', 'time', 'enzymemap')


def report(out):
    plan = json.loads((out / 'protocol.json').read_text())
    mode = plan.get('biological_mode', 'attraction')
    biological_weight = plan.get('biological_weight', .1)
    targets_path = CROSS / 'goal_primary_comparators_20260921.json'
    targets = json.loads(targets_path.read_text())
    parent = json.loads((CROSS / 'shared_recipe_alpha04_cap05_v1/case1_freeze.json').read_text())
    records = []
    for variant in VARIANTS:
        rows, models = [], []
        for name in NAMES:
            evaluation = out / 'followup' / name
            result_path = evaluation / name / variant / 'test_summary.json'
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())['summary']
            task, = json.loads((evaluation / 'protocol.json').read_text())['tasks']
            if name == 'enzymemap':
                for table, metrics in targets['enzymemap'].items():
                    for metric, target in metrics.items():
                        if table == 'table2' and metric == 'ef0.1':
                            target = max(target, 7.81)
                        rows.append(dict(benchmark='EnzymeMap', setting=table, metric=metric,
                            value=result[table][metric], target=target, passes=result[table][metric] > target,
                            source=str(result_path), source_sha256=sha(result_path)))
            else:
                for metric, target in targets['reactzyme'][name].items():
                    value = result[metric]['all']['reactzyme_mrr']
                    rows.append(dict(benchmark='ReactZyme', setting=name, metric=metric, value=value,
                        target=target, passes=value > target, source=str(result_path), source_sha256=sha(result_path)))
            model = copy.deepcopy(task['model'])
            model['phase2_checkpoint'] = str(evaluation / name / variant / 'training/step0100.pt')
            model['phase2_checkpoint_sha256'] = sha(model['phase2_checkpoint'])
            model['benchmark_result'] = str(result_path)
            source = Path(task['source_phase2'])
            model['dictionary_sha256'] = sha(source / 'anchors.pt')
            model['feature_manifest_sha256'] = sha(source / 'features/manifest.json')
            models.append(model)
        method = out / 'methods' / variant; method.mkdir(parents=True, exist_ok=True)
        record = dict(variant=variant, available_cells=len(rows), passed_cells=sum(r['passes'] for r in rows),
            required_cells=14, all_results_available=len(rows) == 14,
            all_primary_targets_exceeded=len(rows) == 14 and all(r['passes'] for r in rows), rows=rows,
            parent_protocol_sha256=sha(out / 'protocol.json'), target_registry_sha256=sha(targets_path),
            fixed_epoch=10, same_architecture=True, biological_annotations_added=True,
            interpretation='Exploratory repeated-test point comparisons, not independent confirmatory superiority.')
        records.append(record)
        freeze_path = method / 'case1_freeze.json'
        if freeze_path.exists():
            freeze = json.loads(freeze_path.read_text())
            if sha(method / 'qualification.json') != freeze['benchmark_qualification_sha256']:
                raise ValueError('Frozen qualification changed')
        else:
            atomic_json(method / 'qualification.json', record)
            if record['all_primary_targets_exceeded']:
                audit = json.loads((out / 'architecture_and_data_audit.json').read_text())
                if {r['target'] for r in audit['rows'] if r['status'] == 'verified'} != set(NAMES):
                    raise ValueError('Architecture/data audit incomplete')
                freeze = copy.deepcopy(parent)
                freeze.update(frozen_utc=datetime.now(timezone.utc).isoformat(), models=models,
                    benchmark_qualification_sha256=sha(method / 'qualification.json'))
                freeze['recipe']['biological_geometry'] = dict(phase1=biological_weight,
                    phase2=.1 if variant == 'f3_phase2_biology' else 0., families=['ec', 'cofactor', 'mechanism'],
                    additional_learned_parameters=0, mode=mode, margin=plan.get('biological_margin', .1))
                atomic_json(freeze_path, freeze)
    atomic_json(out / 'comparison.json', dict(methods=records, updated_utc=datetime.now(timezone.utc).isoformat()))
    lines = [f'# Fresh F3 biological supervision ({mode}): unchanged V4 architecture', '',
        'The fixed V4 recipe is retrained from scratch for each target, with biological supervision on the existing '
        '512-dimensional embeddings. SLEEC remains frozen and retained. The first method applies the biological loss '
        'during F3 training; the second also applies it during the existing phase-2 refinement. Neither adds model '
        'parameters, inference inputs, heads, slots, or ensembles.', '',
        f'Both use seed 42, 10 epochs, batch 512, the original decoupled all-positive contrastive loss, and F3 biological '
        f'weight {biological_weight:g}. The optional phase-2 biological weight is 0.1. Epoch 10 and both phase-2 variants were declared before these tests. The 3× inference fusion, '
        '100 phase-2 steps, residual multiplier 0.5 and semantic weight 0.4 are unchanged.', '',
        ('The relative loss compares within-category distance with distance to other annotated endpoints, '
         'using a confidence-weighted margin of 0.1. It compares relative separation instead of absolute within-category compactness. '
         'Category-complement examples are representation references, not asserted activity negatives.'
         if mode == 'relative' else 'The attraction loss minimizes confidence-weighted within-category cosine distance.'), '',
        '| Method | Available primary metrics | Targets exceeded |', '| --- | ---: | ---: |']
    for r in records:
        lines.append(f"| {r['variant']} | {r['available_cells']}/14 | {r['passed_cells']}/14 |")
    lines += ['', '| Benchmark / setting / metric | Comparator | Biology in F3 | Biology in F3 + phase 2 |',
        '| --- | ---: | ---: | ---: |']
    keys = list(dict.fromkeys(tuple(x[k] for k in ('benchmark', 'setting', 'metric')) for r in records for x in r['rows']))
    for key in keys:
        entries = [next((x for x in r['rows'] if tuple(x[k] for k in ('benchmark', 'setting', 'metric')) == key), None)
                   for r in records]
        target = next(x['target'] for x in entries if x is not None)
        values = [f"{x['value']:.6f}" if x else 'pending' for x in entries]
        lines.append('| ' + ' / '.join(key) + f' | {target:.6f} | ' + ' | '.join(values) + ' |')
    failed = [(r['variant'], x) for r in records for x in r['rows'] if not x['passes']]
    if failed:
        lines += ['', 'The following comparisons do not exceed their targets:']
        for variant, x in failed:
            lines += ['', f"- {variant}: {x['setting']} / {x['metric']} = {x['value']:.6f}, target {x['target']:.6f}."]
    ties_path = out / 'tie_sensitivity.json'
    if ties_path.exists():
        ties = json.loads(ties_path.read_text())
        lines += ['', '## Reaction-score numerical sensitivity', '',
            'These secondary E→R bounds treat competitors within 10⁻⁶ of each positive score as potentially '
            'tied. They do not replace the official metric or its qualification decision. Wide intervals mean '
            'small changes in the official score should not be interpreted as equivalent changes in discrimination.', '',
            '| Split / method | Pessimistic MRR | Optimistic MRR |', '| --- | ---: | ---: |']
        for r in ties['records']:
            bound = next(x for x in r['bounds'] if x['tolerance'] == 1e-6)
            lines.append(f"| {r['target']} / {r['variant']} | {bound['pessimistic_mrr']:.6f} | {bound['optimistic_mrr']:.6f} |")
        lines += ['', f'[All tolerances and score hashes]({ties_path})']
    chemistry_path = out / 'tie_chemistry_audit.json'
    if chemistry_path.exists():
        chemistry = json.loads(chemistry_path.read_text())
        lines += ['', 'Canonicalizing the released participant sets explains much of this sensitivity. '
            'Canonical equality preserves stereochemistry and multiplicity, while removing atom-map identifiers '
            'and component ordering. Physical reaction sides are absent from these released inputs.', '',
            '| Split | Positive associations with a near competitor | Those with identical canonical participants |',
            '| --- | ---: | ---: |']
        for r in chemistry['records']:
            lines.append(f"| {r['split']} | {r['near_competitor_edges']} | {r['edges_with_canonical_participant_alias']} |")
        lines += ['', 'Different reaction identifiers can therefore demand distinct ranks for chemically equivalent '
            'released inputs. This diagnostic does not merge candidates, alter labels or change the official '
            'comparison. It limits the biological interpretation of small score movements.', '',
            f'[Reaction pairs, original strings and source hashes]({chemistry_path})']
    audit_path = out / 'enzymemap/paired_vs_v4.json'
    if audit_path.exists():
        paired = json.loads(audit_path.read_text())
        lines += ['', '## Comparison with original V4', '',
            'Paired query and rule-cluster resampling describe uncertainty for these fixed trained models. '
            'They do not measure variability across seeds or correct for repeated experimentation. Rules sharing '
            'a test query are joined before cluster resampling.', '',
            '| Table / metric | Change vs V4, F3 + phase 2 | Rule-cluster bootstrap 95% interval |', '| --- | ---: | --- |']
        for table, block in paired['results'].items():
            for metric, r in block['metrics'].items():
                lo, hi = r['rule_cluster_bootstrap_95']
                lines.append(f"| {table} / {metric} | {r['paired_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    geometry_path = out / 'validation_biological_geometry.json'
    if geometry_path.exists():
        geometry = json.loads(geometry_path.read_text())
        lines += ['', '## Representation diagnostic on unseen validation rules', '',
            'This post-fit analysis uses validation annotations only for interpretation, never for training or selection. '
            'Relative category separation is (shuffled mean distance − observed category distance) / shuffled mean distance. '
            'Higher values indicate closer within-category neighborhoods relative to random assignments.', '',
            '| Family | Annotated validation reactions /2652 | V4 separation | Biological F3 separation |',
            '| --- | ---: | ---: | ---: |']
        for family in ('ec', 'cofactor', 'mechanism'):
            rows = {r['model']: r for r in geometry['records'] if r['family'] == family}
            lines.append(f"| {family} | {rows['v4']['coverage']} | {rows['v4']['relative_category_separation']:.6f} | {rows['biological_f3']['relative_category_separation']:.6f} |")
        movements = []
        for family in ('ec', 'mechanism'):
            rows = {r['model']: r for r in geometry['records'] if r['family'] == family}
            delta = rows['biological_f3']['relative_category_separation'] - rows['v4']['relative_category_separation']
            movements.append(f'{family}: {delta:+.6f}')
        lines += ['', 'Changes in relative separation are ' + ', '.join(movements) + '. '
            'Only five validation reactions receive recognized cofactor categories, so that comparison is too sparse '
            'to support a broad conclusion. Better screening scores must not be translated into a claim that the '
            'slots acquired explicit biochemical meanings.', '']
    lines += ['', '## Contribution experiments', '',
        'The matched EnzymeMap experiments remove EC, cofactor or mechanism supervision throughout both training '
        'stages, or shuffle all annotation profiles across training endpoints. The surviving loss terms keep their '
        'original coefficients and divisor of three. Their original association rows, model architecture, seed, '
        'batch and training duration are unchanged. Results are reported without selecting a control by test score.', '',
        '| Annotation control, F3 + phase 2 | Table 1 BEDROC85 | Table 2 BEDROC85 |', '| --- | ---: | ---: |']
    ablations = Path(plan.get('contribution_campaign', CROSS / ('v4_relative_biology_f3_contributions_20260921_v1' if mode == 'relative'
                        else 'v4_biological_f3_contributions_20260921_v1')))
    for variant in ('without_ec', 'without_cofactor', 'without_mechanism', 'shuffled'):
        path = ablations / variant / 'followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json'
        if path.exists():
            s = json.loads(path.read_text())['summary']
            lines.append(f"| {variant} | {s['table1']['bedroc85']:.6f} | {s['table2']['bedroc85']:.6f} |")
        else:
            lines.append(f'| {variant} | pending | pending |')
    seeds = CROSS / ('v4_relative_biology_f3_seed_controls_20260921_v1' if mode == 'relative'
                    else 'v4_biological_f3_seed_controls_20260921_v1')
    if seeds.exists():
        lines += ['', '## Matched seed controls', '',
            'Two additional seeds repeat both unannotated and biologically supervised F3 training with identical '
            'settings within each pair. This comparison uses the original unannotated phase-2 objective for both, '
            'isolating the F3 supervision change. No seed is selected by its test result.', '',
            '| Seed / F3 supervision | Table 1 BEDROC85 | Table 2 BEDROC85 |', '| --- | ---: | ---: |']
        for seed in (43, 44):
            for kind in ('control', 'biology'):
                path = seeds / f'{kind}_seed{seed}' / 'followup/enzymemap/enzymemap/f3_biology/test_summary.json'
                if path.exists():
                    s = json.loads(path.read_text())['summary']
                    lines.append(f"| {seed} / {kind} | {s['table1']['bedroc85']:.6f} | {s['table2']['bedroc85']:.6f} |")
                else:
                    lines.append(f'| {seed} / {kind} | pending | pending |')
        if (seeds / 'seed_comparison.json').exists():
            lines += ['', 'The complete three-seed results show substantial variation across seeds, larger '
                'than most paired biological-loss gains. The three-seed mean is not an ensemble. '
                '[All metrics, variability and paired changes](v4_biology_seed_sensitivity_20260921.md).']
    if mode == 'relative' and (CROSS / 'v4_relative_biology_seed_replication_20260921_v1/protocol.json').exists():
        lines += ['', 'Both relative-loss weights have separate matched seed43/44 replications using the same '
            'completed unannotated controls. [All relative-loss seeds and metrics](v4_relative_biology_seed_sensitivity_20260921.md).']
    if (ROOT / 'documents/v4_biological_query_audit_20260921.md').exists():
        lines += ['', '[Existing-query behavior on 128 unseen validation proteins](v4_biological_query_audit_20260921.md) '
            'measures broad attention, gate usage and permutation invariance. It does not assign biological '
            'names or catalytic-site identities to individual queries.']
    lines += ['', '## Case1', '',
        'Only configurations with all 14 primary comparisons above target are frozen for this follow-up. '
        'The 144 entries are scored and the separate 123-sequence analysis is preserved. The literature panel '
        'was previously examined; it remains retrospective, with no new activity measurements.', '',
        '| Method / target model | Unique paper catalysts @25 /12 (123 sequences) | Paper entries @25 /15 (144 entries) | Conditional activity AUROC (123 sequences) |',
        '| --- | ---: | ---: | ---: |']
    for variant in VARIANTS:
        path = out / 'methods' / variant / 'case1/summary.json'
        if not path.exists():
            record = next(r for r in records if r['variant'] == variant)
            status = ('not run: primary targets not all exceeded' if record['all_results_available']
                      and not record['all_primary_targets_exceeded'] else 'pending qualification/evaluation')
            lines.append(f'| {variant} | {status} | — | — |'); continue
        for name, r in json.loads(path.read_text())['methods'].items():
            if name.endswith('/selected'):
                lines.append(f"| {variant} / {name.removesuffix('/selected')} | {r['primary_papers']['recovered_at_25']} | "
                    f"{r['entry_level_144']['primary_papers']['recovered_at_25']} | {r['broad_assay_conditional_discrimination']['auc']:.6f} |")
    lines += ['', '## Interpretation boundaries', '',
        'EC labels describe functional ancestry. Cofactor labels describe participant presence, not proven dependence. '
        'Mechanism labels are coarse atom-mapped bond-change descriptors, not established complete catalytic mechanisms. '
        'Annotations are restricted to training endpoints and do not enter inference. They add supervision relative '
        'to unannotated models; matching downstream associations does not match all pretraining or annotation resources.', '',
        'ReactZyme Enzyme-Sim and Time E→R scores can be sensitive to nearly tied reaction scores; the earlier phase-2 '
        'study documents this explicitly. Fresh-F3 tie checks must be inspected before attributing large changes '
        'to biology. Beating published point estimates does not establish broad wet-lab generalization.', '',
        '[Architecture and vector interpretation](v4_biological_signal_architecture.md) · '
        '[Earlier phase-2-only contribution study](v4_biological_geometry_20260921.md)', '',
        f'[Protocol]({out}/protocol.json) · [Results]({out}/comparison.json) · '
        f'[Architecture and data audit]({out}/architecture_and_data_audit.json)', '']
    document = plan.get('report_filename', 'v4_relative_biology_f3_20260921.md' if mode == 'relative' else 'v4_biological_f3_20260921.md')
    (ROOT / 'documents' / document).write_text('\n'.join(lines))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    a = p.parse_args(); out = a.campaign.resolve()
    while True:
        subprocess.run([sys.executable, str(ROOT / 'scripts/generalization_biological_f3_audit.py'),
            '--campaign', str(out)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        records = report(out)
        if not a.watch:
            break
        if all(r['all_results_available'] for r in records):
            for r in records:
                method = out / 'methods' / r['variant']
                if r['all_primary_targets_exceeded'] and not (method / 'case1/complete.json').exists():
                    with (method / 'case1.log').open('a') as log:
                        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/generalization_shared_recipe_case1.py'),
                            '--campaign', str(method), '--device', 'cuda:3'], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, check=True)
            report(out)
            atomic_json(out / 'report_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat()))
            break
        for name in NAMES:
            path = out / name / 'followup_state.json'
            if path.exists() and json.loads(path.read_text())['stage'] == 'failed':
                raise RuntimeError(f'Follow-up failed for {name}; report remains partial')
        time.sleep(60)


if __name__ == '__main__':
    main()
