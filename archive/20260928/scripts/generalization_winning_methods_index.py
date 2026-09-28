#!/usr/bin/env python3
"""Build an auditable comparison of completed shared-recipe versions."""
from datetime import datetime, timezone
import fcntl
import json

from generalization_screen_replication import ROOT, sha
from generalization_winning_method_report import case1_paper_coverage


def main():
    cross = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
    names = ['shared_recipe_beta5_b512_e10_fusion3_v2', 'shared_recipe_alpha04_cap1_v1',
             'shared_recipe_alpha05_cap1_v1', 'shared_recipe_alpha04_cap05_v1']
    for name in ('shared_fusion03_beta5_b1024_epoch10_fixed_test_v1', 'shared_fusion03_beta5_b1024_v1'):
        if (cross / name / 'method_report.json').exists():
            names.append(name)
    methods = []
    for index, name in enumerate(names, 1):
        d = cross / name
        qualification = json.loads((d / 'qualification.json').read_text())
        complete = json.loads((d / 'case1/complete.json').read_text())
        if not qualification['all_primary_targets_exceeded'] or qualification['passed_cells'] != 14:
            raise ValueError('Index entry does not meet every primary benchmark target')
        for path, key in [('case1/summary.json', 'summary_sha256'), ('case1_freeze.json', 'freeze_sha256')]:
            if sha(d / path) != complete[key]:
                raise ValueError('Case1 source changed')
        plan = json.loads((d / 'protocol.json').read_text())
        frozen = json.loads((d / 'case1_freeze.json').read_text())
        methods.append(dict(version=f'V{index}', campaign=d, qualification=qualification,
            recipe=plan.get('recipe', frozen['recipe']), plan=plan,
            case=json.loads((d / 'case1/summary.json').read_text())))
    preferred = None
    preference_path = cross / 'preferred_method.json'
    if preference_path.exists():
        preference = json.loads(preference_path.read_text())
        preferred = next(m for m in methods if str(m['campaign']) == preference['campaign'])
        assert preferred['version'] == preference['version']
        assert sha(preferred['campaign'] / 'case1_freeze.json') == preference['frozen_artifacts_sha256']
        assert sha(preferred['campaign'] / 'qualification.json') == preference['qualification_sha256']
    stamp = datetime.now(timezone.utc).isoformat()
    lines = [f'# {len(methods)} shared configurations winning both primary benchmark comparisons', '',
        f'Updated {stamp}. Each version exceeds all **14 primary competitor point estimates**: '
        'six ReactZyme MRR cells and eight EnzymeMap screening cells. Case 1 has been evaluated for every version '
        'on all 144 entries, with a separate analysis of its 123 unique sequences.', '',
        'All versions use the SLEEC residue-view dual encoder, beta 5, seed 42 and 100 full-graph positive-CE phase-2 updates. '
        'V1–V4 share fresh target-specific F3 fits with batch 512 and a fixed epoch-10 checkpoint; they apply a 3× internal fusion multiplier at inference. '
        'The stronger-fusion family uses a separate set of fresh fits, batch 1024 and an initial residue contribution of 0.3, '
        'with no inference fusion adjustment. Versions within a family are correlated configurations, not independent statistical replications. '
        'There is no ensemble or hubness correction.', '',
        '| Version | Base training | Inference fusion multiplier | Semantic weight | Residual cap | Detailed method and Case1 report |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    if preferred:
        lines[4:4] = [f"**User-preferred configuration: {preferred['version']}.** Recorded after reviewing the completed results. "
            f"[Detailed method and frozen checkpoints]({ROOT / 'documents' / (preferred['campaign'].name + '.md')}). "
            'This preference leaves the original evaluation protocols and artifacts unchanged.', '']
    for m in methods:
        training = ('batch 1024; scale 0.3; ' + ('epoch 10' if m['plan'].get('fixed_epoch') else 'validation-selected epoch')
                    if 'initial_residual_scale' in m['plan'] else 'batch 512; scale 0.1; epoch 10')
        lines.append(f"| {m['version']} | {training} | {m['recipe']['inference_fusion_multiplier']} | "
            f"{m['recipe']['semantic_alpha']} | {m['recipe']['residual_cap']} | "
            f"[{m['campaign'].name}]({ROOT / 'documents' / (m['campaign'].name + '.md')}) |")
    columns = ' | '.join(m['version'] for m in methods)
    separator = '| --- | ---: | ' + ' | '.join('---:' for _ in methods) + ' |'
    lines += ['', '## ReactZyme test: all-positive MRR', '',
        '| Split / direction | Strongest primary competitor | ' + columns + ' |', separator]
    for benchmark in ('ReactZyme', 'EnzymeMap'):
        if benchmark == 'EnzymeMap':
            lines += ['', '## EnzymeMap test: official screening library', '',
                'Table 1: 261,907 candidate IDs and 1,521 eligible unique test queries. '
                'Table 2: 252,113 candidate IDs after excluding training IDs, and 1,337 eligible test queries. '
                'These query counts follow deduplication and eligibility rules; the original test split contains 4,642 associations.', '',
                '| Setting / metric | Strongest primary competitor | ' + columns + ' |', separator]
        for row in methods[0]['qualification']['rows']:
            if row['benchmark'] != benchmark:
                continue
            values = []
            for m in methods:
                r = next(r for r in m['qualification']['rows'] if
                         (r['benchmark'], r['setting'], r['metric']) ==
                         (row['benchmark'], row['setting'], row['metric']))
                assert r['value'] > r['target']
                values.append(f"{r['value']:.6f}")
            target = max(row['target'], 7.81) if (benchmark, row['setting'], row['metric']) == ('EnzymeMap', 'table2', 'ef0.1') else row['target']
            assert all(float(v) > target for v in values)
            lines.append(f"| {row['setting']} / {row['metric']} | {target:.6f} | " + ' | '.join(values) + ' |')
    lines += ['', '## Case 1: predeclared Reaction-Sim deployment checkpoint', '',
        'Recovery counts below use the top 25. This target model was designated before these Case1 predictions; '
        'all other target-trained models and native pre-phase-2 scores are included in each detailed report.', '',
        '| Version | Unique paper catalysts / 12 | Unique paper + patent catalysts / 24 | Paper entries / 15 (144-entry ranking) | Paper + patent entries / 36 | Conditional AUROC |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for m in methods:
        c = m['case']['methods']['reaction_smi/selected']
        counts = [c['primary_papers']['recovered_at_25'], c['primary_papers_and_patents']['recovered_at_25'],
            c['entry_level_144']['primary_papers']['recovered_at_25'],
            c['entry_level_144']['primary_papers_and_patents']['recovered_at_25']]
        lines.append('| ' + m['version'] + ' | ' + ' | '.join(map(str, counts)) +
            f" | {c['broad_assay_conditional_discrimination']['auc']:.6f} |")
    lines += ['', 'The same frozen Reaction-Sim checkpoint at smaller screening budgets:', '',
        '| Version | Paper catalysts @5 / 12 | @10 / 12 | @25 / 12 | Primary papers represented @25 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for m in methods:
        c = m['case']['methods']['reaction_smi/selected']['primary_papers']
        paper = case1_paper_coverage(m['campaign'], m['case']['methods'])['reaction_smi/selected']
        counts = [c[f'recovered_at_{k}'] for k in (5, 10, 25)]
        lines.append('| ' + m['version'] + ' | ' + ' | '.join(map(str, counts)) +
            f" | {paper['recovered_papers_at_25']}/{paper['paper_count']} |")
    best_recovery = max(m['case']['methods']['reaction_smi/selected']['primary_papers']['recovered_at_25'] for m in methods)
    best_versions = ', '.join(m['version'] for m in methods if
        m['case']['methods']['reaction_smi/selected']['primary_papers']['recovered_at_25'] == best_recovery)
    lines += ['', 'Primary-paper coverage is descriptive, not independent replication. '
        + ('V5 recovers two catalysts in its top five, but both are related constructs from paper S16. ' if len(methods) >= 5 else '')
        + f'Highest paper-catalyst recovery at 25: {best_versions} ({best_recovery}/12). '
        'These retrospective results do not change any frozen deployment choice.']
    if len(methods) == 6:
        fixed = methods[4]['case']['methods']['reaction_smi/selected']['primary_papers']['recovered_at_25']
        later = methods[5]['case']['methods']['reaction_smi/selected']['primary_papers']['recovered_at_25']
        native = methods[5]['case']['methods']['reaction_smi/native_before_phase2']['primary_papers']['recovered_at_25']
        lines += ['', '**Longer-training transfer check.** V6 uses validation-selected checkpoints (epoch 20 for all '
            'ReactZyme splits; epoch 15 for EnzymeMap), from the same fresh trajectories as V5. '
            f'Its Reaction-Sim Case1 paper recovery falls from V5’s {fixed}/12 to {later}/12 at 25; '
            f'the native epoch-20 encoder recovers {native}/12 before phase 2. '
            'All four paper catalysts recovered by V6 have retrieved training homologs at ≥90% identity. '
            'The stronger ReactZyme test scores therefore do not translate into stronger transfer on this literature panel.']
    emap = [m['case']['methods']['enzymemap/selected'] for m in methods]
    paper_counts = [m['case']['methods']['reaction_smi/selected']['primary_papers']['recovered_at_25'] for m in methods]
    emap_counts = [m['primary_papers']['recovered_at_25'] for m in emap]
    mlp_ablation_exceeded = [m['version'] for m in methods if next(r['value'] for r in m['qualification']['rows'] if
        (r['benchmark'], r['setting'], r['metric']) == ('ReactZyme', 'reaction_smi', 'enzyme_to_reaction')) > .543]
    ablation_context = (', '.join(mlp_ablation_exceeded) + ' also exceeds 0.543 on this MRR cell; '
        'the SwissProt+DGN condition remains higher. This does not compare every ablation metric.'
        if mlp_ablation_exceeded else 'None of these versions exceeds those two separate MRR values.')
    lines += ['', f"Across the EnzymeMap-trained checkpoints, Case1 paper recovery is **{min(emap_counts)}–{max(emap_counts)}/12** "
        'at 25; conditional AUROC remains about **0.47**. '
        'This negative result is retained. Benchmark gains have not removed the dependence on the training dataset.', '',
        'For V1–V4, four recovered paper catalysts have retrieved Reaction-Sim training homologs at ≥90% identity. '
        'Their additional paper catalyst H006 moves from native rank 33 into the top 25, but its lack of a qualifying '
        'MMseqs hit does not prove absence of a homolog. Related chemistry also occurs in training. '
        'This is retrospective recovery of literature-supported catalysts for one reaction, not new wet-lab validation or proof of broad generalization.', '',
        '## Comparison boundaries', '',
        '- Primary ReactZyme targets are the strongest main-model results in '
        '[TIGER Table 1](https://arxiv.org/html/2605.24489v1#S3.T1) and the applicable published comparisons. '
        'TIGER main Reaction-Sim E→R MRR is **0.518**. Its **0.543** MLP ablation and **0.632** SwissProt+DGN condition '
        'remain separate context. ' + ablation_context,
        '- EnzymeMap targets combine the published [FGW-CLIP results](https://arxiv.org/pdf/2512.08508) '
        'and the locally reproduced released CLIPZyme checkpoint, taking the stronger value for each cell. '
        'FGW-CLIP has not been reproduced from a released checkpoint.',
        '- Table-2 EF10 uses the published CLIPZyme **7.81** in this presentation; the original frozen qualification '
        'used its exact local reproduction **7.80813533427353**. All listed versions exceed both.',
        '- Downstream training associations are matched. Frozen pretrained protein, molecular and SLEEC inputs differ '
        'from competitors, so this is not a comparison controlling every pretraining resource.',
        '- Versions within each training family share a trajectory and seed. Repeated benchmark inspection makes the findings '
        'exploratory; margins are point-estimate improvements, not demonstrated statistical superiority.',
        '- Pending studies are excluded until all benchmark tests, qualification and Case1 reporting are complete.', '',
        f"[Exact comparator registry]({cross / 'goal_primary_comparators_20260921.json'}) · "
        f"[Training and result evidence audit]({cross / 'shared_winners_evidence_audit_20260921.json'}) · "
        f"[Full chronological findings]({ROOT / 'findings.md'}).", '']
    output = ROOT / 'documents/shared_winning_methods_20260921.md'
    audit_links = [f"[{m['version']} training-family audit]({m['campaign'] / 'evidence_audit.json'})"
                   for m in methods if (m['campaign'] / 'evidence_audit.json').exists()]
    if audit_links:
        lines += ['Additional completed-family audits: ' + ' · '.join(audit_links) + '.', '']
    output.write_text('\n'.join(lines))
    with (ROOT / 'findings.md').open() as handle:
        findings = handle.read()
    start = findings.index('**Latest benchmark update,')
    end = findings.index('\n\n', start)
    opening = (f"**Latest benchmark update, {datetime.now(timezone.utc).strftime('%Y-%m-%d, %H:%M UTC')}:** {len(methods)} shared SLEEC residue-view configurations now "
        'exceed all **14 primary benchmark point estimates** on ReactZyme and EnzymeMap. Each has completed Case1 '
        'scoring on **144 entries / 123 unique sequences**, and each has its own detailed method section below. '
        f"The predeclared Reaction-Sim checkpoint recovers **{min(paper_counts)}/12 to {max(paper_counts)}/12** unique paper catalysts at 25, depending "
        f"on the version; EnzymeMap-trained checkpoints recover **{min(emap_counts)}–{max(emap_counts)}/12** and have conditional AUROC near **0.47**. "
        'The reports distinguish training families from inference-only variants. All use seed 42, with exploratory benchmark comparisons; '
        'broad catalytic generalization is not established. Pending studies are not counted as winners. '
        '[All versions, exact test metrics, and Case1 results](documents/shared_winning_methods_20260921.md).')
    if preferred:
        opening += (f" **User preference: {preferred['version']}**, recorded after the completed benchmark and Case1 results; "
            f"[method details](documents/{preferred['campaign'].name}.md).")
    with open('/tmp/enzymediscovery_findings_append.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = (ROOT / 'findings.md').read_text()
        if current != findings:
            raise RuntimeError('Concurrent findings update; rerun rather than overwrite it')
        (ROOT / 'findings.md').write_text(findings[:start] + opening + findings[end:])
    print(output)


if __name__ == '__main__':
    main()
