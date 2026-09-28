#!/usr/bin/env python3
"""Write a detailed, source-linked method report and append it to findings.md."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from generalization_screen_replication import ROOT, sha


def case1_paper_coverage(campaign, summaries):
    """Describe rank-cutoff and source coverage without selecting a model."""
    with (campaign / 'case1/unique_sequence_rankings.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    coverage = {}
    for method, summary in summaries.items():
        positives = [r for r in rows if r['method'] == method and r['primary_paper'] == 'True']
        assert len(positives) == summary['primary_papers']['positive_count']
        for cutoff in (5, 10, 25):
            assert sum(int(r['rank']) <= cutoff for r in positives) == summary['primary_papers'][f'recovered_at_{cutoff}']
        sources = {s for r in positives for s in r['primary_source_ids'].split(';') if s}
        recovered = {s for r in positives if int(r['rank']) <= 25
                     for s in r['primary_source_ids'].split(';') if s}
        coverage[method] = dict(paper_count=len(sources), recovered_papers_at_25=len(recovered))
    return coverage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args(); campaign = a.campaign.resolve()
    plan = json.loads((campaign / 'protocol.json').read_text())
    qualification = json.loads((campaign / 'qualification.json').read_text())
    freeze = json.loads((campaign / 'case1_freeze.json').read_text())
    case = json.loads((campaign / 'case1/summary.json').read_text())
    done = json.loads((campaign / 'case1/complete.json').read_text())
    if not qualification['all_primary_targets_exceeded'] or sha(campaign / 'case1/summary.json') != done['summary_sha256']:
        raise ValueError('A winning report requires all benchmark results and completed Case1 evidence')
    if sha(campaign / 'case1_freeze.json') != done['freeze_sha256']:
        raise ValueError('Case1 was scored with another frozen method')
    label = campaign.name
    recipe = plan.get('recipe', freeze['recipe'])
    if 'tasks' in plan and 'initial_residual_scale' in plan:
        stage1 = (f"Anchor-balanced all-positive decoupled InfoNCE, beta {plan['base_beta']}, equal R→E/E→R weights, "
            f"seed {plan['seed']}, batch {plan['batch_size']}, maximum {plan['epochs']} epochs, "
            f"initial fused residue scale {plan['initial_residual_scale']}. AdamW learning rate 0.0001 and weight decay 0.01. "
            'Each benchmark is independently trained from initialization. Checkpoints at epochs 5/10/15/20 are selected using '
            'the predeclared target validation criterion, with all four phase-2 fits reported. Selected epochs: ' +
            ', '.join(f"{m['name']}={m['selected_epoch']}" for m in freeze['models']) + '.')
        selection_text = ('The checkpoint grid is selected by ReactZyme mean bidirectional seen/unseen validation MRR '
            'or EnzymeMap full-library Table-1 BEDROC85. The phase-2 and semantic composition recipe is fixed. ')
        if plan.get('fixed_epoch'):
            stage1 = (f"Anchor-balanced all-positive decoupled InfoNCE, beta {plan['base_beta']}, equal R→E/E→R weights, "
                f"seed {plan['seed']}, batch {plan['batch_size']}, fixed epoch {plan['fixed_epoch']} snapshot and "
                f"initial fused residue scale {plan['initial_residual_scale']}. AdamW learning rate 0.0001 and weight decay 0.01. "
                'Snapshots come from independent fresh target fits with constant learning rate; later training cannot alter these saved weights.')
            selection_text = ('All four targets use the same fixed epoch, with no fallback within this evaluation. '
                'This epoch-10 experiment was declared after inspecting epoch-10 validation and the separate EnzymeMap epoch-15 test. '
                'It is an exploratory follow-up, not an independent prospective confirmation. '
                'It does not alter the separate predeclared 5/10/15/20-epoch validation selector. ')
    else:
        stage1 = ('Anchor-balanced all-positive decoupled InfoNCE, beta 5, equal R→E/E→R weights, seed 42, batch 512, '
            '10-epoch checkpoint, AdamW learning rate 0.0001 and weight decay 0.01. Reaction-Sim recovered its optimizer state '
            'after a memory collision; it is not described as uninterrupted training.')
        selection_text = 'This fixed candidate has no EnzymeMap validation fallback or seed selection. '
    lines = [f'### Winning configuration: {label}', '',
        f"Recorded {datetime.now(timezone.utc).isoformat()}. This configuration exceeds all **14 primary benchmark point estimates** "
        '(six ReactZyme cells and eight EnzymeMap screening cells). The comparison is exploratory after repeated benchmark inspection; '
        'it does not establish statistical superiority or control for differing pretrained resources.', '',
        '**Architecture and training.** One dual encoder retains the frozen SLEEC scorer, a global protein view and four learned residue views. '
        'The compact reaction tower uses the same model configuration across targets. Each benchmark/split has its own freshly trained F3 weights '
        'and its own training-only semantic dictionary. Frozen protein/reaction encoder and SLEEC pretraining differ from competitors; '
        'matched downstream associations do not eliminate that pretraining difference.', '',
        '**Stage 1.** ' + stage1, '',
        '**Stage 2.** A residual dual encoder fits the full target training graph for 100 positive-CE updates, temperature 0.2, '
        'identity weight 10 and learning rate 0.0001. The semantic dictionary uses training associations only. '
        f"Inference uses residual cap **{recipe['residual_cap']}**, semantic weight **{recipe['semantic_alpha']}**, "
        f"and a **{recipe['inference_fusion_multiplier']:g}×** multiplier on the internal fused residue contribution. Endpoint embeddings remain independently encodable; "
        'there is no model ensemble, candidate-pool hubness correction, or RefSeq search.', '',
        '**Protocol.** ReactZyme uses the official three target splits and all-positive MRR in both directions. EnzymeMap uses '
        '34,427/7,287/4,642 original train/dev/test associations, with the released 261,907-ID screening pool; '
        'Table 2 excludes training enzyme IDs, leaving 252,113 candidates. Validation reports BEDROC85, BEDROC20, EF5 and EF10; '
        'MRR is disabled for EnzymeMap. ' + selection_text +
        'ReactZyme reaction inputs follow its participant-set representation; EnzymeMap retains physical reactant/product sides.', '',
        '| Benchmark | Setting | Metric | Method ↑ | Comparison target | Difference |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for row in qualification['rows']:
        target = max(row['target'], 7.81) if (row['benchmark'], row['setting'], row['metric']) == ('EnzymeMap', 'table2', 'ef0.1') else row['target']
        if row['value'] <= target:
            raise ValueError('Method does not exceed the conservative published comparison target')
        lines.append(f"| {row['benchmark']} | {row['setting']} | {row['metric']} | {row['value']:.6f} | {target:.6f} | {row['value']-target:+.6f} |")
    lines += ['', 'For Table-2 EF10, the comparison above uses the published CLIPZyme value **7.81** '
        '([FGW-CLIP Table 2](https://arxiv.org/html/2512.08508v1#S5.T2)); the immutable original qualification used '
        'the locally reproduced **7.80813533427353**. This configuration exceeds both.']
    reaction_e2r = next(r['value'] for r in qualification['rows'] if
        (r['benchmark'], r['setting'], r['metric']) == ('ReactZyme', 'reaction_smi', 'enzyme_to_reaction'))
    lines += ['', '**Separate TIGER ablation context.** The primary main-model Reaction-Sim E→R target is 0.518. '
        f"This configuration's {reaction_e2r:.6f} MRR is {'above' if reaction_e2r > .543 else 'below'} the "
        'two-layer-MLP ablation’s 0.543 and below the SwissProt+DGN condition’s 0.632. '
        'These are separate published conditions; this statement concerns that MRR cell only. '
        '[TIGER Tables 1–3](https://arxiv.org/html/2605.24489v1).']
    lines += ['', '**Case 1.** Choices were frozen before scoring this version. All 144 literature-panel entries are retained, '
        'representing 123 unique sequences. The table below counts unique-sequence recovery; the companion CSV also ranks every original entry. '
        'Each target-trained checkpoint is reported independently, alongside its native encoder before phase 2.', '',
        '| Target model / score | Paper catalysts @25 / 12 | Papers + patents @25 / 24 | Workbook actives @25 / 81 | Conditional AUROC |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, summary in case['methods'].items():
        counts = [summary[k]['recovered_at_25'] for k in ('primary_papers', 'primary_papers_and_patents', 'broad_workbook_active')]
        lines.append('| ' + name + ' | ' + ' | '.join(map(str, counts)) + f" | {summary['broad_assay_conditional_discrimination']['auc']:.6f} |")
    selected_case = case['methods']['reaction_smi/selected']
    native_case = case['methods']['reaction_smi/native_before_phase2']
    before = native_case['primary_papers']['recovered_at_25']
    after = selected_case['primary_papers']['recovered_at_25']
    lines += ['', f'**Predeclared deployment result.** Reaction-Sim paper-catalyst recovery changes from '
        f'**{before}/12 before phase 2 to {after}/12 after phase 2** at 25. '
        + ('This is a negative transfer result despite the benchmark win. ' if after < before else '')
        + 'The other target-trained checkpoints are reported without substituting one based on Case1 performance.']
    coverage = case1_paper_coverage(campaign, case['methods'])
    lines += ['', '**Early literature-catalyst recovery.** All three recorded cutoffs are shown to make screening-budget tradeoffs visible. '
        'The final column counts primary papers represented by at least one recovered catalyst; it is descriptive source coverage, '
        'not a count of independent validation experiments. These results do not alter the frozen model choices.', '',
        '| Target model / score | Paper catalysts @5 / 12 | @10 / 12 | @25 / 12 | Primary papers represented @25 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, summary in case['methods'].items():
        counts = [summary['primary_papers'][f'recovered_at_{k}'] for k in (5, 10, 25)]
        paper = coverage[name]
        lines.append('| ' + name + ' | ' + ' | '.join(map(str, counts)) +
            f" | {paper['recovered_papers_at_25']}/{paper['paper_count']} |")
    lines += ['', '**Requested 144-entry panel.** The same sequence can represent several literature entries; '
        'these counts are not independent confirmations. Rankings retain all entries and use the recorded deterministic tie order.', '',
        '| Target model / score | Paper entries @25 / 15 | Papers + patents @25 / 36 | Workbook active entries @25 / 102 |',
        '| --- | ---: | ---: | ---: |']
    for name, summary in case['methods'].items():
        counts = [summary['entry_level_144'][k]['recovered_at_25'] for k in
                  ('primary_papers', 'primary_papers_and_patents', 'broad_workbook_active')]
        lines.append('| ' + name + ' | ' + ' | '.join(map(str, counts)) + ' |')
    homology = campaign / 'case1/homology_context.json'
    if homology.exists():
        context = json.loads(homology.read_text())
        lines += ['', '**Reaction-Sim training homology.** ' + context['interpretation'], '',
            '| Paper catalyst | Selected rank | Native rank | Best retrieved training identity |',
            '| --- | ---: | ---: | ---: |']
        for row in context['paper_catalysts']:
            identity = row['max_retrieved_identity']
            lines.append(f"| {row['representative_id']} | {row['selected_rank']} | {row['native_rank']} | "
                         + ('unknown (no qualifying hit)' if identity is None else f'{identity:.1%}') + ' |')
    lines += ['', 'Case 1 is one previously examined reaction, with dependent constructs and heterogeneous literature assays. '
        'The 42 non-detect-only sequences are conditional assay observations, not universally inactive enzymes. '
        'These results are retrospective evidence, not new wet-lab validation or proof of broad catalytic generalization. '
        'No model is selected by Case1 results.', '', '**Frozen artifacts.**', '']
    for model in freeze['models']:
        lines.append(f"- {model['name']}: F3 [{Path(model['checkpoint']).name}]({model['checkpoint']}); "
            f"[training config]({model['config']}); [phase-2 head]({model['phase2_checkpoint']}).")
    lines += ['', f"[Exact protocol]({campaign / 'protocol.json'}) · [Benchmark source records]({campaign / 'qualification.json'}) · "
        f"[Case1 metrics]({campaign / 'case1/summary.json'}) · [All 144 rankings]({campaign / 'case1/all_144_entry_rankings.csv'}) · "
        f"[123 unique-sequence rankings]({campaign / 'case1/unique_sequence_rankings.csv'}).", '']
    if (campaign / 'evidence_audit.json').exists():
        lines += [f"[Architecture, training-edge and artifact audit]({campaign / 'evidence_audit.json'}).", '']
    text = '\n'.join(lines)
    report = ROOT / 'documents' / (label + '.md')
    old_text = report.read_text() if report.exists() else None
    old_receipt = campaign / 'method_report.json'
    if old_text is not None and old_receipt.exists() and sha(report) != json.loads(old_receipt.read_text())['sha256']:
        raise ValueError('Existing report was edited; preserve it and reconcile the update explicitly')
    report.write_text(text)
    with Path('/tmp/enzymediscovery_findings_append.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        findings = ROOT / 'findings.md'
        marker = lines[0]
        current = findings.read_text()
        if old_text and old_text in current:
            findings.write_text(current.replace(old_text, text, 1))
        elif marker not in current:
            with findings.open('a') as handle:
                handle.write('\n' + text)
        elif old_text != text:
            raise ValueError('Existing findings section differs from its source report; update it explicitly')
    (campaign / 'method_report.json').write_text(json.dumps(dict(report=str(report), sha256=sha(report),
        findings=str(ROOT / 'findings.md'), all_primary_targets_exceeded=True, case1_completed=True), indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    main()
