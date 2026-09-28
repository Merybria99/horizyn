#!/usr/bin/env python3
"""Write a compact, source-linked status for the ongoing benchmark experiments."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/generalization_20260919_2251'
CROSS=RUN/'cross_paper_retraining'
METRICS=('bedroc85','bedroc20','ef0.05','ef0.1')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def link(path,label='source'):
    return f'[{label}]({path.relative_to(RUN)})'


def shared_recipe_status():
    lines = ['## Shared configurations passing both primary benchmarks', '',
        '[Full comparison and Case1 results](../../documents/shared_winning_methods_20260921.md). '
        'Each listed version is one encoder configuration; no ensemble is used. Versions within each training family share a trajectory and seed.', '',
        '| Configuration | Benchmark cells passed | Case1 completion | Paper catalysts @25 / 12 |',
        '| --- | ---: | --- | ---: |']
    for campaign in sorted([*CROSS.glob('shared_recipe*'), *CROSS.glob('shared_fusion03*')]):
        result = read(campaign / 'qualification.json')
        if not result:
            continue
        case = read(campaign / 'case1/summary.json')
        done = read(campaign / 'case1/complete.json')
        count = str(case['methods']['reaction_smi/selected']['primary_papers']['recovered_at_25']) if case and done else 'pending'
        lines.append(f"| {link(campaign / 'comparison.md', campaign.name)} | {result['passed_cells']}/14 | "
            f"{'144 entries / 123 unique sequences scored' if done else 'pending'} | {count} |")
    lines += ['', '## Fresh stronger-fusion training study', '',
        'Initial fused contribution 0.3; beta 5; batch 1024; fresh target-specific weights. '
        'The 5/10/15/20-epoch grid receives the fixed phase-2 recipe. EnzymeMap selection uses full-library BEDROC85.', '',
        '| Target | Base epochs logged / 20 | Trainer | Phase-2 follower | Stage | Validated epochs |',
        '| --- | ---: | --- | --- | --- | --- |']
    campaign = CROSS / 'shared_fusion03_beta5_b1024_v1'
    for target in ('reaction_smi', 'enzyme_smi', 'time', 'enzymemap'):
        base = campaign / target
        if not base.exists():
            continue
        epoch = 'starting'
        logs = list((base / 'training').glob('logs/**/metrics.csv'))
        if logs:
            with max(logs, key=lambda f:f.stat().st_mtime).open() as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                epoch = str(int(float(rows[-1]['epoch'])) + 1)
        live = []
        for name in ('trainer_execution.json', 'phase2_follower_execution.json'):
            execution = read(base / name)
            pid = execution['pid'] if execution else None
            live.append(f'live PID {pid}' if pid and Path(f'/proc/{pid}/cmdline').exists() else 'exited / not started')
        follow = base / 'phase2_followup'
        state = read(follow / 'state.json')
        epochs = sorted(int(read(f)['epoch']) for f in follow.glob('epoch*/validation_result.json'))
        lines.append(f"| {target} | {epoch} | {live[0]} | {live[1]} | "
            f"{state['stage'] if state else 'not started'} | {', '.join(map(str, epochs)) or 'pending'} |")
    result = read(campaign / 'qualification.json')
    if result:
        lines += ['', f"Stronger-fusion study: **{result['passed_cells']}/14** targets passed. "
            f"{link(campaign / 'comparison.md', 'Completed comparison')}."]
    lines += ['', 'The following sections retain earlier studies and controls, separate from the completed shared configurations above.', '']
    return lines


def matched_temperature_report():
    """Report every matched seed; never turn partial results into a final mean."""
    control=CROSS/'sleec_beta10_matched18_replication_v1'
    if not (control/'paired_config_audit.json').exists():return
    sources=[];results={}
    for beta,folder in ((5,'sleec_beta5_fresh_replication_v1'),(10,control.name)):
        results[beta]={}
        for seed in (17,73,42):
            path=CROSS/folder/f'seed{seed}/phase2/composition_v1/test_evaluation/summary.json'
            result=read(path)
            if result:
                results[beta][seed]=result
                sources.append(dict(beta=beta,seed=seed,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    complete=all(len(results[b])==3 for b in (5,10))
    pairs=sorted(set(results[5])&set(results[10]))
    deltas={str(seed):{table:{metric:results[5][seed]['summary'][table][metric]-results[10][seed]['summary'][table][metric]
        for metric in METRICS} for table in ('table1','table2')} for seed in pairs}
    means={str(beta):{table:{metric:dict(mean=statistics.mean(results[beta][s]['summary'][table][metric] for s in (17,73,42)),
        sample_sd=statistics.stdev(results[beta][s]['summary'][table][metric] for s in (17,73,42))) for metric in METRICS}
        for table in ('table1','table2')} for beta in (5,10)} if complete else None
    record=dict(updated_utc=datetime.now(timezone.utc).isoformat(),complete=complete,seeds=[17,73,42],
        completed_pairs=pairs,paired_delta_beta5_minus_beta10=deltas,three_seed_summary=means,sources=sources,
        ensembles=False,test_seed_selection=False,exploratory_repeated_benchmark_evaluation=True)
    lines=['# Matched inverse-temperature comparison','',
        'Fresh SLEEC residue-view models use the same training associations, seeds, fixed 18 epochs, optimizer and phase 2. '
        'The only training-math change is inverse temperature beta 5 versus beta 10. '
        '[Configuration audit](paired_config_audit.json) · [Protocol](protocol.json).','',
        'All three seed pairs are complete.' if complete else f'{len(pairs)} of 3 seed pairs are complete; the final three-seed comparison is pending.','',
        'Validation reports BEDROC85, BEDROC20, EF5 and EF10. This replication uses a fixed recipe, with no MRR selection, seed selection or ensemble.','']
    for table in ('table1','table2'):
        lines += [f'## {table}','','| Seed / beta | BEDROC85 ↑ | BEDROC20 ↑ | EF5 ↑ | EF10 ↑ |','| --- | ---: | ---: | ---: | ---: |']
        for seed in (17,73,42):
            for beta in (5,10):
                result=results[beta].get(seed)
                cells=[f"{result['summary'][table][m]:.6f}" for m in METRICS] if result else ['pending']*4
                lines.append(f'| {seed} / {beta} | '+' | '.join(cells)+' |')
            if seed in pairs:
                lines.append(f'| {seed} / beta5 − beta10 | '+' | '.join(f'{deltas[str(seed)][table][m]:+.6f}' for m in METRICS)+' |')
        if complete:
            for beta in (5,10):
                lines.append(f'| Beta {beta} mean ± sample SD | '+' | '.join(f"{means[str(beta)][table][m]['mean']:.6f} ± {means[str(beta)][table][m]['sample_sd']:.6f}" for m in METRICS)+' |')
        lines.append('')
    lines += ['The repeated benchmark evaluations are exploratory. Three seeds do not establish statistical superiority; every seed and negative difference is retained. '
        'Frozen encoder/SLEEC pretraining still differs from the published competitors. '
        '[Exact source files and hashes](paired_summary.json).','']
    for name,payload in [('paired_summary.json',json.dumps(record,indent=2)+'\n'),('comparison.md','\n'.join(lines))]:
        path=control/name;temporary=path.with_name(path.name+'.tmp');temporary.write_text(payload);temporary.replace(path)


def report():
    matched_temperature_report()
    now=datetime.now(timezone.utc).isoformat()
    lines=['# Current benchmark experiments','',f'Updated {now}','',
        'EnzymeMap reports full-library BEDROC85, BEDROC20, EF5 and EF10. Primary studies select by validation BEDROC85; the separately recorded joint selector uses all four metrics. No EnzymeMap MRR selection.',
        'ReactZyme uses all-positive MRR in both directions. Frozen-backbone/SLEEC pretraining differs from the published competitors even where downstream training associations match. Repeated tests are exploratory.','',
        *shared_recipe_status(),
        '## EnzymeMap: softer base contrastive logits, beta 5','',
        'The recovered seed-42 beta-5 base was selected at epoch 18 by full-library validation BEDROC85. It receives the same fixed 100-update phase-2 recipe and alpha 0.25 as the other geometry arms. Its test exceeds both comparators in all eight cells. Three uninterrupted fresh-seed replications are reported below as they finish; no seed is selected or ensembled.','']
    clip=read(CROSS/'clipzyme_released_screen_evaluation_v1/summary.json')
    fgw={'table1':[.4866,.6669,14.91,8.18],'table2':[.4514,.6143,13.57,7.61]}
    beta5_paths={'Recovered seed 42':CROSS/'sleec_multiview_temperature5_fixed_phase2_v1/composition_v1/test_evaluation/summary.json',
        **{f'Fresh seed {seed}':CROSS/f'sleec_beta5_fresh_replication_v1/seed{seed}/phase2/composition_v1/test_evaluation/summary.json' for seed in (17,73,42)}}
    beta5_results={name:read(path) for name,path in beta5_paths.items()}
    for table in ('table1','table2'):
        lines += [f'### Beta 5, {table}','','| Model | BEDROC85 ↑ | BEDROC20 ↑ | EF5 ↑ | EF10 ↑ |','| --- | ---: | ---: | ---: | ---: |']
        for name,values in [('Released CLIPZyme',[clip['summary'][table][m] for m in METRICS]),('Published FGW-CLIP',fgw[table])]:
            lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
        for name,result in beta5_results.items():
            if result:lines.append('| '+link(beta5_paths[name],name)+' | '+' | '.join(f"{result['summary'][table][m]:.6f}" for m in METRICS)+' |')
        fresh=[result for name,result in beta5_results.items() if name.startswith('Fresh')]
        if len(fresh)==3 and all(fresh):
            cells=[]
            for metric in METRICS:
                values=[r['summary'][table][metric] for r in fresh]
                cells.append(f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}')
            lines.append('| Fresh-seed mean ± sample SD | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['[Fixed geometry-arm protocol](cross_paper_retraining/sleec_geometry_fixed_phase2_v1/protocol.json) · [Fresh beta-5 replication protocol](cross_paper_retraining/sleec_beta5_fresh_replication_v1/protocol.json). Repeated benchmark inspection makes these exploratory results.','',
        '### Fresh beta-5 replication progress','',
        '| Seed | Current stage | Base validation BEDROC85 | BEDROC20 | EF5 | EF10 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for seed in (17,73,42):
        p=CROSS/f'sleec_beta5_fresh_replication_v1/seed{seed}'
        state=read(p/'state.json');validation=read(p/'screen_epoch17/validation_evaluation/summary.json')
        values=[f"{validation['summary']['table1'][m]:.6f}" for m in METRICS] if validation else ['pending']*4
        lines.append('| '+str(seed)+' | '+(state['stage'] if state else 'queued')+' | '+' | '.join(values)+' |')
    lines += ['', '[Matched beta-5 versus beta-10 comparison at 18 epochs](cross_paper_retraining/sleec_beta10_matched18_replication_v1/comparison.md). This separate control isolates inverse temperature from training length.','',
        '## EnzymeMap: earlier beta-10 recipe, alpha 0.25','',
        'Seed 42 exceeds published FGW-CLIP and released CLIPZyme on all eight point estimates. The fresh-seed results show substantial variation; the all-cell FGW win is not stable across seeds.','']
    paths={42:CROSS/'sleec_multiview_phase2_v1/composition_all_metrics_v1/test_evaluation/summary.json',
        **{seed:CROSS/f'sleec_replication_seed{seed}_composition_v1/composition_joint_recipe_v1/test_evaluation/summary.json' for seed in (17,73)}}
    results={seed:read(path) for seed,path in paths.items()}
    for table in ('table1','table2'):
        lines += [f'### {table}','','| Model | BEDROC85 | BEDROC20 | EF5 | EF10 |','| --- | ---: | ---: | ---: | ---: |']
        for name,values in [('Released CLIPZyme',[clip['summary'][table][m] for m in METRICS]),('Published FGW-CLIP',fgw[table])]:
            lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
        for seed,result in results.items():
            if result:lines.append('| '+link(paths[seed],f'F3 + phase 2, seed {seed}')+' | '+' | '.join(f"{result['summary'][table][m]:.6f}" for m in METRICS)+' |')
        if all(results.values()):
            cells=[]
            for metric in METRICS:
                values=[r['summary'][table][metric] for r in results.values()]
                cells.append(f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}')
            lines.append('| Three-seed mean ± sample SD | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['The means average independently evaluated models; no ensemble predictions are used. The new seeds use exact storage-precision input transport and four loader workers. The same-seed replay matched early losses but diverged later; input/export parity does not establish an identical full-run random trajectory.','',
        '## ReactZyme: retained phase-2 reference','',
        '| Split | R→E all-positive MRR | E→R all-positive MRR |','| --- | ---: | ---: |']
    for name,r,e in [('Reaction-Sim',.415174186230,.523980677128),('Enzyme-Sim',.697468400002,.973675847054),('Time',.599849462509,.835166871548)]:
        lines.append(f'| {name} | {r:.6f} | {e:.6f} |')
    lines += ['',"These exceed TIGER's main comparison table, but Reaction-Sim E→R is below its 0.543 two-layer-MLP ablation. Fresh-head, warm-start soft-CE and linear-alignment follow-ups have not established an improvement in both directions.",'',
        '## Active base training','',
        '| Run | Latest logged epoch | Step | Completion |','| --- | ---: | ---: | --- |']
    training=[]
    for p in (CROSS/'reactzyme_architecture_extension_v1').glob('*/training'):training.append((p.parent.name+' / Reaction-Sim continuation',p,p.parent/'complete.json'))
    for p in (CROSS/'reactzyme_architecture_extension30_v1').glob('*/training'):training.append((p.parent.name+' / Reaction-Sim continuation to 30',p,p.parent/'complete.json'))
    for p in (CROSS/'reactzyme_multiview_all_splits_v1').glob('*/training'):training.append(('Residue views / '+p.parent.name,p,p.parent/'complete.json'))
    for p in (CROSS/'reactzyme_beta5_all_splits_v1').glob('*/training'):training.append(('Residue views beta5 / '+p.parent.name,p,p.parent/'complete.json'))
    for p in (CROSS/'reactzyme_multiview_large_batch_v1').glob('*/training'):training.append(('Residue views / Reaction-Sim batch2048',p,p.parent/'complete.json'))
    for p in (CROSS/'reactzyme_multiview_beta5_v1').glob('*/training'):training.append(('Residue views / Reaction-Sim beta5',p,p.parent/'complete.json'))
    for seed in (17,73,42):
        p=CROSS/f'sleec_beta5_fresh_replication_v1/seed{seed}/training'
        if p.exists():training.append((f'EnzymeMap beta5 / seed{seed}',p,p.parent/'train_complete.json'))
        p=CROSS/f'sleec_beta10_matched18_replication_v1/seed{seed}/training'
        if p.exists():training.append((f'EnzymeMap matched beta10 / seed{seed}',p,p.parent/'train_complete.json'))
    p=CROSS/'sleec_multiview_seed42_io_replay_v1/seed42/training'
    if p.exists():training.append(('EnzymeMap seed42 loader replay',p,p.parent/'train_complete.json'))
    for name,path,done in training:
        if (path.parent/'recovery_training').exists():
            path=path.parent/'recovery_training'
            name+=' (recovered)'
        logs=list(path.glob('logs/**/metrics.csv'));epoch=step='starting'
        if logs:
            with max(logs,key=lambda f:f.stat().st_mtime).open() as f:rows=list(csv.DictReader(f))
            if rows:epoch=str(int(float(rows[-1]['epoch']))+1);step=rows[-1]['step']
        state=read(done)
        completion='training / evaluating'
        if state is not None:completion='finished' if state.get('exit_code',0)==0 else 'failed'
        else:
            execution=read(path.parent/'execution.json')
            if execution and not Path(f"/proc/{execution['pid']}").exists():
                log=path.parent/'training.log'
                tail=log.read_text()[-4096:] if log.exists() else ''
                completion='finished (trainer log)' if '`max_epochs=' in tail and 'reached' in tail else 'process exited; inspect log'
        lines.append(f'| {name} | {epoch} | {step} | {completion} |')
    lines += ['', '## New ReactZyme phase-2 campaigns','','| Campaign | State | Selected test |','| --- | --- | --- |']
    for campaign in sorted(CROSS.glob('reactzyme_*phase2*')):
        protocol=read(campaign/'protocol.json')
        if not protocol or 'tasks' not in protocol:continue
        complete=read(campaign/'complete.json');test=read(campaign/'selected_test_summary.json')
        states=[read(campaign/task['arm']/'state.json') for task in protocol['tasks']]
        state='; '.join(sorted({s['stage'] if s else 'waiting for checkpoint' for s in states}))
        if complete:state='original parent retained' if complete['parent_retained'] else 'new model selected'
        result='pending' if not complete else 'not run; parent retained'
        if test:result=' / '.join(f"{test['summary'][d]['all']['reactzyme_mrr']:.6f}" for d in ('reaction_to_enzyme','enzyme_to_reaction'))
        lines.append(f'| {link(campaign/"protocol.json",campaign.name)} | {state} | {result} |')
    lines += ['', '[Fixed EnzymeMap-recipe transfer to all ReactZyme splits](cross_paper_retraining/reactzyme_fixed_enzymemap_recipe_transfer_v1/comparison.md): all declared beta-5/10 and epoch-10/20 tests are reported, including validation failures.','',
        '## Important negative controls','',
        'The early temperature-20 base had higher validation BEDROC85 than the residue-view base. Its selected phase-2 composition reached validation BEDROC85 0.560179, but test Table 1 was only 0.474812 / 0.633992 / 14.095749 / 7.850042. This is a concrete validation/test rank reversal, not evidence that the strongest validation run generalizes best.',
        '', 'All studies and recovery events are recorded in [findings.md](../../findings.md). '
        'The [screening comparison](cross_paper_retraining/sleec_multiview_phase2_v1/comparison.md) retains the primary BEDROC85-selected model and uncertainty. '
        'The [joint selector protocol](cross_paper_retraining/sleec_multiview_phase2_v1/all_metrics_selector_protocol.json) identifies the secondary exploratory analysis.','']
    path=RUN/'benchmark_status.md';temporary=path.with_suffix('.tmp.md');temporary.write_text('\n'.join(lines));temporary.replace(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hours',type=float,default=0);a=p.parse_args();start=time.monotonic()
    while True:
        report()
        if time.monotonic()-start>=a.hours*3600:break
        time.sleep(30)
