#!/usr/bin/env python3
"""EC1--4 enzyme-neighborhood diagnostics from fixed existing embeddings.

From EnzymeDiscovery:
 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .capability-run-py/bin/python horizyn/scripts/cersei_enzyme_ec_levels.py compute
 OPENBLAS_NUM_THREADS=8 python3 horizyn/scripts/cersei_enzyme_ec_levels.py report
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
from cersei_neighborhood_curves import (ROOT, OLD, SOURCE, SPLITS, LABELS, NAMES,
    COLORS, STYLES, KS, read, dump, digest, models, select, write_csv, table)

OUT = SOURCE / 'ec_levels'


def compute():
    import torch
    from cersei_embedding_metrics import nearest, labels_matrix
    from cersei_embedding_organization import prefix
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    families = read(OLD / 'homology/families.json')
    paths = {OLD / 'homology/families.json', Path(__file__),
             ROOT / 'scripts/cersei_neighborhood_curves.py', OUT / 'chart_contract.json'}
    curves, scope, per_class, audits = [], [], [], []
    for split in SPLITS:
        meta = read(OLD / split / 'metadata.json')
        sub = read(OLD / split / 'subsets.json')
        bank = np.asarray(meta['neighborhood_indices'])
        lookup = {int(i): j for j, i in enumerate(bank)}
        sampled = np.asarray([lookup[i] for i in sub['protein_indices']], dtype=int)
        raw_labels = [meta['enzyme_ec'][i] for i in bank]
        complete = [prefix(x, 4) for x in raw_labels]
        family50 = np.asarray([families[meta['protein_ids'][i]][1] for i in bank])
        existing = read(SOURCE / split / 'neighborhoods_summary.json')
        paths.add(SOURCE / split / 'neighborhoods_summary.json')
        for name in ['metadata.json', 'subsets.json', 'embeddings.npz']:
            paths.add(OLD / split / name)
        embeddings = {}
        for model in models(split):
            path = SOURCE / split / f'{model}_embeddings.npz'; paths.add(path)
            with np.load(path) as z:
                embeddings[model] = z['enzyme']
                if 'bank' in z: assert np.array_equal(z['bank'], bank)
        with np.load(OLD / split / 'embeddings.npz') as z:
            assert np.array_equal(z['prott5_indices'], bank)
            embeddings['prott5'] = z['prott5']
        for z in embeddings.values():
            assert len(z) == len(bank) and np.isfinite(z).all()
        ec4_neighbors = {}
        common_previous = {}
        for mode in ['level_specific', 'complete_ec4']:
            for level in [1, 2, 3, 4]:
                labs = [prefix(x, level) for x in (raw_labels if mode == 'level_specific' else complete)]
                valid = np.asarray([bool(x) for x in labs])
                q = sampled[valid[sampled]]
                if mode == 'complete_ec4':
                    assert np.array_equal(valid, np.asarray([bool(x) for x in complete]))
                membership, classes = labels_matrix([labs[i] for i in q])
                counts = np.asarray(membership.sum(0)).ravel()
                attrs = dict(split=split, cohort=mode, ec_level=level, starting_bank=len(bank),
                             eligible_bank=int(valid.sum()), sampled_queries=len(sampled), queries=len(q), classes=len(classes))
                scope.append(attrs)
                sets = [set(x) for x in labs]
                for model, z in embeddings.items():
                    if mode == 'level_specific':
                        nn = nearest(z, valid, family50, 'cpu', k=50, queries=q)[q]
                        if level == 4: ec4_neighbors[model] = (q.copy(), nn.copy())
                    else:
                        saved_q, nn = ec4_neighbors[model]
                        assert np.array_equal(q, saved_q)
                    assert np.all(nn >= 0) and np.all(nn != q[:, None])
                    assert np.all(valid[nn]) and np.all(family50[q, None] != family50[nn])
                    matches = np.asarray([[bool(sets[i] & sets[j]) for j in row] for i, row in zip(q, nn)])
                    values = np.cumsum(matches, axis=1) / KS
                    means = (membership.T @ values) / counts[:, None]
                    macro, micro = means.mean(0), values.mean(0)
                    assert np.isfinite(macro).all() and np.all((macro >= 0) & (macro <= 1 + 1e-12))
                    if mode == 'complete_ec4':
                        if level > 1: assert np.all(values <= common_previous[model] + 1e-12)
                        common_previous[model] = values
                    for j, k in enumerate(KS):
                        curves.append(dict(**attrs, model=model, k=int(k), macro=float(macro[j]), micro=float(micro[j])))
                    for j in [0, 9, 49]:
                        for c, n, v in zip(classes, counts, means[:, j]):
                            per_class.append(dict(split=split, cohort=mode, ec_level=level, model=model,
                                                  k=int(KS[j]), label=c, queries=int(n), agreement=float(v)))
                    np.savez_compressed(OUT / f'{split}_{mode}_ec{level}_{model}.npz', query_indices=q,
                        neighbors=nn, agreement=values, class_names=np.asarray(classes), class_means=means, class_counts=counts)
                    if mode == 'level_specific':
                        for k in [10, 50]:
                            old = select(existing, model=model, ec_level=level, k=k,
                                         exclude_homologs=True, exclude_training_overlap=False)
                            error = abs(old['macro'] - macro[k - 1])
                            assert error < 1e-5, (split, level, model, k, error)
                            assert (old['evaluated_queries'], old['classes'], old['eligible_bank']) == (len(q), len(classes), int(valid.sum()))
                            audits.append(dict(split=split, ec_level=level, model=model, k=k, absolute_error=float(error)))
                        if level == 3:
                            old_file = SOURCE / 'neighborhood_curves' / f'{split}_enzyme_{model}.npz'; paths.add(old_file)
                            with np.load(old_file) as old:
                                assert np.array_equal(q, old['query_indices'])
                                assert np.array_equal(values, old['agreement'])
                    else:
                        if level == 4:
                            with np.load(OUT / f'{split}_level_specific_ec4_{model}.npz') as old:
                                assert np.array_equal(values, old['agreement'])
                    print(split, mode, f'EC{level}', model, round(100 * macro[9], 4), flush=True)
    write_csv(OUT / 'curves.csv', curves)
    write_csv(OUT / 'per_class.csv', per_class)
    dump(OUT / 'scope.json', scope)
    clean_rows = []
    for split in SPLITS:
        path = SOURCE / split / 'clean_neighborhoods_summary.json'; paths.add(path)
        for r in read(path):
            if r.get('model') in ['cersei', 'clean', 'esm1b'] and r.get('ec_level') in [1, 2, 3, 4] and r.get('k') in [10, 50] and r.get('exclude_homologs') and r.get('exclude_training_overlap'):
                clean_rows.append(dict(split=split, model=r['model'], ec_level=r['ec_level'], k=r['k'],
                    macro=r['macro'], queries=r['evaluated_queries'], classes=r['classes'], eligible_bank=r['eligible_bank']))
    assert len(clean_rows) == 96
    write_csv(OUT / 'clean_transfer.csv', clean_rows)
    dump(OUT / 'sources.json', {str(p): digest(p) for p in sorted(paths)})
    dump(OUT / 'validation.json', dict(passed=True, device='cpu', curve_rows=len(curves), saved_summary_checks=audits,
        common_cohort_pointwise_hierarchy_nested=True, common_cohort_neighbors_fixed_across_levels=True,
        all_ec3_per_query_curves_reproduce=True, query_self_and_close_component_exclusions_verified=True,
        fixed_query_denominators_across_k=True, clean_transfer_rows=len(clean_rows)))


def report():
    sys.path.insert(0, str(ROOT / '.deps/ablation-figures'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    with (OUT / 'curves.csv').open() as f: rows = list(csv.DictReader(f))
    def curve(split, level, model, cohort='level_specific', metric='macro'):
        rr = [r for r in rows if (r['split'], r['ec_level'], r['model'], r['cohort']) == (split, str(level), model, cohort)]
        rr.sort(key=lambda r: int(r['k']))
        assert [int(r['k']) for r in rr] == KS.tolist()
        return np.array([float(r[metric]) for r in rr]) * 100
    plt.style.use(['science', 'no-latex'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'xtick.top': False, 'ytick.right': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    def legend(fig):
        from matplotlib.lines import Line2D
        mm = ['cersei', 'horizyn', 'creep', 'clipzyme', 'prott5']
        hh = [Line2D([], [], color=COLORS[m], marker=STYLES[m][1], linestyle=STYLES[m][0],
                      mfc='white', markersize=4, linewidth=1.3, label=NAMES[m]) for m in mm]
        fig.legend(handles=hh, ncol=5, loc='upper center', frameon=False, bbox_to_anchor=(.52, 1.005), handlelength=2.5)
    def save(fig, name):
        for ext in ['pdf', 'svg', 'png']: fig.savefig(OUT / f'{name}.{ext}', dpi=250, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    def dots(name, cohort, metric, ylabel):
        fig, axes = plt.subplots(1, 4, figsize=(11, 3.05), sharey=True)
        for ax, split in zip(axes, SPLITS):
            for off, model in zip([-.225, -.075, .075, .225], models(split) + ['prott5']):
                yy = [curve(split, l, model, cohort, metric)[9] for l in [1, 2, 3, 4]]
                ax.plot(np.arange(1, 5) + off, yy, linestyle='none', marker=STYLES[model][1],
                        color=COLORS[model], mfc='white', ms=4.8, mew=1.15)
            ax.set_xticks([1, 2, 3, 4], ['EC1', 'EC2', 'EC3', 'EC4'])
            ax.set_xlim(.55, 4.45); ax.set_ylim(0, 102); ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.set_xlabel(LABELS[split]); ax.grid(axis='y', alpha=.15, lw=.5); ax.minorticks_off()
        axes[0].set_ylabel(ylabel); legend(fig)
        fig.tight_layout(rect=(0, 0, 1, .91), w_pad=1.2); save(fig, name)
    dots('07_enzyme_ec1_ec4_agreement', 'level_specific', 'macro', 'Macro agreement@10 (%)')
    dots('09_enzyme_ec_complete_cohort', 'complete_ec4', 'micro', 'Query-mean agreement@10 (%)')
    fig, axes = plt.subplots(4, 4, figsize=(11.5, 9.8), sharey=True)
    for i, split in enumerate(SPLITS):
        for j, level in enumerate([1, 2, 3, 4]):
            ax = axes[i, j]
            for model in models(split) + ['prott5']:
                ls, mk = STYLES[model]
                ax.plot(KS, curve(split, level, model), color=COLORS[model], ls=ls, marker=mk,
                        markevery=[0, 9, 24, 49], ms=2.7, mfc='white', mew=.7, lw=1.15)
            ax.set_xlim(.5, 50.5); ax.set_xticks([1, 10, 25, 50]); ax.set_ylim(0, 102)
            ax.set_yticks([0, 25, 50, 75, 100]); ax.minorticks_off(); ax.grid(axis='y', alpha=.15, lw=.5)
            ax.set_xlabel(f'EC{level} — neighbors, $k$')
            if j == 0: ax.set_ylabel(LABELS[split] + '\nMacro agreement (%)')
    legend(fig); fig.tight_layout(rect=(0, 0, 1, .965), h_pad=1.7, w_pad=1.3)
    save(fig, '08_enzyme_ec_neighborhood_curves')

    main, gaps, common, persistence = [], [], [], []
    for split in SPLITS:
        for level in [1, 2, 3, 4]:
            mm = models(split)
            vals = {m: curve(split, level, m)[9] for m in mm + ['prott5']}
            main.append([LABELS[split], f'EC{level}', f"{vals['cersei']:.2f}", f"{vals['horizyn']:.2f}",
                         f"{vals['creep']:.2f}" if 'creep' in vals else '—',
                         f"{vals['clipzyme']:.2f}" if 'clipzyme' in vals else '—', f"{vals['prott5']:.2f}"])
            best = max(mm[1:], key=lambda m: vals[m]); d = vals['cersei'] - vals[best]
            gaps.append([LABELS[split], f'EC{level}', NAMES[best], f'{d:+.2f}'])
            cv = {m: curve(split, level, m, 'complete_ec4', 'micro')[9] for m in mm + ['prott5']}
            common.append([LABELS[split], f'EC{level}', f"{cv['cersei']:.2f}", f"{cv['horizyn']:.2f}",
                           f"{cv['creep']:.2f}" if 'creep' in cv else '—',
                           f"{cv['clipzyme']:.2f}" if 'clipzyme' in cv else '—', f"{cv['prott5']:.2f}"])
            for m in mm[1:]:
                diff = curve(split, level, 'cersei') - curve(split, level, m)
                persistence.append([LABELS[split], f'EC{level}', 'CERSEI − ' + NAMES[m],
                    f'{diff[9]:+.2f}', f'{diff[49]:+.2f}', f'{int((diff>1e-9).sum())}/50', f'{diff.min():+.2f} to {diff.max():+.2f}'])
    scope = read(OUT / 'scope.json')
    scope_table = [[LABELS[s['split']], f"EC{s['ec_level']}", s['starting_bank'], s['eligible_bank'],
                    s['queries'], s['classes']] for s in scope if s['cohort'] == 'level_specific']
    with (OUT / 'clean_transfer.csv').open() as f: clean = list(csv.DictReader(f))
    clean_table = []
    for split in SPLITS:
        for level in [1, 2, 3, 4]:
            vals = [select(clean, split=split, ec_level=str(level), k='10', model=m) for m in ['cersei', 'clean', 'esm1b']]
            assert len({(v['queries'], v['eligible_bank'], v['classes']) for v in vals}) == 1
            clean_table.append([LABELS[split], f'EC{level}', vals[0]['queries'], vals[0]['eligible_bank']] +
                               [f"{100*float(v['macro']):.2f}" for v in vals])
    headers = ['Partition', 'Level', 'CERSEI (%)', 'Horizyn (%)', 'CREEP (%)', 'CLIPZyme (%)', 'ProtT5 (%)']
    paragraphs = [
        '# Enzyme neighborhoods at every EC level',
        'Completed 24 September 2026 using the frozen embeddings already evaluated in the public-checkpoint analysis. This extends the EC3-only curves to EC1, EC2, EC3 and EC4; no training, checkpoint selection or reaction-space analysis is added.',
        '**The enzyme-space advantage depends on functional resolution.** CERSEI generally improves broad and intermediate EC neighborhoods, but its advantage over Horizyn is much smaller at EC4. On the time split, Horizyn has higher EC4 agreement@10; on the enzyme split and EnzymeMap, the EC4 differences are only +0.03 and +0.02 percentage points for CERSEI. These are descriptive near-ties, not evidence of a meaningful win.',
        '## What the four levels ask',
        table(['Level', 'Label example', 'Question'], [
            ['EC1', '1', 'Do nearby proteins share a broad enzyme class?'],
            ['EC2', '1.1', 'Does proximity preserve the finer subclass?'],
            ['EC3', '1.1.1', 'Does proximity preserve the sub-subclass used in the earlier analysis?'],
            ['EC4', '1.1.1.1', 'Do nearby proteins share the complete annotated EC number?']]),
        'These examples show truncation of an annotation, not selected classes. All represented labels are included. EC4 is a finer functional identifier, but matching it still does not verify activity on the exact target substrate, reaction conditions or literature case. The experiment evaluates local same-modality neighborhoods, not a fitted EC classifier or global clustering algorithm.',
        '## Main result at k=10',
        '![EC1–EC4 agreement](07_enzyme_ec1_ec4_agreement.png)',
        '**Figure 7.** Class-balanced enzyme-neighborhood agreement@10 at EC1–EC4. Panels show Time, Enzyme-Similarity, Reaction-Similarity and EnzymeMap. All models within a panel and EC level share queries, candidate IDs and the detected 50%-identity component exclusion. Dots have no uncertainty intervals. CREEP is shown on ReactZyme and CLIPZyme on EnzymeMap; ProtT5 is a frozen-input control. The vertical scale is identical across panels.',
        table(headers, main),
        table(['Partition', 'Level', 'Higher-scoring trained comparator at k=10', 'CERSEI difference (pp)'], gaps),
        'The comparator in the difference table is whichever of the two available trained alternatives scores higher in that cell. It is a descriptive comparison, not a new model-selection rule. The frozen ProtT5 control is excluded from that choice.',
        '## Complete neighborhood-size curves',
        '![Curves at every EC level](08_enzyme_ec_neighborhood_curves.png)',
        '**Figure 8.** Macro agreement for k=1–50. Rows are Time, Enzyme-Similarity, Reaction-Similarity and EnzymeMap; columns are EC1–EC4. Identical axes make the change in scale visible. Curves use the level-specific eligible banks described below. Small gaps, especially at EC4, should be read with the exact table rather than inferred from line overlap.',
        table(['Partition', 'Level', 'Difference', 'Δ@10 (pp)', 'Δ@50 (pp)', 'CERSEI higher at k', 'Δ range (pp)'], persistence),
        'The nested k values are correlated. Higher at 50/50 k values means persistence of the observed ordering, not 50 independent confirmations. A drop across EC levels cannot be attributed solely to functional specificity: the class weights, annotation coverage and number of available same-label neighbors also change.',
        '## Protocol and annotation coverage',
        'The [full protocol](../neighborhood_curves/README.md#protocol-from-a-fixed-checkpoint-to-a-plotted-point) defines the checkpoint provenance, original query samples, cosine ranking, sequence-component exclusion and macro averaging. The present extension changes the label prefix length only in its primary analysis. For each level, a protein is eligible if at least one annotation has that many complete components; a prefix containing a missing `-` is excluded. A query and neighbor agree if their retained prefix sets intersect. Multi-label queries retain the any-shared-label criterion and contribute to every query-class average to which they belong.',
        'Primary query samples are the same fixed ID-hash samples as before: at most 2,000 proteins per partition. Candidate banks remain the same starting pools. Eligibility is applied separately at each EC level, so EC4 can have fewer queries and candidates than EC1. The table gives candidate counts before query-self and same-sequence-component exclusions. Every retained query has at least 50 candidates afterward. Within each level, the same queries and exclusion masks apply to every model and to all k.',
        table(['Partition', 'Level', 'Starting bank', 'Annotated candidates', 'Evaluated queries', 'Represented labels'], scope_table),
        'EnzymeMap has 720 annotated proteins at all four levels within its 1,357-sequence test-positive bank. It is not a geometry analysis of the full 261,907-accession screening library. On ReactZyme, the smaller EC4 cohorts make an additional control useful. Excluding close query–candidate sequence components is not an additional audit of overlap with training sequences.',
        '## Control: the same completely annotated proteins at all four levels',
        'To separate annotation coverage from hierarchy depth, we restrict queries and candidates to proteins carrying at least one complete EC4 label. We derive EC1–EC3 prefixes **only from their complete EC4 annotations**, discarding other incomplete annotations for this control. Thus the protein IDs, neighbor rankings and exclusions are exactly the same at every level. We recompute both macro and ordinary query-mean agreement; the latter is plotted because it also keeps query weights fixed across levels.',
        '![Complete-annotation cohort control](09_enzyme_ec_complete_cohort.png)',
        '**Figure 9.** Ordinary query-mean agreement@10 on the common EC4-annotated cohorts. The evaluated query counts are 1,746, 1,891, 1,609 and 720 for Time, Enzyme-Similarity, Reaction-Similarity and EnzymeMap. Unlike Figure 7, each query has equal weight. This is an explicit change of estimand for the hierarchy-depth control, not a replacement for the primary class-balanced result. All candidate-bank counts equal the corresponding EC4 row above.',
        table(headers, common),
        'The common-cohort control confirms that the finer-resolution limitation is not removed simply by fixing annotation coverage. At EC4, ordinary query-mean agreement favors Horizyn on Time (49.01% versus 47.14% CERSEI) and slightly on the enzyme split (60.94% versus 60.54%). CERSEI is higher on the reaction split (85.10% versus 83.70% Horizyn) and EnzymeMap (22.39% versus 22.15%). These percentages use query averaging and must not replace the macro results above when making the primary comparison.',
        'Weighting also matters at intermediate resolution: on Time, CERSEI leads the common-cohort EC2 query mean, whereas its common-cohort EC2 macro mean is below both trained alternatives (57.90% CERSEI, 58.65% Horizyn, 58.59% CREEP; all values are in curves.csv). Giving each function equal weight can therefore change the ordering compared with giving each protein equal weight. Both estimands are retained rather than selecting the favorable one.',
        'For this control, agreement for each individual query is guaranteed to stay the same or decrease as the EC prefix becomes more specific: matching a complete EC number implies matching its prefixes. The implementation verifies this nesting for every query and k. The primary macro curves need not obey that property because their populations and class weights differ. The control is conditional on complete annotations and can favor common functions; it is not an unbiased estimate for unannotated proteins.',
        '## Interpretation by level',
        'At **EC1**, CERSEI has the highest observed macro agreement@10 among the displayed trained methods in all four partitions. This supports organization by broad enzyme function. It does not establish fine substrate specificity.',
        'At **EC2**, CERSEI leads on the enzyme split, reaction split and EnzymeMap. CREEP leads on Time: 58.35% versus 57.84%. Broad-class success therefore does not imply an advantage at every intermediate hierarchy level.',
        'At **EC3**, the earlier findings are reproduced exactly: the strongest gains are on the enzyme and reaction splits; Time is close; EnzymeMap shows a smaller advantage.',
        'At **EC4**, Time favors Horizyn (27.94% versus 27.11% CERSEI). The enzyme split is essentially tied with Horizyn (38.07% versus 38.04%), as is EnzymeMap (8.99% versus 8.97%). CERSEI retains a larger observed advantage on the reaction split (34.50% versus 33.27% Horizyn and 32.53% CREEP). Thus the results support stronger coarse-to-intermediate organization more consistently than a universal advantage at complete EC resolution.',
        'EC4 has many more represented labels and fewer same-label candidates, so agreement at a fixed k has a lower attainable ceiling for many queries. These percentages are not chance-adjusted or ceiling-normalized. The per-class exports allow inspection of support sizes; no label-permutation test, prospective wet-lab experiment or new training-seed replication was performed here.',
        '## CLEAN transfer control at all levels',
        'The earlier CLEAN transfer analysis already recorded EC1–EC4. The table below extracts its matched-bank cosine agreement@10 after removing exact CLEAN training-sequence matches from both query and candidate pools, and then excluding detected close query–candidate sequence components. All three models in each row use the same smaller pool. These values are not comparable directly with Figure 7 because the pools differ. CLEAN has different supervised training data and its own ESM-1b input; this remains a separate transfer diagnostic, not an equally trained architecture benchmark. The prior native unnormalized-L2 sensitivity was EC3-only and is not relabeled as an all-level analysis.',
        table(['Partition', 'Level', 'Queries', 'Annotated bank', 'CERSEI (%)', 'CLEAN cosine (%)', 'ESM-1b (%)'], clean_table),
        'EnzymeCAGE remains in its separate native pocket/pair-score analysis because pocket coverage does not support these common benchmark banks. No EC1–EC4 result is fabricated for its full pair-conditioned architecture. CREEP EnzymeMap has no final export in this frozen comparison; the training run continues separately.',
        '## Reproducibility and checks',
        'Computation ran on CPU and left the four-GPU CREEP refit untouched. All 128 primary k=10/50 comparisons are checked against the earlier saved EC1–EC4 summaries. The EC3 per-query curves reproduce the previous curve artifacts exactly. All levels verify finite values, identical eligible IDs across models, query-self exclusion, sequence-component exclusion and at least 50 retained candidates. The common-cohort analysis additionally verifies identical neighbors and pointwise hierarchy nesting.',
        'Files: [all curves](curves.csv), [per-class agreement at k=1/10/50](per_class.csv), [coverage](scope.json), [CLEAN source table](clean_transfer.csv), [numerical checks](validation.json), [source hashes](sources.json). Per-query and all-k per-class arrays are stored in the accompanying NPZ files. Reproduce with [`cersei_enzyme_ec_levels.py`](../../../scripts/cersei_enzyme_ec_levels.py) using `compute` in the capability environment and `report` in the system SciencePlots environment.',
        'Vector figures: [EC-level overview](07_enzyme_ec1_ec4_agreement.pdf), [complete curve matrix](08_enzyme_ec_neighborhood_curves.pdf), [common-cohort control](09_enzyme_ec_complete_cohort.pdf). Each also has SVG and PNG exports.',
    ]
    (OUT / 'README.md').write_text('\n\n'.join(paragraphs) + '\n')
    write_csv(OUT / 'summary_k10.csv', [dict(zip(headers, r)) for r in main])
    dump(OUT / 'plot_receipt.json', dict(style='SciencePlots science/no-latex', matplotlib=matplotlib.__version__,
                                       curves_sha256=digest(OUT / 'curves.csv')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['compute', 'report'])
    args = p.parse_args(); compute() if args.action == 'compute' else report()
