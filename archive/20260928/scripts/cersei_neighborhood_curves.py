#!/usr/bin/env python3
"""Frozen-checkpoint neighborhood curves; CPU analysis and SciencePlots exports.

From the workspace root:
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .capability-run-py/bin/python horizyn/scripts/cersei_neighborhood_curves.py compute
  OPENBLAS_NUM_THREADS=8 python3 horizyn/scripts/cersei_neighborhood_curves.py plot

Original embeddings/checkpoints and previous diagnostics are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/public_embedding_comparison_20260924'
OLD = ROOT / 'runs/cersei_embedding_organization_20260923_v1'
OUT = SOURCE / 'neighborhood_curves'
SPLITS = ['time', 'enzyme_smi', 'reaction_smi', 'enzymemap']
LABELS = dict(time='Time', enzyme_smi='Enzyme split', reaction_smi='Reaction split', enzymemap='EnzymeMap')
NAMES = dict(cersei='CERSEI', horizyn='Horizyn', creep='CREEP', clipzyme='CLIPZyme', prott5='ProtT5', reactiont5='ReactionT5')
COLORS = dict(cersei='#0072B2', horizyn='#E69F00', creep='#CC79A7', clipzyme='#009E73', prott5='#666666', reactiont5='#666666')
STYLES = dict(cersei=('-', 'o'), horizyn=('--', 's'), creep=('-.', '^'), clipzyme=('-.', '^'), prott5=(':', 'D'), reactiont5=(':', 'D'))
KS = np.arange(1, 51)


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def models(split):
    return ['cersei', 'horizyn', 'clipzyme' if split == 'enzymemap' else 'creep']


def select(rows, **filters):
    found = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
    assert len(found) == 1, (filters, len(found))
    return found[0]


def write_csv(path, rows):
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def compute():
    import torch
    from cersei_embedding_metrics import nearest, labels_matrix
    from cersei_embedding_organization import norm, prefix

    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    families = read(OLD / 'homology/families.json')
    curves, class_rows, audits, scopes = [], [], [], []
    input_paths = {OLD / 'homology/families.json', OUT / 'chart_contract.json', Path(__file__)}
    for split in SPLITS:
        meta = read(OLD / split / 'metadata.json')
        sub = read(OLD / split / 'subsets.json')
        bank = np.array(meta['neighborhood_indices'])
        where = {int(g): i for i, g in enumerate(bank)}
        qprotein = np.array([where[i] for i in sub['protein_indices'] if i in where], dtype=int)
        family50 = np.array([families[meta['protein_ids'][i]][1] for i in bank])
        enzyme_labels = [prefix(meta['enzyme_ec'][i], 3) for i in bank]
        reaction_labels = meta['reaction_rules'] if split == 'enzymemap' else [prefix(x, 3) for x in meta['reaction_ec']]
        with np.load(OLD / split / 'chemistry.npz') as chemistry:
            tanimoto = chemistry['tanimoto']
            chem_valid = chemistry['valid']
        stages = {}
        for model in models(split):
            path = SOURCE / split / f'{model}_embeddings.npz'
            input_paths.add(path)
            with np.load(path) as z:
                assert len(z['enzyme']) == len(bank)
                assert len(z['reaction']) == len(meta['reaction_ids'])
                if 'bank' in z:
                    assert np.array_equal(z['bank'], bank)
                stages[model] = dict(enzyme=z['enzyme'], reaction=z['reaction'])
        with np.load(OLD / split / 'embeddings.npz') as z:
            assert np.array_equal(z['prott5_indices'], bank)
            controls = dict(enzyme=z['prott5'], reaction=z['raw_reaction'])
        for filename in ['metadata.json', 'subsets.json', 'chemistry.npz', 'embeddings.npz']:
            input_paths.add(OLD / split / filename)

        for modality, labs in [('enzyme', enzyme_labels), ('reaction', reaction_labels)]:
            valid = np.array([bool(x) for x in labs])
            if modality == 'reaction':
                valid &= chem_valid
            q = qprotein[valid[qprotein]] if modality == 'enzyme' else np.array(sub['reaction_indices'])
            q = q[valid[q]]
            sets = [set(x) for x in labs]
            membership, classes = labels_matrix([labs[i] for i in q])
            class_counts = np.asarray(membership.sum(0)).ravel()
            ctrl = 'prott5' if modality == 'enzyme' else 'reactiont5'
            existing_file = SOURCE / split / ('neighborhoods_summary.json' if modality == 'enzyme' else 'reaction_summary.json')
            input_paths.add(existing_file)
            existing = read(existing_file)
            scope = dict(split=split, modality=modality, bank_size=len(labs), eligible_bank=int(valid.sum()), queries=len(q), classes=len(classes))
            scopes.append(scope)
            for model in models(split) + [ctrl]:
                z = controls[modality] if model == ctrl else stages[model][modality]
                assert np.all(np.isfinite(z))
                if modality == 'enzyme':
                    neighbors = nearest(z, valid, family50, 'cpu', k=50, queries=q)[q]
                else:
                    z = norm(z)
                    scores = (z @ z.T)[q]
                    scores[:, ~valid] = -np.inf
                    scores[np.arange(len(q)), q] = -np.inf
                    scores[tanimoto[q] >= .5] = -np.inf
                    neighbors = np.argsort(-scores, axis=1, kind='stable')[:, :50]
                    neighbors[~np.isfinite(np.take_along_axis(scores, neighbors, axis=1))] = -1
                # Keep identical query/class denominators for all 50 k values.
                assert np.all(neighbors >= 0), (split, modality, model, 'fewer than 50 neighbors')
                assert np.all(neighbors != q[:, None])
                assert np.all(valid[neighbors])
                if modality == 'enzyme':
                    assert np.all(family50[q, None] != family50[neighbors])
                else:
                    assert np.all(tanimoto[q[:, None], neighbors] < .5)
                matches = np.array([[bool(sets[i] & sets[j]) for j in nn] for i, nn in zip(q, neighbors)])
                values = np.cumsum(matches, axis=1) / KS
                means = (membership.T @ values) / class_counts[:, None]
                macro = np.asarray(means.mean(0)).ravel()
                assert np.all((macro >= 0) & (macro <= 1))
                np.savez_compressed(OUT / f'{split}_{modality}_{model}.npz', query_indices=q,
                                    neighbors=neighbors, agreement=values, class_names=np.array(classes),
                                    class_means=means, class_counts=class_counts)
                for j, k in enumerate(KS):
                    curves.append(dict(**scope, model=model, k=int(k), macro=float(macro[j]), micro=float(values[:, j].mean())))
                    for c, n, v in zip(classes, class_counts, means[:, j]):
                        class_rows.append(dict(split=split, modality=modality, model=model, k=int(k), label=c, queries=int(n), agreement=float(v)))
                for k in [10, 50]:
                    filters = dict(model=model, k=k)
                    filters.update(dict(ec_level=3, exclude_homologs=True, exclude_training_overlap=False) if modality == 'enzyme' else dict(exclude_similar_participants=True))
                    previous = select(existing, **filters)['macro']
                    error = abs(float(macro[k - 1]) - previous)
                    audits.append(dict(split=split, modality=modality, model=model, k=k, previous=previous, current=float(macro[k - 1]), absolute_error=error))
                    assert error < 1e-5, audits[-1]
                print(split, modality, model, 'k10', round(100 * macro[9], 4), 'k50', round(100 * macro[49], 4), flush=True)
    write_csv(OUT / 'curves.csv', curves)
    write_csv(OUT / 'per_class.csv', class_rows)
    dump(OUT / 'scope.json', scopes)
    dump(OUT / 'validation.json', dict(passed=True, device='cpu', common_query_counts_across_k=True,
         exclusions_verified=True, parity_tolerance=1e-5, parity_checks=audits,
         uncertainty='Descriptive point curves, no interval or significance claim.'))
    dump(OUT / 'sources.json', {str(p): digest(p) for p in sorted(input_paths)})


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def plot():
    sys.path.insert(0, str(ROOT / '.deps/ablation-figures'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    with (OUT / 'curves.csv').open() as f:
        data = list(csv.DictReader(f))
    def curve(split, modality, model):
        rows = [r for r in data if (r['split'], r['modality'], r['model']) == (split, modality, model)]
        rows.sort(key=lambda r: int(r['k']))
        assert [int(r['k']) for r in rows] == KS.tolist()
        return np.array([float(r['macro']) for r in rows]) * 100
    plt.style.use(['science', 'no-latex'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False, 'xtick.top': False,
                         'ytick.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    def panel(ax, split, modality):
        control = 'prott5' if modality == 'enzyme' else 'reactiont5'
        for model in models(split) + [control]:
            ls, mk = STYLES[model]
            ax.plot(KS, curve(split, modality, model), color=COLORS[model], ls=ls, marker=mk,
                    markevery=[0, 9, 19, 29, 39, 49], ms=3, mfc='white', mew=.8, lw=1.35,
                    label=NAMES[model] if model != control else 'Frozen backbone')
        ax.set_xlim(.5, 50.5)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        upper = (60 if split == 'enzymemap' else 90) if modality == 'enzyme' else 35
        ax.set_ylim(0, upper)
        ax.set_yticks(range(0, upper + 1, 20 if modality == 'enzyme' else 10))
        for line in ax.lines:
            assert np.max(line.get_ydata()) < upper, 'A curve would be clipped by the y axis'
        ax.set_xlabel('Neighbors, $k$')
        ax.set_ylabel('Enzyme EC3 agreement (%)' if modality == 'enzyme' else
                      ('Reaction-rule agreement (%)' if split == 'enzymemap' else 'Reaction EC3 agreement (%)'))
        ax.grid(axis='y', alpha=.16, lw=.55)
        ax.minorticks_off()
    def save(fig, stem):
        for ext in ['pdf', 'svg', 'png']:
            fig.savefig(OUT / f'{stem}.{ext}', dpi=250, facecolor='white', bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))
    for ax, modality in zip(axes, ['enzyme', 'reaction']):
        panel(ax, 'enzymemap', modality)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', frameon=False, bbox_to_anchor=(.51, 1.01), handlelength=2.4)
    fig.tight_layout(rect=(0, 0, 1, .91), w_pad=2)
    save(fig, '05_enzymemap_neighborhood_curves')
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6), sharey='row')
    for j, split in enumerate(SPLITS[:3]):
        for i, modality in enumerate(['enzyme', 'reaction']):
            panel(axes[i, j], split, modality)
            axes[i, j].set_xlabel(LABELS[split] + ' — neighbors, $k$')
            if j:
                axes[i, j].set_ylabel('')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', frameon=False, bbox_to_anchor=(.52, 1.005), handlelength=2.4)
    fig.tight_layout(rect=(0, 0, 1, .945), h_pad=2, w_pad=1.4)
    save(fig, '06_reactzyme_neighborhood_curves')

    overview, differences = [], []
    for split in SPLITS:
        for modality in ['enzyme', 'reaction']:
            for model in models(split):
                y = curve(split, modality, model)
                overview.append([LABELS[split], modality, NAMES[model], f'{y[9]:.2f}', f'{y[49]:.2f}'])
            for model in models(split)[1:]:
                d = curve(split, modality, 'cersei') - curve(split, modality, model)
                differences.append([LABELS[split], modality, 'CERSEI − ' + NAMES[model], f'{d[9]:+.2f}',
                                    f'{d[49]:+.2f}', f'{int((d > 1e-9).sum())}/50', f'{d.min():+.2f} to {d.max():+.2f}'])
    report = [
        '# Functional neighborhood analysis',
        'Created 24 September 2026 from the saved, validation-selected checkpoints. No fitting, checkpoint selection, embedding projection or reranking was performed for these plots.',
        '**CERSEI shows stronger local functional organization in several comparisons, but the evidence does not establish universally better embeddings.** Protein EC3 neighborhoods provide the clearest advantage on the ReactZyme enzyme and reaction splits. The time split is close, reaction-space rankings are mixed, and functional agreement is not the same as exact reaction–enzyme retrieval.',
        (OUT / 'protocol_explanation.md').read_text().strip(),
        '## EnzymeMap',
        '![EnzymeMap neighborhood curves](05_enzymemap_neighborhood_curves.png)',
        '**Figure 5.** Macro-averaged functional agreement among the first $k$ cosine neighbors in each model’s original embedding space. Left: enzymes share at least one three-level EC prefix; right: reactions share at least one native EnzymeMap rule ID. Protein neighbors in the same detected 50%-identity sequence component are excluded; reaction neighbors with participant-fingerprint Tanimoto ≥0.5 are excluded. The gray input controls are frozen ProtT5 (left) and ReactionT5 (right). Higher agreement indicates more functionally coherent local neighborhoods. Lines are descriptive point estimates, without confidence bands.',
        'At k=10, enzyme agreement is **28.71% for CERSEI, 27.53% for Horizyn, and 25.91% for CLIPZyme**. Reaction-rule agreement is **19.26%, 17.36%, and 16.94%**, respectively. These are modest differences. The complete curves below record any ordering changes rather than selecting a favorable neighborhood size.',
        '**Across neighborhood sizes:** CERSEI has higher enzyme agreement than both comparators at every k from 1 to 50. In reaction space, Horizyn leads CERSEI at k=1–2 and CLIPZyme leads at k=1–3; CERSEI leads both from k=4 through 50. The evidence is therefore stronger for wider functional neighborhoods than for the single closest reaction neighbor.',
        '**Scope:** the protein geometry bank contains 1,357 test-positive unique sequences, of which 720 are EC3-annotated and eligible as queries/candidates. This is not a clustering analysis of all 261,907 screening accessions. The reaction bank contains 1,521 test reactions and the fixed query sample contains 500 reactions. CREEP is omitted here because its EnzymeMap refit has no completed final test embedding export at figure creation.',
        '## ReactZyme',
        '![ReactZyme neighborhood curves](06_reactzyme_neighborhood_curves.png)',
        '**Figure 6.** The same matched-query, matched-bank analysis on the Time, Enzyme-Similarity and Reaction-Similarity partitions (columns). The top row evaluates enzyme EC3 neighborhoods; the bottom row evaluates reaction EC3 labels inherited from held-out associations. These reaction labels describe associated functions, not independently measured reaction mechanisms. Gray denotes frozen ProtT5 above and ReactionT5 below. The exclusions and class-balanced averaging are the same as in Figure 5.',
        'The strongest k=10 enzyme-space results are **57.86% vs 54.23% CREEP / 53.51% Horizyn** on the enzyme split, and **60.52% vs 55.75% / 55.12%** on the reaction split. On the time split, **51.61% vs 51.28% / 51.27%** is nearly tied. In reaction space, **CREEP slightly exceeds CERSEI on the reaction split at k=10: 8.44% vs 8.28%**. The figures retain this exception.',
        '**Across neighborhood sizes:** on the enzyme and reaction splits, CERSEI leads both comparators in enzyme space at every k from 1 to 50. Reaction-space differences are less consistent: CERSEI leads both comparators throughout the time-split curve, whereas CREEP leads CERSEI at 46 of the 50 neighborhood sizes on the reaction split. CREEP also overtakes CERSEI at k=45–50 in reaction space on the enzyme split. Thus the strongest claim concerns enzyme embeddings, not uniform dominance across both modalities.',
        '## Exact values and persistence across k',
        table(['Partition', 'Space', 'Model', 'Agreement@10 (%)', 'Agreement@50 (%)'], overview),
        table(['Partition', 'Space', 'Difference', 'Δ@10 (pp)', 'Δ@50 (pp)', 'CERSEI higher at k', 'Δ range (pp)'], differences),
        'The 50 neighborhood sizes are nested and strongly correlated. The count of positive differences is descriptive persistence, not 50 independent tests or a significance measure. No assumption that agreement must monotonically decrease with k is imposed.',
        '## What can be concluded',
        '1. **Local functional organization:** on the stronger protein-space comparisons, nearby CERSEI embeddings more often share EC3 function after removing detected close sequence components. This supports a functional-neighborhood claim in the original representation space, without depending on a visually favorable UMAP.',
        '2. **Beyond simple similarity:** the exclusion analyses reduce obvious close-sequence and close-participant explanations. They do not remove all homology, all chemical similarity, or pretrained-model exposure; consequently they do not isolate the causal architectural component responsible for the differences.',
        '3. **Useful but incomplete generalization evidence:** the measurements concern entities in the benchmark test evaluation pools. These are not necessarily unseen sequences or reactions under every split; the neighbor exclusions do not independently audit training overlap. They do not demonstrate wet-lab activity, performance on every functional class, or generalization over the full screening library.',
        '4. **Clustering and retrieval can disagree:** the existing complete ReactZyme evaluations place the local Horizyn refit above the current dictionary-free CERSEI in all six all-positive MRR cells. Better coarse functional neighborhoods therefore cannot be described as universally better reaction–enzyme ranking. See the [complete comparison](../README.md).',
        '## Metric, provenance and uncertainty',
        'For a query, agreement@k is the fraction of its k eligible cosine neighbors sharing at least one label. Each label receives equal weight after averaging its member queries; multi-label queries contribute to each applicable label. All displayed models within a panel use the same IDs, annotation eligibility, exclusions and query sample. No query lacks 50 eligible neighbors, so the query and class denominators stay fixed across k.',
        'CERSEI is the current dictionary-free manuscript model (fusion multiplier 2, residual cap 0.5, no added biological-label supervision). Horizyn and ReactZyme CREEP are local validation-selected refits on the corresponding downstream training graph; CLIPZyme uses the verified official export. Equal downstream data do not equalize pretrained backbones, optimization budgets or loss functions. Frozen input controls are not fully trained retrieval competitors. The source export receipts in the parent directory record the checkpoint and protocol provenance.',
        'Sequence exclusion uses detected MMseqs 50%-identity connected components with ≥80% bidirectional coverage and E≤1e−3. This is an operational filter, not proof of absence of homology. Reaction chemistry exclusion uses the existing participant-fingerprint Tanimoto matrix. ReactZyme query subsets were fixed by an ID hash before this analysis, independently of outcomes and labels.',
        '**Uncertainty limitation:** the earlier fixed-class, family-weighted bootstrap becomes degenerate for EnzymeMap reaction-rule agreement because rule members occur within the same resampling component. Its zero-width intervals do not establish perfect certainty. These new figures deliberately show point estimates only. The earlier report retains both interval conventions; differences between them, especially for the small EnzymeMap enzyme-space advantage over Horizyn, preclude a broad significance claim. These are single-checkpoint diagnostics and contain no seed-to-seed training uncertainty.',
        '## Suggested paper wording',
        '> We assess local functional organization directly in the learned embedding spaces using class-balanced neighborhood agreement. After excluding detected close sequence components, CERSEI achieves higher enzyme EC3 agreement than the compared refits throughout k=1–50 on the ReactZyme enzyme and reaction splits. EnzymeMap shows smaller but consistent enzyme-space gains over CLIPZyme and Horizyn, with higher reaction-rule agreement from k=4 onward. The time-split enzyme neighborhoods are nearly tied, and reaction-space improvements are not uniform: CREEP leads on most neighborhood sizes of the ReactZyme reaction split. These diagnostics support improved functional organization in specific settings rather than universal retrieval superiority.',
        '## Reproducibility',
        'The [curve table](curves.csv), [per-class table](per_class.csv), per-query NPZ archives, [source hashes](sources.json), [cohort sizes](scope.json), and [numerical validation](validation.json) accompany the figures. The standalone script is [`cersei_neighborhood_curves.py`](../../../scripts/cersei_neighborhood_curves.py). All analysis ran on CPU while the four-GPU CREEP training continued.',
        'Vector exports: [EnzymeMap PDF](05_enzymemap_neighborhood_curves.pdf), [EnzymeMap SVG](05_enzymemap_neighborhood_curves.svg), [ReactZyme PDF](06_reactzyme_neighborhood_curves.pdf), [ReactZyme SVG](06_reactzyme_neighborhood_curves.svg).',
        'The expanded protocol is maintained in [protocol_explanation.md](protocol_explanation.md) and included here by the report generator. The [documentation update receipt](documentation_update.json) records the reporting-only code change separately from the original computation-time source hashes and verifies that the numerical artifacts and figures are unchanged.',
    ]
    (OUT / 'README.md').write_text('\n\n'.join(report) + '\n')
    dump(OUT / 'plot_receipt.json', dict(style='SciencePlots science/no-latex', versions=dict(matplotlib=matplotlib.__version__, numpy=np.__version__),
                                       curves_sha256=digest(OUT / 'curves.csv'), figures=['05_enzymemap_neighborhood_curves', '06_reactzyme_neighborhood_curves']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['compute', 'plot'])
    args = parser.parse_args()
    compute() if args.mode == 'compute' else plot()
