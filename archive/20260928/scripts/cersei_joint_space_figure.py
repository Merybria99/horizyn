#!/usr/bin/env python3
"""Fixed-cohort t-SNE illustration with existing cross-modal recovery evidence.

No model fitting, checkpoint selection, or projection selection using EC labels.
"""
from pathlib import Path
import argparse
from collections import Counter
import csv
import json
import sys
import time
import numpy as np
from cersei_crossmodal_functional_recovery import ROOT, sha, dump, read

OUT = ROOT / 'runs/cersei_joint_space_figure_20260925'
OLD = ROOT / 'runs/cersei_embedding_organization_20260923_v1/enzymemap'
PUBLIC = ROOT / 'runs/public_embedding_comparison_20260924/enzymemap'
RECOVERY = ROOT / 'runs/cersei_crossmodal_functional_recovery_20260925'
PAPER = ROOT.parent / '-ICLR2027---Enzyme-Reaction-Retrieval'
MODELS = ['horizyn', 'creep', 'clipzyme', 'cersei']
NAMES = {'horizyn': 'Horizyn', 'creep': 'CREEP', 'clipzyme': 'CLIPZyme', 'cersei': 'CERSEI', 'random': 'Random'}
COLORS = {'horizyn': '#E69F00', 'creep': '#CC79A7', 'clipzyme': '#009E73', 'cersei': '#0072B2', 'random': '#666666'}
EC_COLORS = {'1': '#0072B2', '2': '#E69F00', '3': '#009E73', '4': '#CC79A7', '5': '#56B4E9', '6': '#D55E00', 'Multiple': '#777777'}


def prepare():
    if (OUT / 'prepared.json').exists():
        assert read(OUT / 'prepared.json')['protocol_sha256'] == sha(OUT / 'protocol.json')
        return
    protocol = read(OUT / 'protocol.json')
    meta = read(OLD / 'metadata.json')
    bank = np.asarray(meta['neighborhood_indices'])
    labels = [sorted({x.split('.')[0] for x in meta['enzyme_ec'][i] if x.split('.')[0].isdigit()}) for i in bank]
    eligible = [i for i, ls in enumerate(labels) if ls]
    # A single common ordering also fixes the overplotting order independently of labels.
    import hashlib
    eligible.sort(key=lambda i: hashlib.sha256(f"25092026:{meta['protein_ids'][bank[i]]}".encode()).hexdigest())
    keep = np.asarray(eligible)
    assert len(keep) == 720
    selected = bank[keep]
    ids = [meta['protein_ids'][i] for i in selected]
    ec = [labels[i] for i in keep]
    groups = [ls[0] if len(ls) == 1 else 'Multiple' for ls in ec]
    assert set(groups) == set(EC_COLORS)
    assert len(set(ids)) == len(ids)
    expanded = np.load(OLD / 'expanded.npy')
    # Released CLIPZyme can assign different vectors to accession aliases.
    # Reuse the existing, model-independent lexical test-positive alias rule.
    clip = np.load(PUBLIC / 'clipzyme_embeddings.npz')
    assert np.array_equal(clip['bank'], bank)
    rep = clip['representative'][keep]
    assert (rep >= 0).all() and np.array_equal(expanded[rep], selected)
    truth = np.load(OLD / 'truth.npy')
    aliases = {}
    for j in np.unique(truth[:, 1]):
        aliases.setdefault(int(expanded[j]), []).append(int(j))
    assert rep.tolist() == [min(aliases[int(i)],key=lambda j:meta['candidate_ids'][j]) for i in selected]
    score_sources = read(RECOVERY / 'enzymemap/components/complete.json')['sources']
    embeddings = {}
    sources = {}
    parity = {}
    query_indices = np.arange(0, len(meta['reaction_ids']), 73)
    for model in MODELS:
        if model == 'creep':
            run = ROOT / 'runs/enzymemap_public_creep_20260924_seed42'
            cat = read(run / 'catalog.json')
            assert cat['protein_ids'] == meta['protein_ids']
            assert cat['test_ids'] == meta['reaction_ids']
            assert cat['candidate_ids'] == meta['candidate_ids']
            ep = run / 'test/protein.npy'
            rp = run / 'test/reaction.npy'
            e = np.load(ep, mmap_mode='r')[selected]
            r = np.load(rp, mmap_mode='r')[query_indices]
            sources[model] = dict(enzyme=str(ep), enzyme_sha256=sha(ep), reaction=str(rp), reaction_sha256=sha(rp), selection=read(run / 'selection.json'))
        else:
            path = PUBLIC / f'{model}_embeddings.npz'
            z = np.load(path)
            assert np.array_equal(z['bank'], bank)
            e = z['enzyme'][keep]
            r = z['reaction'][query_indices]
            sources[model] = dict(embeddings=str(path), embeddings_sha256=sha(path))
        assert e.shape[0] == len(ids) and np.isfinite(e).all()
        assert np.max(np.abs(np.linalg.norm(e, axis=1) - 1)) < 1e-5
        scores = np.load(score_sources[model]['scores'], mmap_mode='r')
        error = float(np.max(np.abs(r @ e.T - scores[np.ix_(query_indices, rep)])))
        assert error < 3e-5, (model, error)
        parity[model] = error
        embeddings[model] = e.astype(np.float64) / np.linalg.norm(e.astype(np.float64), axis=1, keepdims=True)
    old_receipt = read(ROOT / 'runs/clipzyme_embedding_comparison_20260924/prepared.json')
    current = read(RECOVERY / 'enzymemap/stages/complete.json')
    assert old_receipt['receipts']['CERSEI']['checkpoint_sha256'] == current['checkpoint_sha256']
    assert old_receipt['receipts']['CERSEI']['head_sha256'] == current['head_sha256']
    assert current['stages']['cersei']['alpha'] == 2 and current['stages']['cersei']['kappa'] == .1
    np.savez_compressed(OUT / 'embeddings.npz', **embeddings)
    dump(OUT / 'cohort.json', dict(protein_ids=ids, bank_indices=selected.tolist(), representative_accession_indices=rep.tolist(),
         representative_accessions=[meta['candidate_ids'][i] for i in rep], annotation_ec1=ec,
         color_group=groups, counts=dict(Counter(groups)), original_test_positive_bank=len(bank),
         excluded_without_ec1=len(bank)-len(keep), sequence_deduplicated=True))
    with (OUT / 'cohort.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['protein_id', 'global_sequence_index', 'ec1', 'display_group'])
        writer.writerows(zip(ids, selected, [';'.join(x) for x in ec], groups))
    dump(OUT / 'prepared.json', dict(protocol_sha256=sha(OUT / 'protocol.json'),
         cohort_sha256=sha(OUT / 'cohort.json'), embeddings_sha256=sha(OUT / 'embeddings.npz'),
         metadata_sha256=sha(OLD / 'metadata.json'), sources=sources, dimensions={m: embeddings[m].shape[1] for m in MODELS},
         score_reconstruction_max_error=parity, score_parity_queries=len(query_indices),
         label_selection=False, sample_selection_by_model=False))
    print('Prepared identical cohort', Counter(groups), 'score errors', parity, flush=True)


def project():
    from sklearn.manifold import TSNE, trustworthiness
    from sklearn.metrics import pairwise_distances
    import sklearn
    protocol = read(OUT / 'protocol.json')['projection']
    data = np.load(OUT / 'embeddings.npz')
    records = []
    for seed in [protocol['primary_seed']] + protocol['sensitivity_seeds']:
        path = OUT / f'projections_{seed}.npz'
        receipt = path.with_suffix('.json')
        if path.exists() and receipt.exists():
            saved = read(receipt)
            assert saved['protocol_sha256'] == sha(OUT / 'protocol.json')
            assert saved['embeddings_sha256'] == sha(OUT / 'embeddings.npz')
            records.extend(saved['models'])
            continue
        projections = {}
        stats = []
        initial = np.random.default_rng(seed).normal(0, 1e-4, size=(len(data[MODELS[0]]), 2)).astype(np.float32)
        for model in MODELS:
            started = time.monotonic()
            x = data[model]
            estimator = TSNE(n_components=2, perplexity=protocol['perplexity'], early_exaggeration=protocol['early_exaggeration'],
                learning_rate=protocol['learning_rate'], max_iter=protocol['max_iter'],
                n_iter_without_progress=protocol['n_iter_without_progress'], min_grad_norm=protocol['min_grad_norm'],
                metric=protocol['metric'], init=initial.copy(), random_state=seed, method=protocol['method'],
                angle=protocol['angle'], n_jobs=protocol['n_jobs'])
            y = estimator.fit_transform(x)
            assert y.shape == (720, 2) and np.isfinite(y).all()
            projections[model] = y
            d = pairwise_distances(x, metric='cosine');np.fill_diagonal(d, np.inf)
            d2 = pairwise_distances(y);np.fill_diagonal(d2, np.inf)
            neighbors = np.argsort(d, axis=1, kind='stable')[:, :10]
            neighbors2 = np.argsort(d2, axis=1, kind='stable')[:, :10]
            preservation = np.mean([len(set(a) & set(b))/10 for a,b in zip(neighbors, neighbors2)])
            stats.append(dict(model=model, seed=seed, kl_divergence=float(estimator.kl_divergence_),
                iterations=int(estimator.n_iter_), trustworthiness_at10=float(trustworthiness(x, y, n_neighbors=10, metric='cosine')),
                original_neighbors_retained_at10=float(preservation), seconds=time.monotonic()-started))
            print('Projection', stats[-1], flush=True)
        np.savez_compressed(path, **projections)
        dump(receipt, dict(models=stats, sklearn_version=sklearn.__version__, protocol_sha256=sha(OUT / 'protocol.json'),
            embeddings_sha256=sha(OUT / 'embeddings.npz'), projections_sha256=sha(path)))
        records.extend(stats)
    dump(OUT / 'projection_diagnostics.json', records)


def recovery_values():
    path = RECOVERY / 'enzymemap/components/curves.csv'
    assert sha(path) == read(path.with_name('complete.json'))['curves_sha256']
    with path.open() as f:
        values = {r['model']:r for r in csv.DictReader(f) if int(r['k']) == 10 and r['model'] in MODELS + ['random']}
    assert set(values) == set(MODELS + ['random'])
    for model in MODELS + ['random']:
        z = np.load(RECOVERY / 'enzymemap/components' / (f'{model}_per_query.npz' if model != 'random' else 'controls_per_query.npz'))
        score = float(z['macro_weights'] @ z['agreement' if model != 'random' else 'random'][:,9])
        assert abs(score - float(values[model]['macro'])) < 1e-12
        assert len(z['query_indices']) == 1067
    return values


def render():
    from cersei_joint_space_clarity import export_data, render as render_clear
    export_data()
    render_clear()


def report():
    cohort=read(OUT/'cohort.json');prepared=read(OUT/'prepared.json');diagnostics=read(OUT/'projection_diagnostics.json')
    values=recovery_values();protocol=read(OUT/'protocol.json')
    primary=protocol['projection']['primary_seed'];seeds=[primary]+protocol['projection']['sensitivity_seeds']
    assert len(diagnostics)==12 and len(cohort['protein_ids'])==720
    assert sum(cohort['counts'].values())==720 and cohort['counts']['Multiple']==28
    assert prepared['protocol_sha256']==sha(OUT/'protocol.json')
    assert prepared['embeddings_sha256']==sha(OUT/'embeddings.npz')
    for seed in seeds:
        path=OUT/f'projections_{seed}.npz';receipt=read(path.with_suffix('.json'))
        assert receipt['projections_sha256']==sha(path)
        z=np.load(path)
        assert set(z.files)==set(MODELS)
        assert all(z[m].shape==(720,2) and np.isfinite(z[m]).all() for m in MODELS)
    lines=['# Enzyme-space maps and cross-modal functional recovery',
        '25 September 2026. Full four-model diagnostic archive. At the user\'s request, the current manuscript shows CREEP, CLIPZyme and CERSEI in a [four-panel comparison](manuscript_comparison/README.md). The archived full comparison below retains all original measurements. Existing model checkpoints, inference coefficients, benchmark scores and training jobs are unchanged.',
        '## What the figure supports',
        'CERSEI has the highest measured EnzymeMap EC3 functional recovery@10 among the four evaluated systems after recorded partners and their detected close-sequence components are removed. The maps supply an intuitive view of broad enzyme function; they do not independently establish that CERSEI has the globally best-separated embedding space. CLIPZyme also shows substantial functional organization, and class mixing remains in every model.',
        '![Archived full comparison](joint_space_composite.png)',
        '**Figure 1.** (a–d) Independent t-SNE maps of the same 720 EC-annotated, test-associated enzyme sequences for Horizyn, CREEP, CLIPZyme and CERSEI. Colors indicate EC1; gray crosses retain proteins with multiple top-level classes. Labels are used only for coloring. Coordinates, gaps and areas cannot be compared numerically across independent projections. (e) Class-macro cross-modal EC3 recovery@10 in the original spaces, after excluding recorded partners, exact-sequence aliases and detected close-sequence components. This panel uses 1,067 reactions and 131,409 annotated accession candidates before query-specific exclusions. Whiskers are 95% reaction-family multiplier intervals conditional on the fitted models; random retrieval is an exact expectation. The sequence exclusion applies to retrieval, not the maps.',
        '## Source data and clearer rendering',
        'The archived five-panel comparison remains in one horizontal row; the current manuscript uses the four-panel display linked above. The renderer imports SciencePlots and retains its serif typography, with tighter square map windows, visible point boundaries, capped confidence intervals and direct percentage labels. The saved coordinates, cohort identities, projection settings and measured scores are unchanged. Larger standalone [maps](enzyme_maps.pdf) and [recovery panel](functional_recovery_at10.pdf) support detailed inspection.',
        'The [source-data ZIP](figure_source_data.zip) contains the original normalized embeddings, exact projection arrays and auditable CSVs. [Projection coordinates](plot_coordinates.csv) give all 8,640 model/seed/protein rows; filter primary_projection=True for the main figure. [Recovery values](recovery_at10.csv) give the plotted percentages and intervals. [Per-query contributions](recovery_per_query.csv) reproduce each class-macro percentage by summation within a method. The [data guide](DATA_README.md) explains units and the two distinct cohorts; [export checks](data_export_validation.json) verify exact coordinate round trips and aggregation.',
        '## Quantitative panel',
        '| Method | Recovery@10 (%) | 95% family-multiplier interval |\n| --- | ---: | --- |']
    for model in MODELS+['random']:
        r=values[model]
        interval=f'[{100*float(r["ci_low"]):.2f}, {100*float(r["ci_high"]):.2f}]' if model!='random' else 'Exact expected reference'
        lines[-1]+=f'\n| {NAMES[model]} | {100*float(r["macro"]):.2f} | {interval} |'
    lines.extend([
        'The paired CERSEI−CLIPZyme difference is +4.65 percentage points, with interval [+3.04, +6.87]. CERSEI leads throughout k=1–50 on EnzymeMap. These are functional compatibility measurements, not confirmed new catalytic associations, and the confidence intervals do not include training-seed variation. The [complete four-partition analysis](../cersei_crossmodal_functional_recovery_20260925/README.md) retains the small/uncertain Time and Reaction-Sim differences, the Enzyme-Sim gain, both exclusion policies and all stage interventions.',
        '## Identical enzyme cohort and annotation scope',
        'All 720 EC1-annotated sequences in the existing 1,357-sequence EnzymeMap test-associated geometry bank are retained. The 637 without EC1 labels are excluded; there is no subsampling for favorable cluster appearance, balancing of the display by class, or selection based on model predictions. These are sequences associated with test reactions, not necessarily sequences unseen during training. This cohort is smaller than the screening bank and is used only for the maps.',
        '| Display group | Sequences |\n| --- | ---: |\n'+'\n'.join(f'| {"EC "+c if c!="Multiple" else c} | {cohort["counts"][c]} |' for c in EC_COLORS),
        'Each sequence appears once in each panel. CLIPZyme uses the lexicographically first test-positive accession per sequence, following the pre-existing geometry protocol; its released accession vectors can differ for identical sequences. The other systems encode sequences. The same representative accession was used to check reconstructed cross-modal scores for all four models. The initial preparation check caught a mismatch from using the first bank accession; the existing lexical test-positive rule was restored before any projection was fitted. This is recorded in the frozen protocol and its earlier snapshot.',
        '## Fixed projection protocol',
        'The EC labels do not enter projection fitting. Embeddings are L2-normalized, then each model is projected independently with scikit-learn 1.7.2 t-SNE: cosine metric, perplexity 30, early exaggeration 12, learning rate 200, 1,500 iterations, Barnes–Hut angle 0.5, no-progress threshold 300 and minimum gradient norm 1e−7. Every model starts from the same 720×2 Gaussian array (standard deviation 1e−4) for a given seed. No PCA or other learned preprocessing is applied.',
        'The primary seed 25092026 and two sensitivity seeds 25092027/25092028 were specified before projection. All twelve resulting maps are retained. Models do not share a coordinate frame; each panel uses equal axis aspect and a square plotting window centered on its coordinate bounds and containing every point. No point is cropped to make clusters appear cleaner. Draw order uses the same fixed hash ordering for every model; multiple-class proteins use gray crosses. Display formatting was adjusted for legibility without changing projection coordinates.',
        '![All prespecified seeds](projection_seed_sensitivity.png)',
        '**Figure 2.** Rows correspond to seeds 25092026, 25092027 and 25092028; columns correspond to Horizyn, CREEP, CLIPZyme and CERSEI. The first row supplies the main figure. Seed changes alter the two-dimensional layout and do not represent independent training replicates.',
        '## Projection fidelity',
        'The following checks compare original-space cosine neighborhoods with two-dimensional Euclidean neighborhoods. They measure faithfulness of the projection, not enzyme function or retrieval performance. EC labels are unused by these calculations. A projection retaining about 70–74% of the original ten neighbors still changes a substantial fraction of them; this is why the model claim rests on the separate original-space recovery measurement.',
        '| Method | Original dimension | Primary: original top-10 neighbors retained (%) | Range across all three seeds (%) | Primary trustworthiness@10 |\n| --- | ---: | ---: | --- | ---: |'])
    for model in MODELS:
        ds=[r for r in diagnostics if r['model']==model];d=next(r for r in ds if r['seed']==primary)
        vs=[100*r['original_neighbors_retained_at10'] for r in ds]
        lines[-1]+=f'\n| {NAMES[model]} | {prepared["dimensions"][model]} | {100*d["original_neighbors_retained_at10"]:.2f} | {min(vs):.2f}–{max(vs):.2f} | {d["trustworthiness_at10"]:.4f} |'
    lines.extend(['## Paper placement and reproducibility',
        'The main analysis section contains the three-model composite in manuscript_comparison/ and the cross-modal recovery definition/results. Complete k=1–50 curves, both exclusion policies, projection seeds, same-modality neighborhoods, EC-depth controls and exact-association counterexamples are retained in the appendix. Existing exact-association results were not deleted or replaced by the functional-compatibility metric.',
        'The design follows the EC-colored enzyme-map idea in [FGW-CLIP Figure 2](https://arxiv.org/pdf/2512.08508v2#page=11), but compares four trained methods and pairs the qualitative maps with original-space quantitative recovery. It is not an exact replication of that paper’s sampled protein cohort or unspecified t-SNE configuration.',
        'Sources: [protocol](protocol.json), [chart contract](chart_contract.json), [cohort IDs](cohort.csv), [source hashes and score parity](prepared.json), [all projection diagnostics](projection_diagnostics.json), [validation](validation.json), [export receipt](export_receipt.json). [Composite PDF](joint_space_composite.pdf) and [SVG](joint_space_composite.svg), plus [seed sensitivity PDF](projection_seed_sensitivity.pdf), are standalone exports.',
        'Run from the horizyn directory: `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ../.capability-run-py/bin/python scripts/cersei_joint_space_figure.py all`. Preparation/projection artifacts are reused only if their frozen protocol and source hashes match. For the final figures, run `OPENBLAS_NUM_THREADS=4 python3 scripts/cersei_joint_space_figure.py render` with system Matplotlib 3.6.3; this imports SciencePlots 2.2.1 natively and calls its science/no-latex styles. The renderer also supports Matplotlib 3.11 by loading the same library style files directly when its registration hook is incompatible. The export receipt records the actual loader, version and serif font family. Regenerate this report with `python3 scripts/cersei_joint_space_figure.py report`.',
        'No extra training or GPU runs are needed for this figure. The interpretation remains exploratory because these benchmark tests and the earlier representation results had already been inspected.'
    ])
    (OUT/'README.md').write_text('\n\n'.join(lines)+'\n')
    prior=read(OUT/'validation.json') if (OUT/'validation.json').exists() else {}
    projection_audit={}
    if 'independent_primary_projection_rerun_max_coordinate_error' in prior:
        projection_audit['independent_primary_projection_rerun_max_coordinate_error']=prior['independent_primary_projection_rerun_max_coordinate_error']
    dump(OUT/'validation.json',dict(**projection_audit,passed=True,cohort_sequences=720,missing_ec1_excluded=637,
        class_counts=cohort['counts'],same_sequences_all_models=True,all_three_prespecified_seeds_retained=True,
        no_projection_selection=True,no_labels_used_in_projection=True,projection_shape_checks=12,
        source_parity=prepared['score_reconstruction_max_error'],retrieval_panel_recomputed_from_per_query=True,
        retrieval_queries=1067,script_sha256=sha(__file__),report_sha256=sha(OUT/'README.md')))
    print('Saved report and validation',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','project','render','report','all']);args=parser.parse_args()
    for action,fn in [('prepare',prepare),('project',project),('render',render),('report',report)]:
        if args.action in [action,'all']:fn()
