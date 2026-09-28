"""Fixed-cohort enzyme neighborhood and clustering diagnostics; no model fitting.

Only the descriptive KMeans partition is fitted, never the retrieval encoders.
The protocol is recorded before computing these diagnostic values.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, adjusted_rand_score, adjusted_mutual_info_score
import sklearn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/cersei_enzyme_cluster_metrics_20260925'
MAPS = ROOT / 'runs/cersei_joint_space_figure_20260925'
MODELS = ['creep', 'clipzyme', 'cersei']
NAMES = {'creep': 'CREEP', 'clipzyme': 'CLIPZyme', 'cersei': 'CERSEI'}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table(p, rows):
    with Path(p).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prefix(labels, depth):
    return sorted({'.'.join(x.split('.')[:depth]) for x in labels
                   if len(x.split('.')) >= depth and '-' not in x.split('.')[:depth]})


def membership(labels):
    classes = sorted({v for row in labels for v in row})
    matrix = np.array([[c in row for c in classes] for row in labels], dtype=float)
    assert (matrix.sum(1) > 0).all()
    return matrix, classes


def macro_weights(labels):
    matrix, classes = membership(labels)
    weights = (matrix / matrix.sum(0)).sum(1) / len(classes)
    assert abs(weights.sum() - 1) < 1e-12
    return weights


def multiplier_weights(labels, families):
    matrix, classes = membership(labels)
    unique, inverse = np.unique(families, return_inverse=True)
    family_weights = np.random.default_rng(25092026).exponential(size=(1000, len(unique)))
    query_weights = family_weights[:, inverse]
    denominators = query_weights @ matrix
    result = ((1 / denominators) @ matrix.T) * query_weights / len(classes)
    assert np.max(np.abs(result.sum(1) - 1)) < 1e-12
    return result


def estimate(values, labels, boot):
    weights = macro_weights(labels)
    samples = boot @ values
    direct = np.mean([np.mean(values[[c in row for row in labels]])
                      for c in sorted({v for row in labels for v in row})])
    score = float(weights @ values)
    assert abs(score - direct) < 1e-12
    ci = np.quantile(samples, [.025, .975])
    return dict(macro=score, mean=float(np.mean(values)), ci_low=float(ci[0]), ci_high=float(ci[1]))


def agreement_curve(sim, compatible, allowed, kmax=50):
    """Tie expectations over complete sorted candidate lists, before truncation."""
    result = np.empty((len(sim), kmax))
    ties = []
    for i in range(len(sim)):
        candidates = np.flatnonzero(allowed[i])
        order = candidates[np.argsort(-sim[i, candidates], kind='stable')]
        scores = sim[i, order]
        starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
        counts = np.diff(np.r_[starts, len(scores)])
        hits = np.add.reduceat(compatible[i, order].astype(float), starts)
        expected = np.repeat(hits / counts, counts)
        result[i] = np.cumsum(expected[:kmax]) / np.arange(1, kmax+1)
        ties.append(int(counts[np.searchsorted(starts, 9, side='right')-1]))
        # Independent threshold expression for k=10, including the full boundary tie.
        threshold = scores[9]
        above = allowed[i] & (sim[i] > threshold)
        equal = allowed[i] & (sim[i] == threshold)
        check = (compatible[i, above].sum() + (10-above.sum()) * compatible[i, equal].mean()) / 10
        assert abs(check - result[i, 9]) < 1e-12
    return result, np.asarray(ties)


def main():
    protocol = read(OUT/'protocol.json')
    assert protocol['models'] == MODELS
    for source in protocol['sources'].values():
        assert sha(source['path']) == source['sha256']
    cohort = read(protocol['sources']['cohort']['path'])
    prepared = read(MAPS/'prepared.json')
    assert sha(MAPS/'embeddings.npz') == prepared['embeddings_sha256']
    meta = read(protocol['sources']['labels']['path'])
    families = read(protocol['sources']['families']['path'])
    ids = cohort['protein_ids']; indices = cohort['bank_indices']
    assert len(ids) == len(set(ids)) == 720
    assert [meta['protein_ids'][i] for i in indices] == ids
    raw_labels = [meta['enzyme_ec'][i] for i in indices]
    labels = {depth: [prefix(row, depth) for row in raw_labels] for depth in [1, 3]}
    assert labels[1] == cohort['annotation_ec1']
    assert all(labels[3])
    fam30 = np.array([families[p][0] for p in ids])
    fam50 = np.array([families[p][1] for p in ids])
    data = np.load(MAPS/'embeddings.npz')
    embeddings = {m: data[m] for m in MODELS}
    similarities = {}
    for model, x in embeddings.items():
        assert np.isfinite(x).all() and x.shape[0] == 720
        assert np.max(np.abs(np.linalg.norm(x, axis=1)-1)) < 1e-10
        similarities[model] = np.clip(x @ x.T, -1, 1)
    boot = {depth: multiplier_weights(ls, fam30) for depth, ls in labels.items()}
    matches = {depth: (membership(ls)[0] @ membership(ls)[0].T) > 0 for depth, ls in labels.items()}
    allowed = {'self_only': ~np.eye(720, dtype=bool),
               'exclude_component50': fam50[:, None] != fam50[None, :]}
    assert min(mask.sum(1).min() for mask in allowed.values()) >= 50
    summaries = []; curves = []; per_query = []; paired = []; per_class = []
    cache = {}
    for depth in [1, 3]:
        weights = macro_weights(labels[depth])
        for policy, mask in allowed.items():
            random = (matches[depth] & mask).sum(1) / mask.sum(1)
            summaries.append(dict(model='Random', ec_level=depth, exclusion=policy, k=10,
                **estimate(random, labels[depth], boot[depth])))
            for model in MODELS:
                values, ties = agreement_curve(similarities[model], matches[depth], mask)
                cache[depth, policy, model] = values
                summary = dict(model=NAMES[model], ec_level=depth, exclusion=policy, k=10,
                    **estimate(values[:, 9], labels[depth], boot[depth]))
                summaries.append(summary)
                for k in range(1, 51):
                    curves.append(dict(model=NAMES[model], ec_level=depth, exclusion=policy, k=k,
                        macro=float(weights @ values[:, k-1]), mean=float(values[:, k-1].mean())))
                for i, pid in enumerate(ids):
                    per_query.append(dict(model=NAMES[model], ec_level=depth, exclusion=policy,
                        protein_id=pid, family30=int(fam30[i]), family50=int(fam50[i]),
                        labels=';'.join(labels[depth][i]), eligible_candidates=int(mask[i].sum()),
                        agreement_at10=float(values[i, 9]), exact_tie_size_at10=int(ties[i]),
                        random_expectation=float(random[i]), macro_weight=float(weights[i])))
                for c in membership(labels[depth])[1]:
                    keep = np.array([c in row for row in labels[depth]])
                    per_class.append(dict(model=NAMES[model], ec_level=depth, exclusion=policy,
                        ec_class=c, queries=int(keep.sum()), agreement_at10=float(values[keep, 9].mean())))
                print('Neighborhood', summary, flush=True)
            for comparator in MODELS[:-1]:
                diff = cache[depth, policy, 'cersei'] - cache[depth, policy, comparator]
                paired.append(dict(metric='neighbor_agreement_at10', comparator=NAMES[comparator],
                    ec_level=depth, exclusion=policy, **estimate(diff[:, 9], labels[depth], boot[depth]),
                    positive_ks=int(np.count_nonzero(weights @ diff > 0))))
    table(OUT/'neighborhood_at10.csv', summaries)
    table(OUT/'neighborhood_curves.csv', curves)
    table(OUT/'per_query.csv', per_query)
    table(OUT/'per_class.csv', per_class)
    table(OUT/'paired_neighborhood_differences.csv', paired)
    np.savez_compressed(OUT/'neighbor_agreements.npz',
        **{f'ec{d}_{p}_{m}': v for (d, p, m), v in cache.items()})

    # Conventional silhouette and disjoint cluster labels need one class per point.
    single = np.array([len(x) == 1 for x in labels[1]])
    assert single.sum() == 692
    y = np.array([row[0] for row in np.array(labels[1], dtype=object)[single]])
    single_labels = [[c] for c in y]
    single_boot = multiplier_weights(single_labels, fam30[single])
    silhouettes = []; silhouette_values = {}; assignments = []; cluster_scores = []
    for model in MODELS:
        distance = 1-similarities[model][np.ix_(single, single)]
        np.fill_diagonal(distance, 0)
        v = silhouette_samples(distance, y, metric='precomputed')
        silhouette_values[model] = v
        # Independent per-point implementation for the whole 692-point cohort.
        direct = []
        for i in range(len(y)):
            same = (y == y[i]); same[i] = False
            a = distance[i, same].mean()
            b = min(distance[i, y == c].mean() for c in np.unique(y) if c != y[i])
            direct.append((b-a)/max(a, b))
        assert np.max(np.abs(v-direct)) < 1e-12
        record = dict(model=NAMES[model], space='original_cosine',
            **estimate(v, single_labels, single_boot))
        silhouettes.append(record); print('Silhouette', record, flush=True)
        for seed in protocol['kmeans']['seeds']:
            km = KMeans(n_clusters=6, n_init=20, random_state=seed, max_iter=300, algorithm='lloyd')
            group = km.fit_predict(embeddings[model][single])
            record = dict(model=NAMES[model], seed=seed, k=6,
                ari=float(adjusted_rand_score(y, group)),
                ami=float(adjusted_mutual_info_score(y, group)), inertia=float(km.inertia_))
            cluster_scores.append(record); print('Clustering', record, flush=True)
            for pid, label, cluster in zip(np.array(ids)[single], y, group):
                assignments.append(dict(model=NAMES[model], seed=seed, protein_id=pid,
                    ec1=label, fitted_cluster=int(cluster)))
    table(OUT/'silhouette.csv', silhouettes)
    table(OUT/'clustering_scores.csv', cluster_scores)
    table(OUT/'cluster_assignments.csv', assignments)
    sil_paired = [dict(comparator=NAMES[m], **estimate(silhouette_values['cersei']-silhouette_values[m],
                   single_labels, single_boot)) for m in MODELS[:-1]]
    table(OUT/'paired_silhouette_differences.csv', sil_paired)

    projection_rows = []
    for seed in protocol['projection_seeds']:
        z = np.load(MAPS/f'projections_{seed}.npz')
        for model in MODELS:
            x = z[model].astype(float)
            dist = np.linalg.norm(x[:, None]-x[None, :], axis=-1)
            curve, _ = agreement_curve(-dist, matches[1], allowed['self_only'])
            v = silhouette_samples(dist[np.ix_(single, single)], y, metric='precomputed')
            projection_rows.append(dict(model=NAMES[model], seed=seed,
                ec1_agreement_at10_macro=float(macro_weights(labels[1]) @ curve[:, 9]),
                ec1_silhouette_mean=float(v.mean()),
                ec1_silhouette_macro=float(macro_weights(single_labels) @ v)))
    table(OUT/'projection_diagnostics.csv', projection_rows)
    table(OUT/'cohort.csv', [dict(protein_id=p, ec1=';'.join(labels[1][i]),
        ec3=';'.join(labels[3][i]), family30=int(fam30[i]), family50=int(fam50[i]),
        used_for_disjoint_clustering=bool(single[i])) for i,p in enumerate(ids)])
    dump(OUT/'scope.json', dict(queries=720,ec1_classes=len(membership(labels[1])[1]),
        ec3_classes=len(membership(labels[3])[1]),sequence_families30=len(set(fam30)),
        sequence_families50=len(set(fam50)),single_ec1_queries=int(single.sum()),
        excluded_multilabel_ec1=int((~single).sum()),
        eligible_neighbors={p:[int(mask.sum(1).min()),int(mask.sum(1).max())] for p,mask in allowed.items()}))
    dump(OUT/'validation.json', dict(passed=True, protocol_sha256=sha(OUT/'protocol.json'),
        script_sha256=sha(__file__), sklearn_version=sklearn.__version__,
        matched_ids=True, source_hashes_verified=True, no_model_training=True,
        independent_macro_checks=True, independent_exact_tie_threshold_checks=720*3*2*2+720*3*3,
        independent_silhouette_checks=692*3, all_prespecified_metrics_and_seeds_retained=True,
        csv_sha256={p.name:sha(p) for p in sorted(OUT.glob('*.csv'))}))
    print('Saved all metrics and validation:', OUT, flush=True)


if __name__ == '__main__':
    main()
