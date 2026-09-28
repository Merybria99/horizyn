#!/usr/bin/env python3
"""Sensitivity to score ties, shared reactants, sequence support and query dependence."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from run_f3_diagnostics_20260918 import (BASE,PILOT,OUT,K,KEYS,align_scores,canonical_reaction,
    ordinal_ranks,expected_tie_hits,paired,save)


def cluster_interval(values, groups, rng):
    unique=sorted(set(groups));sums=[];counts=[]
    for group in unique:
        mask=np.array(groups)==group;sums.append(np.asarray(values)[mask].sum());counts.append(mask.sum())
    indices=rng.integers(0,len(unique),(10000,len(unique)))
    draws=np.array(sums)[indices].sum(axis=1)/np.array(counts)[indices].sum(axis=1)
    return dict(clusters=len(unique),mean=float(np.mean(values)),ci95=np.quantile(draws,[.025,.975]).tolist())


def main():
    panel=pd.read_csv(BASE/'data/test_P450.csv');mapping=pd.read_csv(BASE/'f3_features/reaction_mapping.csv')
    rxns=mapping.raw_reaction.tolist(); qids=mapping.reaction_id.tolist();pids=panel.UniprotID.drop_duplicates().tolist()
    qi={q:i for i,q in enumerate(rxns)};pi={p:i for i,p in enumerate(pids)}
    labels=np.zeros((191,490),bool)
    for q,p,y in panel[KEYS+['Label']].itertuples(index=False,name=None):labels[qi[q],pi[p]]=bool(y)
    full=[canonical_reaction(s) for s in rxns];reactants=[s.split('>>')[0] for s in full]
    same=np.array([[a==b for b in reactants] for a in reactants])
    different=np.array([[a!=b for b in full] for a in full])
    disjoint=labels.astype(int)@labels.astype(int).T==0
    eligible=same&different&disjoint
    valid=eligible.any(axis=1);donors=eligible.argmax(axis=1)
    swaps=pd.read_csv(OUT/'swaps_per_query.csv');nov=pd.read_csv(OUT/'reaction_novelty.csv')
    transfer=pd.read_csv(OUT/'nearest_neighbor_transfer.csv');summary=json.loads((OUT/'summary.json').read_text())
    rng=np.random.default_rng(84);results=dict(reactant_groups=len(set(reactants)),
        same_reactant_disjoint_queries=int(valid.sum()),same_reactant_disjoint_groups=len(set(np.array(reactants)[valid])),models={})
    paths={'original':BASE/'scores/f3_enzymecage_seed42_epoch06.csv','control':PILOT/'p450/control_scores.csv','combined':PILOT/'p450/combined_scores.csv'}
    details=[]
    for name,path in paths.items():
        f=align_scores(panel,pd.read_csv(path)); mat=np.zeros((191,490))
        for q,p,s in f[KEYS+['pred']].itertuples(index=False,name=None):mat[qi[q],pi[p]]=s
        ranks=ordinal_ranks(mat); best=np.array([ranks[:,labels[i]].min(axis=1) for i in range(191)])
        correct=best.diagonal();swapped=best[np.arange(191),donors]
        rows=swaps[swaps.model==name].set_index('reaction_id').loc[qids]
        stats={}
        for k in K:
            obs=(correct<=k).astype(float)
            similar=(rows.similar_rank.to_numpy()<=k).astype(float)
            random=((best<=k).sum(axis=1)-obs)/190
            expected=np.array([expected_tie_hits(mat[i],labels[i],k) for i in range(191)])
            stats[f'top{k}']=dict(official_stable_order=float(obs.mean()),randomized_score_ties=float(expected.mean()),
                correct_minus_random_cluster=cluster_interval(obs-random,reactants,rng),
                correct_minus_similar_cluster=cluster_interval(obs-similar,reactants,rng),
                same_reactant_disjoint_correct=float(obs[valid].mean()),
                same_reactant_disjoint_swapped=float((swapped[valid]<=k).mean()),
                same_reactant_disjoint_difference=cluster_interval((obs-(swapped<=k))[valid],np.array(reactants)[valid],rng))
        results['models'][name]=stats
        for i in np.flatnonzero(valid):details.append(dict(model=name,reaction_id=qids[i],donor=qids[donors[i]],correct_rank=int(correct[i]),swapped_rank=int(swapped[i])))
    slices=[]
    for b in ['low','medium','high']:
        mask=(nov.bin==b).to_numpy()
        slices.append(dict(bin=b,n=int(mask.sum()),nn_candidate_queries=int((~transfer.abstains.to_numpy()&mask).sum()),
            nn_positive_queries=int(((transfer.supported_positives.to_numpy()>0)&mask).sum()),
            nn_top4=float(transfer.expected_top4.to_numpy()[mask].mean()),nn_top24=float(transfer.expected_top24.to_numpy()[mask].mean()),
            queries_with_training_positive_sequence=int(((nov.training_sequence_positive_count.to_numpy()>0)&mask).sum())))
    results['novelty_support']=slices
    results['similarity_sensitivity']=dict(full_reactant_low=int((nov.max_similarity<.5).sum()),largest_fragment_low=int((nov.largest_fragment_max_similarity<.5).sum()),
        changed_bin=int(sum(a!=('low' if b<.5 else 'medium' if b<.7 else 'high') for a,b in zip(nov.bin,nov.largest_fragment_max_similarity))))
    # Paired model changes within each fixed full-reactant stratum.
    raw=pd.read_csv(OUT/'prior_per_query.csv').pivot(index='reaction_id',columns='model',values='raw_rank').loc[qids]
    results['combined_minus_original_by_bin']={}
    for b in ['low','medium','high']:
        mask=(nov.bin==b).to_numpy()
        d=(raw.combined.to_numpy()<=24).astype(float)-(raw.original.to_numpy()<=24)
        results['combined_minus_original_by_bin'][b]=paired(d[mask],rng)
    results['notes']=['Post-primary-result robustness checks; no model choices changed.',
        'Cluster bootstrap groups exactly matching canonical reactants; other chemical and enzyme dependence remains.',
        'Score-tie sensitivity averages only model score ties; primary metrics preserve the released ranking rule.',
        'Label-disjoint swaps are a conditional sensitivity analysis, not proof of inactive cross-pairs.']
    pd.DataFrame(details).to_csv(OUT/'same_reactant_swaps.csv',index=False)
    save('robustness.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
