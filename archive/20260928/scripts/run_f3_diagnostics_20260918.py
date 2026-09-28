#!/usr/bin/env python3
"""Three fixed-checkpoint F3 diagnostics; CPU only, no training or inference."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem

ROOT = Path('/datastor2/deep-proteins/EnzymeDiscovery/horizyn')
BASE = ROOT/'runs/enzymecage_p450_reproduction_20260918'
PILOT = ROOT/'runs/enzymecage_f3_combined_20260918'
OUT = ROOT/'runs/f3_diagnostics_20260918'
sys.path.insert(0,str(PILOT))
from p450_protocol import DATA_SHA, KEYS, align_scores, apply_official_prior, digest, validate_panel, score_metrics
K = (4,14,24)


def save(name, data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def ordinal_ranks(scores):
    order=np.argsort(-scores,axis=1,kind='stable')
    ranks=np.empty_like(order)
    np.put_along_axis(ranks,order,np.broadcast_to(np.arange(1,scores.shape[1]+1),order.shape),axis=1)
    return ranks


def positive_ranks(scores, labels):
    ranks=ordinal_ranks(scores)
    if not labels.any(axis=1).all(): raise ValueError('Query without positive')
    return np.where(labels,ranks,scores.shape[1]+1).min(axis=1)


def metrics(best):
    best=np.asarray(best)
    return {**{f'top{k}':float(np.mean(best<=k)) for k in K},'mrr':float(np.mean(1/best))}


def transform(best, key):
    return 1/best if key=='mrr' else (best<=int(key[3:])).astype(float)


def paired(values, rng):
    values=np.asarray(values,dtype=float)
    if len(values)==0: return dict(n=0,mean=None,ci95=None)
    idx=rng.integers(0,len(values),(10000,len(values)))
    return dict(n=len(values),mean=float(values.mean()),ci95=np.quantile(values[idx].mean(axis=1),[.025,.975]).tolist())


def derangements(n, repeats, rng):
    result=[]
    while len(result)<repeats:
        p=rng.permutation(n)
        if np.all(p!=np.arange(n)): result.append(p)
    return np.array(result)


def transitions(before, after, k):
    a,b=before<=k,after<=k
    result=dict(both=int((a&b).sum()),lost=int((a&~b).sum()),gained=int((~a&b).sum()),neither=int((~a&~b).sum()))
    assert sum(result.values())==len(a)
    assert int(b.sum())-int(a.sum())==result['gained']-result['lost']
    return result


def expected_tie_hits(votes, positives, k):
    """Abstention-aware transfer: rank only enzymes with nonzero votes."""
    used=0; miss=1.0
    for value in sorted(set(votes[votes>0]),reverse=True):
        mask=votes==value; n=int(mask.sum()); p=int(positives[mask].sum())
        take=min(k-used,n)
        if take<=0: break
        if take>n-p: miss=0.0
        else: miss*=math.comb(n-p,take)/math.comb(n,take)
        used+=take
        if used>=k: break
    return 1-miss


def molecule(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f'Unparseable SMILES: {smiles}')
    return mol


def fingerprint(smiles, largest=False):
    reactants=smiles.split('>')[0]
    mol=Chem.MolFromSmiles(reactants)
    fallback=False
    if mol is None:
        mols=[molecule(x) for x in reactants.split('.')]
        fallback=True
    else: mols=list(Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True))
    if largest:
        mols=[max(mols,key=lambda m:(sum(a.GetAtomicNum()==6 for a in m.GetAtoms()),m.GetNumHeavyAtoms(),Chem.MolToSmiles(m)))]
    fp=None
    for m in mols:
        part=AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048,useChirality=False)
        fp=part if fp is None else fp|part
    return fp,fallback


def canonical_reaction(smiles):
    parts=smiles.split('>')
    if len(parts)!=3: raise ValueError('Not a reaction')
    def side(s):
        result=[]
        for part in s.split('.'):
            if not part: continue
            m=molecule(part)
            for a in m.GetAtoms(): a.SetAtomMapNum(0)
            result.append(Chem.MolToSmiles(m,isomericSmiles=True))
        return '.'.join(sorted(result))
    return side(parts[0])+'>>'+side(parts[2])


def fasta(path):
    key=None; seq=[]
    with path.open() as f:
        for line in f:
            if line.startswith('>'):
                if key is not None: yield key,''.join(seq).upper()
                key=line[1:].split()[0];seq=[]
            else: seq.append(line.strip())
        if key is not None: yield key,''.join(seq).upper()


class Contracts(unittest.TestCase):
    def test_swaps_detect_query_information(self):
        labels=np.eye(4,dtype=bool)
        scores=np.eye(4)
        self.assertTrue((positive_ranks(scores,labels)==1).all())
        self.assertTrue((positive_ranks(scores[[1,0,3,2]],labels)>1).all())
        invariant=np.tile(np.arange(4),(4,1))
        np.testing.assert_array_equal(positive_ranks(invariant,labels),positive_ranks(invariant[::-1],labels))
    def test_prior_transition_accounting(self):
        self.assertEqual(transitions(np.array([1,1,7,7]),np.array([1,7,1,7]),4),dict(both=1,lost=1,gained=1,neither=1))
    def test_derangement(self):
        p=derangements(8,100,np.random.default_rng(42))
        self.assertTrue((p!=np.arange(8)).all())
        self.assertTrue((np.sort(p,axis=1)==np.arange(8)).all())
    def test_transfer_ties_and_abstention(self):
        self.assertEqual(expected_tie_hits(np.zeros(4),np.array([1,0,0,0],bool),2),0.)
        self.assertAlmostEqual(expected_tie_hits(np.ones(4),np.array([1,0,0,0],bool),2),.5)
        self.assertEqual(expected_tie_hits(np.array([0,1,1,0]),np.array([1,0,0,0],bool),4),0.)
    def test_fingerprints_ignore_product(self):
        a,_=fingerprint('CCO>>CC=O');b,_=fingerprint('CCO>>CC')
        self.assertEqual(DataStructs.TanimotoSimilarity(a,b),1.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-only',action='store_true')
    args=parser.parse_args()
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not tests.wasSuccessful(): raise RuntimeError('Contract tests failed')
    if args.test_only:return
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists(): raise FileExistsError('Completed output already exists')
    # Prespecified choices are persisted before calculating diagnostic results.
    protocol=dict(seed=42,random_derangements=10000,bootstrap_replicates=10000,cutoffs=list(K),
        novelty_bins=['low: <0.5','medium: >=0.5 and <0.7','high: >=0.7'],
        fingerprint='RDKit Morgan radius2 2048bit, chirality off; all reactant components OR-combined; products excluded',
        similarity_sensitivity='Largest fragment by carbon count, then heavy atoms; not necessarily the substrate',
        similar_swap='Nearest different canonical reaction in the same panel, chosen by reactant similarity without labels; ties by reaction ID',
        disjoint_swap='Sensitivity only: exclude donors sharing any listed positive with recipient',
        nn_transfer='Top10 training reactions, enzyme frequency votes; exact sequence mapping into same490 pool; zero votes abstain; boundary ties averaged exactly',
        scope='Three saved F3 score matrices, no retraining, no new model inference; raw swaps isolate model from prior',
        inference='Exploratory diagnostics; bootstrap assumes independent queries; no seed uncertainty or multiplicity adjustment')
    save('protocol.json',protocol)
    panel_path=BASE/'data/test_P450.csv';assert digest(panel_path)==DATA_SHA
    panel=pd.read_csv(panel_path);counts=validate_panel(panel,released=True)
    mapping=pd.read_csv(BASE/'f3_features/reaction_mapping.csv')
    qids=mapping.reaction_id.tolist(); rxns=mapping.raw_reaction.tolist()
    pids=panel.UniprotID.drop_duplicates().tolist(); qi={q:i for i,q in enumerate(rxns)};pi={p:i for i,p in enumerate(pids)}
    assert len(set(rxns))==191 and len(pids)==490
    labels=np.zeros((191,490),bool)
    for q,p,y in panel[KEYS+['Label']].itertuples(index=False,name=None):labels[qi[q],pi[p]]=bool(y)
    prior_path=BASE/'data/corr_score_map.pkl'
    assert digest(prior_path)==json.loads((BASE/'prior.receipt.json').read_text())['prior_sha256']
    with prior_path.open('rb') as f: prior=pickle.load(f)
    sources={'panel':dict(path=str(panel_path),sha256=digest(panel_path)),'prior':dict(path=str(prior_path),sha256=digest(prior_path))}
    paths={'original':BASE/'scores/f3_enzymecage_seed42_epoch06.csv',
           'control':PILOT/'p450/control_scores.csv','combined':PILOT/'p450/combined_scores.csv'}
    expected_pilot=json.loads((PILOT/'p450/comparison.json').read_text())['results']
    original=pd.read_csv(BASE/'comparison.csv')
    matrices={}; raw_ranks={};prior_ranks={}; summary=dict(counts=counts,models={})
    prior_rows=[]; positive_rows=[]
    for name,path in paths.items():
        receipt_path=path.with_suffix('.receipt.json') if name=='original' else path.with_name(name+'_receipt.json')
        receipt=json.loads(receipt_path.read_text());assert digest(path)==receipt['scores_sha256']
        sources[name]=dict(path=str(path),sha256=digest(path),checkpoint_sha256=receipt['checkpoint_sha256'])
        frame=align_scores(panel,pd.read_csv(path)); mat=np.zeros((191,490))
        for q,p,s in frame[KEYS+['pred']].itertuples(index=False,name=None):mat[qi[q],pi[p]]=s
        matrices[name]=mat;raw=positive_ranks(mat,labels);raw_ranks[name]=raw
        gated=apply_official_prior(frame,prior); after=np.zeros(191,int);retained=np.zeros(191,int)
        raw_all=ordinal_ranks(mat)
        for q,g in gated.groupby(KEYS[0]):
            i=qi[q];ranking=sorted(zip(g.UniprotID,g.pred),key=lambda x:x[1],reverse=True)
            ranks={p:j for j,(p,_) in enumerate(ranking,1)}
            after[i]=min(ranks[p] for p in pids if labels[i,pi[p]])
            retained[i]=int(((g.pred>0)&(g.Label==1)).sum())
            for p in pids:
                if labels[i,pi[p]]:
                    row=g[g.UniprotID==p].iloc[0]
                    positive_rows.append(dict(model=name,reaction_id=qids[i],enzyme=p,raw_rank=int(raw_all[i,pi[p]]),
                        prior_rank=ranks[p],retained=bool(row.pred>0)))
            prior_rows.append(dict(model=name,reaction_id=qids[i],raw_rank=int(raw[i]),prior_rank=int(after[i]),
                known_positives=int(labels[i].sum()),retained_positives=int(retained[i]),
                **{f'top{k}_change':int(after[i]<=k)-int(raw[i]<=k) for k in K}))
        prior_ranks[name]=after
        for mode, data,best in [('raw',frame,raw),('official_prior',gated,after)]:
            official=score_metrics(data,panel,ROOT/'.deps/enzymecage_p450_official')
            if name=='original':
                previous=original[(original.method=='f3_enzymecage_epoch06')&(original.evaluation==('raw_diagnostic' if mode=='raw' else mode))].iloc[0]
            else:previous=next(x for x in expected_pilot if x['arm']==name and x['evaluation']==mode)
            for k,label in [(4,'Top 1.0%'),(14,'Top 3.0%'),(24,'Top 5.0%')]:
                assert abs(float(np.mean(best<=k))-official[label])<1e-12
                assert abs(official[label]-previous[label])<1e-12
        summary['models'][name]=dict(raw=metrics(raw),official_prior=metrics(after),
            prior_transitions={str(k):transitions(raw,after,k) for k in K},
            all_positives_zeroed_queries=int((retained==0).sum()),retained_positive_pairs=int(retained.sum()),
            raw_score_tied_queries=int(sum(len(np.unique(row))<490 for row in mat)))
        print('PRIOR',name,json.dumps(summary['models'][name]),flush=True)
    pd.DataFrame(prior_rows).to_csv(OUT/'prior_per_query.csv',index=False)
    pd.DataFrame(positive_rows).to_csv(OUT/'prior_per_positive.csv',index=False)
    save('prior_summary.json',summary)
    # Reaction chemistry and training support; independent of model scores.
    train_dir=ROOT/'runs/enzymecage_f3_seed42/data'
    pairs=pd.read_csv(train_dir/'train_pairs.csv');train=pd.read_csv(train_dir/'normalized/train_rxns.csv')
    train=train[train.reaction_id.isin(set(pairs.reaction_id))].sort_values('reaction_id').reset_index(drop=True)
    assert set(train.reaction_id)==set(pairs.reaction_id) and not train.reaction_id.duplicated().any()
    fingerprints={};novelty={}; panel_similarity={};fall={}
    for mode in ['full_reactant','largest_fragment']:
        largest=mode=='largest_fragment'
        tq=[fingerprint(s,largest) for s in train.reaction_smiles]
        pq=[fingerprint(s,largest) for s in rxns]
        fall[mode]=dict(training=sum(x[1] for x in tq),panel=sum(x[1] for x in pq))
        t=[x[0] for x in tq];p=[x[0] for x in pq]
        sim=np.array([DataStructs.BulkTanimotoSimilarity(x,t) for x in p])
        np.save(OUT/f'{mode}_train_similarity.npy',sim)
        novelty[mode]=sim
        panel_similarity[mode]=np.array([DataStructs.BulkTanimotoSimilarity(x,p) for x in p])
        assert np.allclose(np.diag(panel_similarity[mode]),1.)
    canonical=[canonical_reaction(s) for s in rxns]
    traincanonical=[canonical_reaction(s) for s in train.reaction_smiles]
    exact=defaultdict(list)
    for i,c in enumerate(traincanonical): exact[c].append(i)
    seq_to_panel=defaultdict(list)
    for p,s in panel[['UniprotID','sequence']].drop_duplicates().itertuples(index=False,name=None):seq_to_panel[s.upper()].append(p)
    trainids=set(pairs.protein_id); protein_map={}
    for p,s in fasta(train_dir/'proteins.fasta'):
        if p in trainids and s in seq_to_panel:protein_map[p]=seq_to_panel[s]
    associations=defaultdict(set)
    for q,p in pairs[['reaction_id','protein_id']].itertuples(index=False,name=None):
        associations[q].update(protein_map.get(p,[]))
    candidate_overlap=sorted({x for v in protein_map.values() for x in v})
    novelty_rows=[];transfer_rows=[];votes_all=[]
    for i,q in enumerate(qids):
        sim=novelty['full_reactant'][i];order=np.argsort(-sim,kind='stable');near=order[:10]
        votes=np.zeros(490,int)
        for j in near:
            for p in associations[train.reaction_id.iloc[j]]:votes[pi[p]]+=1
        votes_all.append(votes)
        supported_positive=int(((votes>0)&labels[i]).sum())
        exact_enzymes=set().union(*(associations[train.reaction_id.iloc[j]] for j in exact.get(canonical[i],[])))
        exact_positive=sum(labels[i,pi[p]] for p in exact_enzymes)
        row=dict(reaction_id=q,max_similarity=float(sim[order[0]]),nearest_training_reaction=train.reaction_id.iloc[order[0]],
            largest_fragment_max_similarity=float(novelty['largest_fragment'][i].max()),
            bin='low' if sim.max()<.5 else ('medium' if sim.max()<.7 else 'high'),
            exact_training_reactions=len(exact.get(canonical[i],[])),exact_known_positive_pairs=int(exact_positive),
            training_sequence_positive_count=sum(p in candidate_overlap for p in np.array(pids)[labels[i]]))
        novelty_rows.append(row)
        transfer_rows.append(dict(reaction_id=q,neighbors=10,supported_candidates=int((votes>0).sum()),
            supported_positives=supported_positive,abstains=not bool((votes>0).any()),
            **{f'expected_top{k}':expected_tie_hits(votes,labels[i],k) for k in K}))
    pd.DataFrame(novelty_rows).to_csv(OUT/'reaction_novelty.csv',index=False)
    pd.DataFrame(transfer_rows).to_csv(OUT/'nearest_neighbor_transfer.csv',index=False)
    save('sequence_overlap.json',dict(training_proteins=len(trainids),panel_enzymes=490,
        exact_sequence_mapped_training_ids=len(protein_map),panel_enzymes_seen_in_training=len(candidate_overlap),
        candidate_ids=candidate_overlap,exact_training_positive_pairs=sum(r['exact_known_positive_pairs'] for r in novelty_rows),
        note='Sequence identity mapping only; no sequence-similarity or homology transfer is invented.'))
    summary['chemistry']=dict(training_reactions=len(train),fallbacks=fall,
        bins={b:sum(r['bin']==b for r in novelty_rows) for b in ['low','medium','high']},
        exact_training_reaction_queries=sum(r['exact_training_reactions']>0 for r in novelty_rows),
        exact_training_positive_pairs=sum(r['exact_known_positive_pairs'] for r in novelty_rows),
        panel_enzymes_seen_in_training=len(candidate_overlap))
    summary['nearest_neighbor']=dict(queries_with_candidates=sum(not r['abstains'] for r in transfer_rows),
        queries_with_supported_positive=sum(r['supported_positives']>0 for r in transfer_rows),
        **{f'expected_top{k}':float(np.mean([r[f'expected_top{k}'] for r in transfer_rows])) for k in K})
    print('CHEMISTRY',json.dumps(summary['chemistry']),flush=True)
    print('TRANSFER',json.dumps(summary['nearest_neighbor']),flush=True)
    rng=np.random.default_rng(42); permutations=derangements(191,10000,rng)
    save('randomization.json',dict(derangements_sha256=hashlib.sha256(permutations.tobytes()).hexdigest(),n=10000,seed=42))
    swap_rows=[];novelty_metrics=[]
    diff_canonical=np.array([[a!=b for b in canonical] for a in canonical])
    disjoint=(labels.astype(int)@labels.astype(int).T)==0
    valid=diff_canonical&~np.eye(191,dtype=bool)
    # Donors are chosen from structure alone. Label-disjoint donors are secondary.
    donors={}
    for mode, sim in panel_similarity.items():
        donors[mode]=np.argmax(np.where(valid,sim,-np.inf),axis=1)
    donors['disjoint']=np.argmax(np.where(valid&disjoint,panel_similarity['full_reactant'],-np.inf),axis=1)
    assert (valid&disjoint).any(axis=1).all()
    for name,mat in matrices.items():
        ranks=ordinal_ranks(mat)
        # best[i,j] evaluates donor reaction j's ordering using recipient i's positives.
        best=np.array([ranks[:,labels[i]].min(axis=1) for i in range(191)])
        correct=best.diagonal();np.testing.assert_array_equal(correct,raw_ranks[name])
        random_values=best[np.arange(191)[None,:],permutations]
        common=(mat.sum(axis=0)[None,:]-mat)/(len(mat)-1)
        common_best=positive_ranks(common,labels)
        comparisons={}; donor=donors['full_reactant']; similar=best[np.arange(191),donor]
        # Pairwise top24 overlap is descriptive; score correlation can be high while ranks matter.
        top=ranks<=24;inter=top.astype(int)@top.astype(int).T
        jaccard=inter/(48-inter);offdiag=~np.eye(191,dtype=bool)
        rankcorr=np.corrcoef(ranks)
        for key in ['top4','top14','top24','mrr']:
            obs=transform(correct,key);random_metric=transform(random_values,key).mean(axis=1)
            counterfactual=transform(best,key)
            random_query_mean=(counterfactual.sum(axis=1)-obs)/190
            sd=donors['disjoint'];sens=best[np.arange(191),sd]
            comparisons[key]=dict(correct=float(obs.mean()),random_mean=float(random_metric.mean()),
                random_reference95=np.quantile(random_metric,[.025,.975]).tolist(),
                randomization_p_ge_correct=float((1+(random_metric>=obs.mean()).sum())/(len(random_metric)+1)),
                correct_minus_random_mean=paired(obs-random_query_mean,rng),
                similar=float(transform(similar,key).mean()),correct_minus_similar=paired(obs-transform(similar,key),rng),
                similar_disjoint=float(transform(sens,key).mean()),
                largest_fragment_similar=float(transform(best[np.arange(191),donors['largest_fragment']],key).mean()),
                common_enzyme_preference=float(transform(common_best,key).mean()),
                correct_minus_common=paired(obs-transform(common_best,key),rng))
        summary['models'][name]['swaps']=dict(metrics=comparisons,
            mean_pairwise_top24_jaccard=float(jaccard[offdiag].mean()),mean_pairwise_rank_correlation=float(rankcorr[offdiag].mean()),
            nearest_donor_similarity_quantiles=np.quantile(panel_similarity['full_reactant'][np.arange(191),donor],[0,.25,.5,.75,1]).tolist(),
            similar_donors_sharing_positive=int((~disjoint[np.arange(191),donor]).sum()),
            similar_donors_at_least_point5=int((panel_similarity['full_reactant'][np.arange(191),donor]>=.5).sum()))
        for i in range(191):
            swap_rows.append(dict(model=name,reaction_id=qids[i],correct_rank=int(correct[i]),similar_donor=qids[donor[i]],
                similar_rank=int(similar[i]),similarity=float(panel_similarity['full_reactant'][i,donor[i]]),
                similar_shares_positive=bool(not disjoint[i,donor[i]]),
                disjoint_donor=qids[donors['disjoint'][i]],disjoint_rank=int(best[i,donors['disjoint'][i]]),
                common_preference_rank=int(common_best[i]),random_mean_top24=float((np.delete(best[i],i)<=24).mean()),
                random_mean_mrr=float((1/np.delete(best[i],i)).mean())))
        for mode,sim in novelty.items():
            maximum=sim.max(axis=1)
            for b,mask in [('low',maximum<.5),('medium',(maximum>=.5)&(maximum<.7)),('high',maximum>=.7)]:
                for evaluation,br in [('raw',raw_ranks[name]),('official_prior',prior_ranks[name])]:
                    novelty_metrics.append(dict(model=name,fingerprint=mode,bin=b,evaluation=evaluation,n=int(mask.sum()),
                        **(metrics(br[mask]) if mask.any() else {**{f'top{k}':None for k in K},'mrr':None})))
        print('SWAPS',name,json.dumps(comparisons),flush=True)
    pd.DataFrame(swap_rows).to_csv(OUT/'swaps_per_query.csv',index=False)
    pd.DataFrame(novelty_metrics).to_csv(OUT/'novelty_metrics.csv',index=False)
    summary['novelty_metrics']=novelty_metrics
    for f in [train_dir/'train_pairs.csv',train_dir/'normalized/train_rxns.csv',train_dir/'proteins.fasta']:
        sources[f.name]=dict(path=str(f),sha256=digest(f))
    save('sources.json',sources)
    summary['complete']=True;summary['checks']=dict(contract_tests=tests.testsRun,official_metric_parity=True,
        full_candidate_coverage=True,prior_transition_reconciliation=True,versions=dict(rdkit=rdBase.rdkitVersion,pandas=pd.__version__,numpy=np.__version__))
    save('summary.json',summary)
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__': main()
