#!/usr/bin/env python3
"""Independent aggregation, exclusion and exact-tie checks for functional recovery."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from cersei_crossmodal_functional_recovery import OUT,SPLITS,KS,read,dump,sha

def main(selected=None):
    records=[];family_path=OUT/'homology/families.json'
    families=read(family_path) if family_path.exists() else None
    for split in SPLITS:
        c=read(OUT/split/'cohort.json');candidate_ec=[set(x) for x in c['candidate_ec3']]
        for exclusion in ['partners','components']:
            if selected is not None and exclusion!=selected:continue
            root=OUT/split/exclusion
            if not (root/'complete.json').exists():continue
            complete=read(root/'complete.json')
            with (root/'curves.csv').open() as f:summary=list(csv.DictReader(f))
            with (root/'query_scope.csv').open() as f:scope=list(csv.DictReader(f))
            for model,source in complete['sources'].items():
                assert sha(source['scores'])==source['sha256']
                z=np.load(root/f'{model}_per_query.npz');q=z['query_indices'];y=z['agreement'];nn=z['stable_neighbors']
                assert len(q)==complete['scope']['queries'] and y.shape==(len(q),50)
                assert q.tolist()==[int(x['query_index']) for x in scope]
                assert np.all(np.isfinite(y)) and np.all((y>=0)&(y<=1+1e-12))
                increments=np.diff(np.c_[np.zeros(len(y)),y*KS],axis=1)
                assert increments.min()>-1e-10 and increments.max()<1+1e-10
                query_ec=[set(c['reaction_ec3'][i]) for i in q]
                classes=sorted(set.union(*query_ec));means=[]
                for label in classes:means.append(y[[label in ls for ls in query_ec]].mean(0))
                macro=np.mean(means,axis=0)
                reported=sorted([r for r in summary if r['model']==model],key=lambda r:int(r['k']))
                assert len(reported)==50
                aggregate_error=float(np.max(np.abs(macro-np.asarray([float(r['macro']) for r in reported]))))
                assert aggregate_error<1e-12,(split,exclusion,model,aggregate_error)
                for i,(index,neighbors) in enumerate(zip(q,nn)):
                    assert len(set(neighbors))==50
                    assert all(candidate_ec[j] for j in neighbors)
                    assert not (set(neighbors)&set(c['blocked_partner_indices'][index]))
                    pkeys=set(c['partner_sequence_keys'][index])
                    assert not pkeys.intersection(c['candidate_sequence_keys'][j] for j in neighbors)
                    if exclusion=='components':
                        assert not c['partners_missing_sequences'][index]
                        pf={families[p] for p in pkeys}
                        assert not pf.intersection(families[c['candidate_sequence_keys'][j]] for j in neighbors)
                # A separate full-sort/tie-block computation on fixed hash-selected queries.
                spot_queries=sorted(range(len(q)),key=lambda i:hashlib.sha256(c['reaction_ids'][q[i]].encode()).hexdigest())[:7]
                scores=np.load(source['scores'],mmap_mode='r');ranking_error=0.
                for qi in spot_queries:
                    index=q[qi];valid=np.asarray([bool(x) for x in candidate_ec]);valid[c['blocked_partner_indices'][index]]=False
                    if exclusion=='components':
                        pf={families[p] for p in c['partner_sequence_keys'][index]}
                        valid &= np.asarray([families.get(p,-1) not in pf for p in c['candidate_sequence_keys']])
                    candidates=np.flatnonzero(valid);values=np.asarray(scores[index,candidates])
                    hits=np.asarray([bool(query_ec[qi]&candidate_ec[j]) for j in candidates])
                    assert len(candidates)==int(scope[qi]['eligible_candidates'])
                    assert int(hits.sum())==int(scope[qi]['compatible_candidates'])
                    order=np.argsort(-values,kind='stable')
                    assert np.array_equal(candidates[order[:50]],nn[qi])
                    for k in [1,10,50]:
                        cutoff=values[order[k-1]];above=values>cutoff;tied=values==cutoff
                        expected=(hits[above].sum()+(k-above.sum())*hits[tied].mean())/k
                        ranking_error=max(ranking_error,abs(float(expected)-float(y[qi,k-1])))
                    assert ranking_error<1e-12
                records.append(dict(split=split,exclusion=exclusion,model=model,queries=len(q),classes=len(classes),
                    independent_macro_error=aggregate_error,full_sort_spotcheck_queries=len(spot_queries),independent_tie_error=ranking_error,
                    all_saved_neighbors_annotated=True,all_saved_neighbors_pass_exclusions=True))
                print(split,exclusion,model,'PASS',flush=True)
    if selected is not None and (OUT/'validation.json').exists():
        records=[r for r in read(OUT/'validation.json')['records'] if r['exclusion']!=selected]+records
    dump(OUT/'validation.json',dict(passed=True,completed_panels=len({(r['split'],r['exclusion']) for r in records}),
        all_eight_panels_complete=len({(r['split'],r['exclusion']) for r in records})==8,
        records=records,checker_sha256=sha(__file__),algorithm_checks=read(OUT/'algorithm_checks.json')))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--exclusion',choices=['partners','components']);a=p.parse_args();main(a.exclusion)
