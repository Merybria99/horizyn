#!/usr/bin/env python3
"""Independent spot checks of geometry summaries and source provenance."""
import argparse
import csv
from collections import defaultdict
import json
import numpy as np

from cersei_embedding_organization import OUT,SPLITS,ROOT,read,dump,sha,prefix,norm
from cersei_embedding_metrics import summarize


def table(path):
    with path.open() as f:return list(csv.DictReader(f))


def macro(values,labels):
    byclass=defaultdict(list)
    for v,labs in zip(values,labels):
        if np.isfinite(v):
            for lab in labs:byclass[lab].append(v)
    return np.mean([np.mean(v) for v in byclass.values()])


def main():
    p=argparse.ArgumentParser();p.add_argument('--include-views',action='store_true');a=p.parse_args()
    checks=[];fam=read(OUT/'homology/families.json')
    for split in SPLITS:
        d=OUT/split;meta=read(d/'metadata.json');sub=read(d/'subsets.json');z=np.load(d/'embeddings.npz')
        lineage=read(d/'lineage.json');m=lineage['task']['model']
        assert sha(m['checkpoint'])==m['checkpoint_sha256']
        assert sha(m['phase2_checkpoint'])==m['phase2_checkpoint_sha256']
        assert sha(d/'embeddings.npz')==read(d/'prepared.json')['embeddings_sha256']
        assert len(sub['protein_indices'])<=2000 and len(sub['reaction_indices'])<=500
        for key in ('phase1_e','phase1_r','refined_e','refined_r'):
            x=z[key];assert np.isfinite(x).all();assert np.max(np.abs(np.linalg.norm(x,axis=1)-1))<2e-6
        bank=np.array(meta['neighborhood_indices']);idx={int(g):i for i,g in enumerate(bank)}
        nn=np.load(d/'analysis1/neighbors_ec3_refined_exclude1.npy')
        records=[r for r in table(d/'analysis1/per_protein.csv') if r['stage']=='refined' and r['ec_level']=='3' and r['k']=='10' and r['exclude_homologs']=='True']
        ecs=[set(prefix(meta['enzyme_ec'][i],3)) for i in bank];lookup={p:i for i,p in enumerate(meta['protein_ids'])}
        for row in records[:40]:
            qi=idx[lookup[row['protein_id']]];neighbors=nn[qi,:10]
            assert qi not in neighbors and np.all(neighbors>=0)
            assert all(fam[meta['protein_ids'][bank[j]]][1]!=fam[row['protein_id']][1] for j in neighbors)
            assert abs(np.mean([bool(ecs[qi]&ecs[j]) for j in neighbors])-float(row['agreement']))<1e-12
            # Check saved neighbors against independent NumPy dot products.
            scores=norm(z['refined_e'][bank]) @ norm(z['refined_e'][bank[qi]:bank[qi]+1])[0]
            eligible=np.array([bool(e) and fam[meta['protein_ids'][bank[j]]][1]!=fam[row['protein_id']][1] for j,e in enumerate(ecs)])
            cut=np.sort(scores[eligible])[-10]
            assert np.min(scores[neighbors])>=cut-3e-6
        ys=np.array([float(r['agreement']) for r in records]);labels=[prefix(meta['enzyme_ec'][lookup[r['protein_id']]],3) for r in records]
        official=next(r for r in read(d/'analysis1/summary.json') if r['stage']=='refined' and r['ec_level']==3 and r['k']==10 and r['exclude_homologs'])
        assert abs(macro(ys,labels)-official['macro'])<1e-12
        truth=np.load(d/'truth.npy');pos=defaultdict(list)
        for r,e in truth:pos[int(r)].append(int(e))
        scores=np.load(meta['final_scores'],mmap_mode='r')
        rrecords=[r for r in table(d/'analysis2/per_query_R2E.csv') if r['stage']=='final_score']
        for row in rrecords[:20]:
            qi=int(row['query_index']);line=np.asarray(scores[qi]);ps=line[pos[qi]]
            unknown=np.delete(line,pos[qi]);assert abs(float(ps.mean()-unknown.max())-float(row['hardest_margin']))<1e-7
            # Independent rank-based treatment of ties, rather than the
            # implementation's kth-score threshold calculation.
            for k in (10,50):
                probabilities=[max(0,min(1,(k-int(np.sum(line>s)))/int(np.sum(line==s)))) for s in ps]
                assert abs(np.mean(probabilities)-float(row[f'coverage{k}']))<1e-12
        if a.include_views:
            for k in (1,2,4,8):
                vd=d/f'learned_views/k{k}';receipt=read(vd/'export_receipt.json');complete=read(vd/'complete.json')
                assert receipt['cache_reproduction_max_abs_error']<3e-6
                assert sha(receipt['task']['model']['checkpoint'])==receipt['checkpoint_sha256']
                assert sha(receipt['task']['head_path'])==receipt['head_sha256']
                assert sha(vd/'export.npz')==receipt['export_sha256']
                vz=np.load(vd/'export.npz');assert np.array_equal(vz['selected'],sub['protein_indices'])
                assert np.isfinite(vz['embeddings']).all()
                if k==1:assert np.max(np.abs(vz['embeddings'][:,1]-vz['embeddings'][:,2]))==0
                att=table(vd/'per_protein_attention.csv')
                assert all(abs(sum(float(r[x]) for x in ['global_gate_mass','sleec_gate_mass','learned_gate_mass'])-1)<2e-6 for r in att)
                checks.append(dict(split=split,K=k,export_reproduced=True,proteins=complete['queries']))
        checks.append(dict(split=split,checkpoint_immutable=True,norms_valid=True,neighborhood_spot_checks=min(40,len(records)),
            class_macro_recomputed=True,alignment_spot_checks=20,annotation_coverage=read(d/'prepared.json')['protein_ec_coverage']))
    # Independently reproduce a small multilabel family bootstrap, including
    # a family spanning two EC classes and repeated members of one family.
    y=np.array([.1,.4,.9,.3,.8]);labs=[['a'],['a','b'],['b'],['c'],['b','c']];ff=np.array([0,0,1,2,2])
    result=summarize(y,labs,ff,replicates=100)
    rng=np.random.default_rng(23092026);boot=[]
    for counts in rng.multinomial(3,[1/3]*3,size=100):
        ii=[i for i in range(len(y)) for _ in range(counts[ff[i]])]
        boot.append(macro(y[ii],[labs[i] for i in ii]))
    assert np.allclose(np.quantile(boot,[.025,.975]),result['ci95'])
    coverage=read(OUT/'enzymemap/analysis3/quality_matching_coverage.json');assert coverage['max_tanimoto_gap']<=.02000001
    report=dict(status='Share with caveats',include_views=a.include_views,passed=True,checks=checks,
        independent_multilabel_family_bootstrap_check=True,
        caveats=['One fit per target/K; intervals resample query families with fixed candidate banks, not training seeds.',
            'In percentile replicates, absent annotation classes are omitted; sparse-class intervals are exploratory.',
            'EnzymeMap EC3 annotations cover 720/1357 test-positive unique sequences in the native ec2uniprot source; unknown labels are excluded from macro EC summaries.',
            'Protein families and homology exclusions are operational MMseqs connected components, not known evolutionary families.',
            'ReactZyme reaction EC classes are a proxy derived from known positive proteins; directed transformation analysis is unavailable.',
            'Unannotated pairs are not verified inactive pairs. Positive coverage measures observed labels only.',
            'K fits are separate matched-seed refits; they do not replace the primary benchmark checkpoints.',
            'Repeated access to these benchmarks makes this exploratory evidence, not an untouched confirmatory test.'],
        incomplete=['Learned-view package not yet included in this validation'] if not a.include_views else [],
        code_sha256=sha(__file__))
    dump(OUT/('validation.json' if a.include_views else 'main_validation.json'),report)
    print('VALIDATION PASSED',len(checks),'records',flush=True)


if __name__=='__main__':main()
