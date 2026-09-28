"""Native EnzymeCAGE score/profile diagnostics; no invented shared latent space."""
from collections import defaultdict
import csv
import numpy as np
import torch
from cersei_embedding_organization import ROOT,read,dump,sha,fingerprints,tanimoto,components,SEED
from cersei_embedding_metrics import nearest,neighbor_agreement,summarize,query_metrics,write_table
from cersei_public_embedding_stats import summarize
OUT=ROOT/'runs/public_embedding_comparison_20260924/enzymecage_p450'

def main():
    torch.set_num_threads(4);m=read(OUT/'metadata.json');z=np.load(OUT/'embeddings.npz');pids=m['protein_ids'];pi={p:i for i,p in enumerate(pids)}
    edges30=[];edges50=[]
    with (OUT/'homology.tsv').open() as f:
        for line in f:
            a,b,ident,qcov,tcov,evalue=line.rstrip().split('\t')
            if a==b or float(qcov)<.8 or float(tcov)<.8 or float(evalue)>1e-3:continue
            if float(ident)>=.3:edges30.append((pi[a],pi[b]))
            if float(ident)>=.5:edges50.append((pi[a],pi[b]))
    fam=components(len(pids),edges30);fam50=components(len(pids),edges50)
    truth=np.load(OUT/'truth.npy');labels=[[] for _ in pids]
    for r,e in truth:labels[e].append(str(r))
    valid=np.array([bool(x) for x in labels]);summaries=[];records=[];yy={}
    for exclude in [False,True]:
        for model,key in [('EnzymeCAGE','enzyme'),('ESMC pocket','esm_pocket')]:
            nn=nearest(z[key],valid,fam50 if exclude else None,'cuda:1',k=50)
            for k in [10,50]:
                y=neighbor_agreement(nn,labels,k);yy[(model,exclude,k)]=y
                summaries.append(dict(model=model,k=k,exclude_homologs=exclude,**summarize(y,labels,fam)))
                records.extend(dict(model=model,k=k,exclude_homologs=exclude,protein_id=pids[i],value=float(v)) for i,v in enumerate(y) if np.isfinite(v))
        for k in [10,50]:summaries.append(dict(model='EnzymeCAGE-ESMC pocket',k=k,exclude_homologs=exclude,**summarize(yy[('EnzymeCAGE',exclude,k)]-yy[('ESMC pocket',exclude,k)],labels,fam)))
    dump(OUT/'neighborhoods_summary.json',summaries);write_table(OUT/'neighborhoods_per_protein.csv',records)
    bits,validr=fingerprints(m['reaction_smiles']);chem=tanimoto(bits);rfam=components(len(bits),np.argwhere(np.triu(chem>=.7,1)))
    scores=np.load(OUT/'scores.npy');ss=[]
    for direction in ['R2E','E2R']:
        pos=defaultdict(list)
        for r,e in truth:pos[int(r if direction=='R2E' else e)].append(int(e if direction=='R2E' else r))
        qq=sorted(pos);nc=scores.shape[1 if direction=='R2E' else 0];rr=[]
        for q in qq:
            unknown=np.setdiff1d(np.arange(nc),pos[q]);rng=np.random.default_rng(SEED+q)
            y=query_metrics(scores[q] if direction=='R2E' else scores[:,q],pos[q],rng.choice(unknown,min(64,len(unknown)),replace=False))
            y['positive_score']=y.pop('positive_cosine');rr.append(dict(query_index=q,direction=direction,**y))
        for metric in ['positive_score','hardest_margin','random_margin','coverage10','coverage50']:
            ss.append(dict(direction=direction,metric=metric,**summarize([r[metric] for r in rr],[['all'] for _ in qq],(rfam if direction=='R2E' else fam)[qq])))
        write_table(OUT/f'alignment_{direction}_per_query.csv',rr)
    dump(OUT/'alignment_summary.json',ss)
    train=ROOT.parent/'EnzymeCAGE/dataset/training/train.csv';sequences=set(m['protein_sequences']);seen=set()
    test_positive={(m['reaction_smiles'][r],m['protein_sequences'][e]) for r,e in truth}
    seen_positive=set();seen_negative=set()
    with train.open() as f:
        for row in csv.DictReader(f):
            if row['sequence'] in sequences:seen.add(row['sequence'])
            key=(row['CANO_RXN_SMILES'],row['sequence'])
            if key in test_positive:
                (seen_positive if float(row['Label'])>0 else seen_negative).add(key)
    dump(OUT/'coverage.json',dict(proteins=490,reactions=191,positive_edges=len(truth),positive_enzymes=int(valid.sum()),positive_reactions=len(set(truth[:,0])),
        exact_training_sequence_overlap=sum(s in seen for s in m['protein_sequences']),training_sha256=sha(train),
        exact_test_positive_pairs_seen_as_training_positive=len(seen_positive),exact_test_positive_pairs_seen_as_training_negative=len(seen_negative),
        association_overlap_definition='Exact sequence and canonical reaction string; does not establish absence of chemically equivalent aliases',
        protein_families30=len(set(fam)),protein_families50=len(set(fam50)),reaction_families=len(set(rfam)),
        family_definition='MMseqs detected >=30/50% identity, >=80% both-side coverage, e<=1e-3; reaction Morgan Tanimoto>=.7 components',
        profile_definition='Share at least one listed positive reaction; unannotated proteins excluded from profile neighbor bank',
        scores='Native pair sigmoid scores, not cosine; no prior filtering; fixed released pretrained seed42'))
    print('ENZYMECAGE P450 ANALYSIS COMPLETE',flush=True)

if __name__=='__main__':main()
