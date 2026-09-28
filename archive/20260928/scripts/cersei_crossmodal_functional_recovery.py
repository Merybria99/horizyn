#!/usr/bin/env python3
"""Frozen-checkpoint EC3 recovery after removing all locally recorded partners.

Prepare fixes labels, eligible banks and comparisons before reading scores.
No model parameters or training jobs are changed by this analysis.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'runs/cersei_crossmodal_functional_recovery_20260925'
OLD = ROOT/'runs/cersei_embedding_organization_20260923_v1'
PUBLIC = ROOT/'runs/public_embedding_comparison_20260924'
RUN = ROOT/'runs/generalization_20260919_2251'
CROSS = RUN/'cross_paper_retraining'
COEFF = ROOT/'runs/cersei_dictionary_free_sensitivity_20260924'
CURRENT = ROOT/'runs/cersei_horizyn_challenge_20260923/no_dictionary_validation_20260923/reference_shared_test'
SPLITS = ['time','enzyme_smi','reaction_smi','enzymemap']
KS = np.arange(1,51)
SEED = 25092026

def read(p): return json.loads(Path(p).read_text())
def stamp(): return datetime.now(timezone.utc).isoformat()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def rows(p,sep=','):
    with Path(p).open() as f:return list(csv.DictReader(f,delimiter=sep))
def prefix(xs):
    return sorted({'.'.join(x.split('.')[:3]) for x in xs
        if len(x.split('.'))>=3 and all(v.isdigit() for v in x.split('.')[:3])})
def seqkey(s):return 's_'+hashlib.sha256(s.encode()).hexdigest()
def fasta(p):
    result={};ident=None;parts=[]
    for line in Path(p).open():
        if line.startswith('>'):
            if ident is not None:result[ident]=''.join(parts)
            ident=line[1:].split()[0];parts=[]
        else:parts.append(line.strip())
    if ident is not None:result[ident]=''.join(parts)
    return result
def csvwrite(p,records):
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def prepare():
    from cersei_embedding_organization import ec_labels
    if (OUT/'prepared.json').exists():raise FileExistsError('Prepared cohort is immutable')
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=stamp(),exploratory=True,test_sets_previously_inspected=True,
        estimand='Reaction-to-enzyme EC3 compatibility beyond all locally recorded partner identities',
        queries='All test reaction queries with at least one partner-derived specified EC3 and >=50 eligible candidates; no performance-based query sampling',
        labels='Native ReactZyme UniProt-Rhea EC annotations; released CLIPZyme ec2uniprot mapping for EnzymeMap; annotation unions over identical full sequences',
        partners='Union of all released train/validation/test association records for the exact reaction identifier/string, including partners outside the candidate bank',
        exclusions=['all recorded partners and their exact-sequence aliases','add all candidates in detected 50%-identity components of any recorded partner'],
        candidate_unit='Benchmark candidate identifiers: sequence IDs for ReactZyme, accessions for EnzymeMap; no unannotated candidate is ranked',
        class_macro='Mean of per-EC3-class query means; multilabel queries enter every class they carry and use any-label overlap',
        primary_ties='Expected agreement under uniform permutations within exact score ties; stable candidate-index result retained as sensitivity',
        k=list(map(int,KS)),primary_k=10,
        reference='Exact expected random retrieval in each eligible candidate bank; accounts for multilabel EC3 frequencies and exclusions',
        ceiling='min(k, compatible candidate count)/k per query',
        stage_coefficients={'phase1_native':[1.,0.],'phase2_native':[1.,.2],
            'phase1_fusion2':[2.,0.],'phase2_fusion2':[2.,.2],'cersei':[2.,.1]},
        baselines={'reactzyme':['horizyn','creep'],'enzymemap':['horizyn','creep','clipzyme']},
        uncertainty='1000 paired family-level exponential multiplier draws; fixed observed EC3 classes, models and banks; not training-seed uncertainty',
        seed=SEED,training=False,checkpoint_selection=False,
        interpretation='Functionally compatible candidates, not validated new associations or evidence of exact substrate specificity; existing exact-retrieval comparisons remain')
    dump(OUT/'protocol.json',protocol)
    catalog_path=ROOT/'runs/reactzyme_public_baselines_20260921/features/catalog.json'
    pairs_path=catalog_path.with_name('pairs.npz');catalog=read(catalog_path)
    seqs=dict(zip(catalog['protein_ids'],catalog['protein_sequences']))
    global_ec=defaultdict(set)
    native=ROOT/'data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv'
    for row in rows(native,'\t'):
        pid='prot_'+hashlib.sha1(row['Sequence'].encode()).hexdigest()[:16]
        global_ec[pid].update(ec_labels(row['EC number']))
        if pid in seqs:assert seqs[pid]==row['Sequence']
    allpairs=defaultdict(set)
    with np.load(pairs_path) as z:
        for key in z.files:
            for r,e in z[key]:allpairs[catalog['reaction_ids'][r]].add(catalog['protein_ids'][e])
    acquisition=read(CROSS/'clipzyme_data_acquisition.json')
    native_map=Path(acquisition['files']['cached_enzymemap.p']['path'])
    ecmap=Path(acquisition['files']['ec2uniprot.p']['path'])
    with native_map.open('rb') as f:emap=pickle.load(f)
    with ecmap.open('rb') as f:ec2uni=pickle.load(f)
    ue=defaultdict(set)
    for ec,ids in ec2uni.items():
        for u in ids:ue[u].update(ec_labels(ec))
    map_partners=defaultdict(set)
    for row in emap:
        map_partners[row['reaction_string']].update(row.get('protein_refs',[]))
        for key in ['uniprot_id','protein_id']:
            if row.get(key):map_partners[row['reaction_string']].add(row[key])
    mapseq=fasta(CROSS/'clipzyme_f3_catalog_v1/screening_proteins.fasta')
    mapping=rows(CROSS/'clipzyme_f3_catalog_v1/screening_candidate_map.csv')
    uniseq={r['uniprot_id']:mapseq[r['protein_id']] for r in mapping}
    for part in ['train','dev','test']:
        for row in rows(CROSS/f'clipzyme_manifests_v2/{part}_associations.csv'):
            if row['sequence']:
                if row['protein_id'] in uniseq:assert uniseq[row['protein_id']]==row['sequence']
                uniseq[row['protein_id']]=row['sequence']
    sequence_universe={};source_paths={native,catalog_path,pairs_path,native_map,ecmap,
        CROSS/'clipzyme_f3_catalog_v1/screening_proteins.fasta',
        CROSS/'clipzyme_f3_catalog_v1/screening_candidate_map.csv',Path(__file__)}
    summaries=[]
    for split in SPLITS:
        dest=OUT/split;dest.mkdir(exist_ok=True)
        path=OLD/split/'metadata.json';meta=read(path);source_paths.add(path)
        ids=meta['candidate_ids'];ec=[prefix(x) for x in meta['candidate_ec']]
        if split=='enzymemap':
            assert ids==[r['uniprot_id'] for r in mapping]
            candidate_seqs=[uniseq[p] for p in ids]
            psets=[map_partners[s] for s in meta['reaction_smiles']]
            labels=[sorted(set().union(*(set(prefix(ue[p])) for p in ps))) for ps in psets]
            partner_sequences=[[uniseq[p] for p in ps if p in uniseq] for ps in psets]
            missing_sequences=[sorted(p for p in ps if p not in uniseq) for ps in psets]
            # Preserve the existing sequence-aggregated labels of the main diagnostics.
            byseq=defaultdict(set)
            for s,ls in zip(candidate_seqs,ec):byseq[s].update(ls)
            labels=[sorted(set(ls).union(*(byseq[s] for s in ss))) for ls,ss in zip(labels,partner_sequences)]
        else:
            candidate_seqs=[seqs[p] for p in ids]
            psets=[allpairs[r.removesuffix('_f')] for r in meta['reaction_ids']]
            labels=[sorted(set().union(*(set(prefix(global_ec[p])) for p in ps))) for ps in psets]
            partner_sequences=[[seqs[p] for p in ps] for ps in psets]
            missing_sequences=[[] for ps in psets]
        keys=[seqkey(s) for s in candidate_seqs]
        partners=[sorted({seqkey(s) for s in ss}) for ss in partner_sequences]
        annotated=np.asarray([bool(x) for x in ec]);blocked=[]
        candidate_lookup={p:i for i,p in enumerate(ids)};sequence_lookup=defaultdict(set)
        for i,key in enumerate(keys):sequence_lookup[key].add(i)
        for ps,pk in zip(psets,partners):
            excluded={candidate_lookup[p] for p in ps if p in candidate_lookup}
            for key in pk:excluded.update(sequence_lookup[key])
            blocked.append(sorted(excluded))
        truth=np.load(OLD/split/'truth.npy')
        for q,idx in truth:assert int(idx) in blocked[int(q)],(split,q,idx)
        valid=np.asarray([bool(ls) and int(annotated.sum())-sum(annotated[b])>=50 for ls,b in zip(labels,blocked)])
        for s,keep in zip(candidate_seqs,annotated):
            if keep:sequence_universe[seqkey(s)]=s
        for ss in partner_sequences:
            for s in ss:sequence_universe[seqkey(s)]=s
        cohort=dict(split=split,candidate_ids=ids,candidate_ec3=ec,candidate_sequence_keys=keys,
            reaction_ids=meta['reaction_ids'],reaction_ec3=labels,partner_ids=[sorted(ps) for ps in psets],
            partner_sequence_keys=partners,blocked_partner_indices=blocked,
            partners_missing_sequences=missing_sequences,
            reaction_families=meta['reaction_families'],primary_query_indices=np.flatnonzero(valid).tolist())
        dump(dest/'cohort.json',cohort)
        summary=dict(split=split,test_queries=len(labels),ec3_queries=sum(bool(x) for x in labels),
            primary_queries=int(valid.sum()),classes=len(set().union(*(set(labels[i]) for i in np.flatnonzero(valid)))),
            original_candidates=len(ids),annotated_candidates=int(annotated.sum()),
            known_partner_edges=sum(map(len,psets)),excluded_bank_edges=sum(map(len,blocked)),
            listed_test_edges=len(truth),partner_sequence_coverage=sum(len(s) for s in partner_sequences),
            partners_missing_sequences=sum(map(len,missing_sequences)),
            minimum_remaining_candidates=int(min(int(annotated.sum())-sum(annotated[b]) for b in blocked)),
            cohort_sha256=sha(dest/'cohort.json'))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    with (OUT/'component_universe.fasta').open('w') as f:
        for key,s in sorted(sequence_universe.items()):f.write(f'>{key}\n{s}\n')
    dump(OUT/'prepared.json',dict(created_utc=stamp(),scopes=summaries,sequence_universe=len(sequence_universe),
        universe_sha256=sha(OUT/'component_universe.fasta'),protocol_sha256=sha(OUT/'protocol.json'),
        sources={str(p):sha(p) for p in source_paths}))

def homology(threads):
    dest=OUT/'homology';dest.mkdir(exist_ok=True)
    command=[str(ROOT/'tools/mmseqs/bin/mmseqs'),'easy-search',str(OUT/'component_universe.fasta'),
        str(OUT/'component_universe.fasta'),str(dest/'hits.tsv'),str(dest/'tmp'),
        '--threads',str(threads),'-s','7.5','-e','0.001','--min-seq-id','0.5','-c','0.8','--cov-mode','0',
        '--max-seqs','10000','--split-memory-limit','96G','--format-output','query,target,fident,qcov,tcov,evalue']
    dump(dest/'command.json',dict(command=command,started_utc=stamp(),
        universe='All EC3-annotated candidate sequences and available recorded-partner sequences across the four targets'))
    if not (dest/'search_complete.json').exists():
        with (dest/'search.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        dump(dest/'search_complete.json',dict(completed_utc=stamp(),hits_sha256=sha(dest/'hits.tsv')))
    ids=[line[1:].strip() for line in (OUT/'component_universe.fasta').open() if line.startswith('>')]
    index={x:i for i,x in enumerate(ids)};parent=np.arange(len(ids));size=np.ones(len(ids),int);counts=defaultdict(int)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    edges=0
    for line in (dest/'hits.tsv').open():
        q,t,identity,qcov,tcov,ev=line.rstrip().split('\t');counts[q]+=1
        if float(identity)<.5 or min(float(qcov),float(tcov))<.8 or float(ev)>.001:continue
        a,b=find(index[q]),find(index[t]);edges+=1
        if a!=b:
            if size[a]<size[b]:a,b=b,a
            parent[b]=a;size[a]+=size[b]
    families={x:int(find(i)) for i,x in enumerate(ids)}
    dump(dest/'families.json',families)
    dump(dest/'complete.json',dict(completed_utc=stamp(),sequences=len(ids),components=len(set(families.values())),
        accepted_hits=edges,max_hits_per_query=max(counts.values(),default=0),search_cap=10000,
        families_sha256=sha(dest/'families.json'),qualification='Heuristic search-derived transitive components; absence of a hit does not establish absence of homology'))

def export_stages(device):
    import torch
    from torch.nn import functional as F
    from generalization_reactzyme_architecture_phase2 import load_head,canonical_dot
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    plan=read(COEFF/'protocol.json');stages=read(OUT/'protocol.json')['stage_coefficients']
    for split in SPLITS:
        dest=OUT/split/'stages';dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():continue
        cohort=read(OUT/split/'cohort.json');task=plan['tasks'][split]
        cache=COEFF/split/'fusion_components.pt';receipt=read(cache.with_suffix('.receipt.json'))
        assert sha(cache)==receipt['sha256']
        assert sha(task['head'])==task['head_sha256']
        with torch.inference_mode():
            data=torch.load(cache,map_location=device,weights_only=False)
            assert data['checkpoint_sha256']==task['checkpoint_sha256']
            g,f,scale=data['g'],data['f'],data['scale']
            head=load_head(Path(task['head']),task['manifest_sha256'],device)
            if split=='enzymemap':
                mapping=rows(CROSS/'clipzyme_f3_catalog_v1/screening_candidate_map.csv')
                assert [x['uniprot_id'] for x in mapping]==cohort['candidate_ids']
                lookup={p:i for i,p in enumerate(data['ids'])};expanded=np.asarray([lookup[x['protein_id']] for x in mapping])
                native=Path(task['phase'])/'selected_test/test_embeddings'
                assert (native/'query_ids.txt').read_text().splitlines()==cohort['reaction_ids']
                br=torch.tensor(np.load(native/'reaction_embeddings.npy'),device=device)
            else:
                native=Path(task['phase'])/'test_features';cat=read(native/'catalog.json')
                assert cat['proteins']==data['ids']==cohort['candidate_ids']
                assert cat['reactions']==cohort['reaction_ids']
                br=torch.tensor(np.load(native/'f3_features.npz')['reactions'],device=device);expanded=None
            reference=np.load(CURRENT/split/'scores.npy',mmap_mode='r');records={}
            for name,(a,k) in stages.items():
                b=F.normalize(g+(scale*a)*f,dim=-1);cap=k/head.scale
                e=(torch.cat([F.normalize(v+cap*head.scale*head.enzyme(v),dim=-1) for v in b.split(8192)])
                    if split=='enzymemap' else F.normalize(b+cap*head.scale*head.enzyme(b),dim=-1))
                r=F.normalize(br+cap*head.scale*head.reaction(br),dim=-1)
                path=dest/(name+'.npy');scores=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=reference.shape)
                error=0.
                if split=='enzymemap':
                    for start in range(0,len(r),32):
                        block=(r[start:start+32]@e.T).float().cpu().numpy()[:,expanded]
                        scores[start:start+len(block)]=block
                        if name=='cersei':error=max(error,float(np.max(np.abs(block-reference[start:start+len(block)]))))
                else:
                    scores[:]=canonical_dot(r,e).cpu().numpy()
                    if name=='cersei':error=float(np.max(np.abs(scores-reference)))
                scores.flush();del scores
                if name=='cersei':assert error==0.,(split,error)
                records[name]=dict(alpha=a,kappa=k,sha256=sha(path),reference_score_error=error if name=='cersei' else None)
                print(split,'EXPORTED',name,'parity',error,flush=True)
            dump(dest/'complete.json',dict(completed_utc=stamp(),checkpoint=task['checkpoint'],checkpoint_sha256=task['checkpoint_sha256'],
                head=task['head'],head_sha256=task['head_sha256'],components_sha256=receipt['sha256'],stages=records))
            del data,g,f,b,e,r,br,head;torch.cuda.empty_cache()

def functional_curve(values,hits,kmax=50):
    """Partial top-k sorting with exact-tie expectation over the whole tie block."""
    assert len(values)>=kmax and np.isfinite(values).all()
    threshold=np.partition(values,len(values)-kmax)[len(values)-kmax]
    chosen=np.flatnonzero(values>=threshold)
    order=chosen[np.argsort(-values[chosen],kind='stable')]
    scores=values[order];y=hits[order].astype(float)
    starts=np.r_[0,np.flatnonzero(scores[1:]!=scores[:-1])+1]
    sizes=np.diff(np.r_[starts,len(order)])
    proportions=np.add.reduceat(y,starts)/sizes
    expected=np.cumsum(np.repeat(proportions,sizes)[:kmax])/np.arange(1,kmax+1)
    stable=np.cumsum(y[:kmax])/np.arange(1,kmax+1)
    return expected,stable,order[:kmax],int(sizes[np.searchsorted(starts,min(9,kmax-1),side='right')-1])

def macro_weights(labels,families):
    from scipy import sparse
    classes=sorted(set().union(*map(set,labels)));lookup={x:i for i,x in enumerate(classes)}
    row=[];col=[]
    for i,ls in enumerate(labels):
        for c in ls:row.append(i);col.append(lookup[c])
    membership=sparse.csr_matrix((np.ones(len(row)),(row,col)),shape=(len(labels),len(classes)))
    counts=np.asarray(membership.sum(0)).ravel()
    weights=np.asarray(membership@(1/counts)).ravel()/len(classes)
    _,fi=np.unique(families,return_inverse=True);nf=int(fi.max())+1
    family_draw=np.random.default_rng(SEED).exponential(size=(1000,nf))
    query_draw=family_draw[:,fi]
    denom=np.asarray(membership.T@query_draw.T).T
    boot=query_draw*np.asarray(membership@(1/denom).T).T/len(classes)
    assert np.allclose(weights.sum(),1) and np.allclose(boot.sum(1),1)
    return membership,classes,counts,weights,boot,nf

def score_sources(split,cohort):
    result={name:OUT/split/'stages'/f'{name}.npy' for name in read(OUT/'protocol.json')['stage_coefficients']}
    if split=='enzymemap':
        for model,folder in [('horizyn','enzymemap_public_horizyn_20260923_seed42'),('creep','enzymemap_public_creep_20260924_seed42')]:
            root=ROOT/'runs'/folder;cat=read(root/'catalog.json')
            assert cat['candidate_ids']==cohort['candidate_ids'] and cat['test_ids']==cohort['reaction_ids']
            result[model]=root/'test/scores.npy'
        path=CROSS/'clipzyme_released_screen_v1'
        assert (path/'candidate_ids.txt').read_text().splitlines()==cohort['candidate_ids']
        assert (path/'query_ids.txt').read_text().splitlines()==cohort['reaction_ids']
        result['clipzyme']=path/'scores.npy'
    else:
        for name in ['horizyn','creep']:result[name]=PUBLIC/split/f'{name}_scores.npy'
    return result

def evaluate(exclusion,selected=None):
    families=read(OUT/'homology/families.json') if exclusion=='components' else None
    targets=SPLITS if selected is None else (SPLITS[:3] if selected=='reactzyme' else [selected])
    for split in targets:
        out=OUT/split/exclusion;out.mkdir(exist_ok=True)
        if (out/'complete.json').exists():continue
        assert (OUT/split/'stages/complete.json').exists(), 'Stage exports must finish before evaluation'
        c=read(OUT/split/'cohort.json');candidate_labels=c['candidate_ec3'];n=len(candidate_labels)
        valid=np.asarray([bool(x) for x in candidate_labels]);bylabel=defaultdict(list)
        for i,ls in enumerate(candidate_labels):
            for label in ls:bylabel[label].append(i)
        byfamily=defaultdict(list)
        if families is not None:
            for i,key in enumerate(c['candidate_sequence_keys']):
                if valid[i]:byfamily[families[key]].append(i)
        contexts=[];eligible_queries=[];baseline=[];ceiling=[];coverage=[]
        for q in c['primary_query_indices']:
            if families is not None and c['partners_missing_sequences'][q]:continue
            blocked=set(c['blocked_partner_indices'][q]);allowed=valid.copy()
            if families is not None:
                partner_families={families[p] for p in c['partner_sequence_keys'][q]}
                for fam in partner_families:blocked.update(byfamily[fam])
            allowed[list(blocked)]=False;indices=np.flatnonzero(allowed)
            if len(indices)<50:continue
            compatible=np.zeros(n,bool)
            for label in c['reaction_ec3'][q]:compatible[bylabel[label]]=True
            hits=compatible[indices];num=int(hits.sum())
            assert not (set(indices)&set(c['blocked_partner_indices'][q]))
            if families is not None:assert not any(families[c['candidate_sequence_keys'][i]] in partner_families for i in indices)
            eligible_queries.append(q);contexts.append((indices,hits));baseline.append(np.full(50,num/len(indices)));ceiling.append(np.minimum(KS,num)/KS)
            coverage.append(dict(query_index=q,query_id=c['reaction_ids'][q],ec3=';'.join(c['reaction_ec3'][q]),
                recorded_partners=len(c['partner_ids'][q]),excluded_candidates=len(blocked),eligible_candidates=len(indices),compatible_candidates=num,
                random_expectation=num/len(indices),max_possible_at10=min(num,10)/10,
                reaction_family=c['reaction_families'][q]))
        q=np.asarray(eligible_queries);labels=[c['reaction_ec3'][i] for i in q]
        mem,classes,counts,w,boot,nf=macro_weights(labels,np.asarray(c['reaction_families'])[q])
        scope=dict(split=split,exclusion=exclusion,queries=len(q),classes=len(classes),families=nf,
            original_candidates=n,annotated_candidates=int(valid.sum()),
            eligible_candidate_min=min(r['eligible_candidates'] for r in coverage),
            eligible_candidate_max=max(r['eligible_candidates'] for r in coverage),
            queries_without_alternative=sum(r['compatible_candidates']==0 for r in coverage),
            missing_partner_sequence_queries=sum(bool(x) for x in c['partners_missing_sequences']),
            cohort_sha256=sha(OUT/split/'cohort.json'))
        dump(out/'scope.json',scope);csvwrite(out/'query_scope.csv',coverage)
        curves=[];class_rows=[];values={'random':np.asarray(baseline),'ceiling':np.asarray(ceiling)};stable_values={};receipts={}
        sources=score_sources(split,c)
        for model,path in sources.items():
            scores=np.load(path,mmap_mode='r');assert scores.shape==(len(c['reaction_ids']),n)
            expected=[];stable=[];neighbors=[];ties=[]
            for index,(candidate,hits) in zip(q,contexts):
                y,ys,order,tie=functional_curve(np.asarray(scores[index,candidate]),hits)
                expected.append(y);stable.append(ys);neighbors.append(candidate[order]);ties.append(tie)
            values[model]=np.asarray(expected);stable_values[model]=np.asarray(stable)
            np.savez_compressed(out/f'{model}_per_query.npz',query_indices=q,agreement=values[model],stable_agreement=stable_values[model],
                stable_neighbors=np.asarray(neighbors),tie_block_at10=np.asarray(ties),macro_weights=w)
            receipts[model]=dict(scores=str(path),sha256=sha(path),tied_queries_at10=sum(t>1 for t in ties),
                max_stable_vs_expected_difference=float(abs(w@(values[model]-stable_values[model])).max()))
            print(split,exclusion,model,'@10',float(w@values[model][:,9]),flush=True)
        comparisons={model:v for model,v in values.items()}
        for model in sources:
            if model!='cersei':comparisons['cersei-minus-'+model]=values['cersei']-values[model]
        comparisons['phase2-native-minus-phase1-native']=values['phase2_native']-values['phase1_native']
        comparisons['cersei-minus-random']=values['cersei']-values['random']
        for model,y in comparisons.items():
            macro=w@y;micro=y.mean(0);draws=boot@y
            low,high=np.quantile(draws,[.025,.975],axis=0)
            for j,k in enumerate(KS):
                curves.append(dict(**scope,model=model,k=int(k),macro=float(macro[j]),micro=float(micro[j]),
                    ci_low=float(low[j]),ci_high=float(high[j]),
                    stable_macro=float(w@stable_values[model][:,j]) if model in stable_values else float(macro[j])))
            if model in values:
                class_means=np.asarray(mem.T@y)/counts[:,None]
                for i,label in enumerate(classes):
                    for k in [1,10,50]:class_rows.append(dict(split=split,exclusion=exclusion,model=model,ec3=label,queries=int(counts[i]),k=k,agreement=float(class_means[i,k-1])))
        csvwrite(out/'curves.csv',curves);csvwrite(out/'per_class.csv',class_rows)
        np.savez_compressed(out/'controls_per_query.npz',query_indices=q,random=values['random'],ceiling=values['ceiling'],macro_weights=w)
        dump(out/'complete.json',dict(completed_utc=stamp(),scope=scope,sources=receipts,script_sha256=sha(__file__),
            protocol_sha256=sha(OUT/'protocol.json'),curves_sha256=sha(out/'curves.csv'),per_class_sha256=sha(out/'per_class.csv')))

def checks():
    # Every prefix of a completely tied ranking has the global hit fraction.
    y,s,idx,tie=functional_curve(np.zeros(100),np.arange(100)%2==0)
    assert np.all(y==.5) and tie==100 and s[0]==1
    # The boundary block includes tied candidates below position 50.
    x=np.r_[np.ones(10),np.zeros(100)];h=np.r_[np.ones(10,bool),np.arange(100)%4==0]
    y,_,_,_=functional_curve(x,h);assert abs(y[-1]-(10+40*.25)/50)<1e-12
    labels=[['a'],['a','b'],['b']];_,_,_,w,boot,_=macro_weights(labels,[0,1,2])
    assert np.allclose(w,[.25,.5,.25]) and np.allclose(boot.sum(1),1)
    dump(OUT/'algorithm_checks.json',dict(exact_ties=True,boundary_tie_expansion=True,multilabel_macro=True,bootstrap_weights_normalized=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','homology','export','evaluate','check']);p.add_argument('--threads',type=int,default=24)
    p.add_argument('--device',default='cuda:0');p.add_argument('--exclusion',choices=['partners','components'],default='partners')
    p.add_argument('--split',choices=SPLITS+['reactzyme']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='homology':homology(a.threads)
    elif a.action=='export':export_stages(a.device)
    elif a.action=='evaluate':evaluate(a.exclusion,a.split)
    else:checks()
