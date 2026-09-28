#!/usr/bin/env python3
"""Evaluation-only CERSEI geometry audit. Existing fits are immutable inputs.

Run with .capability-run-py/bin/python from EnzymeDiscovery. See protocol.json
and README.md in the output directory for estimands and interpretation limits.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import h5py
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/generalization_20260919_2251'
CROSS = RUN / 'cross_paper_retraining'
OLD = CROSS / 'v4_biological_geometry_20260921_v1'
KROOT = ROOT / 'runs/v4_residue_view_count_20260922_seed42'
OUT = ROOT / 'runs/cersei_embedding_organization_20260923_v1'
SPLITS = ('reaction_smi', 'enzyme_smi', 'time', 'enzymemap')
SEED = 23092026


def fixed_subset(ids, limit):
    """Uniform pseudorandom ordering, independent of annotations and results."""
    return np.array(sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f'{SEED}:{ids[i]}'.encode()).hexdigest())[:limit], dtype=np.int64)


def subsets():
    amendment=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        user_request='use like 2000 proteins and 500 reactions', proteins=2000,reactions=500,
        learned_view_proteins=2000,selection='SHA256(seed:entity ID), independent of labels and results',
        full_candidate_banks_retained=True,base_protocol_sha256=sha(OUT/'protocol.json'))
    dump(OUT/'subset_amendment.json',amendment)
    for split in SPLITS:
        d=OUT/split;meta=read(d/'metadata.json');truth=np.load(d/'truth.npy');expanded=np.load(d/'expanded.npy')
        bank=np.array(meta['neighborhood_indices']);pids=[meta['protein_ids'][i] for i in bank]
        proteins=bank[fixed_subset(pids,2000)];reactions=fixed_subset(meta['reaction_ids'],500)
        # For accession-based EnzymeMap retrieval, retain all known-positive
        # accessions corresponding to sampled sequences; <=2000 here.
        equeries=np.array(sorted(set(truth[:,1]) & set(np.flatnonzero(np.isin(expanded,proteins)))))
        assert len(equeries)<=2000
        sample=dict(protein_indices=proteins.tolist(),reaction_indices=reactions.tolist(),
            enzyme_query_indices=equeries.tolist(),protein_ids=[meta['protein_ids'][i] for i in proteins],
            reaction_ids=[meta['reaction_ids'][i] for i in reactions],
            protein_bank_count=len(bank),reaction_bank_count=len(meta['reaction_ids']))
        dump(d/'subsets.json',sample)
        print('SUBSET',split,len(proteins),'proteins',len(reactions),'reactions',len(equeries),'enzyme queries',flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def read(path): return json.loads(Path(path).read_text())


def rows(path, sep=','):
    with Path(path).open() as f: return list(csv.DictReader(f, delimiter=sep))


def norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def ec_labels(s):
    return sorted(set(re.findall(r'\b[1-7]\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-)', str(s))))


def prefix(labels, level):
    return sorted({'.'.join(s.split('.')[:level]) for s in labels
                   if len(s.split('.')) >= level and '-' not in s.split('.')[:level]})


def components(n, pairs):
    if not len(pairs): return np.arange(n)
    pairs = np.asarray(pairs, np.int64)
    return connected_components(sparse.coo_matrix((np.ones(len(pairs)),
        (pairs[:, 0], pairs[:, 1])), shape=(n, n)), directed=False)[1]


def label_components(labels):
    first, edges = {}, []
    for i, lab in enumerate(labels):
        for v in lab:
            if v in first: edges.append((i, first[v]))
            else: first[v] = i
    return components(len(labels), edges)


def checked_cache(path):
    expected = read(path.with_suffix('.receipt.json'))['sha256']
    assert sha(path) == expected, f'Cache changed: {path}'
    return torch.load(path, map_location='cpu', weights_only=False)


@torch.inference_mode()
def refine(x, model, endpoint, device='cpu'):
    return torch.cat([F.normalize(b.to(device) + .5 * model.scale *
        getattr(model, endpoint)(b.to(device)), dim=-1).cpu()
        for b in torch.as_tensor(x).split(4096)]).numpy()


def get_head(task, device='cpu'):
    from horizyn.generalization_residual import FrozenGeometryResidual
    path = Path(task.get('head_path', Path(task['source_phase2']) / 'training/step0100.pt'))
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['registry']['feature_manifest_sha256'] == sha(Path(task['source_phase2']) / 'features/manifest.json')
    assert not state['registry']['test_used']
    if 'phase2_checkpoint_sha256' in task['model']:
        assert sha(path) == task['model']['phase2_checkpoint_sha256']
    model = FrozenGeometryResidual(**state['model_config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['state_dict'], strict=True)
    return model


def fingerprints(smiles):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    result, valid = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s.replace('>>', '.').replace('>', '.'))
        if mol is None or mol.GetNumAtoms() == 0:
            result.append(np.zeros(2048, np.uint8)); valid.append(False); continue
        for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
        x = np.zeros(2048, np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), x)
        result.append(x); valid.append(True)
    return np.stack(result), np.asarray(valid)


def tanimoto(x, y=None):
    a = torch.as_tensor(x.astype(np.float32)); b = a if y is None else torch.as_tensor(y.astype(np.float32))
    intersection = (a @ b.T).numpy()
    union = np.asarray(x.sum(1))[:, None] + np.asarray(x.sum(1) if y is None else y.sum(1))[None] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def prepare():
    OUT.mkdir(exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
        scope='Evaluation only; no parameter optimization or checkpoint selection',
        splits=list(SPLITS), stages=['frozen_ProtT5 (enzyme only)', 'phase1_native (ReactZyme)',
            'phase1_deployed_fusion', 'phase2_neural', 'final_score (alignment only, separate)'],
        analysis1=dict(population='Within each test enzyme bank; EnzymeMap uses unique sequences with a test-positive association',
            ec_levels=[1,2,3,4], neighbors=[10,50], classes='EC prefix at evaluated depth; multilabel macro average',
            homology='Report unrestricted and excluding same 50%-identity MMseqs connected component',
            annotation_policy='Only native EC annotations, no homology imputation. Candidate bank restricted to annotated proteins at each EC depth.'),
        analysis2=dict(population='All official test query endpoints and official candidate axes; EnzymeMap retains candidate accessions',
            metrics=['mean-positive minus mean of 64 fixed random unannotated distractors',
                'mean-positive minus maximum unannotated score', 'mean-positive minus maximum same-EC1 unannotated score',
                'mean-positive minus maximum similar-participant unannotated score (E2R)',
                'known-positive coverage@10,@50', 'all-known-positives retrieved@50'],
            degrees=['1','2-5','6+'], chemistry_hard_threshold=.5,
            unknown_pairs='Unannotated distractors are not experimentally verified negatives'),
        analysis3=dict(population='Official EnzymeMap test reactions', labels='Native reaction-rule IDs; ambiguity and mapping-quality counts retained',
            metrics=['rule agreement@10,@50', 'rule agreement excluding participant Tanimoto >=0.5',
                'same-rule versus different-rule embedding similarity within 0.1-wide participant-Tanimoto bins',
                'same-rule low-participant pairs versus different-rule high-participant pairs'],
            thresholds=dict(dissimilar=.3,similar=.7),
            ReactZyme='Directed transformation analysis not applicable to unordered participant sets'),
        learned_views=dict(counts=[1,2,4,8], sample=256,
            sampling='Fixed SHA256 ordering of test-positive proteins, same identities across K within each split',
            metrics=['attention cosine, centered cosine, top-10%-residue Jaccard, entropy support fraction',
                'projected view cosine and linear CKA', 'feature-wise gate mass',
                'frozen view suppression with remaining feature-wise gates renormalized: retrieval and neighborhoods'],
            interpretation='One training seed. Suppression measures reliance, not causal biochemical function.'),
        uncertainty=dict(replicates=1000, level=.95, method='Percentile cluster bootstrap; resample whole query families, keep candidate bank fixed',
            enzyme_families='Connected components of retrieved MMseqs hits >=30% identity, >=80% both-sequence coverage, E<=1e-3',
            reaction_families='EnzymeMap: connected components of shared rule IDs; ReactZyme: participant fingerprint Tanimoto >=0.7 components',
            estimand='Equal weight to each represented annotation class, equal query weight within class; multiclass queries enter each annotated class',
            limitations='Conditional on these fits and candidate banks; not variability across training seeds. MMseqs search is heuristic.'),
        exploratory=True, benchmark_tests_previously_inspected=True,
        source_protocol_sha256=sha(OLD/'protocol.json'), code_sha256=sha(__file__))
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')['seed'] == SEED
    else: dump(OUT/'protocol.json', protocol)
    native = ROOT/'data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv'
    native_ec, sequences = defaultdict(set), {}
    for row in rows(native, '\t'):
        seq = row['Sequence']; key = 'prot_' + hashlib.sha1(seq.encode()).hexdigest()[:16]
        native_ec[key].update(ec_labels(row['EC number'])); sequences[key] = seq
    acquisition = read(CROSS/'clipzyme_data_acquisition.json')
    with open(acquisition['files']['cached_enzymemap.p']['path'], 'rb') as f: emap = pickle.load(f)
    with open(acquisition['files']['ec2uniprot.p']['path'], 'rb') as f: ec2uni = pickle.load(f)
    uec = defaultdict(set)
    for ec, ids in ec2uni.items():
        for u in ids: uec[u].update(ec_labels(ec))
    ecbyreaction, rulebyreaction, qualitybyreaction = defaultdict(set), defaultdict(set), defaultdict(list)
    for row in emap:
        ecbyreaction[row['reaction_string']].update(ec_labels(row['ec']))
        if row.get('rule_id') is not None: rulebyreaction[row['reaction_string']].add(str(row['rule_id']))
        qualitybyreaction[row['reaction_string']].append(float(row.get('quality', 0)))
    native_sources = {str(native):sha(native)}
    for key in ('cached_enzymemap.p', 'ec2uniprot.p'):
        path=Path(acquisition['files'][key]['path']); assert sha(path)==acquisition['files'][key]['sha256']; native_sources[str(path)]=sha(path)
    tasks = {x['name']:x for x in read(OLD/'protocol.json')['tasks']}
    feature_paths = {x['split']:Path(x['task']['fixed_test_features']) for x in read(CROSS/'shared_semantic_strength_variants_v1/protocol.json')['reactzyme']}
    needed_sequences = {}
    for split in SPLITS:
        dest=OUT/split; dest.mkdir(exist_ok=True)
        if (dest/'prepared.json').exists():
            meta=read(dest/'metadata.json'); needed_sequences.update(meta['sequences']); continue
        task=tasks[split]; m=task['model']; assert sha(m['checkpoint'])==m['checkpoint_sha256']
        cache=checked_cache(OLD/split/'test_cache.pt'); head=get_head(task)
        if split != 'enzymemap':
            fp=feature_paths[split]; complete=read(fp/'complete.json')
            assert complete['checkpoint_sha256']==m['checkpoint_sha256']
            assert sha(fp/'f3_features.npz')==complete['f3_features_sha256']
            catalog=read(fp/'catalog.json'); official=read(RUN/f'features_test_{split}/catalog.json'); assert catalog==official
            pids=catalog['proteins']; rids=catalog['reactions']
            with np.load(RUN/f'features_test_{split}/pairs.npz') as z: truth=np.unique(z['test'],axis=0)
            cached_edges=np.stack([cache['truth']['reaction_index'],cache['truth']['enzyme_index']],1)
            assert np.array_equal(np.unique(cached_edges,axis=0),truth)
            with np.load(fp/'f3_features.npz') as z: native_e=z['proteins']; assert np.max(np.abs(z['reactions']-cache['base_r'].numpy()))<3e-6
            with h5py.File(fp/'protein_mean.h5') as h:
                assert h['ids'].asstr()[:].tolist()==pids and h['complete'][:].all(); means=h['vectors'][:]
            with np.load(fp/'reaction_features.npz') as z: raw_r=z['t5v2']
            smiles_by_id={x['reaction_id']:x['reaction_smiles'] for x in rows(ROOT/f'data/revised_protocols/reactzyme_paper/{split}/test_rxns.csv')}
            smiles=[smiles_by_id[r.removesuffix('_f')] for r in rids]
            ecs=[sorted(native_ec[p]) for p in pids]; recs=[set() for _ in rids]
            for r,e in truth: recs[r].update(ecs[e])
            recs=[sorted(x) for x in recs]; rules=[[] for _ in rids]
            seqs={p:sequences[p] for p in pids}; selected=np.arange(len(pids)); expanded=np.arange(len(pids))
            candidate_ids=pids; candidate_ecs=ecs
            base_e=cache['base_e'].numpy(); base_r=cache['base_r'].numpy()
            final_scores=str(CROSS/f'shared_semantic_strength_variants_v1/{split}/alpha04_cap05/scores.npy')
            lineage=dict(task=task, features=str(fp), feature_receipt=complete,
                test_cache_sha256=sha(OLD/split/'test_cache.pt'), official_pairs_sha256=sha(RUN/f'features_test_{split}/pairs.npz'))
        else:
            lib=checked_cache(OLD/split/'screening_library.pt'); assert lib['checkpoint_sha256']==m['checkpoint_sha256']
            pids=lib['protein_ids']; candidate_ids=lib['candidate_ids']; expanded=lib['expanded']
            u2index={u:i for i,u in enumerate(candidate_ids)}
            qrows=rows(CROSS/'clipzyme_screening_evaluation_protocol_v2/queries.csv')
            rids=[x['reaction_id'] for x in qrows]; assert rids==cache['ids']
            smiles=[x['reaction'] for x in qrows]
            truth=np.asarray([(r,u2index[u]) for r,row in enumerate(qrows) for u in json.loads(row['positive_uniprot_ids_json'])], np.int64)
            truth=np.unique(truth,axis=0); selected=np.unique(expanded[truth[:,1]])
            ecs=[set() for _ in pids]
            for i,u in enumerate(candidate_ids): ecs[expanded[i]].update(uec[u])
            ecs=[sorted(x) for x in ecs]; candidate_ecs=[ecs[i] for i in expanded]
            recs=[sorted(ecbyreaction[s]) for s in smiles]; rules=[sorted(rulebyreaction[s]) for s in smiles]
            seqs={}
            with open(acquisition['files']['uniprot2sequence.p']['path'],'rb') as f: uniseq=pickle.load(f)
            for i in np.unique(truth[:,1]):
                u=candidate_ids[i]; p=pids[expanded[i]]
                if u in uniseq: seqs[p]=uniseq[u]
            for row in rows(CROSS/'clipzyme_manifests_v2/test_associations.csv'):
                if row['sequence'] and row['protein_id'] in u2index:
                    seqs[pids[expanded[u2index[row['protein_id']]]]]=row['sequence']
            assert all(pids[i] in seqs for i in selected)
            base_e=lib['base_e'].numpy(); base_r=cache['base_r'].numpy(); native_e=None
            means_path=CROSS/'clipzyme_phase2_enzymemap_v1/screen_protein_mean.h5'
            with h5py.File(means_path) as h:
                assert h['ids'].asstr()[:].tolist()==pids and h['complete'][:].all(); means=h['vectors'][selected]
            feature_path=CROSS/'clipzyme_f3_catalog_v1/features/reactiont5v2.forward.h5'
            from generalization_export import reaction_block
            raw_r,mask=reaction_block(feature_path,rids,False); assert mask.all()
            final_scores=str(Path(task['source_phase2'])/'shared_variant_alpha04_cap05/scores.npy')
            lineage=dict(task=task, test_cache_sha256=sha(OLD/split/'test_cache.pt'),
                screening_cache_sha256=sha(OLD/split/'screening_library.pt'),
                queries_sha256=sha(CROSS/'clipzyme_screening_evaluation_protocol_v2/queries.csv'))
            dump(dest/'mapping_quality.json',dict(reactions=len(rids), rule_annotated=sum(bool(x) for x in rules),
                multi_rule=sum(len(x)>1 for x in rules),
                min_native_quality=[min(qualitybyreaction[s],default=0) for s in smiles],
                note='Rule IDs are native transformation classes; mapping quality is audited separately, not proof of detailed mechanism.'))
        assert len(base_r)==len(rids) and len(base_e)==len(pids)
        assert np.isfinite(base_e).all() and np.isfinite(base_r).all()
        np.savez(dest/'embeddings.npz', phase1_e=base_e, phase1_r=base_r,
            refined_e=refine(base_e,head,'enzyme'),refined_r=refine(base_r,head,'reaction'),
            prott5=norm(means),prott5_indices=selected,raw_reaction=norm(raw_r),
            **({'native_e':native_e} if native_e is not None else {}))
        np.save(dest/'truth.npy',truth); np.save(dest/'expanded.npy',expanded)
        bits,valid=fingerprints(smiles); sim=tanimoto(bits)
        np.savez(dest/'chemistry.npz',bits=bits,valid=valid,tanimoto=sim)
        family_r=label_components(rules) if split=='enzymemap' else components(len(rids),np.argwhere(np.triu(sim>=.7,1)))
        meta=dict(protein_ids=pids,reaction_ids=rids,candidate_ids=candidate_ids,
            enzyme_ec=ecs,candidate_ec=candidate_ecs,reaction_ec=recs,reaction_rules=rules,
            reaction_smiles=smiles,sequences=seqs,neighborhood_indices=selected.tolist(),
            reaction_families=family_r.tolist(),final_scores=final_scores,
            reaction_ec_source='native EnzymeMap EC' if split=='enzymemap' else 'union of native EC annotations on known associated test proteins; coarse class proxy')
        dump(dest/'metadata.json',meta); dump(dest/'lineage.json',lineage)
        dump(dest/'prepared.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
            proteins=len(pids),reaction_queries=len(rids),known_edges=len(truth),
            neighborhood_proteins=len(selected),chemical_fingerprint_valid=int(valid.sum()),
            protein_ec_coverage={str(l):sum(bool(prefix(ecs[i],l)) for i in selected) for l in (1,2,3,4)},
            reaction_ec_coverage={str(l):sum(bool(prefix(ec,l)) for ec in recs) for l in (1,2,3,4)},
            reaction_rule_coverage=sum(bool(x) for x in rules),reaction_families=len(set(family_r.tolist())),
            embeddings_sha256=sha(dest/'embeddings.npz')))
        needed_sequences.update(seqs)
        print('PREPARED',split,read(dest/'prepared.json'),flush=True)
    dump(OUT/'annotation_sources.json',native_sources)
    with (OUT/'proteins.fasta').open('w') as f:
        for p,s in sorted(needed_sequences.items()):f.write(f'>{p}\n{s}\n')
    dump(OUT/'sequence_manifest.json',dict(proteins=len(needed_sequences),fasta_sha256=sha(OUT/'proteins.fasta')))


def homology():
    d=OUT/'homology'; d.mkdir(exist_ok=True)
    if not (d/'hits.tsv').exists():
        cmd=[str(ROOT/'tools/mmseqs/bin/mmseqs'),'easy-search',str(OUT/'proteins.fasta'),str(OUT/'proteins.fasta'),
            str(d/'hits.tsv'),str(d/'tmp'),'--threads','8','-s','7.5','-e','0.001','--min-seq-id','0.3',
            '-c','0.8','--cov-mode','0','--max-seqs','10000','--format-output','query,target,fident,qcov,tcov,evalue']
        dump(d/'command.json',dict(command=cmd,version=subprocess.check_output([cmd[0],'version'],text=True).strip()))
        with (d/'mmseqs.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    ids=[line[1:].strip() for line in (OUT/'proteins.fasta').read_text().splitlines() if line.startswith('>')]
    index={p:i for i,p in enumerate(ids)}; pairs30=[]; pairs50=[]
    maxhits=defaultdict(int)
    with (d/'hits.tsv').open() as f:
        for line in f:
            q,t,identity,qcov,tcov,ev=line.rstrip().split('\t'); maxhits[q]+=1
            if min(float(qcov),float(tcov))<.8 or float(identity)<.3 or float(ev)>.001: continue
            pairs30.append((index[q],index[t]))
            if float(identity)>=.5:pairs50.append((index[q],index[t]))
    c30=components(len(ids),pairs30);c50=components(len(ids),pairs50)
    dump(d/'families.json',{p:[int(c30[i]),int(c50[i])] for i,p in enumerate(ids)})
    dump(d/'complete.json',dict(proteins=len(ids),families30=len(set(c30)),families50=len(set(c50)),
        largest_family30=int(np.bincount(c30).max()),max_hits_per_query=max(maxhits.values()),
        search_cap=10000,hits_sha256=sha(d/'hits.tsv'),
        definition='Operational sequence families, connected components; no detected hit does not establish sequence dissimilarity.'))
    print('HOMOLOGY',read(d/'complete.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','homology','subsets']);a=p.parse_args()
    torch.set_num_threads(8)
    if a.action=='prepare':prepare()
    elif a.action=='homology':homology()
    elif a.action=='subsets':subsets()


if __name__=='__main__':main()
