"""Artifact QA: identities, source checksums, finite vectors and table arithmetic."""
import csv
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
from cersei_embedding_organization import ROOT,OUT as OLD,read,dump,sha,prefix
from cersei_public_embedding_analysis import OUT,CURRENT
from cersei_public_embedding_stats import summarize

def main():
    checks=[];hashes={};counts={}
    for split in ['time','enzyme_smi','reaction_smi','enzymemap']:
        dest=OUT/split;m=read(OLD/split/'metadata.json');bank=np.array(m['neighborhood_indices']);where={p:i for i,p in enumerate(m['protein_ids'])}
        models=['cersei','horizyn']+(['clipzyme'] if split=='enzymemap' else ['creep'])
        for model in models:
            p=dest/f'{model}_embeddings.npz';z=np.load(p)
            for key in ['enzyme','reaction']:
                assert np.isfinite(z[key]).all() and np.max(np.abs(np.linalg.norm(z[key],axis=1)-1))<3e-5,(split,model,key)
            assert z['enzyme'].shape[0]==len(bank) and z['reaction'].shape[0]==len(m['reaction_ids'])
            if model!='clipzyme' and not (split=='enzymemap' and model=='cersei'):
                rec=read(dest/f'{model}_receipt.json');assert rec['embeddings_sha256']==sha(p)
                assert rec['parity_max_error']<3e-5
            hashes[str(p.relative_to(OUT))]=sha(p)
        if split!='enzymemap':
            public=ROOT/'runs/reactzyme_public_baselines_20260921/features';pcat=read(public/'catalog.json')
            base=ROOT/'runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1'/split/'phase2_followup/epoch20/features'
            ccat=read(base/'catalog.json');ce=np.load(base/'pairs.npz')['train'];pe=np.load(public/'pairs.npz')[split+'_train']
            cedges={(ccat['reactions'][r].removesuffix('_f'),ccat['proteins'][e]) for r,e in ce}
            pedges={(pcat['reaction_ids'][r],pcat['protein_ids'][e]) for r,e in pe}
            assert cedges==pedges,(split,len(cedges-pedges),len(pedges-cedges))
            for model in ['horizyn','creep']:
                rec=read(dest/f'{model}_receipt.json')
                assert rec['protocol']['data_sha256']==sha(public/'pairs.npz') and not rec['selection']['test_used']
            checks.append(split+f': identical downstream training association sets ({len(cedges)} unique pairs); validation-only baseline selection')
        # Recompute the primary class-balanced result directly from per-protein
        # rows, without calling the summary/statistics implementation.
        summary=read(dest/'neighborhoods_summary.json');classes=defaultdict(lambda:defaultdict(list));querysets=defaultdict(set)
        with (dest/'neighborhoods_per_protein.csv').open() as f:
            for r in csv.DictReader(f):
                if r['ec_level']!='3' or r['k']!='10' or r['exclude_homologs']!='True' or r['exclude_training_overlap']!='False':continue
                i=where[r['protein_id']];querysets[r['model']].add(r['protein_id'])
                for ec in prefix(m['enzyme_ec'][i],3):classes[r['model']][ec].append(float(r['value']))
        for model,cc in classes.items():
            independent=float(np.mean([np.mean(v) for v in cc.values()]));selected=next(r for r in summary if r['model']==model and r['ec_level']==3 and r['k']==10 and r['exclude_homologs'])
            assert abs(independent-selected['macro'])<1e-12
        assert all(q==querysets['cersei'] for q in querysets.values())
        for control in ['clean','enzymecage']:
            z=np.load(dest/f'{control}_embeddings.npz');rec=read(dest/f'{control}_receipt.json');assert rec['embeddings_sha256']==sha(dest/f'{control}_embeddings.npz')
            assert np.isfinite(z['enzyme']).all() and len(z['enzyme'])==len(z['bank'])==rec['proteins']
            assert len(set(z['bank'].tolist()))==len(z['bank']) and set(z['bank'].tolist())<=set(bank.tolist())
            assert int(z['training_sequence_overlap'].sum())==rec['exact_training_sequence_overlap']
        counts[split]=dict(protein_bank=len(bank),reaction_bank=len(m['reaction_ids']),screening_candidates=len(m['candidate_ids']),primary_annotated_queries=len(querysets['cersei']))
        for f in ['metadata.json','subsets.json','truth.npy','expanded.npy','chemistry.npz']:hashes[str((OLD/split/f).relative_to(ROOT))]=sha(OLD/split/f)
        for f in dest.glob('*summary.json'):
            for r in read(f):
                if r.get('macro') is not None:assert np.isfinite(r['macro'])
                if r.get('ci95') is not None:assert np.isfinite(r['ci95']).all() and r['ci95'][0]<=r['ci95'][1]
            hashes[str(f.relative_to(OUT))]=sha(f)
        checks.append(split+': ID coverage, normalized vectors, checkpoint/score parity receipt, common queries, independent macro arithmetic, control coverage')
    # A fixed paired effect must be preserved by class and family weighting.
    r=summarize([.2,.2,.2,.2],[['a'],['a','b'],['b'],['c']],[0,0,1,2])
    assert abs(r['macro']-.2)<1e-12 and np.max(np.abs(np.array(r['ci95'])-.2))<1e-12
    r=summarize([1.,0.,np.nan],[['a'],['a'],['b']],[0,1,2]);assert r['macro']==.5 and r['evaluated_queries']==2
    checks.append('Family-weighted statistic preserves a constant paired effect and excludes missing observations')
    native=OUT/'enzymecage_p450';c=read(native/'coverage.json');s=np.load(native/'scores.npy');t=np.load(native/'truth.npy');z=np.load(native/'embeddings.npz')
    assert s.shape==(191,490) and t.shape==(318,2) and z['enzyme'].shape==(490,512)
    assert np.isfinite(s).all() and np.isfinite(z['enzyme']).all() and len(set(map(tuple,t)))==318
    assert sha(Path(read(native/'metadata.json')['score_source']))==read(native/'metadata.json')['score_receipt']['scores_sha256']
    checks.append('EnzymeCAGE native panel: all 93,590 pairs, 318 distinct positives, 490 vectors, verified source scores')
    for name in ['01_matched_functional_neighborhoods','02_matched_alignment','03_clean_transfer_control','04_enzymecage_native_panel']:
        for ext in ['pdf','png','svg']:
            p=OUT/f'{name}.{ext}';assert p.stat().st_size>1000;hashes[p.name]=sha(p)
    for p in (ROOT/'scripts').glob('cersei_public_embedding*.py'):hashes[str(p.relative_to(ROOT))]=sha(p)
    for name in ['cersei_clean_embedding_control.py','cersei_enzymecage_embedding_control.py','cersei_enzymecage_p450_geometry.py','cersei_enzymecage_p450_analysis.py']:hashes['scripts/'+name]=sha(ROOT/'scripts'/name)
    dump(OUT/'validation.json',dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,counts=counts,sha256=hashes,
        visual_qa='All four PNG exports inspected: labels, uncertainty, palette, bounds, legends and missing-data representation checked',
        supported_scope='Matched-data dual encoders; CLEAN transfer; restricted EnzymeCAGE coverage and complete native P450 diagnostic',
        unsupported_claims=['Full ReactZyme/EnzymeMap EnzymeCAGE comparison','Same-data CLEAN architecture comparison','A win over all competitors','Wet-lab generalization','Multi-seed significance']))
    print('VALIDATION PASSED',len(checks),'groups',flush=True)

if __name__=='__main__':main()
