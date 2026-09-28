#!/usr/bin/env python3
"""Audit the finished Horizyn EnzymeMap run and compare with fixed CERSEI."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CROSS=ROOT/'runs/generalization_20260919_2251/cross_paper_retraining'
CERSEI=CROSS/'shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/phase2/shared_variant_alpha04_cap05/test_evaluation/summary.json'
METRICS=['bedroc85','bedroc20','ef0.05','ef0.1']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run
    protocol=read(out/'protocol.json');complete=read(out/'complete.json');selection=read(out/'selection.json')
    for path,digest in protocol['inputs'].items():assert sha(path)==digest, f'Input changed: {path}'
    assert sha(out/'evaluation_source.py')==protocol['code_sha256']
    assert complete['protocol_sha256']==sha(out/'protocol.json')
    assert selection['test_used'] is False and selection['checkpoint_sha256']==sha(selection['checkpoint'])
    vals=[read(out/'validation'/f'epoch{e:03d}'/'summary.json') for e in protocol['validation_epochs']]
    for val in vals:
        assert val['checkpoint_sha256']==sha(out/'checkpoints'/f"epoch{val['epoch']:03d}.pt")
        source=out/'validation'/f"epoch{val['epoch']:03d}"/'per_query.jsonl'
        assert sha(source)==val['per_query_sha256']
        rows=[json.loads(x) for x in source.read_text().splitlines()]
        for table in ['table1','table2']:
            used=[r[table] for r in rows if r[table] is not None]
            assert len(used)==val['summary'][table]['queries']
            for m in METRICS:assert abs(np.mean([r[m] for r in used])-val['summary'][table][m])<1e-12
    selected=max(vals,key=lambda x:x['selection_value'])
    assert selection['selected_epoch']==selected['epoch'] and selection['validation_bedroc85']==selected['selection_value']
    prediction=read(out/'test/score_receipt.json');assert prediction['selection_sha256']==sha(out/'selection.json')
    assert prediction['checkpoint_sha256']==selection['checkpoint_sha256'] and prediction['test_labels_read'] is False
    assert prediction['created_utc']>=selection['selection_saved_utc']
    official=read(out/'test/evaluation/summary.json');assert official['scores_sha256']==prediction['scores_sha256']==sha(out/'test/scores.npy')
    assert official['summary']==complete['summary']
    scores=np.load(out/'test/scores.npy',mmap_mode='r');assert scores.shape==(1521,261907)
    for block in np.array_split(np.arange(1521),24):assert np.isfinite(scores[block]).all()
    with np.load(out/'test/embeddings.npz') as z:rxn=z['reactions'];proteins=z['unique_proteins']
    axes=dict(np.load(out/'axes.npz')); expanded=axes['expanded']
    q=np.linspace(0,1520,17,dtype=int);c=np.linspace(0,261906,67,dtype=int)
    expected=rxn[q].astype(np.float64)@proteins[expanded[c]].astype(np.float64).T
    score_error=float(np.max(np.abs(expected-scores[q[:,None],c[None,:]])));assert score_error<2e-6
    first={};aliases=[]
    for i,k in enumerate(expanded):
        if int(k) in first:aliases.append((first[int(k)],i))
        else:first[int(k)]=i
    alias_rows=np.array(aliases,dtype=int)
    assert np.array_equal(scores[q[:,None],alias_rows[None,:,0]],scores[q[:,None],alias_rows[None,:,1]])
    rows=[json.loads(x) for x in (out/'test/evaluation/per_query.jsonl').read_text().splitlines()]
    for table in ['table1','table2']:
        used=[r[table] for r in rows if r[table] is not None]
        assert len(used)==official['summary'][table]['queries']
        for m in METRICS:assert abs(np.mean([r[m] for r in used])-official['summary'][table][m])<1e-12
    reference=read(CERSEI);comparison=[]
    for table in ['table1','table2']:
        h=official['summary'][table];c=reference['summary'][table]
        assert h['queries']==c['queries'] and h['candidate_ids']==c['candidate_ids']
        for m in METRICS:comparison.append(dict(library=table,metric=m,horizyn=h[m],cersei=c[m],horizyn_minus_cersei=h[m]-c[m]))
    validation=dict(passed=True,selected_epoch=selection['selected_epoch'],validation_checkpoints=10,
        selection_verified_without_test=True,summary_means_verified=True,full_score_matrix_finite=True,
        independent_float64_submatrix_error=score_error,accession_alias_pairs_checked=len(aliases),
        horizyn_source_sha256=sha(out/'complete.json'),cersei_source=str(CERSEI),cersei_source_sha256=sha(CERSEI),comparison=comparison)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    headers=['# Horizyn on EnzymeMap','',
        f"Completed native directed-input Horizyn retraining, seed 42. Validation selected epoch {selection['selected_epoch']} of 100 (full-library validation BEDROC85 {selection['validation_bedroc85']:.6f}). Test labels did not enter training or checkpoint selection.",'',
        '| Library | Metric | Horizyn | CERSEI default | Horizyn − CERSEI |','|---|---|---:|---:|---:|']
    for row in comparison:headers.append(f"| {'Full' if row['library']=='table1' else 'Training-ID-excluded'} | {row['metric']} | {row['horizyn']:.6f} | {row['cersei']:.6f} | {row['horizyn_minus_cersei']:+.6f} |")
    headers+=['','The full library has 261,907 accession candidates and 1,521 queries. The training-ID-excluded library has 252,113 candidates and 1,337 queries. Both methods use the same query labels, candidate order and released BEDROC/EF definitions. BEDROC uses the 0–1 scale; EF reports fold enrichment.','',
        'Horizyn uses the original public ReLU towers, 2,048-dimensional RDKit+/DRFP reaction fingerprints, frozen 1,024-dimensional ProtT5 means, normalized 512-dimensional outputs and original MLNCE. Physical reactant/product sides are retained. Standardization follows the native SOTA configuration, including uncharging. All 34,427 training association rows are preserved, with 34,180 unique sequence/reaction edges.','',
        'Training: fresh initialization, AdamW lr=1e-4, weight decay=.01, batch=16,384, fixed beta=10, 100 epochs; BF16 tower training and FP32 loss/inference. Checkpoints are evaluated at epochs 10,20,...,100 against the full validation library, using BEDROC85 alone for selection. ProtT5 means reuse the verified label-free residue cache and average all stored residues.','',
        'This is a single-seed comparison of complete methods. Horizyn and CERSEI have different representations, capacities, objectives and training budgets (100 versus 10 base epochs plus residual training). The result does not isolate an architectural component. No test-driven tuning or replacement of the CERSEI default was performed.','',
        '[Protocol](protocol.json) · [Selection](selection.json) · [Test metrics](test/evaluation/summary.json) · [Validation audit](validation.json) · [Training trajectory](metrics.jsonl)']
    (out/'report.md').write_text('\n'.join(headers)+'\n');print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
