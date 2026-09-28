#!/usr/bin/env python3
"""Secondary frozen-score screening audit excluding all training-sequence aliases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time

import numpy as np

from generalization_clipzyme_screening_evaluate import notebook_metrics,sha256

METRICS=('bedroc85','bedroc20','ef0.05','ef0.1')


def read_csv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def write(path,value):
    temporary=path.with_suffix('.tmp.json');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();out=a.campaign.resolve();plan=json.loads((out/'protocol.json').read_text());cross=out.parent
    protocol=cross/'clipzyme_screening_evaluation_protocol_v2'
    pool=read_csv(cross/'clipzyme_f3_catalog_v1/screening_candidate_map.csv')
    train={r['protein_id'] for r in read_csv(cross/'clipzyme_f3_catalog_v1/train_pairs.csv')}
    train_ids=set((protocol/'train_uniprot_ids.txt').read_text().splitlines())
    candidates=[r['uniprot_id'] for r in pool];index={key:i for i,key in enumerate(candidates)}
    official_kept=np.array([i for i,r in enumerate(pool) if r['uniprot_id'] not in train_ids],dtype=np.int64)
    kept=np.array([i for i,r in enumerate(pool) if r['protein_id'] not in train],dtype=np.int64)
    if len(kept)!=249828 or len(official_kept)!=252113:raise ValueError('Declared candidate exclusions changed')
    if any(pool[i]['protein_id'] in train for i in kept):raise ValueError('Training sequence survives exclusion')
    queries=read_csv(protocol/'queries.csv');query_ids=[r['reaction_id'] for r in queries]
    labels=[]
    for r in queries:
        row=np.zeros(len(candidates),dtype=bool)
        row[[index[k] for k in json.loads(r['positive_uniprot_ids_json'])]]=True;labels.append(row)
    if sum(bool(row[kept].any()) for row in labels)!=1333:raise ValueError('Declared query denominator changed')
    (out/'excluded_candidate_ids.txt').write_text(''.join(r['uniprot_id']+'\n' for r in pool if r['protein_id'] in train))
    inputs=[cross/'clipzyme_f3_catalog_v1/screening_candidate_map.csv',cross/'clipzyme_f3_catalog_v1/train_pairs.csv',
            protocol/'queries.csv',protocol/'train_uniprot_ids.txt']
    write(out/'exclusion_receipt.json',dict(candidate_ids=len(kept),queries=1333,training_sequences=len(train),
        extra_alias_ids_excluded=len(official_kept)-len(kept),source_hashes={str(f):sha256(f) for f in inputs},
        selection_used_scores=False,definition='Exclude every candidate internal sequence key appearing in the original training pairs. Preserve official query ordering, positive labels and metric definitions.'))
    pending=list(plan['methods']);results={}
    while pending:
        ready=[m for m in pending if Path(m['official_summary']).exists()]
        if not ready:
            time.sleep(10);continue
        for method in ready:
            dest=out/method['name'];dest.mkdir()
            summary=json.loads(Path(method['official_summary']).read_text())
            score_path=Path(method['scores']);scores=np.load(score_path,mmap_mode='r')
            if sha256(score_path)!=summary['scores_sha256']:raise ValueError('Frozen score matrix changed')
            if Path(method['query_ids']).read_text().splitlines()!=query_ids:raise ValueError('Query order mismatch')
            if Path(method['candidate_ids']).read_text().splitlines()!=candidates:raise ValueError('Candidate order mismatch')
            if scores.shape!=(1521,261907) or scores.dtype!=np.float32:raise ValueError('Score shape or precision mismatch')
            original=[json.loads(line) for line in Path(method['official_summary']).with_name('per_query.jsonl').read_text().splitlines()]
            def evaluate(i):
                values=np.asarray(scores[i]);label=labels[i]
                if not np.isfinite(values).all():raise ValueError('Nonfinite scores')
                official_labels=label[official_kept]
                official=notebook_metrics(official_labels[np.argsort(-values[official_kept])]) if official_labels.any() else None
                old=original[i]
                if old['reaction_id']!=query_ids[i] or (official is None)!=(old['table2'] is None):raise ValueError('Official query/label mismatch')
                error=max(abs(official[k]-old['table2'][k]) for k in METRICS) if official else 0.
                if error>1e-12:raise ValueError('Official Table-2 parity failed before secondary evaluation')
                reduced=label[kept]
                metrics=notebook_metrics(reduced[np.argsort(-values[kept])]) if reduced.any() else None
                return dict(query_index=i,reaction_id=query_ids[i],positives=int(reduced.sum()),metrics=metrics,official_parity_error=error)
            with ThreadPoolExecutor(max_workers=8) as executor:rows=list(executor.map(evaluate,range(len(queries))))
            measured=[r['metrics'] for r in rows if r['metrics'] is not None]
            value=dict(queries=len(measured),candidate_ids=len(kept),**{k:float(np.mean([r[k] for r in measured])) for k in METRICS})
            result=dict(summary=value,official_table2_parity_max_abs_error=max(r['official_parity_error'] for r in rows),
                frozen_scores_sha256=summary['scores_sha256'],official_summary_sha256=sha256(Path(method['official_summary'])),
                protocol_sha256=sha256(out/'protocol.json'),exclusion_receipt_sha256=sha256(out/'exclusion_receipt.json'),
                source_sha256=sha256(Path(__file__)),test_used_for_model_selection=False,exploratory_secondary_setting=True)
            (dest/'per_query.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));write(dest/'summary.json',result)
            results[method['name']]=value;pending.remove(method);write(out/'status.json',dict(completed=results,pending=[m['name'] for m in pending]))
            print(json.dumps(dict(method=method['name'],summary=value)),flush=True)
            lines=['# Secondary EnzymeMap screening: exact training sequences excluded','',
                'Same frozen scores and notebook BEDROC/EF definitions, with all candidate aliases of exact training sequences removed. This differs from the paper\'s official Table 2 and is reported separately. FGW-CLIP has no predictions for this setting, so no numerical comparison to its published table is made.','',
                'The pool contains 249,828 candidate IDs and 1,333 queries with a surviving positive. Excluding 2,285 additional aliases removes four queries beyond the official training-ID filter. Exact-sequence exclusion does not establish remote-homology separation or measured catalytic generalization.','',
                '| Model | BEDROC85 ↑ | BEDROC20 ↑ | EF5 ↑ | EF10 ↑ |','| --- | ---: | ---: | ---: | ---: |']
            for name,v in results.items():lines.append('| '+f'[{name}]({name}/summary.json)'+' | '+' | '.join(f'{v[k]:.6f}' for k in METRICS)+' |')
            fresh=[results.get(f'beta5_fresh_seed{s}') for s in (17,73,42)]
            if all(fresh):
                cells=[f"{statistics.mean([v[k] for v in fresh]):.6f} ± {statistics.stdev([v[k] for v in fresh]):.6f}" for k in METRICS]
                lines.append('| Three fresh seeds, mean ± sample SD | '+' | '.join(cells)+' |')
            lines+=['','Means summarize independently evaluated models; there are no ensemble predictions. Every method reproduces its official Table-2 per-query metrics before applying the stricter exclusion. This is exploratory analysis of already inspected test data, with no model or threshold tuned on the new outcomes.','',
                '[Predeclared audit methods](protocol.json) · [Exclusion sources and hashes](exclusion_receipt.json)','']
            (out/'comparison.md').write_text('\n'.join(lines))
    write(out/'complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),methods=list(results)))


if __name__=='__main__':main()
