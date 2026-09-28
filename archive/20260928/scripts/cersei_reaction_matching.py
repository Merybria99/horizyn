#!/usr/bin/env python3
"""Quality-screened reaction-rule sensitivity with tight chemistry matching."""
import numpy as np
from cersei_embedding_organization import OUT,read,dump,norm
from cersei_embedding_metrics import summarize,write_table


def main():
    d=OUT/'enzymemap';dest=d/'analysis3';meta=read(d/'metadata.json');z=np.load(d/'embeddings.npz')
    simchem=np.load(d/'chemistry.npz')['tanimoto'];rules=meta['reaction_rules']
    quality=np.array(read(d/'mapping_quality.json')['min_native_quality']);q=np.array(read(d/'subsets.json')['reaction_indices'])
    # Added as a sensitivity analysis after the unfiltered primary analysis,
    # prompted by the annotation-quality audit, not by a search over outcomes.
    protocol=dict(primary_analysis='Unchanged; native rule labels and 0.1-wide participant similarity bins',
        sensitivity='Both reactions have one native rule ID and minimum native atom-mapping quality >=0.5',
        matching='For each same-rule candidate, choose the different-rule candidate with nearest participant Tanimoto; allow reuse; maximum absolute gap 0.02',
        aggregation='Average matched pair differences within each query, then macro over native rule classes',
        addition_reason='Quality audit found 127/1521 reactions with a minimum mapping quality below 0.5; this diagnostic was added after the primary results',
        causal_claim=False)
    dump(dest/'quality_matching_protocol.json',protocol)
    valid=(quality>=.5)&np.array([len(x)==1 for x in rules]);rule=np.array([x[0] if len(x)==1 else '' for x in rules]);n=len(rule)
    stages=dict(participant_fingerprint=simchem,ReactionT5v2=norm(z['raw_reaction'])@norm(z['raw_reaction']).T,
        phase1=norm(z['phase1_r'])@norm(z['phase1_r']).T,refined=norm(z['refined_r'])@norm(z['refined_r']).T)
    values={s:np.full(len(q),np.nan) for s in stages};records=[];counts=[];gaps=[]
    for qi,i in enumerate(q):
        if not valid[i]:continue
        same=np.flatnonzero(valid & (rule==rule[i]) & (np.arange(n)!=i));different=np.flatnonzero(valid & (rule!=rule[i]))
        if not len(same) or not len(different):continue
        order=different[np.argsort(simchem[i,different],kind='stable')];vv=simchem[i,order]
        p=np.searchsorted(vv,simchem[i,same]);left=np.clip(p-1,0,len(vv)-1);right=np.clip(p,0,len(vv)-1)
        choice=np.where(np.abs(vv[left]-simchem[i,same])<=np.abs(vv[right]-simchem[i,same]),left,right)
        gap=np.abs(vv[choice]-simchem[i,same]);keep=gap<=.02;a=same[keep];b=order[choice[keep]]
        if not len(a):continue
        counts.append(len(a));gaps.extend(gap[keep].tolist())
        row=dict(query_id=meta['reaction_ids'][i],rule=rule[i],matched_pairs=len(a),mean_tanimoto_gap=float(gap[keep].mean()))
        for s,sim in stages.items():
            values[s][qi]=float(np.mean(sim[i,a]-sim[i,b]));row[s]=values[s][qi]
        records.append(row)
    labels=[rules[i] for i in q];families=np.array(meta['reaction_families'])[q]
    results=[dict(stage=s,**summarize(y,labels,families)) for s,y in values.items()]
    for s in ('phase1','refined'):
        results.append(dict(stage=s+'-ReactionT5v2',**summarize(values[s]-values['ReactionT5v2'],labels,families)))
    dump(dest/'quality_matching_summary.json',results);write_table(dest/'quality_matching_per_query.csv',records)
    dump(dest/'quality_matching_coverage.json',dict(candidate_reactions=n,eligible_candidates=int(valid.sum()),
        query_sample=len(q),eligible_queries=int(valid[q].sum()),matched_queries=len(counts),
        matched_comparisons=sum(counts),mean_tanimoto_gap=float(np.mean(gaps)),max_tanimoto_gap=float(np.max(gaps))))
    for r in results:print(r['stage'],r['macro'],r['ci95'],r['evaluated_queries'])


if __name__=='__main__':main()
