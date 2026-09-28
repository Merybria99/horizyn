#!/usr/bin/env python3
"""Frozen-checkpoint EnzymeMap geometry comparison, with identical evaluation IDs.

Uses released CLIPZyme and the current dictionary-free CERSEI reference.
No fitting, model selection, or changes to the original analysis artifacts.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from torch.nn import functional as F

from cersei_embedding_organization import ROOT, CROSS, OUT as PREVIOUS, read, dump, sha, norm, prefix
from cersei_embedding_metrics import summarize, nearest, neighbor_agreement, query_metrics, write_table

OUT = ROOT / 'runs/clipzyme_embedding_comparison_20260924'
OLD = PREVIOUS / 'enzymemap'
CURRENT = ROOT / 'runs/cersei_horizyn_challenge_20260923/no_dictionary_validation_20260923'
RELEASE = CROSS / 'clipzyme_released_screen_v1'
BASE = CROSS / 'shared_fusion03_beta5_b1024_v1/enzymemap'
PHASE2 = BASE / 'phase2_followup/epoch15'
MODELS = ['CLIPZyme', 'CERSEI']
SEED = 23092026


def shared():
    meta = read(OLD / 'metadata.json')
    sub = read(OLD / 'subsets.json')
    truth = np.load(OLD / 'truth.npy')
    expanded = np.load(OLD / 'expanded.npy')
    return meta, sub, truth, expanded


@torch.inference_mode()
def prepare(device):
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        task='Apply the same embedding diagnostics to a released competitor checkpoint',
        models=MODELS, benchmark='Official CLIPZyme EnzymeMap rule split',
        training_associations=34427, validation_associations=7287, test_associations=4642,
        query_protocol='Reuse original fixed SHA256 subsets: all 1357 test-positive unique sequences and 500 reactions',
        candidates='Full 261907 accessions for R2E; all 1521 test reactions for E2R',
        neural_geometry='Cosine in each model native dimension: CLIPZyme 1280, CERSEI 512',
        sequence_alias_policy='Protein neighborhoods use lexicographically first test-positive accession per unique sequence for CLIPZyme; CERSEI is sequence-based. Alignment retains every official accession.',
        analyses=['EC1-4 agreement@10/50, with/without 50%-identity-component exclusion',
                  'Both-direction coverage@10/50; random, hardest, same-EC1 and similar-participant margins',
                  'Reaction-rule neighborhoods and participant-similarity bins; quality-screened tight matching'],
        uncertainty='1000 paired family-bootstrap replicates, class-balanced means, fixed candidate banks',
        selection='No fitting or selection; fixed current manuscript recipe and official released competitor',
        caveats=['Previously inspected benchmark; exploratory, not confirmatory',
                 'Same downstream data protocol; different pretrained encoders and input resources',
                 'Model cosine-margin scales are not directly calibrated; use retrieval and neighborhood statistics as primary comparisons',
                 'Unannotated pairs are not experimentally verified inactive pairs',
                 'No learned-view suppression for CLIPZyme, which has a different architecture'],
        cersei_recipe={'fusion_multiplier':2.0,'residual_cap':0.5,'dictionary_weight':0.0},
        public_source='https://github.com/pgmikhael/clipzyme',
        release='https://zenodo.org/records/15161343', source_sha256=sha(__file__))
    dump(OUT / 'protocol.json', protocol)
    meta, sub, truth, expanded = shared()
    bank = np.asarray(meta['neighborhood_indices'])
    assert len(bank) == 1357 and len(sub['reaction_indices']) == 500
    current = CURRENT / 'reference_shared_test/enzymemap'
    receipts = {}
    for name, path in [('CLIPZyme', RELEASE), ('CERSEI', current)]:
        assert (path/'candidate_ids.txt').read_text().splitlines() == meta['candidate_ids']
        assert (path/'query_ids.txt').read_text().splitlines() == meta['reaction_ids']
        scores = np.load(path/'scores.npy', mmap_mode='r')
        assert scores.shape == (1521,261907)
        receipt = read(path/('receipt.json' if name=='CLIPZyme' else 'score_receipt.json'))
        assert sha(path/'scores.npy') == receipt['scores_sha256']
        receipts[name] = receipt
    acquisition = read(CROSS/'clipzyme_data_acquisition.json')
    screen_path = Path(acquisition['files']['clipzyme_screening_set.p']['path'])
    assert sha(screen_path) == receipts['CLIPZyme']['screening_pickle_sha256']
    with screen_path.open('rb') as f:
        screen = pickle.load(f)
    assert screen['uniprots'] == meta['candidate_ids']
    protein = screen['hiddens'].cpu().numpy()
    assert protein.shape == (261907,1280) and np.isfinite(protein).all()
    aliases = defaultdict(list)
    for index in np.unique(truth[:,1]):
        aliases[int(expanded[index])].append(int(index))
    representative = np.asarray([min(aliases[int(i)],key=lambda j:meta['candidate_ids'][j]) for i in bank])
    # Verify selected accession sequences against the same unique sequence IDs.
    with Path(acquisition['files']['uniprot2sequence.p']['path']).open('rb') as f:
        sequences = pickle.load(f)
    for i,j in zip(bank,representative):
        assert sequences[meta['candidate_ids'][j]] == meta['sequences'][meta['protein_ids'][i]]
    clip_e = norm(protein[representative])
    clip_r = norm(np.load(RELEASE/'reaction_embeddings.npy'))
    sampled_r = np.asarray(sub['reaction_indices'])[:12]
    expected = np.load(RELEASE/'scores.npy',mmap_mode='r')[sampled_r][:,representative]
    clip_error = float(np.max(np.abs(clip_r[sampled_r] @ clip_e.T - expected)))
    assert clip_error < 3e-5
    clip_checkpoint = ROOT/'runs/cyp_external_v1/assets/clipzyme_models/model_ckpts/clipzyme_model.ckpt'
    assert sha(clip_checkpoint) == receipts['CLIPZyme']['checkpoint_sha256']
    np.savez(OUT/'clipzyme_embeddings.npz', enzyme=clip_e,reaction=clip_r,bank=bank,representative=representative)
    del screen, protein, sequences
    print('CLIPZYME features verified',flush=True)

    from generalization_clipzyme_f3_screen import model_from_checkpoint
    from generalization_multiview_calibration import components
    from horizyn.generalization_residual import FrozenGeometryResidual
    checkpoint = BASE/'checkpoints/screen_selection/screen-epoch=14.ckpt'
    head_path = PHASE2/'training/step0100.pt'
    assert sha(checkpoint) == receipts['CERSEI']['checkpoint_sha256']
    assert sha(head_path) == receipts['CERSEI']['head_sha256']
    freeze = read(CURRENT/'reference_shared_selection.json')
    assert freeze['selected'] == protocol['cersei_recipe']
    assert sha(CURRENT/'reference_shared_selection.json') == receipts['CERSEI']['selection_sha256']
    model, config = model_from_checkpoint(BASE/'configs/train.yaml', checkpoint, device)
    config.data.protein_residue_embeds_path = str(CROSS/'clipzyme_f3_catalog_v1/features/proteins_prott5_residue.local.h5')
    keys = [meta['protein_ids'][i] for i in bank]
    g,f,scale,native,reconstruction_error = components(model,config,keys,device)
    state = torch.load(head_path,map_location=device,weights_only=False)
    assert not state['registry']['test_used']
    assert state['registry']['feature_manifest_sha256'] == sha(PHASE2/'features/manifest.json')
    head = FrozenGeometryResidual(**state['model_config']).to(device).eval().requires_grad_(False)
    head.load_state_dict(state['state_dict'],strict=True)
    base_e = F.normalize(g+2.0*scale*f,dim=-1)
    enzyme = F.normalize(base_e+0.5*head.scale*head.enzyme(base_e),dim=-1)
    source = PHASE2/'selected_test/test_embeddings'
    rr = read(source/'reaction_receipt.json')
    assert rr['checkpoint_sha256'] == sha(checkpoint)
    assert (source/'query_ids.txt').read_text().splitlines() == meta['reaction_ids']
    base_r = torch.as_tensor(np.load(source/'reaction_embeddings.npy'),device=device)
    reaction = F.normalize(base_r+0.5*head.scale*head.reaction(base_r),dim=-1)
    cersei_e = enzyme.cpu().numpy(); cersei_r = reaction.cpu().numpy()
    expected = np.load(current/'scores.npy',mmap_mode='r')[sampled_r][:,representative]
    cersei_error = float(np.max(np.abs(cersei_r[sampled_r] @ cersei_e.T-expected)))
    assert cersei_error < 3e-5
    np.savez(OUT/'cersei_embeddings.npz',enzyme=cersei_e,reaction=cersei_r,bank=bank,
             phase1_enzyme=base_e.cpu().numpy(),phase1_reaction=base_r.cpu().numpy())
    dump(OUT/'prepared.json',dict(complete=True,receipts=receipts,
        score_parity_max_error={'CLIPZyme':clip_error,'CERSEI':cersei_error},
        cersei_reconstruction_error=reconstruction_error, unique_proteins=len(bank),
        test_positive_accessions=len(np.unique(truth[:,1])),reaction_queries=500,
        shared_input_hashes={p:sha(OLD/p) for p in ['metadata.json','subsets.json','truth.npy','expanded.npy','chemistry.npz','mapping_quality.json']},
        export_sha256={n:sha(OUT/n) for n in ['clipzyme_embeddings.npz','cersei_embeddings.npz']},
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print('CURRENT CERSEI features verified',cersei_error,flush=True)


def neighbors(device):
    meta, sub, _, _ = shared(); bank=np.asarray(meta['neighborhood_indices'])
    families=read(PREVIOUS/'homology/families.json')
    fam=np.asarray([families[meta['protein_ids'][i]][0] for i in bank])
    fam50=np.asarray([families[meta['protein_ids'][i]][1] for i in bank])
    stages={m:np.load(OUT/f'{m.lower()}_embeddings.npz')['enzyme'] for m in MODELS}
    z=np.load(OLD/'embeddings.npz');stages['ProtT5']=z['prott5']
    assert np.array_equal(z['prott5_indices'],bank)
    summaries=[];records=[]
    for level in (1,2,3,4):
        labels=[prefix(meta['enzyme_ec'][i],level) for i in bank]; permitted=np.array([bool(x) for x in labels])
        for exclude in (False,True):
            results={}
            for model,embedding in stages.items():
                nn=nearest(embedding,permitted,fam50 if exclude else None,device,k=50)
                np.save(OUT/f'neighbors_{model}_ec{level}_exclude{int(exclude)}.npy',nn)
                for k in (10,50):
                    y=neighbor_agreement(nn,labels,k);results[(model,k)]=y
                    summaries.append(dict(model=model,ec_level=level,k=k,exclude_homologs=exclude,**summarize(y,labels,fam)))
                    records.extend(dict(model=model,protein_id=meta['protein_ids'][bank[i]],ec_level=level,k=k,exclude_homologs=exclude,agreement=float(y[i])) for i in np.flatnonzero(np.isfinite(y)))
            for k in (10,50):
                summaries.append(dict(model='CERSEI-CLIPZyme',ec_level=level,k=k,exclude_homologs=exclude,
                    **summarize(results[('CERSEI',k)]-results[('CLIPZyme',k)],labels,fam)))
    write_table(OUT/'neighborhoods_per_protein.csv',records);dump(OUT/'neighborhoods_summary.json',summaries)
    print('NEIGHBORHOODS complete',flush=True)


def alignment():
    import cersei_embedding_metrics as metrics
    meta,sub,truth,expanded=shared();meta['_split']='enzymemap'
    chem=np.load(OLD/'chemistry.npz')['tanimoto'];all_summaries=[]
    for direction in ['R2E','E2R']:
        ids,labels,fam,pos,coarse,query_ec=metrics.direction_context(meta,truth,direction)
        selected=sub['reaction_indices' if direction=='R2E' else 'enzyme_query_indices']
        queries=sorted(set(selected)&set(pos));nc=len(meta['candidate_ids']) if direction=='R2E' else len(meta['reaction_ids'])
        fixed={}
        for q in queries:
            rng=np.random.default_rng(SEED+q);unknown=np.setdiff1d(np.arange(nc),pos[q],assume_unique=True)
            random=rng.choice(unknown,min(64,len(unknown)),replace=False)
            hard=sorted(set.union(set(),*(coarse[v] for v in prefix(query_ec[q],1))))
            chemical=np.flatnonzero(chem[pos[q]].max(0)>=.5) if direction=='E2R' else None
            fixed[q]=(random,hard,chemical)
        records=[];by_model={}
        for model,path in [('CLIPZyme',RELEASE),('CERSEI',CURRENT/'reference_shared_test/enzymemap')]:
            scores=np.load(path/'scores.npy')
            rr=[]
            for q in queries:
                vector=scores[q] if direction=='R2E' else scores[:,q]
                rr.append(dict(query_index=q,query_id=ids[q],stage=model,direction=direction,
                               **query_metrics(vector,pos[q],*fixed[q])))
            by_model[model]=rr;records.extend(rr);del scores
            print('ALIGNMENT',direction,model,len(rr),flush=True)
        summaries=metrics.alignment_summary(records,meta,truth,direction)
        qi=np.array(queries)
        for metric in [k for k in records[0] if k not in ['query_index','query_id','stage','direction','degree']]:
            delta=np.array([b[metric]-a[metric] for a,b in zip(by_model['CLIPZyme'],by_model['CERSEI'])])
            summaries.append(dict(stage='CERSEI-CLIPZyme',direction=direction,degree_group='all',metric=metric,
                **summarize(delta,[labels[i] for i in qi],fam[qi])))
        write_table(OUT/f'alignment_per_query_{direction}.csv',records);all_summaries.extend(summaries)
    dump(OUT/'alignment_summary.json',all_summaries)


def reactions():
    from cersei_embedding_metrics import labels_matrix
    meta,sub,_,_=shared();ch=np.load(OLD/'chemistry.npz');chem=ch['tanimoto']
    labels=meta['reaction_rules'];fam=np.asarray(meta['reaction_families']);q=np.asarray(sub['reaction_indices']);n=len(labels)
    z=np.load(OLD/'embeddings.npz');r=norm(z['raw_reaction'])
    stages={'Participant fingerprint':chem,'ReactionT5v2':r@r.T}
    for name in MODELS:
        r=norm(np.load(OUT/f'{name.lower()}_embeddings.npz')['reaction']);stages[name]=r@r.T
    membership,_=labels_matrix(labels);same=(membership@membership.T).toarray()>0
    valid=np.array([bool(x) for x in labels])&ch['valid'];pair_valid=valid[:,None]&valid[None,:]&~np.eye(n,dtype=bool)
    summary=[];records=[];metrics={}
    def add(model,metric,y,**extra):
        summary.append(dict(model=model,metric=metric,**extra,**summarize(y,[labels[i] for i in q],fam[q])))
        metrics[(model,metric,tuple(extra.items()))]=y
        for j in np.flatnonzero(np.isfinite(y)):
            records.append(dict(model=model,metric=metric,query_id=meta['reaction_ids'][q[j]],value=float(y[j]),**extra))
    for model,sim in stages.items():
        for exclude in (False,True):
            work=sim.copy();work[~pair_valid]=-np.inf
            if exclude:work[chem>=.5]=-np.inf
            order=np.argsort(-work,axis=1,kind='stable')[:,:50]
            order[~np.isfinite(np.take_along_axis(work,order,axis=1))]=-1
            for k in (10,50):add(model,f'rule_agreement{k}',neighbor_agreement(order,labels,k)[q],exclude_similar_participants=exclude)
        for lo,hi,metric in [(i/10,(i+1)/10,'similarity_bin') for i in range(10)]+[(0.3,0.7,'cross_regime')]:
            y=np.full(len(q),np.nan)
            for j,i in enumerate(q):
                if metric=='cross_regime':yes=pair_valid[i]&same[i]&(chem[i]<=lo);no=pair_valid[i]&~same[i]&(chem[i]>=hi)
                else:
                    mask=pair_valid[i]&(chem[i]>=lo)&(chem[i]<hi if hi<1 else chem[i]<=1)
                    yes=mask&same[i];no=mask&~same[i]
                if yes.any() and no.any():y[j]=sim[i,yes].mean()-sim[i,no].mean()
            add(model,metric,y,bin_low=lo,bin_high=hi)
    quality=np.array(read(OLD/'mapping_quality.json')['min_native_quality'])
    eligible=(quality>=.5)&np.array([len(x)==1 for x in labels]);rule=np.array([x[0] if len(x)==1 else '' for x in labels])
    matched={s:np.full(len(q),np.nan) for s in stages};counts=[];gaps=[]
    for j,i in enumerate(q):
        if not eligible[i]:continue
        same_idx=np.flatnonzero(eligible&(rule==rule[i])&(np.arange(n)!=i));different=np.flatnonzero(eligible&(rule!=rule[i]))
        if not len(same_idx) or not len(different):continue
        order=different[np.argsort(chem[i,different],kind='stable')];vv=chem[i,order]
        position=np.searchsorted(vv,chem[i,same_idx]);left=np.clip(position-1,0,len(vv)-1);right=np.clip(position,0,len(vv)-1)
        chosen=np.where(abs(vv[left]-chem[i,same_idx])<=abs(vv[right]-chem[i,same_idx]),left,right)
        gap=abs(vv[chosen]-chem[i,same_idx]);keep=gap<=.02;a=same_idx[keep];b=order[chosen[keep]]
        if not len(a):continue
        counts.append(len(a));gaps.extend(gap[keep].tolist())
        for model,sim in stages.items():matched[model][j]=np.mean(sim[i,a]-sim[i,b])
    for model,y in matched.items():add(model,'quality_matched_gap',y)
    for model,metric,extra in list(metrics):
        if model=='CERSEI':add('CERSEI-CLIPZyme',metric,metrics[(model,metric,extra)]-metrics[('CLIPZyme',metric,extra)],**dict(extra))
    assert len(counts)==427 and sum(counts)==55704
    dump(OUT/'reaction_summary.json',summary);write_table(OUT/'reaction_per_query.csv',records)
    dump(OUT/'reaction_matching_coverage.json',dict(matched_queries=len(counts),comparisons=sum(counts),mean_tanimoto_gap=float(np.mean(gaps)),max_tanimoto_gap=float(np.max(gaps))))
    print('REACTION ORGANIZATION complete',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','neighbors','alignment','reactions']);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    if args.stage=='prepare':prepare(args.device)
    else:
        assert read(OUT/'prepared.json')['complete']
        if args.stage=='neighbors':neighbors(args.device)
        elif args.stage=='alignment':alignment()
        else:reactions()


if __name__=='__main__':main()
