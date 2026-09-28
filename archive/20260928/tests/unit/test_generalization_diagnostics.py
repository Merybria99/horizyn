import json
import numpy as np
import pytest

from horizyn.generalization_diagnostics import (
    associations, band, chemistry_similarity, cluster_interval, decorate_rows,
    novelty, rank_metrics, reaction_components, read_pairs, read_sequence_hits,
    score_matrix_rows, summarize, transfer_baselines,
)


def test_known_unseen_and_mixed_are_distinct():
    assert novelty(['a'],{'a'})=='seen'
    assert novelty(['b'],{'a'})=='unseen'
    assert novelty(['a','b'],{'a'})=='mixed'
    assert band(None,[0,.5,1.000001])=='unknown'
    assert band(1,[0,.5,1.000001])=='0.5-1'


def test_cluster_bridges_and_query_weighting():
    rc,pc=reaction_components([('a','p'),('b','p'),('c','q')])
    assert rc['a']==rc['b']==pc['p']
    assert rc['c']!=rc['a']
    r=cluster_interval([0,0,1],['a','a','c'],repeats=50)
    assert r['mean']==pytest.approx(1/3)
    assert r['clusters']==2
    assert cluster_interval([0,1],['a','a'])['ci95'] is None
    assert r==cluster_interval([0,0,1],['a','a','c'],repeats=50)


def test_recall_is_not_hit_rate_and_ceiling_is_not_one():
    scores=np.arange(20,0,-1)
    result=rank_metrics(scores,[0,19])
    assert result['top_10']==1
    assert result['recall_10']==.5
    assert result['reactzyme_mrr']==pytest.approx((1+1/20)/2)
    assert rank_metrics(np.ones(3),[2])['all_scores_tied']


def test_strata_do_not_reduce_candidate_pool_or_impute_unknown_identity():
    train=[('a','t')];val=[('a','p'),('b','p'),('b','q')]
    rows=score_matrix_rows(np.array([[.9,.1],[.8,.7]]),['a','b'],['p','q'],val)
    decorated=decorate_rows(rows,train,val,{'a':{'similarity':1},'b':{'similarity':.3}},
                            {'p':{'identity':None},'q':{'identity':.8}})
    mixed=next(r for r in decorated if r['direction']=='enzyme_to_reaction' and r['query_id']=='p')
    assert mixed['novelty']=='mixed' and mixed['candidate_count']==2
    assert mixed['sequence_identity'] is None
    unseen=next(r for r in decorated if r['direction']=='reaction_to_enzyme' and r['query_id']=='b')
    assert unseen['positive_count']==2 and unseen['candidate_count']==2
    assert unseen['sequence_hit_coverage']==.5
    report=summarize({'none':decorated,'cls002':decorated,'cls005':decorated},repeats=20)
    assert all(x['mean']==0 for x in report['paired_differences'])
    assert any(x['method']=='cls005' and x['reference']=='cls002' for x in report['paired_differences'])
    with pytest.raises(ValueError,match='coverage'):
        summarize({'none':decorated,'cls005':decorated[:-1]},repeats=20)


def test_training_only_transfer_keeps_missing_hit_queries():
    train=[('a','t1'),('b','t2')]
    similarity=np.array([[1,.2],[.3,1]],dtype=np.float32)
    hits={'p':[('t2',.9,100),('t1',.7,50)]}
    result=transfer_baselines(similarity,['a','b'],['v1','v2'],['p','q'],train,hits)
    np.testing.assert_allclose(result['reaction_neighbor_transfer'],[[.7,0],[.9,0]])
    np.testing.assert_allclose(result['enzyme_neighbor_transfer'],[[.2,0],[1,0]])
    invalid=transfer_baselines(similarity,['a','b'],['v1','v2'],['p'],train,hits,np.array([False,True]))
    assert invalid['reaction_neighbor_transfer'][0,0]==0
    no_overlap=transfer_baselines(np.zeros_like(similarity),['a','b'],['v1','v2'],['p'],train,hits)
    assert not no_overlap['reaction_neighbor_transfer'].any()


def test_sequence_search_filters_coverage_and_rejects_external_ids(tmp_path):
    path=tmp_path/'hits.tsv'
    path.write_text('p\tt\t0.7\t0.9\t0.9\t100\nq\tt\t0.8\t0.5\t0.9\t150\n')
    assert list(read_sequence_hits(path,{'t'},{'p','q'}))==['p']
    with pytest.raises(ValueError,match='outside'):
        read_sequence_hits(path,{'other'},{'p','q'})


def test_chemistry_invalid_is_unknown_not_silently_dropped():
    sim,train_ok,valid_ok=chemistry_similarity(['CCO','CCN'],['CCO','not-smiles'])
    assert sim.shape==(2,2)
    assert sim[0,0]==1 and sim[0,1]<1
    assert valid_ok.tolist()==[True,False]
    assert not sim[1].any()


def test_duplicate_edges_rejected(tmp_path):
    path=tmp_path/'pairs.csv'
    path.write_text('reaction_id,protein_id\na,p\na,p\n')
    with pytest.raises(ValueError,match='duplicate'):read_pairs(path)


def test_existing_evaluator_details_preserve_metrics():
    import torch
    from collections import defaultdict
    from scripts.evaluate_protein_pooling import append_retrieval_metrics
    a,b=defaultdict(list),defaultdict(list)
    scores=torch.arange(20,0,-1,dtype=torch.float32);positive=torch.tensor([0,19])
    assert append_retrieval_metrics(a,scores,positive) is None
    row=append_retrieval_metrics(b,scores,positive,collect_details=True)
    assert a==b and row['recall_10']==.5 and row['positive_count']==2
    assert row['reactzyme_mrr']==pytest.approx(.525)


def test_pipeline_sequence_subset_requires_complete_ids(tmp_path):
    from scripts.run_circe_generalization_diagnostics import write_fastas
    path=tmp_path/'source.fa';path.write_text('>t\nACD\n>p description\nACE\n>unrelated\nWWW\n')
    write_fastas(path,tmp_path,{'t'},{'p'})
    assert (tmp_path/'train.fasta').read_text()=='>t\nACD\n'
    assert (tmp_path/'validation.fasta').read_text()=='>p\nACE\n'
    with pytest.raises(ValueError,match='Missing'):
        write_fastas(path,tmp_path,{'missing'},{'p'})


def test_manifest_rejects_changed_files_and_config(tmp_path):
    from scripts.run_circe_generalization_diagnostics import identity, sha, validate_manifest
    from horizyn.generalization_diagnostics import atomic_json
    source=tmp_path/'source';source.write_text('original')
    config=tmp_path/'validation.yaml';config.write_text('config')
    manifest=dict(sources={'source':identity(source)},checkpoints={},evaluation_config_sha256=sha(config))
    atomic_json(tmp_path/'manifest.json',manifest)
    assert validate_manifest(tmp_path)==manifest
    config.write_text('different')
    with pytest.raises(ValueError,match='config changed'):validate_manifest(tmp_path)
    config.write_text('config');source.write_text('different length')
    with pytest.raises(ValueError,match='Input changed'):validate_manifest(tmp_path)


def test_completion_requires_provenance_and_outputs(tmp_path):
    from scripts.run_circe_generalization_diagnostics import completed, sha
    from horizyn.generalization_diagnostics import atomic_json
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    receipt=tmp_path/'complete.json';out=tmp_path/'queries.json'
    assert not completed(receipt,tmp_path,[out])
    atomic_json(receipt,dict(manifest_sha256=sha(manifest)))
    with pytest.raises(ValueError,match='missing outputs'):completed(receipt,tmp_path,[out])
    out.write_text('{}')
    assert completed(receipt,tmp_path,[out])
    manifest.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='Stale'):completed(receipt,tmp_path,[out])


def test_gpu_preflight_never_accepts_occupied_devices(monkeypatch):
    from scripts.run_circe_generalization_diagnostics import check_gpus
    from types import SimpleNamespace
    monkeypatch.setattr('scripts.run_circe_generalization_diagnostics.subprocess.run',
                        lambda cmd,**kw: SimpleNamespace(stdout='0, GPU-zero\n1, GPU-one\n' if 'index,uuid' in cmd[1]
                                                       else 'GPU-zero, 1234\n'))
    with pytest.raises(RuntimeError,match='occupied'):check_gpus(['0','1'])
    check_gpus(['1'])  # An unrelated job on GPU 0 must not block GPU 1.
    with pytest.raises(ValueError,match='do not exist'):check_gpus(['8'])
    with pytest.raises(ValueError,match='distinct'):check_gpus(['1','1'])
    with pytest.raises(ValueError,match='distinct'):check_gpus(['not-a-gpu'])


def test_cpu_stage_and_report_end_to_end_without_gpu(tmp_path,monkeypatch):
    import h5py
    import yaml
    from types import SimpleNamespace
    from horizyn.generalization_diagnostics import atomic_json
    import scripts.run_circe_generalization_diagnostics as pipeline
    root=tmp_path/'diagnostics';root.mkdir()
    train=tmp_path/'train.csv';train.write_text('reaction_id,protein_id\na,t1\nb,t2\n')
    valid=tmp_path/'valid.csv';valid.write_text('reaction_id,protein_id\na,p\nc,q\n')
    fasta=tmp_path/'input.fa';fasta.write_text('>t1\nACD\n>t2\nACE\n>p\nACD\n>q\nACE\n')
    labels=tmp_path/'labels.npz'
    np.savez(labels,ids=np.array(['t1']),mechanism_mask=np.array([[True]]),cofactor_mask=np.array([[False]]))
    train_rxns=tmp_path/'train_rxns.csv';train_rxns.write_text('reaction_id,reaction_smiles\na,CCO\nb,CCN\n')
    valid_rxns=tmp_path/'valid_rxns.csv';valid_rxns.write_text('reaction_id,reaction_smiles\na,CCO\nc,CCC\n')
    h5=tmp_path/'features.h5'
    with h5py.File(h5,'w') as f:f.create_dataset('ids',data=np.array([b'a_f']))
    chemistry=tmp_path/'chemistry.npz';np.savez(chemistry,ids=np.array(['a','c']),mask=np.array([True,False]))
    config={'data':{f'reaction_{m}_embeds_path':str(h5) for m in ('t5v2','unimol2','chiro')}}
    config['data']['reaction_chemistry_vectors_path']=str(chemistry)
    (root/'validation.yaml').write_text(yaml.safe_dump(config))
    inputs={'train_pairs_path':train,'validation_pairs_path':valid,'fasta':fasta,'labels':labels,
            'train_reactions_path':train_rxns,'validation_reactions_path':valid_rxns,'mmseqs':tmp_path/'mock-mmseqs'}
    manifest={'sources':{k:{'path':str(v)} for k,v in inputs.items()},
              'sequence':dict(max_seqs=64,sensitivity=7.5,evalue=.001,min_coverage=.8)}
    atomic_json(root/'manifest.json',manifest)
    def fake_search(command,**kwargs):
        assert 'easy-search' in command and '--cov-mode' in command
        (root/'sequence/hits.partial.tsv').write_text('p\tt1\t0.9\t1\t1\t100\n')
    monkeypatch.setattr(pipeline,'run',fake_search)
    args=SimpleNamespace(output=root,threads=1,bootstrap=20)
    pipeline.cpu(args)
    metadata=json.loads((root/'novelty.json').read_text())
    assert metadata['sequence_no_hit']==1
    assert metadata['proteins']['q']['identity'] is None
    assert metadata['reactions']['c']['missing_modality']
    assert not metadata['reactions']['a']['missing_modality']
    pipeline.cpu(args)  # A completed stage needs no new search.
    for method in pipeline.METHODS:
        out=root/'models'/method;out.mkdir(parents=True)
        (out/'queries.json').write_text((root/'enzyme_neighbor_transfer.json').read_text())
        atomic_json(out/'metrics.json',{})
        atomic_json(out/'complete.json',dict(manifest_sha256=pipeline.sha(root/'manifest.json')))
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'BASE',tmp_path/'none')
    for folder in (tmp_path/'none',tmp_path/'runs/circe_v3_reaction_smi_cls002.DfNq1C',
                   tmp_path/'runs/circe_v3_reaction_smi_cls005_fresh.piabCZ'):
        log=folder/'logs/protein_pooling_training/version_0';log.mkdir(parents=True)
        (log/'metrics.csv').write_text('epoch,train/loss_mlnce_epoch,train/loss_weighted_biofp_epoch\n8,1.2,0.03\n')
    pipeline.report(args)
    summary=json.loads((root/'summary.json').read_text())
    assert len({r['method'] for r in summary['summaries']})==5
    assert (root/'report.md').exists() and (root/'paired_differences.csv').exists()
    curves=json.loads((root/'training_curves.json').read_text())
    assert curves[0]['values']['train/loss_weighted_biofp_epoch']==.03
