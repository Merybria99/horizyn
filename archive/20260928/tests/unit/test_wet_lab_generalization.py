from types import SimpleNamespace
import h5py
import numpy as np
import pytest
import torch

from horizyn.generalization_retrieval import GeneralizationDualEncoder
from wet_lab.generalization import build_candidate_index, rank_index, stored_residue_means, authenticate_contract, record, atomic_json
from wet_lab.query import rank_query_optional_bundle, streaming_topk


class SyntheticIndexModel:
    def __init__(self):
        self.calls = 0
        self.density = SimpleNamespace(encode_enzymes=lambda base, means, batch_size, return_diagnostics:
            (base,dict(support_cosine=means[:,0],gate_scale=torch.ones(len(base)))))
    def encode_enzyme_index(self, base, means, batch_size):
        self.calls += 1
        return dict(dense=base,anchors=means[:,:2].to_sparse_csr())
    score_index = staticmethod(GeneralizationDualEncoder.score_index)


def residues(path):
    with h5py.File(path,"w") as handle:
        handle["ids"] = np.asarray(["P0","P1","P2"],dtype=h5py.string_dtype())
        handle["offsets"] = [0,2,4,6]
        vectors=np.zeros((6,1024),np.float32)
        vectors[:2,:2]=[1,0];vectors[2:4,:2]=[0,1];vectors[4:,:2]=[1,0]
        handle["vectors"]=vectors


def test_sparse_hook_scores_match_independent_vectors_and_catalog_ties(tmp_path):
    path=tmp_path/"residues.h5";residues(path)
    model=SyntheticIndexModel()
    ids=["P2","P0","P1"]  # Preserve this catalog order, not lexical ID order.
    base=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
    directory,receipt,status=build_candidate_index(model,base,ids,path,tmp_path/"cache",dict(sha256="bundle"),"cpu",2)
    query=torch.tensor([[1.,0.,1.,0.]])
    scores,indices,diagnostics=rank_index(model,query,receipt,3,"cpu")
    with h5py.File(path)as h:means=torch.tensor(stored_residue_means(h,ids))
    expected=(query.double()@torch.cat((base,means[:,:2]),dim=1).double().T).float()[0]
    assert torch.equal(scores,expected[indices])
    assert indices.tolist()==[0,1,2]
    assert diagnostics[0]["support_cosine"]==1
    calls=model.calls
    reused,_,hit=build_candidate_index(model,base,ids,path,tmp_path/"cache",dict(sha256="bundle"),"cpu",2)
    assert reused==directory and hit=="hit" and model.calls==calls
    with h5py.File(path,"a")as h:h["vectors"][0,0]=2
    changed,_,status=build_candidate_index(model,base,ids,path,tmp_path/"cache",dict(sha256="bundle"),"cpu",2)
    assert changed!=directory and status=="built"


def test_candidate_cache_rejects_changed_chunk(tmp_path):
    path=tmp_path/"residues.h5";residues(path)
    args=(SyntheticIndexModel(),torch.eye(3),["P0","P1","P2"],path,tmp_path/"cache",dict(sha256="bundle"),"cpu",2)
    directory,receipt,_=build_candidate_index(*args)
    chunk=directory/"chunk_000000000.pt"
    with chunk.open("ab")as handle:handle.write(b"changed")
    with pytest.raises(ValueError,match="hash mismatch"):build_candidate_index(*args)


def test_raw_means_average_all_stored_rows_without_dataset_truncation(tmp_path):
    path=tmp_path/"long.h5"
    with h5py.File(path,"w")as h:
        h["ids"]=np.asarray(["long"],dtype=h5py.string_dtype());h["offsets"]=[0,2000]
        values=np.zeros((2000,1024),np.float32);values[1000:]=2;h["vectors"]=values
    with h5py.File(path)as h:actual=stored_residue_means(h,["long"])
    assert np.array_equal(actual,np.ones((1,1024),np.float32))


def test_query_default_remains_legacy_and_bundle_is_explicit(monkeypatch):
    import wet_lab.generalization as adapter
    query=torch.tensor([[1.,0.]]);targets=torch.tensor([[1.,0.],[0.,1.]])
    args=dict(query_embedding=query,target_embeddings=targets,top_k=2,device="cpu")
    scores,indices,audit=rank_query_optional_bundle(dict(model={},inference={}),**args)
    expected=streaming_topk(query,targets,top_k=2,device="cpu")
    assert torch.equal(scores,expected[0]) and torch.equal(indices,expected[1]) and audit is None
    marker=object()
    monkeypatch.setattr(adapter,"rank_generalized_query",lambda **kwargs:(scores,indices,marker))
    assert rank_query_optional_bundle(dict(model={"generalization_bundle":"explicit"}),**args)[2] is marker


def test_optional_adapter_rejects_schema_outside_frozen_export_lineage(tmp_path):
    schema=tmp_path/"train_schema.json";schema.write_text("train-fitted schema")
    wrong=tmp_path/"other_schema.json";wrong.write_text("different fitted schema")
    source=tmp_path/"manifest.json";atomic_json(source,dict(inputs=dict(chemistry_schema=record(schema))))
    receipt=tmp_path/"feature_receipt.json";atomic_json(receipt,dict(source_receipts=[record(source)]))
    freeze=tmp_path/"freeze.json";atomic_json(freeze,dict(feature_export_receipts=dict(reaction_smi=record(receipt))))
    contract=tmp_path/"contract.json";atomic_json(contract,dict(phase2_frozen_recipe=record(freeze),splits=dict(reaction_smi=dict(
        feature_export_receipt=record(receipt),source_manifest=record(source),chemistry_schema=record(schema),reaction_input_policy="participant_self_reaction"))))
    spec=dict(split="reaction_smi",phase2_frozen_recipe=record(freeze))
    with pytest.raises(ValueError,match="differs from frozen training schema"):
        authenticate_contract(tmp_path/"bundle.json",spec,contract,tmp_path/"checkpoint.pt",
            dict(feature_generation=dict(reaction_chemistry_schema=str(wrong))),{}, {})


def test_bundle_schema_dispatch_retains_phase2_and_authenticates_phase4(tmp_path,monkeypatch):
    import wet_lab.generalization as adapter
    import scripts.generalization_phase4_predict as predictor
    import horizyn.generalization_phase4 as phase4
    bundle=tmp_path/'bundle.json';calls=[];marker=object()
    monkeypatch.setattr(adapter.ComposedPhase2Encoder,'from_bundle',lambda p,d:(calls.append(('phase2',p,d)) or (marker,{'schema':'generalization_phase2_bundle_v1'})))
    atomic_json(bundle,dict(schema='generalization_phase2_bundle_v1'))
    assert adapter.load_frozen_query_bundle(bundle,'cpu')[0] is marker
    assert calls[-1][0]=='phase2'
    monkeypatch.setattr(predictor,'validate_frozen_sources',lambda record,required:calls.append(('guard',record)))
    monkeypatch.setattr(phase4.HybridPhase4Encoder,'from_bundle',lambda p,d:(calls.append(('phase4',p,d)) or (marker,{'schema':'phase4_hybrid_bundle_v1'})))
    atomic_json(bundle,dict(schema='phase4_hybrid_bundle_v1',phase4_frozen_recipe={'sha256':'new'}))
    assert adapter.load_frozen_query_bundle(bundle,'cpu')[0] is marker
    assert [row[0] for row in calls[-2:]]==['guard','phase4']
    atomic_json(bundle,dict(schema='unknown'))
    with pytest.raises(ValueError,match='Unsupported'):adapter.load_frozen_query_bundle(bundle,'cpu')


def test_phase4_adapter_does_not_load_model_after_source_rejection(tmp_path,monkeypatch):
    import wet_lab.generalization as adapter
    import scripts.generalization_phase4_predict as predictor
    import horizyn.generalization_phase4 as phase4
    bundle=tmp_path/'bundle.json';atomic_json(bundle,dict(schema='phase4_hybrid_bundle_v1',phase4_frozen_recipe={}))
    def reject(*args):raise ValueError('Frozen source changed')
    def forbidden(*args):raise AssertionError('Model loaded after source rejection')
    monkeypatch.setattr(predictor,'validate_frozen_sources',reject);monkeypatch.setattr(phase4.HybridPhase4Encoder,'from_bundle',forbidden)
    with pytest.raises(ValueError,match='Frozen source changed'):adapter.load_frozen_query_bundle(bundle,'cpu')
