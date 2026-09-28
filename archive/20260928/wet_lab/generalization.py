"""Optional frozen phase2/phase4 adapter for the existing wet-lab query workflow."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from horizyn.generalization_phase2 import ComposedPhase2Encoder
from horizyn.generalization_retrieval import checked_artifact, sha256
from scripts.generalization_export import reaction_block

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def record(path):
    path = resolve(path)
    return dict(path=str(path), sha256=sha256(path))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def tensor_sha256(values):
    result = hashlib.sha256()
    for start in range(0, len(values), 2048):
        result.update(values[start:start + 2048].detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def stable_topk(scores, indices, count):
    """Retain the existing deterministic catalog index as the tie breaker."""
    order = torch.argsort(indices, stable=True)
    order = order[torch.argsort(scores[order], descending=True, stable=True)[:count]]
    return scores[order], indices[order]


def stored_residue_means(handle, protein_ids, lookup=None, offsets=None):
    """Match the research exporter: all stored rows, FP32 mean, no re-truncation."""
    if lookup is None:
        ids = handle["ids"].asstr()[:].tolist()
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate residue IDs")
        lookup = {key: row for row, key in enumerate(ids)}
    if offsets is None:
        offsets = handle["offsets"][:]
    output = []
    for key in protein_ids:
        if key not in lookup:
            raise ValueError(f"Missing full-residue mean candidate: {key}")
        row = lookup[key]
        values = np.asarray(handle["vectors"][offsets[row]:offsets[row + 1]], np.float32)
        if not len(values) or values.shape[1] != 1024 or not np.isfinite(values).all():
            raise ValueError(f"Invalid stored ProtT5 residues: {key}")
        output.append(values.mean(axis=0, dtype=np.float32))
    return np.asarray(output, np.float32)


def _canonical_t5_name(value):
    value = str(value)
    if "models--sagawa--ReactionT5v2-forward/" in value:
        return "sagawa/ReactionT5v2-forward"
    return value


def authenticate_contract(bundle, specification, contract_path, checkpoint, config, feature_manifest, feature_paths):
    contract = json.loads(contract_path.read_text())
    freeze_path = checked_artifact(contract["phase2_frozen_recipe"], contract_path.parent)
    if contract["phase2_frozen_recipe"]["sha256"] != specification["phase2_frozen_recipe"]["sha256"]:
        raise ValueError("Feature-generation contract belongs to another frozen bundle")
    frozen = json.loads(freeze_path.read_text())
    split = specification["split"]
    settings = contract["splits"][split]
    if settings.get("reaction_input_policy") != "participant_self_reaction":
        raise ValueError("Unsupported frozen query representation policy")
    pinned = frozen["feature_export_receipts"][split]
    if settings["feature_export_receipt"]["sha256"] != pinned["sha256"]:
        raise ValueError("Feature-generation contract is not descended from the frozen export")
    receipt_path = checked_artifact(pinned, freeze_path.parent)
    receipt = json.loads(receipt_path.read_text())
    source_path = checked_artifact(settings["source_manifest"], contract_path.parent)
    if not any(value["sha256"] == settings["source_manifest"]["sha256"] for value in receipt["source_receipts"]):
        raise ValueError("Chemistry source manifest is outside the frozen export lineage")
    source = json.loads(source_path.read_text())
    schema = checked_artifact(source["inputs"]["chemistry_schema"], source_path.parent)
    if record(resolve(config["feature_generation"]["reaction_chemistry_schema"]))["sha256"] != sha256(schema):
        raise ValueError("Query chemistry schema differs from frozen training schema")
    if settings["chemistry_schema"]["sha256"] != sha256(schema):
        raise ValueError("Deployment chemistry schema binding changed")
    cofactor = checked_artifact(contract["cofactor_dictionary"], contract_path.parent)
    if sha256(resolve(config["feature_generation"]["cofactor_dictionary"])) != sha256(cofactor):
        raise ValueError("Query cofactor dictionary differs from deployment contract")
    if sha256(checkpoint) != specification["base_checkpoint"]["sha256"]:
        raise ValueError("Query base checkpoint differs from frozen phase2 bundle")
    if feature_manifest.get("reaction_input_policy", {}).get("policy") != "participant_self_reaction":
        raise ValueError("This frozen F3 bundle requires training-matched participant self-reactions")
    if feature_manifest.get("reaction_feature_policy") != "participant_self_reaction_v1":
        raise ValueError("Query modalities were not prepared using the frozen participant policy")
    for name, input_key in (("reaction_t5", "t5v2"), ("unimol2", "unimol2"), ("chiro", "chiro")):
        reference_path = checked_artifact(source["inputs"][input_key], source_path.parent)
        with h5py.File(reference_path) as original, h5py.File(feature_paths[name]) as current:
            keys = ("model_name", "embedding_dim")
            if name == "unimol2": keys += ("model_size",)
            if name == "chiro": keys += ("embedding_kind", "checkpoint", "params")
            for key in keys:
                expected, actual = original.attrs.get(key), current.attrs.get(key)
                if key == "model_name" and name == "reaction_t5":
                    expected, actual = _canonical_t5_name(expected), _canonical_t5_name(actual)
                if expected is None or actual != expected:
                    raise ValueError(f"Query {name} {key} differs from authenticated training feature family")
    return dict(contract=record(contract_path), bundle=record(bundle), checkpoint=record(checkpoint),
                schema=record(schema), cofactor_dictionary=record(cofactor),
                cofactor_provenance=contract["cofactor_provenance"],
                modality_inputs={key: record(path) for key, path in feature_paths.items()})


def raw_query_blocks(feature_paths, query_id):
    blocks, masks = {}, {}
    for key, path_key, dimension in (("t5v2", "reaction_t5", 768), ("unimol2", "unimol2", 768), ("chiro", "chiro", 256)):
        block, mask = reaction_block(feature_paths[path_key], [query_id], key != "t5v2")
        if block.shape != (1, dimension) or (key == "t5v2" and not mask.all()):
            raise ValueError(f"Missing or invalid query {key} features")
        blocks[key], masks[key] = block, mask
    with np.load(feature_paths["chemistry"], allow_pickle=True) as source:
        keys = [str(key) for key in source["ids"]]
        if keys.count(query_id) != 1:
            raise ValueError("Chemistry query ID must occur exactly once")
        i = keys.index(query_id)
        blocks["chemistry"] = np.asarray(source["vectors"][i:i + 1], np.float32)
        masks["chemistry"] = np.asarray(source["mask"][i:i + 1], bool)
    if blocks["chemistry"].shape != (1, 617) or any(not np.isfinite(value).all() for value in blocks.values()):
        raise ValueError("Invalid query raw modalities")
    return blocks, masks


def load_frozen_query_bundle(bundle, device):
    """Dispatch only explicit supported schemas; authenticate phase4 sources."""
    specification = json.loads(bundle.read_text())
    schema = specification.get("schema")
    if schema == "generalization_phase2_bundle_v1":
        return ComposedPhase2Encoder.from_bundle(bundle, device)
    if schema == "phase4_hybrid_bundle_v1":
        from scripts.generalization_phase4_predict import validate_frozen_sources
        validate_frozen_sources(specification["phase4_frozen_recipe"],
            [ROOT / "horizyn/generalization_phase4.py"])
        from horizyn.generalization_phase4 import HybridPhase4Encoder
        return HybridPhase4Encoder.from_bundle(bundle, device)
    raise ValueError("Unsupported frozen query bundle schema")


def build_candidate_index(model, base, protein_ids, residue_h5, cache_root, bundle_identity, device, batch_size):
    if batch_size < 1 or len(base) != len(protein_ids) or len(set(protein_ids)) != len(protein_ids):
        raise ValueError("Candidate rows must be unique, complete and aligned")
    source = dict(bundle=bundle_identity, residues=record(residue_h5), base_sha256=tensor_sha256(base),
        ordered_ids_sha256=hashlib.sha256(json.dumps(protein_ids, ensure_ascii=False).encode()).hexdigest(),
        count=len(protein_ids), mean_policy="all stored residues; NumPy mean(dtype=float32); no dataset truncation",
        batch_size=batch_size)
    signature = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    directory = cache_root / signature
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "complete.json"
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["source"] != source:
                raise ValueError("Candidate-index cache provenance mismatch")
            for chunk in receipt["chunks"]:
                checked_artifact(chunk["artifact"], directory)
            return directory, receipt, "hit"
        chunks = []
        with h5py.File(residue_h5) as residues:
            keys = residues["ids"].asstr()[:].tolist()
            if len(keys) != len(set(keys)): raise ValueError("Duplicate residue IDs")
            lookup = {key: row for row, key in enumerate(keys)}
            offsets = residues["offsets"][:]
            for start in range(0, len(base), batch_size):
                end = min(start + batch_size, len(base))
                means = torch.tensor(stored_residue_means(residues, protein_ids[start:end], lookup, offsets), device=device)
                native = base[start:end].float().to(device)
                index = model.encode_enzyme_index(native, means, batch_size)
                _, diagnostics = model.density.encode_enzymes(native, means, batch_size, return_diagnostics=True)
                payload = dict(index={key: value.cpu() for key, value in index.items()},
                    diagnostics={key: value.cpu() for key, value in diagnostics.items() if value is not None})
                path = directory / f"chunk_{start:09d}.pt"
                temporary = path.with_suffix(".partial")
                torch.save(payload, temporary)
                os.replace(temporary, path)
                chunks.append(dict(start=start, end=end, artifact=record(path),
                    semantic_nonzero_values=index["anchors"].values().numel(),
                    semantic_columns=index["anchors"].shape[1]))
        receipt = dict(schema="wet_lab_phase2_candidate_index_v1", source=source, chunks=chunks)
        atomic_json(receipt_path, receipt)
    return directory, receipt, "built"


@torch.inference_mode()
def rank_index(model, query, receipt, top_k, device):
    scores, indices = torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    selected_diagnostics = {}
    for chunk in receipt["chunks"]:
        payload = torch.load(checked_artifact(chunk["artifact"]), map_location=device, weights_only=False)
        values = model.score_index(query, payload["index"])[0]
        if not torch.isfinite(values).all(): raise ValueError("Nonfinite frozen-bundle scores")
        rows = torch.arange(chunk["start"], chunk["end"], device=device)
        local, local_indices = stable_topk(values, rows, min(top_k, len(values)))
        for i in local_indices.tolist():
            selected_diagnostics[i] = {key: float(value[i - chunk["start"]]) for key, value in payload["diagnostics"].items()}
        scores, indices = stable_topk(torch.cat((scores, local)), torch.cat((indices, local_indices)), top_k)
    return scores.cpu(), indices.cpu(), [selected_diagnostics[i] for i in indices.tolist()]


def rank_generalized_query(*, config, workspace, checkpoint_path, candidate_keys, residue_h5,
                           target_embeddings, query_embedding, query_id, feature_paths,
                           feature_manifest, top_k, device):
    bundle = resolve(config["model"]["generalization_bundle"])
    if not config["model"].get("generalization_feature_contract"):
        raise ValueError("model.generalization_feature_contract is required with a frozen bundle")
    contract_path = resolve(config["model"]["generalization_feature_contract"])
    model, specification = load_frozen_query_bundle(bundle, device)
    provenance = authenticate_contract(bundle, specification, contract_path, checkpoint_path, config, feature_manifest, feature_paths)
    blocks, masks = raw_query_blocks(feature_paths, query_id)
    blocks = {key: torch.tensor(value, device=device) for key, value in blocks.items()}
    masks = {key: torch.tensor(value, device=device) for key, value in masks.items()}
    batch_size = int(config.get("inference", {}).get("generalization_batch_size", 512))
    query, diagnostics = model.encode_reactions(query_embedding.float().to(device), blocks, masks, batch_size, return_diagnostics=True)
    cache_root = resolve(config["candidate_pool"].get("generalization_cache_dir",
        str(resolve(config["candidate_pool"]["target_cache_dir"]) / "generalization")))
    directory, receipt, status = build_candidate_index(model, target_embeddings, candidate_keys, residue_h5,
        cache_root, record(bundle), device, batch_size)
    scores, indices, candidate_diagnostics = rank_index(model, query, receipt, top_k, device)
    phase = "phase4" if specification["schema"] == "phase4_hybrid_bundle_v1" else "phase2"
    audit = dict(schema=f"wet_lab_frozen_{phase}_query_v1", labels_used=False, calibrated_probability=False,
        bundle_schema=specification["schema"],
        model_freezes={key: specification[key] for key in ("frozen_recipe", "phase2_frozen_recipe", "phase4_frozen_recipe") if key in specification},
        source=provenance, index_receipt=record(directory / "complete.json"), index_status=status,
        mean_policy=receipt["source"]["mean_policy"],
        score="Frozen independent dense+CSR score_index; FP64 accumulation rounded to FP32; existing catalog-index ties",
        query_diagnostics={key: value.cpu().tolist() for key, value in diagnostics.items() if value is not None},
        query_semantic_endpoint=dict(nonzero_values=int((query[:, 512:] != 0).sum()),
            minimum=float(query[:, 512:].min()), maximum=float(query[:, 512:].max())),
        candidate_diagnostics=[dict(protein_id=candidate_keys[int(i)], **row) for i, row in zip(indices, candidate_diagnostics)],
        diagnostics_note="Training-support cosine and gate scale are model diagnostics, not calibrated probabilities")
    atomic_json(workspace / "generalization.json", audit)
    return scores, indices, audit
