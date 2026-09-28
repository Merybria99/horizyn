#!/usr/bin/env python3
"""CPU-only ReactZyme reaction-smi setup for the indexed 85/15 training recipe.

Reuses all frozen embeddings. Writes only into a fresh experiment directory.
The released molecule-set chemistry and original train/validation split remain
unchanged; _f aliases connect those rows to the existing feature-store IDs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
F3 = ROOT / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/train.yaml"
RECIPE = ROOT / "runs/horizyn1_circe_v2_reaction_holdout_h200_paired/configs/train.yaml"
BIO = ROOT / "runs/reactzyme_f3_biological_v2/data/reaction_smi/biofp"
EC = ROOT / "data/standardized/retrieval_training_source_collapse/hyperbolic_ec_labels/nr90_valid_prefix_ec_labels.csv"
SPLIT = ROOT / "data/revised_protocols/reactzyme_paper/reaction_smi"
PAIR_COLUMNS = ("pr_id", "reaction_id", "protein_id")


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def rooted(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def forward_id(value):
    value = value.strip()
    if not value or value.endswith(("_f", "_r")):
        raise ValueError("Expected original, unsuffixed ReactZyme reaction IDs")
    return value + "_f"


def alias_csv(source, destination, columns):
    with source.open(newline="") as src, destination.open("x", newline="") as dest:
        reader = csv.DictReader(src)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {source}")
        writer = csv.DictWriter(dest, fieldnames=columns)
        writer.writeheader()
        seen_pair_ids = set()
        for row in reader:
            row = {key: row[key] for key in columns}
            if "pr_id" in columns:
                pair_id = row["pr_id"]
                if not pair_id or not pair_id.strip() or pair_id in seen_pair_ids:
                    raise ValueError(f"Empty or duplicate pr_id in {source}: {pair_id!r}")
                seen_pair_ids.add(pair_id)
            row["reaction_id"] = forward_id(row["reaction_id"])
            writer.writerow(row)


def check_pair_loader(path):
    """Exercise the same CSV contract used by validation before starting DDP."""
    from horizyn.datasets.csv import CSVDataset

    pairs = CSVDataset(
        file_path=str(path), key_column="pr_id",
        columns=["reaction_id", "protein_id"],
        rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
    )
    print(f"Pair-loader check passed: {path.name}, {len(pairs):,} rows", flush=True)
    return len(pairs)


def safe_label_proteins(source_pairs, target_edges):
    """Reject an aggregate if ANY source edge is absent from current training.

    Only source-train and current-train CSVs are inspected. No validation/test
    edges, labels, or reaction profiles enter the training index.
    """
    seen, unsafe = set(), set()
    with source_pairs.open(newline="") as handle:
        for row in csv.DictReader(handle):
            edge = (row["reaction_id"].strip(), row["protein_id"].strip())
            seen.add(edge[1])
            if edge not in target_edges:
                unsafe.add(edge[1])
    return seen - unsafe, unsafe


def config_for_run(base, recipe, output):
    config = copy.deepcopy(base)
    data = config["data"]
    # Keep the established ReactZyme feature representation, not Horizyn's
    # directional chemistry: released ReactZyme inputs are molecule sets.
    for key, value in list(data.items()):
        if key.endswith("_path") and value:
            data[key] = str(rooted(value))
    data.update(
        train_pairs_path=str(output / "data/train_pairs.csv"),
        train_reactions_path=str(output / "data/train_rxns.csv"),
        validation_pairs_path=str(output / "data/validation_pairs.csv"),
        validation_reactions_path=str(output / "data/validation_rxns.csv"),
        indexed_pairs_dir=str(output / "data/index"), typed_negative_pools_path=None,
        typed_negative_positive_fraction=0.85, typed_negative_biological_fraction=0.5,
        typed_negative_seed=42, train_batch_size=200, retrieval_batch_size=128,
        validation_retrieval_batch_size=128, num_workers=2, persistent_workers=True,
        prefetch_factor=1, worker_num_threads=1, pin_memory=True,
        reaction_direction_mode="forward_only", normalize_molecule_sets_as_self_reactions=True,
    )
    data.pop("validation_retrieval_query_ids_path", None)
    config["model"]["sleec_pooling"].update(
        score_hidden_dim=1024, score_embedding_source="same",
        checkpoint_path=str(rooted(config["model"]["sleec_pooling"]["checkpoint_path"])),
    )
    # No warm-start from a model trained on Horizyn or the ReactZyme test set.
    training = copy.deepcopy(recipe["training"])
    for key in ("init_from_checkpoint", "biofp_pretrain_checkpoint", "training_stage",
                "validation_retrieval_query_ids_path", "limit_train_batches"):
        training.pop(key, None)
    training.update(devices=2, validation_interval_steps=None, check_val_every_n_epoch=1,
                    validation_retrieval_candidate_ids_path=data["validation_retrieval_candidate_ids_path"],
                    validation_retrieval_batch_size=128, validation_retrieval_candidate_chunk_size=8192)
    config["training"] = training
    config["seed"] = 42
    config["logging"] = copy.deepcopy(recipe["logging"])
    config["logging"].update(log_dir=str(output / "configured_logs"),
                             checkpoint_dir=str(output / "configured_checkpoints"),
                             log_every_n_steps=20, recovery_every_n_train_steps=100)
    config["logging"]["wandb"] = {"enabled": False, "mode": "disabled"}
    config["ablation"] = dict(split="reaction_smi", variant="tyrosine_loss_recipe_cached",
                               effective_global_batch_size=400, feature_extraction=False,
                               representation="existing ReactZyme F3 molecule sets",
                               annotations="train-filtered legacy mechanism and native cofactor positives")
    return config


def prepare(output):
    import numpy as np
    import yaml
    from build_indexed_training_pairs import load_graph, load_ec, ProteinLookup, source_record
    from build_annotation_negative_pools import reaction_signature
    from horizyn.datasets.indexed_pairs import write_indexed_pairs, IndexedPairs, IndexedTypedNegativeBatchSampler, SEMANTICS

    if (output / "data").exists() or (output / "configs").exists():
        raise ValueError("Setup output already exists; refuse to overwrite it")
    vocab_path = BIO / "enzyme_biofp_vocab.json"
    npz_path = BIO / "enzyme_biofp_soft_targets.npz"
    vocab = json.loads(vocab_path.read_text())
    if vocab.get("schema_version") != "biofp_minimal_v1":
        raise ValueError("Unrecognized legacy biological-label schema")
    from horizyn.capability.cofactor_vocabulary_v2 import KNOWN_COFACTOR_LABELS
    if tuple(vocab["families"]["cofactor"]) != KNOWN_COFACTOR_LABELS[:10]:
        raise ValueError("Legacy cofactor group order does not match the v2 prefix")
    expected_mechanisms = ["redox_carbonyl_interconversion", "phosphate_transfer", "acyl_transfer",
                           "glycosyl_transfer", "c_n_transfer_transamination", "sulfur_thiol_chemistry",
                           "stereochemical_rearrangement", "hydrolysis_condensation"]
    if vocab["families"]["mechanism"] != expected_mechanisms:
        raise ValueError("Unexpected mechanism vocabulary")
    # These legacy cofactors are documented curated enzyme annotations, not
    # reaction-side participants. Do not infer new metal labels or negative bits.
    if not vocab["sources"].get("enzyme_cofactors"):
        raise ValueError("Curated native-cofactor provenance is required")
    sources = [F3, RECIPE, EC, vocab_path, npz_path]
    for role, path in vocab["sources"].items():
        path = rooted(path)
        print(f"Checking annotation provenance: {role}", flush=True)
        if sha(path) != vocab["source_sha256"][role]:
            raise ValueError(f"Annotation provenance hash changed: {role}")
        sources.append(path)
    sources += [SPLIT / f"{part}_{kind}.csv" for part in ("train", "validation") for kind in ("pairs", "rxns")]
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in sources}
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir()
    (output / "configs").mkdir()
    for part in ("train", "validation"):
        alias_csv(SPLIT / f"{part}_pairs.csv", output / f"data/{part}_pairs.csv", PAIR_COLUMNS)
        alias_csv(SPLIT / f"{part}_rxns.csv", output / f"data/{part}_rxns.csv", ("reaction_id", "reaction_smiles"))
        check_pair_loader(output / f"data/{part}_pairs.csv")
    with (SPLIT / "train_pairs.csv").open(newline="") as handle:
        train_edges = {(row["reaction_id"].strip(), row["protein_id"].strip()) for row in csv.DictReader(handle)}
    safe, unsafe = safe_label_proteins(rooted(vocab["sources"]["train_pairs"]), train_edges)
    print(f"Train-scoped label reuse: {len(safe):,} safe source proteins; {len(unsafe):,} aggregates excluded", flush=True)
    with tempfile.TemporaryDirectory(prefix="index-build-", dir=output / "data") as temp:
        qids, pids, edges = load_graph(output / "data/train_pairs.csv", Path(temp))
        lookup = ProteinLookup(pids)
        protein_ec, prefix, qptr, qec = load_ec(EC, lookup, edges, len(qids), 2)
        mechanisms = np.zeros(len(pids), dtype=np.uint32)
        native = np.zeros(len(pids), dtype=np.uint32)
        with np.load(npz_path, allow_pickle=True) as payload:
            label_ids = payload["ids"].astype(str)
            if len(set(label_ids)) != len(label_ids):
                raise ValueError("Duplicate annotation IDs")
            mapped = lookup.many(label_ids)
            # IDs in these artifacts are the original prot_<hash> IDs.
            safe_canonical = {value.removeprefix("prot_") for value in safe}
            allowed = np.asarray([str(value).removeprefix("prot_") in safe_canonical for value in label_ids]) & (mapped >= 0)
            if len(set(mapped[mapped >= 0].tolist())) != int((mapped >= 0).sum()):
                raise ValueError("Canonical annotation ID collision")
            for family, width, destination in (("mechanism", 8, mechanisms), ("cofactor", 10, native)):
                values, mask = payload[f"{family}_targets"], payload[f"{family}_mask"]
                if values.shape != (len(label_ids), width) or mask.shape != values.shape:
                    raise ValueError(f"Wrong legacy {family} array shape")
                if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                    raise ValueError(f"Invalid legacy {family} targets")
                if not np.isfinite(mask).all() or np.any((mask != 0) & (mask != 1)):
                    raise ValueError(f"Invalid legacy {family} mask")
                positives = (values > 0) & (mask > 0)
                destination[mapped[allowed]] = positives[allowed].astype(np.uint32) @ (np.uint32(1) << np.arange(width, dtype=np.uint32))
        signatures = {}
        with (output / "data/train_rxns.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                # Released ReactZyme chemistry is an undirected molecule set;
                # use the same self-reaction normalization as its F3 loader.
                smiles = row["reaction_smiles"]
                if ">" not in smiles:
                    smiles = smiles + ">>" + smiles
                qid = row["reaction_id"]
                signature = reaction_signature(smiles, qid)
                if qid in signatures and signatures[qid] != signature:
                    raise ValueError("Conflicting reaction records")
                signatures[qid] = signature
        _, signature_codes = np.unique([signatures[str(q)] for q in qids], return_inverse=True)
        provenance = dict(pair_scope="train", annotation_semantics=SEMANTICS,
                          source_files=[source_record(path) for path in (SPLIT / "train_pairs.csv", SPLIT / "train_rxns.csv", EC, vocab_path, npz_path)],
                          label_filter="All source-train edges of a protein must occur in current training; otherwise ALL its BioFP bits are unknown",
                          legacy_positive_policy="Only masked targets > 0 are known positives; zero/unobserved labels are not negative evidence",
                          native_cofactor_policy="Documented curated enzyme cofactor groups, first 10 v2 positions only; remaining groups unknown",
                          reaction_cofactor_policy="No role-specific reaction cofactor evidence imported",
                          unsafe_source_proteins=len(unsafe))
        manifest = write_indexed_pairs(output / "data/index", query_ids=qids, protein_ids=pids, pairs=edges,
            protein_ec=protein_ec, ec_prefix=prefix, query_ec_indptr=qptr, query_ec=qec,
            mechanism_bits=mechanisms, native_cofactor_bits=native, reaction_cofactor_bits=np.zeros(len(pids), dtype=np.uint32),
            ec_eligible=protein_ec >= 0, biological_eligible=(protein_ec >= 0) & (mechanisms != 0),
            query_signature=signature_codes, provenance=provenance, min_biofp_similarity=0.5)
    if {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in sources} != before:
        raise ValueError("Input changed during setup")
    index = IndexedPairs(output / "data/index")
    # CPU-only quota/support/mask check on both intended DDP ranks.
    checks = []
    for rank in (0, 1):
        sampler = IndexedTypedNegativeBatchSampler(index, 200, rank=rank, world_size=2)
        for number, batch in enumerate(sampler):
            positives = {(q, p) for q, p, kind in batch if kind == 0}
            if sum(kind == 0 for _, _, kind in batch) != 170 or len(batch) != 200:
                raise ValueError("85/15 sampler quota failed")
            for q, p, kind in batch:
                if kind and ((q, p) in positives or not any(a == q for a, _ in positives) or not any(b == p for _, b in positives)):
                    raise ValueError("A sampled negative lacks genuine positive support")
            checks.append(dict(rank=rank, batch=number, positives=170, negatives=30))
            if number == 2:
                break
    base, recipe = yaml.safe_load(F3.read_text()), yaml.safe_load(RECIPE.read_text())
    config = config_for_run(base, recipe, output)
    (output / "configs/train.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    report = dict(num_pairs=manifest["num_pairs"], num_proteins=manifest["num_proteins"], num_queries=manifest["num_queries"],
                  random_eligible=int((protein_ec >= 0).sum()), biological_eligible=int(((protein_ec >= 0) & (mechanisms != 0)).sum()),
                  sampler_checks=checks, feature_extraction=False, output=str(output),
                  recipe_sha256=sha(RECIPE), f3_config_sha256=sha(F3),
                  differences_from_tyrosine=["ReactZyme molecule-set chemistry/forward-only benchmark IDs", "train-filtered legacy annotation coverage", "validation every epoch", "2 x 200 rows instead of 4 x 100"])
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_root.resolve()
    if not output.is_relative_to(ROOT / "runs") or output == ROOT / "runs":
        parser.error("Use a dedicated experiment directory under runs/")
    print("Preparing cached-feature ReactZyme training (no extraction)...", flush=True)
    prepare(output)
