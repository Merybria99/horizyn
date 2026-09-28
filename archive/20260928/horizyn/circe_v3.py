"""Train-only preparation for CIRCE-v3. No feature extraction or GPU allocation."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("time", "enzyme_smi", "reaction_smi")
MONITOR = "val/mean_bidirectional_reactzyme_mrr"
F3_ROOT = ROOT / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry"


def rooted(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pairs(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"reaction_id", "protein_id"} <= set(reader.fieldnames or ()):
            raise ValueError(f"Missing pair columns: {path}")
        pairs = set()
        for row in reader:
            if "Label" in row and row["Label"].strip() not in {"1", "1.0"}:
                raise ValueError("Pair files must contain positives only; Label=0 is not verified inactivity")
            pair = row["reaction_id"].strip(), row["protein_id"].strip()
            if not all(pair):
                raise ValueError(f"Empty pair ID in {path}")
            pairs.add(pair)
    return sorted(pairs)


def write_pairs(path, pairs):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pr_id", "reaction_id", "protein_id"])
        writer.writerows((f"v3_{i}", q, p) for i, (q, p) in enumerate(pairs))


def association_holdout(pairs, fraction, seed):
    """Deterministic edge holdout; both endpoints must retain training support."""
    if not 0 <= fraction < 1:
        raise ValueError("holdout fraction must be in [0, 1)")
    pairs = sorted(set(pairs))
    qdegree, pdegree = Counter(q for q, _ in pairs), Counter(p for _, p in pairs)
    order = sorted(pairs, key=lambda edge: hashlib.sha256(
        f"{seed}\0{edge[0]}\0{edge[1]}".encode()).digest())
    hidden = set()
    requested = int(len(pairs) * fraction)
    for q, p in order:
        if len(hidden) >= requested:
            break
        if qdegree[q] > 1 and pdegree[p] > 1:
            hidden.add((q, p))
            qdegree[q] -= 1
            pdegree[p] -= 1
    return [edge for edge in pairs if edge not in hidden], sorted(hidden)


def audited_targets(evidence_path, pairs, family_dims):
    """Only explicit signed evidence; missing/conflicting/one-class labels masked.

    CSV: protein_id,family,label_index,value,confidence,source,evidence_kind,
    support_reaction_id. Values are 0, 1 or unknown. Curated positive and measured
    negative are explicit biological-label evidence, NOT enzyme/reaction Label.
    A training_association supports only a positive, and needs its retained edge.
    Independent evidence requires an empty support_reaction_id.
    """
    proteins = sorted({p for _, p in pairs})
    index = {p: i for i, p in enumerate(proteins)}
    edges = set(pairs)
    cells = defaultdict(list)
    counts = Counter()
    if evidence_path is not None:
        with Path(evidence_path).open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"protein_id", "family", "label_index", "value", "confidence",
                        "source", "evidence_kind", "support_reaction_id"}
            if not required <= set(reader.fieldnames or ()):
                raise ValueError(f"Auxiliary evidence needs columns: {sorted(required)}")
            for row in reader:
                counts["rows"] += 1
                p, family = row["protein_id"], row["family"]
                if p not in index:
                    counts["not_training_protein"] += 1
                    continue
                if family not in family_dims:
                    raise ValueError(f"Unsupported auxiliary family: {family}")
                label = int(row["label_index"])
                if not 0 <= label < family_dims[family]:
                    raise ValueError(f"Invalid label_index for {family}: {label}")
                if row["value"].strip().lower() in {"", "unknown"}:
                    counts["unknown"] += 1
                    continue
                value, confidence = float(row["value"]), float(row["confidence"])
                if value not in (0, 1) or not np.isfinite(confidence) or not 0 < confidence <= 1:
                    raise ValueError("Evidence needs binary value and confidence in (0, 1]")
                kind, support = row["evidence_kind"], row["support_reaction_id"]
                if not row["source"].strip():
                    raise ValueError("Auxiliary evidence must have provenance")
                if kind == "training_association":
                    if value != 1:
                        raise ValueError("Absence in an associated reaction is not a negative enzyme label")
                    if (support, p) not in edges:
                        counts["unretained_support"] += 1
                        continue
                elif kind not in {"curated_positive", "measured_negative"} or support:
                    raise ValueError("Evidence must be independent signed evidence or a retained training association")
                elif (kind == "curated_positive") != (value == 1):
                    raise ValueError("Evidence type and sign disagree")
                cells[p, family, label].append((value, confidence))
    payload = {"ids": np.asarray(proteins, dtype=str)}
    report = {"evidence": dict(counts), "families": {}}
    weights = {}
    for family, width in family_dims.items():
        targets = np.zeros((len(proteins), width), dtype=np.float32)
        confidence = np.zeros_like(targets)
        conflicts = 0
        for (p, f, label), observations in cells.items():
            if f != family:
                continue
            if len({value for value, _ in observations}) > 1:
                conflicts += 1
                continue
            targets[index[p], label] = observations[0][0]
            confidence[index[p], label] = max(weight for _, weight in observations)
        present = confidence > 0
        eligible = ((present & (targets == 1)).any(axis=0)
                    & (present & (targets == 0)).any(axis=0))
        mask = present & eligible[None, :]
        confidence *= mask
        payload.update({f"{family}_targets": targets, f"{family}_mask": mask,
                        f"{family}_denominator": mask.astype(np.float32),
                        f"{family}_confidence": confidence})
        active = bool(mask.any())
        report["families"][family] = {
            "enabled": active, "eligible_label_indices": np.flatnonzero(eligible).tolist(),
            "active_proteins": int(mask.any(axis=1).sum()), "active_cells": int(mask.sum()),
            "conflicting_cells": conflicts,
            "reason": "signed train-only evidence" if active else "no label with both positive and negative evidence",
        }
        if active:
            weights[family] = 1.0
    return payload, report, weights


def configure(base, output, train_pairs, seed, devices, weights):
    config = copy.deepcopy(base)
    if config["model"].get("enzyme_input_mode") != "raw_mean_sleec_biological_factorized":
        raise ValueError("CIRCE-v3 requires the F3 biological-factorized architecture")
    config["seed"] = seed
    data, training, logging = config["data"], config["training"], config["logging"]
    # Fail closed for templates with precomputed indices or alternate sampling.
    for key in list(data):
        if key.startswith(("indexed_", "typed_negative_", "cached_", "hard_negative_")) or key == "source_replay":
            data.pop(key)
    data.update(train_pairs_path=str(train_pairs), enzyme_ec_labels_path=None,
                protein_biofp_targets_path=str(output / "data/biofp_targets.npz") if weights else None,
                protein_biofp_vocab_path=None)
    for key in ("init_from_checkpoint", "biofp_pretrain_checkpoint", "enzyme_base_checkpoint",
                "query_base_checkpoint", "limit_train_batches", "max_steps"):
        training.pop(key, None)
    for key in ("query_encoder_checkpoint_path", "target_encoder_checkpoint_path"):
        config["model"].pop(key, None)
    training.update(max_epochs=30, devices=devices, training_stage="joint", check_val_every_n_epoch=1,
                    validation_interval_steps=None, validation_enabled=True,
                    validation_retrieval_metrics=True,
                    validation_retrieval_directions=["reaction_to_enzyme", "enzyme_to_reaction"])
    training["validation_retrieval_candidate_ids_path"] = data.get("validation_retrieval_candidate_ids_path")
    training["loss"] = dict(name="DecoupledAllPositiveInfoNCELoss", beta=10.0, learn_beta=False,
                            beta_min=0.01, beta_max=100.0, positive_pair_source="all_known_in_batch",
                            lambda_r2e=0.5, lambda_e2r=0.5, unknown_negative_weight=0.5,
                            biofp_aux_weight=0.02 if weights else 0.0,
                            biofp_aux_warmup_epochs=3, biofp_normalize_active_families=True,
                            biofp_family_weights=weights, biofp_confidence_cap=1.0)
    training["early_stopping"] = dict(enabled=True, monitor=MONITOR, mode="max", patience=5, min_delta=0.0001)
    logging.update(log_dir=str(output / "logs"), checkpoint_dir=str(output / "checkpoints"),
                   checkpoint_monitor=MONITOR, checkpoint_mode="max", checkpoint_on_validation_end=True,
                   recovery_every_n_train_steps=50, log_every_n_steps=10)
    logging["wandb"] = {"enabled": False, "mode": "disabled"}
    config["ablation"] = dict(variant="CIRCE-v3", base_architecture="F3", seed=seed,
                               split=base.get("ablation", {}).get("split", "original"),
                               fresh_towers=True, unknown_pairs="weak contrastive competitors, not verified inactivity")
    return config


def prepare_run(template, output, *, test_template=None, seed=42, devices=4,
                holdout_fraction=0.05, evidence=None):
    """Never overwrite a prepared run or touch any existing feature cache."""
    template, output = rooted(template), rooted(output)
    if output.exists():
        raise FileExistsError(f"Use a new run root, or train the existing preparation: {output}")
    base = yaml.safe_load(template.read_text())
    # Resolve data paths before materializing configs in a different directory.
    for section in (base["data"], base["model"].get("sleec_pooling", {})):
        for key, value in section.items():
            if value and isinstance(value, str) and (key.endswith("_path") or key == "checkpoint_path"):
                section[key] = str(rooted(value))
    original = read_pairs(base["data"]["train_pairs_path"])
    validation = read_pairs(rooted(base["data"]["validation_pairs_path"]))
    if set(original) & set(validation):
        raise ValueError("Training/validation positive pairs overlap")
    test_base = None
    if test_template is not None:
        test_base = yaml.safe_load(rooted(test_template).read_text())
        test_pairs = read_pairs(rooted(test_base["data"]["test_pairs_path"]))
        if set(original + validation) & set(test_pairs):
            raise ValueError("Training/validation and released-test positive pairs overlap")
    train, hidden = association_holdout(original, holdout_fraction, seed)
    family_dims = base["model"]["biofp"]["family_dims"]
    if set(family_dims) - {"mechanism", "cofactor"}:
        raise ValueError("Only the existing F3 mechanism/cofactor heads are supported")
    targets, audit, weights = audited_targets(evidence, train, family_dims)
    config = configure(base, output, output / "data/train_pairs.csv", seed, devices, weights)
    output.mkdir(parents=True)
    (output / "data").mkdir()
    (output / "configs").mkdir()
    write_pairs(output / "data/train_pairs.csv", train)
    write_pairs(output / "data/hidden_pairs.csv", hidden)
    np.savez_compressed(output / "data/biofp_targets.npz", **targets)
    configs = {"train": config}
    # Forward evaluation expands bare pair IDs to cache IDs ending in _f.
    valid_config = copy.deepcopy(config)
    valid_config["data"].update(test_pairs_path=base["data"]["validation_pairs_path"],
                                test_reactions_path=base["data"]["validation_reactions_path"])
    configs["validation"] = valid_config
    if hidden:
        hidden_config = copy.deepcopy(config)
        hdata = hidden_config["data"]
        hdata.update(test_pairs_path=str(output / "data/hidden_pairs.csv"),
                     test_reactions_path=base["data"]["train_reactions_path"],
                     evaluation_known_pairs_path=str(output / "data/train_pairs.csv"),
                     evaluation_reaction_candidate_ids_path=str(output / "data/hidden_reaction_candidate_ids.txt"),
                     validation_retrieval_candidate_ids_path=str(output / "data/hidden_candidate_ids.txt"))
        (output / "data/hidden_candidate_ids.txt").write_text("\n".join(sorted({p for _, p in train})) + "\n")
        (output / "data/hidden_reaction_candidate_ids.txt").write_text("\n".join(sorted({q + "_f" for q, _ in train})) + "\n")
        for key, value in base["data"].items():
            if key.startswith("train_reaction_") and key.endswith(("_path",)):
                hdata[key.replace("train_", "validation_", 1)] = value
        configs["hidden"] = hidden_config
    if test_base is not None:
        test_config = copy.deepcopy(config)
        for key, value in test_base["data"].items():
            if key.startswith(("test_", "validation_", "reaction_")):
                test_config["data"][key] = (str(rooted(value)) if isinstance(value, str)
                                             and key.endswith("_path") else value)
        test_config["training"]["validation_retrieval_candidate_ids_path"] = test_config["data"]["validation_retrieval_candidate_ids_path"]
        configs["test"] = test_config
    # Evaluation must use generic paths corresponding to its subset, not train fallbacks.
    for name, cfg in configs.items():
        if name != "train":
            cfg["training"]["validation_retrieval_candidate_ids_path"] = cfg["data"].get("validation_retrieval_candidate_ids_path")
            for key, value in list(cfg["data"].items()):
                if key.startswith("validation_reaction_") and key.endswith("_path"):
                    cfg["data"][key.removeprefix("validation_")] = value
        (output / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    audit.update(policy="Missing/conflicting labels masked; no inferred biological negatives",
                 scope="retained training associations only", desired_aux_weight=0.02)
    (output / "data/auxiliary_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest = dict(version=3, source_config=str(template), source_sha256=sha256(template),
                    seed=seed, original_train_pairs=len(original), train_pairs=len(train),
                    hidden_pairs=len(hidden), requested_holdout_fraction=holdout_fraction,
                    hidden_policy="edge holdout; endpoints remain in training; no withheld edges in positive or label lookups",
                    auxiliary=audit, configs={name: str(output / "configs" / f"{name}.yaml") for name in configs})
    manifest["inputs"] = {base["data"][key]: sha256(base["data"][key]) for key in
                         ("train_pairs_path", "validation_pairs_path", "train_reactions_path", "validation_reactions_path")}
    manifest["inputs"][str(template)] = sha256(template)
    if test_base is not None:
        manifest["inputs"][str(rooted(test_template))] = sha256(rooted(test_template))
        for key in ("test_pairs_path", "test_reactions_path"):
            path = rooted(test_base["data"][key])
            manifest["inputs"][str(path)] = sha256(path)
    for cfg in configs.values():
        candidate_path = cfg["data"].get("validation_retrieval_candidate_ids_path")
        if candidate_path and not rooted(candidate_path).is_relative_to(output):
            manifest["inputs"][str(rooted(candidate_path))] = sha256(rooted(candidate_path))
    if evidence is not None:
        manifest["inputs"][str(rooted(evidence))] = sha256(rooted(evidence))
    manifest["outputs"] = {str(path.relative_to(output)): sha256(path) for path in
                           sorted(output.rglob("*")) if path.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_run(output):
    output = rooted(output)
    manifest = json.loads((output / "manifest.json").read_text())
    for path, digest in manifest["inputs"].items():
        if sha256(path) != digest:
            raise ValueError(f"Preparation source changed: {path}")
    for path, digest in manifest["outputs"].items():
        if sha256(output / path) != digest:
            raise ValueError(f"Prepared artifact changed: {path}")
    return manifest


class RankingConstraints:
    """Optional development-only filtering; never used for released test metrics."""

    def __init__(self, known_pairs=None, family_path=None):
        self.known_r2e, self.known_e2r = defaultdict(set), defaultdict(set)
        for reaction, protein in read_pairs(known_pairs) if known_pairs else []:
            self.known_r2e[reaction + "_f"].add(protein)
            self.known_e2r[protein].add(reaction + "_f")
        self.families = defaultdict(set)
        if family_path:
            with Path(family_path).open(newline="") as stream:
                reader = csv.DictReader(stream)
                if not {"protein_id", "family_id"} <= set(reader.fieldnames or ()):
                    raise ValueError("Family map needs protein_id,family_id columns")
                for row in reader:
                    if not row["protein_id"].strip() or not row["family_id"].strip():
                        raise ValueError("Empty family assignment")
                    self.families[row["protein_id"]].add(row["family_id"])
            if not self.families:
                raise ValueError("Empty family map")

    def restrict(self, scores, positives, anchor, candidate_ids, direction):
        import torch
        known = self.known_r2e if direction == "reaction_to_enzyme" else self.known_e2r
        positive_indices = set(positives.tolist())
        keep = np.ones(len(candidate_ids), dtype=bool)
        if self.families:
            if direction != "reaction_to_enzyme":
                raise ValueError("Within-family development evaluation supports reaction_to_enzyme only")
            missing = set(candidate_ids) - self.families.keys()
            if missing:
                raise ValueError(f"Family map is missing {len(missing)} candidates; refusing reduced coverage")
            families = set().union(*(self.families[candidate_ids[i]] for i in positive_indices))
            keep = np.asarray([bool(self.families[p] & families) for p in candidate_ids])
        excluded = known.get(anchor, set())
        for index, candidate in enumerate(candidate_ids):
            if candidate in excluded and index not in positive_indices:
                keep[index] = False
        if not all(keep[index] for index in positive_indices):
            raise ValueError("Constraints cannot remove evaluation positives")
        indices = torch.as_tensor(np.flatnonzero(keep), device=scores.device, dtype=torch.long)
        remapping = np.cumsum(keep) - 1
        new_positives = torch.as_tensor([remapping[i] for i in positives.tolist()], device=scores.device, dtype=torch.long)
        return scores.index_select(0, indices), new_positives
