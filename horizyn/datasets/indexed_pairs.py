"""Train-only, memory-mapped positive graph and on-demand typed negatives.

No negative edge inventory is stored. Every sampled negative is accompanied by
two genuine training edges, one supporting each endpoint. Missing cross-pairs
are *not* negative labels. The annotation-guided classes are sampling policies,
not experimentally established non-catalysis.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch.distributed as dist
from torch.utils.data import BatchSampler, Dataset

from horizyn.capability.cofactor_vocabulary_v2 import COFACTOR_LABELS

SCHEMA = "circe_indexed_pairs_v1"
SEMANTICS = "positive_only_role_aware_v2"
PAIR_TYPES = ("positive", "biological_negative", "random_negative")
_BYTE_POPCOUNTS = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)
ARRAY_NAMES = frozenset((
    "query_ids", "protein_ids", "pairs", "query_indptr", "protein_indptr", "protein_queries",
    "protein_ec", "ec_prefix", "query_ec_indptr", "query_ec", "query_signature",
    "protein_signature_indptr", "protein_signatures", "query_profile_indptr", "query_profiles",
    "query_prefix_indptr", "query_prefix", "random_candidates", "biological_candidates",
    "biological_prefix", "excluded_indptr", "excluded_targets", "mechanism_bits",
    "native_cofactor_bits", "reaction_cofactor_bits", "ec_eligible", "biological_eligible",
))


def _csr_counts(keys, count):
    return np.concatenate(([0], np.cumsum(np.bincount(keys, minlength=count)))).astype(np.int64)


def _unique_csr(rows, width, dtype=np.uint32):
    """Small per-query temporary arrays, never Python objects per graph edge."""
    offsets = [0]
    values = []
    for row in rows:
        row = np.unique(np.asarray(row, dtype=dtype).reshape(-1, width), axis=0)
        values.append(row)
        offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, dtype=np.int64), np.concatenate(values, axis=0)


def write_indexed_pairs(
    output_dir, *, query_ids, protein_ids, pairs, protein_ec, ec_prefix,
    query_ec_indptr, query_ec, mechanism_bits, native_cofactor_bits,
    reaction_cofactor_bits, ec_eligible, biological_eligible, query_signature,
    provenance, excluded_pairs=None, min_biofp_similarity=0.5,
):
    """Write compact arrays; inputs use sorted string IDs and integer coordinates.

    ``protein_ec`` is a unique internally consistent complete EC code, or -1.
    ``query_ec`` includes ALL complete ECs on positive proteins, including
    multi-EC/ineligible ones. ``ec_prefix[ec_code]`` encodes a complete EC's
    chosen parent prefix. Eligibility is candidate-only. Known cofactor bitsets
    use bits 0..30 and NEVER encode the unknown class (31).
    """
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty indexed directory: {path}")
    if provenance.get("pair_scope") != "train" or provenance.get("annotation_semantics") != SEMANTICS:
        raise ValueError("Indexed pairs require explicitly train-only role-aware provenance")
    if not 0 <= min_biofp_similarity <= 1:
        raise ValueError("min_biofp_similarity must lie in [0, 1]")
    qids, pids = np.asarray(query_ids), np.asarray(protein_ids)
    for name, ids in (("query_ids", qids), ("protein_ids", pids)):
        if ids.ndim != 1 or ids.dtype.kind != "U" or not len(ids) or np.any(ids[1:] <= ids[:-1]):
            raise ValueError(f"{name} must be nonempty, sorted unique Unicode IDs")
    raw_pairs = np.asarray(pairs)
    if raw_pairs.ndim != 2 or raw_pairs.shape[1] != 2 or raw_pairs.dtype.kind not in "iu" or not len(raw_pairs):
        raise ValueError("pairs must be nonempty integer[E,2]")
    if np.any(raw_pairs < 0) or np.any(raw_pairs[:, 0] >= len(qids)) or np.any(raw_pairs[:, 1] >= len(pids)):
        raise ValueError("pairs reference out-of-range entities")
    pairs = np.unique(raw_pairs.astype(np.uint32), axis=0)
    nq, np_ = len(qids), len(pids)
    qptr = _csr_counts(pairs[:, 0], nq)
    porder = np.lexsort((pairs[:, 0], pairs[:, 1]))
    pptr = _csr_counts(pairs[:, 1], np_)
    if np.any(np.diff(qptr) == 0) or np.any(np.diff(pptr) == 0):
        raise ValueError("Every indexed entity must have genuine training positive support")
    protein_ec = np.asarray(protein_ec)
    ec_prefix = np.asarray(ec_prefix)
    if protein_ec.shape != (np_,) or protein_ec.dtype.kind not in "iu" or ec_prefix.ndim != 1 or ec_prefix.dtype.kind not in "iu":
        raise ValueError("Invalid integer EC arrays")
    if np.any(protein_ec < -1) or np.any(protein_ec >= len(ec_prefix)) or np.any(ec_prefix < 0):
        raise ValueError("EC codes out of range")
    fields = {}
    for name, values, max_value in (
        ("mechanism_bits", mechanism_bits, 255),
        ("native_cofactor_bits", native_cofactor_bits, 2**31 - 1),
        ("reaction_cofactor_bits", reaction_cofactor_bits, 2**31 - 1),
        ("ec_eligible", ec_eligible, 1),
        ("biological_eligible", biological_eligible, 1),
    ):
        values = np.asarray(values)
        if values.shape != (np_,) or values.dtype.kind not in "biu" or np.any(values < 0) or np.any(values > max_value):
            raise ValueError(f"Invalid {name}; unknown cannot be a known cofactor bit")
        fields[name] = values.astype(np.uint32 if "bits" in name else np.bool_)
    flags = fields["ec_eligible"]
    if np.any(flags & (protein_ec < 0)):
        raise ValueError("Eligible candidates require one consistent complete EC")
    if np.any(fields["biological_eligible"] & (~flags | (fields["mechanism_bits"] == 0))):
        raise ValueError("Biological eligibility requires EC eligibility and known mechanism")
    qecptr = np.asarray(query_ec_indptr, dtype=np.int64)
    qec = np.asarray(query_ec, dtype=np.uint32)
    if qecptr.shape != (nq + 1,) or qecptr[0] != 0 or qecptr[-1] != len(qec) or np.any(np.diff(qecptr) < 0) or np.any(qec >= len(ec_prefix)):
        raise ValueError("Invalid query EC CSR")
    for q in range(nq):
        ecs = qec[qecptr[q]:qecptr[q + 1]]
        if np.any(ecs[1:] <= ecs[:-1]):
            raise ValueError("Query EC rows must be sorted and unique")
        positive_ec = protein_ec[pairs[qptr[q]:qptr[q + 1], 1]]
        if not np.all(np.isin(positive_ec[positive_ec >= 0], ecs)):
            raise ValueError("Query EC exclusions omit a positive protein's EC")
    signatures = np.asarray(query_signature)
    if signatures.shape != (nq,) or signatures.dtype.kind not in "iu" or np.any(signatures < 0):
        raise ValueError("query_signature must be nonnegative integer[queries]")
    # Signature CSR makes exclusion independent of the candidate's graph degree.
    signature_pairs = np.unique(np.column_stack((pairs[:, 1], signatures[pairs[:, 0]])), axis=0)
    psigptr = _csr_counts(signature_pairs[:, 0], np_)
    psig = signature_pairs[:, 1]
    profiles = np.column_stack((fields["mechanism_bits"], fields["native_cofactor_bits"], fields["reaction_cofactor_bits"]))
    profileptr, query_profiles = _unique_csr(
        (profiles[pairs[qptr[q]:qptr[q + 1], 1]] for q in range(nq)), 3
    )
    prefixptr, query_prefix = _unique_csr(
        (ec_prefix[qec[qecptr[q]:qecptr[q + 1]]] for q in range(nq)), 1
    )
    bio_ids = np.flatnonzero(fields["biological_eligible"]).astype(np.uint32)
    bio_prefixes = ec_prefix[protein_ec[bio_ids]]
    bio_order = np.argsort(bio_prefixes, kind="stable")
    excluded = np.empty((0, 2), dtype=np.uint32) if excluded_pairs is None else np.asarray(excluded_pairs)
    if excluded.ndim != 2 or excluded.shape[1] != 2 or excluded.dtype.kind not in "iu" or np.any(excluded < 0) or np.any(excluded[:, 0] >= nq) or np.any(excluded[:, 1] >= np_):
        raise ValueError("Invalid excluded pair coordinates")
    excluded = np.unique(excluded.astype(np.uint32), axis=0)
    arrays = dict(
        query_ids=qids, protein_ids=pids, pairs=pairs, query_indptr=qptr,
        protein_indptr=pptr, protein_queries=pairs[porder, 0],
        protein_ec=protein_ec.astype(np.int32), ec_prefix=ec_prefix.astype(np.int32),
        query_ec_indptr=qecptr, query_ec=qec, query_signature=signatures.astype(np.uint32),
        protein_signature_indptr=psigptr, protein_signatures=psig,
        query_profile_indptr=profileptr, query_profiles=query_profiles,
        query_prefix_indptr=prefixptr, query_prefix=query_prefix[:, 0],
        random_candidates=np.flatnonzero(flags).astype(np.uint32),
        biological_candidates=bio_ids[bio_order], biological_prefix=bio_prefixes[bio_order],
        excluded_indptr=_csr_counts(excluded[:, 0], nq), excluded_targets=excluded[:, 1],
        **fields,
    )
    if not len(arrays["random_candidates"]):
        raise ValueError("No eligible training negative candidates")
    path.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(path / f"{name}.npy", value, allow_pickle=False)
    manifest = dict(
        schema=SCHEMA, pair_scope="train", annotation_semantics=SEMANTICS,
        cofactor_vocabulary_version="circe_cofactor_v2", cofactor_labels=list(COFACTOR_LABELS),
        cofactor_unknown_index=31, reaction_direction_mode="forward_only",
        num_pairs=len(pairs), num_queries=nq, num_proteins=np_,
        min_biofp_similarity=float(min_biofp_similarity), provenance=provenance,
        negative_claim="annotation-guided contrastive candidates, not verified non-catalysis",
        arrays={name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
    )
    # A complete manifest is the commit marker; partial builds cannot be loaded.
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _contains(sorted_values, value):
    idx = int(np.searchsorted(sorted_values, value))
    return idx < len(sorted_values) and int(sorted_values[idx]) == value


def _contains_many(sorted_values, candidates):
    positions = np.searchsorted(sorted_values, candidates)
    hits = positions < len(sorted_values)
    hits[hits] &= sorted_values[positions[hits]] == candidates[hits]
    return hits


def _popcounts(values):
    # Portable to NumPy 1.x; bitwise_count is only available in newer NumPy.
    values = np.ascontiguousarray(values, dtype=np.uint32)
    return _BYTE_POPCOUNTS[values.view(np.uint8).reshape(*values.shape, 4)].sum(axis=-1)


class _Ids(Sequence):
    def __init__(self, ids, indices):
        self.ids, self.indices = ids, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _Ids(self.ids, self.indices[index])
        return str(self.ids[int(self.indices[index])])


class IndexedPositiveMap(Mapping):
    """Mapping compatibility without per-protein sets or per-edge Python IDs."""
    def __init__(self, index, reverse=False):
        self.index, self.reverse = index, reverse
        self.ids = index.protein_ids if reverse else index.query_ids

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return (str(value) for value in self.ids)

    def __getitem__(self, key):
        i = self.index.find_id(self.ids, key)
        if i is None:
            raise KeyError(key)
        if self.reverse:
            return _Ids(self.index.query_ids, self.index.protein_positive_queries(i))
        return _Ids(self.index.protein_ids, self.index.query_positive_targets(i))

    def batch_positive_indices(self, query_ids, protein_ids):
        if self.reverse:
            raise ValueError("batch_positive_indices requires query-to-protein mapping")
        target_coords = [(j, self.index.find_id(self.index.protein_ids, p)) for j, p in enumerate(protein_ids)]
        valid = [(j, p) for j, p in target_coords if p is not None]
        qi, pi = [], []
        if not valid:
            return qi, pi
        js, ps = map(np.asarray, zip(*valid))
        for i, qid in enumerate(query_ids):
            q = self.index.find_id(self.index.query_ids, qid)
            if q is None:
                continue
            positives = self.index.query_positive_targets(q)
            positions = np.searchsorted(positives, ps)
            hits = positions < len(positives)
            hits[hits] &= positives[positions[hits]] == ps[hits]
            for j in js[hits]:
                qi.append(i)
                pi.append(int(j))
        return qi, pi


class IndexedPairs:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        meta = self.manifest
        if (meta.get("schema") != SCHEMA or meta.get("pair_scope") != "train"
            or meta.get("annotation_semantics") != SEMANTICS
            or meta.get("cofactor_vocabulary_version") != "circe_cofactor_v2"
            or meta.get("cofactor_labels") != list(COFACTOR_LABELS)
            or meta.get("cofactor_unknown_index") != 31
            or meta.get("reaction_direction_mode") != "forward_only"
            or not isinstance(meta.get("arrays"), dict)
            or set(meta["arrays"]) != ARRAY_NAMES
            or meta.get("provenance", {}).get("pair_scope") != "train"
            or meta.get("provenance", {}).get("annotation_semantics") != SEMANTICS):
            raise ValueError("Invalid or non-training indexed-pair manifest")
        for name, spec in meta["arrays"].items():
            if not name.replace("_", "").isalnum():
                raise ValueError("Invalid indexed array name")
            value = np.load(self.directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
                raise ValueError(f"Indexed array contract mismatch: {name}")
            setattr(self, name, value)
        if (self.pairs.shape != (meta.get("num_pairs"), 2)
            or self.query_ids.shape != (meta.get("num_queries"),)
            or self.protein_ids.shape != (meta.get("num_proteins"),)
            or self.pairs.dtype != np.dtype("uint32")):
            raise ValueError("Indexed manifest entity/edge counts do not match arrays")
        for ids in (self.query_ids, self.protein_ids):
            if ids.dtype.kind != "U" or not len(ids) or np.any(ids[1:] <= ids[:-1]):
                raise ValueError("Indexed IDs must be sorted unique Unicode")
        for values in (self.native_cofactor_bits, self.reaction_cofactor_bits):
            if values.dtype != np.dtype("uint32") or values.shape != self.protein_ids.shape or np.any(values >= 2**31):
                raise ValueError("Indexed known cofactors must exclude unknown bit 31")
        self.query_to_targets = IndexedPositiveMap(self)
        self.target_to_queries = IndexedPositiveMap(self, reverse=True)

    def __getstate__(self):
        # Spawned data workers reopen read-only maps instead of pickling arrays.
        return {"directory": str(self.directory)}

    def __setstate__(self, state):
        self.__init__(state["directory"])

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def find_id(ids, value):
        i = int(np.searchsorted(ids, str(value)))
        return i if i < len(ids) and str(ids[i]) == str(value) else None

    def query_positive_targets(self, q):
        return self.pairs[self.query_indptr[q]:self.query_indptr[q + 1], 1]

    def protein_positive_queries(self, p):
        return self.protein_queries[self.protein_indptr[p]:self.protein_indptr[p + 1]]

    def is_positive(self, q, p):
        return _contains(self.query_positive_targets(q), p)

    @staticmethod
    def similarity(a, b):
        def jaccard(x, y):
            union = int(x) | int(y)
            return (int(x) & int(y)).bit_count() / union.bit_count() if union else 0.0
        return sum(w * jaccard(x, y) for w, x, y in zip((0.75, 0.15, 0.10), a, b))

    @lru_cache(maxsize=16384)
    def negative_kind(self, q, p):
        """Deterministic edge category prevents conflicting DDP row labels."""
        if not self.ec_eligible[p] or self.is_positive(q, p):
            return None
        if _contains(self.excluded_targets[self.excluded_indptr[q]:self.excluded_indptr[q + 1]], p):
            return None
        ec = int(self.protein_ec[p])
        if ec < 0 or _contains(self.query_ec[self.query_ec_indptr[q]:self.query_ec_indptr[q + 1]], ec):
            return None
        if _contains(self.protein_signatures[self.protein_signature_indptr[p]:self.protein_signature_indptr[p + 1]], int(self.query_signature[q])):
            return None
        prefixes = self.query_prefix[self.query_prefix_indptr[q]:self.query_prefix_indptr[q + 1]]
        if self.biological_eligible[p] and _contains(prefixes, int(self.ec_prefix[ec])):
            candidate = (self.mechanism_bits[p], self.native_cofactor_bits[p], self.reaction_cofactor_bits[p])
            for profile in self.query_profiles[self.query_profile_indptr[q]:self.query_profile_indptr[q + 1]]:
                if int(profile[0]) & int(candidate[0]) and self.similarity(profile, candidate) >= self.manifest["min_biofp_similarity"]:
                    return 1
        return 2

    def propose_negative(self, rng, biological, max_attempts=128):
        # Anchor sampling is over genuine edges, not a repeated all-protein scan.
        for _ in range(max_attempts):
            anchor = self.pairs[int(rng.integers(len(self.pairs)))]
            q, support_p = map(int, anchor)
            candidates = self.random_candidates
            if biological:
                prefixes = self.query_prefix[self.query_prefix_indptr[q]:self.query_prefix_indptr[q + 1]]
                if not len(prefixes):
                    continue
                prefix = int(prefixes[int(rng.integers(len(prefixes)))])
                start, stop = np.searchsorted(self.biological_prefix, prefix, side="left"), np.searchsorted(self.biological_prefix, prefix, side="right")
                candidates = self.biological_candidates[start:stop]
            if not len(candidates):
                continue
            p = int(candidates[int(rng.integers(len(candidates)))])
            kind = self.negative_kind(q, p)
            if kind != (1 if biological else 2):
                continue
            support_queries = self.protein_positive_queries(p)
            support_q = int(support_queries[int(rng.integers(len(support_queries)))])
            return (q, p, kind), (q, support_p, 0), (support_q, p, 0)
        return None

    def batch_negative_masks(self, query_ids, protein_ids, pair_query_ids, pair_target_ids, pair_types):
        """Optionally reuse sampled negative endpoints, with per-cell checks.

        This is NOT the complement of the positive graph. Only rows/columns
        represented by an actually sampled explicit negative may be reused;
        every proposed cell must independently pass the train-only policy.
        """
        if not (len(pair_query_ids) == len(pair_target_ids) == len(pair_types)):
            raise ValueError("Pair ID/type vectors must have equal lengths")
        selected_queries, selected_proteins = set(), set()
        for qid, pid, kind in zip(pair_query_ids, pair_target_ids, pair_types):
            if kind == "positive":
                continue
            if kind not in PAIR_TYPES[1:]:
                raise ValueError(f"Unsupported indexed pair type: {kind}")
            q, p = self.find_id(self.query_ids, qid), self.find_id(self.protein_ids, pid)
            if q is None or p is None or self.negative_kind(q, p) != PAIR_TYPES.index(kind):
                raise ValueError("Explicit negative does not match indexed training policy")
            selected_queries.add(qid)
            selected_proteins.add(pid)
        return self._negative_masks(query_ids, protein_ids, selected_queries, selected_proteins)

    def _negative_masks(self, query_ids, protein_ids, selected_queries, selected_proteins):
        """Shared train-only policy for candidate proposals and sampled-cell reuse."""
        biological = np.zeros((len(query_ids), len(protein_ids)), dtype=bool)
        random = np.zeros_like(biological)
        query_coords = [self.find_id(self.query_ids, value) for value in query_ids]
        target_coords = [self.find_id(self.protein_ids, value) for value in protein_ids]
        valid_qrows = np.asarray([i for i, q in enumerate(query_coords) if q is not None], dtype=np.int64)
        valid_pcols = np.asarray([j for j, p in enumerate(target_coords) if p is not None], dtype=np.int64)
        if not len(valid_qrows) or not len(valid_pcols):
            return biological, random
        ps = np.asarray([target_coords[j] for j in valid_pcols], dtype=np.uint32)
        qs = np.asarray([query_coords[i] for i in valid_qrows], dtype=np.uint32)
        ecs = self.protein_ec[ps]
        eligible = self.ec_eligible[ps] & (ecs >= 0)
        sampled_columns = np.asarray([protein_ids[j] in selected_proteins for j in valid_pcols])
        # One binary-search vector per candidate protein, independent of its
        # global degree. Avoid 44k+ Python policy calls on a 400x400 batch.
        same_signature = np.zeros((len(query_ids), len(ps)), dtype=bool)
        signatures = self.query_signature[qs]
        for j, p in enumerate(ps):
            known = self.protein_signatures[self.protein_signature_indptr[p]:self.protein_signature_indptr[p + 1]]
            same_signature[valid_qrows, j] = _contains_many(known, signatures)
        candidate_profiles = np.column_stack((self.mechanism_bits[ps], self.native_cofactor_bits[ps], self.reaction_cofactor_bits[ps]))
        for i, q in zip(valid_qrows, qs):
            allowed = eligible & (True if query_ids[i] in selected_queries else sampled_columns)
            if not allowed.any():
                continue
            allowed &= ~_contains_many(self.query_positive_targets(q), ps)
            allowed &= ~_contains_many(self.excluded_targets[self.excluded_indptr[q]:self.excluded_indptr[q + 1]], ps)
            allowed &= ~_contains_many(self.query_ec[self.query_ec_indptr[q]:self.query_ec_indptr[q + 1]], ecs)
            allowed &= ~same_signature[i]
            if not allowed.any():
                continue
            prefixes = self.query_prefix[self.query_prefix_indptr[q]:self.query_prefix_indptr[q + 1]]
            bio_candidates = allowed & self.biological_eligible[ps]
            bio_positions = np.flatnonzero(bio_candidates)
            if len(bio_positions):
                bio_candidates[bio_positions] &= _contains_many(prefixes, self.ec_prefix[ecs[bio_positions]])
            pending = np.flatnonzero(bio_candidates)
            hits = np.zeros(len(ps), dtype=bool)
            profiles = self.query_profiles[self.query_profile_indptr[q]:self.query_profile_indptr[q + 1]]
            # Bound temporary memory even for a reaction with many observed
            # profiles. Each comparison has exactly the scalar policy's roles,
            # weights, empty-set rule and mandatory shared known mechanism.
            for start in range(0, len(profiles), 128):
                if not len(pending):
                    break
                block = profiles[start:start + 128]
                candidates = candidate_profiles[pending]
                shared = (block[:, None, 0] & candidates[None, :, 0]) != 0
                scores = np.zeros(shared.shape, dtype=np.float64)
                for column, weight in enumerate((.75, .15, .10)):
                    intersection = _popcounts(block[:, None, column] & candidates[None, :, column])
                    union = _popcounts(block[:, None, column] | candidates[None, :, column])
                    scores += weight * np.divide(intersection, union, out=np.zeros(scores.shape), where=union != 0)
                threshold = self.manifest["min_biofp_similarity"]
                # NumPy's vectorized division may differ from Python division
                # by one ulp. Preserve the scalar policy exactly at its cutoff
                # so the same explicit edge cannot acquire two DDP categories.
                borderline = shared & (np.abs(scores - threshold) < 1e-12)
                for row, column in zip(*np.nonzero(borderline)):
                    scores[row, column] = self.similarity(block[row], candidates[column])
                matched = (shared & (scores >= threshold)).any(axis=0)
                hits[pending[matched]] = True
                pending = pending[~matched]
            biological[i, valid_pcols] = hits
            random[i, valid_pcols] = allowed & ~hits
        return biological, random


class IndexedPairDataset(Dataset):
    def __init__(self, index, query_dataset, target_dataset):
        self.index = index
        self.query_dataset, self.target_dataset = query_dataset, target_dataset
        self.keys = range(len(index))
        query_keys = set(query_dataset.keys)
        self.query_feature_ids = []
        for qid in index.query_ids:
            qid = str(qid)
            feature_id = qid if qid in query_keys else f"{qid}_f"
            if feature_id not in query_keys:
                raise ValueError(f"Indexed training reaction feature missing: {qid}")
            self.query_feature_ids.append(feature_id)
        # Fail instead of silently changing the graph or its 85/15 contract.
        for ids, dataset, role in ((index.protein_ids, target_dataset, "protein"),):
            try:
                lookup = dataset.key_to_idx
            except AttributeError:
                lookup = set(dataset.keys)
            for value in ids:
                if str(value) not in lookup:
                    raise ValueError(f"Indexed training {role} feature missing: {value}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, row):
        return self._sample(row)

    def __getitems__(self, rows):
        # PyTorch's batched fetch hook: support positives intentionally repeat
        # endpoints. Read each residue/reaction feature once, without dropping
        # any row or changing the sampler's loss/sampling semantics.
        cache = {}
        return [self._sample(row, cache) for row in rows]

    def _sample(self, row, cache=None):
        if isinstance(row, (int, np.integer)):
            q, p = map(int, self.index.pairs[row])
            kind = 0
        else:
            q, p, kind = row
        qid, pid = str(self.index.query_ids[q]), str(self.index.protein_ids[p])
        sample = {"query_id": qid, "target_id": pid, "pair_type": PAIR_TYPES[kind]}
        for key, dataset, field in ((self.query_feature_ids[q], self.query_dataset, "query_vec"), (pid, self.target_dataset, "target_vec")):
            cache_key = (field, key)
            if cache is not None and cache_key in cache:
                value = cache[cache_key]
            else:
                value = dataset[key]
                if cache is not None:
                    cache[cache_key] = value
            if isinstance(value, dict):
                sample.update(value)
            else:
                sample[field] = value
        return sample


class IndexedTypedNegativeBatchSampler(BatchSampler):
    """Exact per-rank 85/15 or 50/50 rows, with genuine positive support.

    At 85/15, 55% of rows traverse all train edges and proposals add supports.
    At 50/50, all positive rows traverse the graph and negatives share their
    endpoints; no additional feature reads are needed for negative rows.
    The traversal visits every train edge once per epoch (with
    bounded epoch-tail padding). A coprime affine permutation is bijective and
    O(1) memory, unlike a Python list/permutation of millions of row objects.
    It is deterministic shuffling, not a uniformly random permutation.
    """
    def __init__(self, index, batch_size, positive_fraction=0.85,
                 biological_negative_fraction=0.5, seed=42, rank=None, world_size=None,
                 max_negative_attempts=128):
        if positive_fraction not in (0.85, 0.5):
            raise ValueError("Indexed positive_fraction must be 0.85 or 0.5")
        divisor = 20 if positive_fraction == 0.85 else 2
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < max(4, divisor) or batch_size % divisor:
            raise ValueError(f"Indexed batch_size must be >= {max(4, divisor)} and divisible by {divisor}")
        if not 0 <= biological_negative_fraction <= 1:
            raise ValueError("biological_negative_fraction must lie in [0, 1]")
        if max_negative_attempts <= 0:
            raise ValueError("max_negative_attempts must be positive")
        self.index, self.batch_size, self.drop_last = index, batch_size, False
        # Lightning discovers epoch setters via batch_sampler.sampler. This
        # also handles resumed/partially consumed epochs, not only exhaustion.
        self.sampler = self
        self.positive_fraction = positive_fraction
        self.biological_fraction = biological_negative_fraction
        self.negative_count = batch_size * 3 // 20 if positive_fraction == 0.85 else batch_size // 2
        self.positive_count = batch_size - self.negative_count
        self.base_count = self.positive_count - 2 * self.negative_count if positive_fraction == 0.85 else self.positive_count
        self.seed, self.epoch = int(seed), 0
        self.rank, self.world_size = rank, world_size
        self.max_negative_attempts = int(max_negative_attempts)
        self.last_counts = {}

    def _rank_info(self):
        rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_available() and dist.is_initialized() else (0, 1)
        rank = rank if self.rank is None else self.rank
        world = world if self.world_size is None else self.world_size
        if not 0 <= rank < world:
            raise ValueError("Invalid indexed sampler rank/world_size")
        return rank, world

    def __len__(self):
        _, world = self._rank_info()
        return math.ceil(len(self.index) / (self.base_count * world))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _shared_support_negatives(self, rows, rng):
        qs = sorted({q for q, _, _ in rows})
        ps = sorted({p for _, p, _ in rows})
        qids = [str(self.index.query_ids[q]) for q in qs]
        pids = [str(self.index.protein_ids[p]) for p in ps]
        masks = self.index._negative_masks(qids, pids, set(qids), set(pids))
        pools = [np.flatnonzero(mask) for mask in masks]
        if not any(len(pool) for pool in pools):
            raise RuntimeError("No eligible 50/50 negatives among positive-supported endpoints; increase batch_size")
        requested_bio = round(self.negative_count * self.biological_fraction)
        counts = [requested_bio, self.negative_count - requested_bio]
        for category in range(2):
            if counts[category] and not len(pools[category]):
                self.last_counts["category_fallbacks"] += counts[category]
                counts[1 - category] += counts[category]
                counts[category] = 0
        for category, count in enumerate(counts):
            if not count:
                continue
            # Avoid repeated negatives when the local policy offers enough.
            selected = rng.choice(pools[category], size=count, replace=len(pools[category]) < count)
            for cell in selected:
                i, j = divmod(int(cell), len(ps))
                rows.append((qs[i], ps[j], category + 1))

    def __iter__(self):
        rank, world = self._rank_info()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        count = len(self.index)
        multiplier = int(rng.integers(1, max(2, count)))
        while math.gcd(multiplier, count) != 1:
            multiplier += 1
        offset = int(rng.integers(count))
        self.last_counts = dict(positive=0, biological_negative=0, random_negative=0, category_fallbacks=0)
        for step in range(len(self)):
            # Rank-addressed RNG avoids reconstructing other ranks' proposals.
            local_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, step, rank]))
            start = (step * world + rank) * self.base_count
            rows = [(*map(int, self.index.pairs[(multiplier * (start + j) + offset) % count]), 0) for j in range(self.base_count)]
            if self.positive_fraction == 0.5:
                self._shared_support_negatives(rows, local_rng)
            requested_bio = round(self.negative_count * self.biological_fraction)
            for j in range(self.negative_count if self.positive_fraction == 0.85 else 0):
                want_bio = j < requested_bio
                proposal = self.index.propose_negative(local_rng, want_bio, self.max_negative_attempts)
                if proposal is None:
                    proposal = self.index.propose_negative(local_rng, not want_bio, self.max_negative_attempts)
                    self.last_counts["category_fallbacks"] += 1
                if proposal is None:
                    raise RuntimeError("Cannot fill exact indexed 85/15 quota with eligible negatives and genuine positive support; bounded proposal budget exhausted")
                rows.extend(proposal)
            local_rng.shuffle(rows)
            for _, _, kind in rows:
                self.last_counts[PAIR_TYPES[kind]] += 1
            yield rows
        self.epoch += 1
