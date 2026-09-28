"""Fixed seed-42 original EnzymeCAGE, using released query-specific features."""
import io
import pickle
from types import SimpleNamespace

from cyp_external_common import ASSETS, ROOT, RUN, digest, finish, inputs, safe_torch, source


class FeatureUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        import numpy as np
        import torch
        import collections
        allowed = {
            ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy.core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("torch.storage", "_load_from_bytes"): lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=True),
        }
        try: return allowed[module, name]
        except KeyError: raise pickle.UnpicklingError(f"Unsupported feature pickle global: {module}.{name}") from None


def feature_load(stream):
    return FeatureUnpickler(stream).load()


def reaction_component_subset(released, full):
    from collections import Counter
    from rdkit import Chem

    equivalent_by_graph = False
    for released_side, full_side in zip(released.split(">>"), full.split(">>")):
        released_counts = Counter(released_side.split("."))
        full_counts = Counter(full_side.split("."))
        unmatched_released = list((released_counts - full_counts).elements())
        unmatched_full = list((full_counts - released_counts).elements())
        for component in unmatched_released:
            molecule = Chem.MolFromSmiles(component)
            match = next((index for index, candidate in enumerate(unmatched_full)
                          if molecule.GetNumAtoms() == (other := Chem.MolFromSmiles(candidate)).GetNumAtoms()
                          and molecule.HasSubstructMatch(other, useChirality=True)
                          and other.HasSubstructMatch(molecule, useChirality=True)), None)
            if match is None:
                return False, False
            unmatched_full.pop(match)
            equivalent_by_graph = True
    return True, equivalent_by_graph


def run(device="cuda", preflight=False):
    import os
    import pandas as pd
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    torch = safe_torch(); source("enzymecage_pretrained")
    from enzymecage.model import EnzymeCAGE
    from enzymecage.dataset import geometric
    from torch_geometric.loader import DataLoader
    geometric.pkl = SimpleNamespace(load=feature_load)
    checkpoint = ROOT.parent / "EnzymeCAGE/checkpoints/pretrain/seed_42/epoch_19.pth"
    model = EnzymeCAGE(use_prods_info=False, esm_model="ESM-C_600M", device=device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    mapped, proteins = inputs()
    groups = {}
    for row in mapped: groups.setdefault(row["query_id"], []).append(row)
    predictions = {}; equivalent_queries = []; model.to(device)
    for query, candidates in sorted(groups.items()):
        directory = ASSETS / "enzymecage_data/dataset/case-study/liz/eval_rxns" / query
        df = pd.read_csv(directory / "liz_p450.csv")
        expected = [r["protein_id"] for r in candidates]
        if df.UniprotID.duplicated().any() or set(df.UniprotID) != set(expected):
            raise ValueError(f"Released EnzymeCAGE pool mismatch: {query}")
        df = df.set_index("UniprotID").loc[expected].reset_index()
        if any(proteins[r.UniprotID] != r.sequence for r in df.itertuples()):
            raise ValueError(f"Released sequence mismatch: {query}")
        from cyp_specificity import canonical_reaction
        released = {canonical_reaction(s) for s in df.CANO_RXN_SMILES}
        full = canonical_reaction(candidates[0]["reaction"])
        # The authors' EnzymeCAGE features deliberately omit co-substrates.
        # Accept only a single reaction whose components are present on the same
        # respective sides of the full query; record this reduced input scope.
        if len(released) != 1:
            raise ValueError(f"Released reaction mismatch: {query}")
        valid, graph_equivalent = reaction_component_subset(next(iter(released)), full)
        if not valid:
            raise ValueError(f"Released reaction mismatch: {query}")
        if graph_equivalent:
            equivalent_queries.append(query)
            print(f"EnzymeCAGE equivalent reaction graph despite different SMILES: {query}", flush=True)
        feat = directory / "feature"
        gvp = torch.load(feat / "protein/gvp_feature/gvp_protein_feature.pt", map_location="cpu", weights_only=True)
        node = torch.load(feat / "protein/ESM-C_600M/pocket_node_feature/esm_node_feature.pt", map_location="cpu", weights_only=True)
        recovered = RUN / "recovery/enzymecage" / query
        if (recovered / "receipt.json").exists():
            import json
            receipt = json.loads((recovered / "receipt.json").read_text())
            for name, target in [("gvp", gvp), ("node", node)]:
                path = recovered / f"{name}.pt"
                if digest(path) != receipt[f"{name}_sha256"]: raise ValueError("Recovered features changed")
                additions = torch.load(path, map_location="cpu", weights_only=True)
                # Replace a pair together only when at least one original feature was absent.
                for pid, value in additions.items(): target[pid] = value
        if not set(expected) <= gvp.keys() & node.keys():
            raise ValueError(f"Missing EnzymeCAGE pocket features: {query}; pool cannot be shrunk")
        dataset = geometric.GeometricDataset(df, gvp,
            str(feat / "reaction/drfp/rxn2fp.pkl"), str(feat / "reaction/molecule_conformation"), node,
            str(feat / "protein/ESM-C_600M/protein_level/seq2feature.pkl"),
            str(feat / "reaction/reacting_center/reacting_center.pkl"))
        if not set(df.sequence) <= dataset.esm_feat_dict.keys():
            raise ValueError(f"Missing full-protein ESM-C means: {query}")
        if preflight:
            # Exercise graph assembly for EVERY candidate, detecting silent zero-feature fallbacks.
            for i in range(len(dataset)): dataset[i]
            first = next(iter(DataLoader(dataset, batch_size=1, follow_batch=["protein", "reaction_feature", "esm_feature", "substrates", "products"])))
            with torch.inference_mode():
                value = model(first.to(device))
                if isinstance(value, tuple): value = value[0]
                if not torch.isfinite(value).all(): raise ValueError("Nonfinite native forward")
        else:
            loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                follow_batch=["protein", "reaction_feature", "esm_feature", "substrates", "products"])
            values = []
            with torch.inference_mode():
                for batch in loader:
                    result = model(batch.to(device))
                    if isinstance(result, tuple): result = result[0]
                    if not model.sigmoid_readout: result = result.sigmoid()
                    values.extend(result.detach().float().cpu().reshape(-1).tolist())
            if len(values) != len(expected): raise ValueError("Wrong prediction count")
            predictions.update({(query, pid): value for pid, value in zip(expected, values)})
        print(f"EnzymeCAGE {'checked' if preflight else 'scored'} {query}: {len(expected)} candidates", flush=True)
    if not preflight:
        finish("enzymecage_pretrained", predictions, checkpoint,
            "Original fixed seed42; author-reduced reactions (co-substrates omitted), released plus explicitly recovered/generated reaction-conditioned pockets (see receipts), ESM-C600M, DRFP, ligand conformers; native sigmoid",
            seed=42, ensemble=False,
            reaction_graph_equivalent_queries=equivalent_queries,
            recovery_receipts={p.parent.name: digest(p) for p in (RUN / "recovery/enzymecage").glob("*/receipt.json")})
