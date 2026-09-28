"""CLIPZyme native graph encoders without candidate-length filtering."""
import argparse
from pathlib import Path

from cyp_external_common import ASSETS, RUN, finish, inputs, safe_torch, source


def structures(mapped):
    import json
    from cyp_external_common import digest
    receipt = RUN / "recovery/clipzyme/structures.json"
    if receipt.exists():
        records = json.loads(receipt.read_text())
        expected = {r["protein_id"] for r in mapped}
        if records.keys() != expected:
            raise ValueError("Structure recovery is incomplete; pool cannot be shrunk")
        for record in records.values():
            if digest(record["path"]) != record["sha256"]: raise ValueError("Recovered structure changed")
        return {pid: Path(record["path"]) for pid, record in records.items()}
    paths = list((ASSETS / "clipzyme_structures").rglob("*.cif"))
    index = {}
    for path in paths:
        index.setdefault(path.name, []).append(path)
    selected = {}
    for row in mapped:
        name = Path(row["cif"]).name
        choices = index.get(name, [])
        if not choices:
            raise ValueError(f"Missing structure for {row['protein_id']}: {name}; pool cannot be shrunk")
        # Same basename may occur in multiple organism folders. Require identical bytes.
        if len(choices) > 1:
            from cyp_external_common import digest
            if len({digest(p) for p in choices}) != 1:
                raise ValueError(f"Ambiguous structure {name}")
        selected[row["protein_id"]] = choices[0]
    return selected


def run(device="cuda", preflight=False):
    torch = safe_torch()
    source("clipzyme_pretrained")
    from clipzyme.models.protmol import EnzymeReactionCLIP
    from clipzyme.utils.screening import process_mapped_reaction
    from clipzyme.utils.protein_utils import read_structure_file, filter_resolution, build_graph, compute_graph_edges
    from torch_geometric.data import Batch
    from torch_geometric.inspector import Inspector
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.Data.IUPACData import protein_letters_3to1
    from esm import pretrained
    # The released EGNN calls PyG's former Inspector.distribute method.
    # PyG 2.6.1 exposes the same signature-filtering operation under
    # collect_param_data; keep the checkpoint and model source unchanged.
    if not hasattr(Inspector, "distribute"):
        Inspector.distribute = Inspector.collect_param_data
    checkpoint = next((ASSETS / "clipzyme_models").rglob("clipzyme_model.ckpt"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    args = ckpt["hyper_parameters"]["args"]
    if args.train_esm_with_graph or args.reaction_clip_model_path or args.use_protein_msa:
        raise ValueError("Unexpected checkpoint architecture")
    model = EnzymeReactionCLIP(args).eval()
    model.load_state_dict({k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}, strict=True)
    del ckpt
    mapped, proteins = inputs(); paths = structures(mapped)
    rxns = {r["query_id"]: r["reaction"] for r in mapped}
    reaction_graphs = {q: process_mapped_reaction(r, use_one_hot_mol_features=args.use_one_hot_mol_features) for q, r in rxns.items()}
    if preflight:
        print(f"CLIPZyme preflight OK: checkpoint, {len(paths)} structures, all mapped reactions", flush=True)
        return
    model.to(device)
    esm_path = next((ASSETS / "clipzyme_models").rglob("esm2_t33_650M_UR50D.pt"))
    esm, alphabet = pretrained.load_model_and_alphabet_local(str(esm_path))
    esm.eval().to(device); converter = alphabet.get_batch_converter()
    pfeatures = {}; parser = MMCIFParser(QUIET=True)
    aa = {k.upper(): v for k, v in protein_letters_3to1.items()}
    aa["UNK"] = "X"
    with torch.inference_mode():
        for n, (pid, seq) in enumerate(sorted(proteins.items())):
            current_parser = PDBParser(QUIET=True) if paths[pid].suffix == ".pdb" else parser
            residues, atoms, positions = read_structure_file(current_parser, str(paths[pid]), pid)
            names, structure_seq, pos = filter_resolution(residues, atoms, positions, protein_resolution="residue")
            aa_seq = "".join(aa[s] for s in structure_seq)
            if aa_seq != seq:
                raise ValueError(f"Structure/benchmark sequence mismatch: {pid}; refusing an implicit sequence substitution")
            graph = compute_graph_edges(build_graph(names, structure_seq, pos, pid), knn_size=10)
            graph["receptor"].pos -= graph["receptor"].pos.mean(0, keepdim=True)
            _, _, tokens = converter([(pid, seq)])
            graph["receptor"].x = esm(tokens.to(device), repr_layers=[33], return_contacts=False)["representations"][33][0, 1:len(seq)+1]
            graph.structure_sequence = seq
            keep = {"receptor", "structure_sequence", ("receptor", "contact", "receptor")}
            for key in list(graph.to_dict()):
                if key not in keep: del graph[key]
            result = model.encode_protein({"graph": Batch.from_data_list([graph]).to(device)})
            pfeatures[pid] = torch.nn.functional.normalize(result, dim=-1)[0].cpu()
            if n % 50 == 0: print(f"CLIPZyme proteins {n+1}/{len(proteins)}", flush=True)
        qfeatures = {}
        for q, (a, b) in reaction_graphs.items():
            result = model.encode_reaction({"reactants": Batch.from_data_list([a]).to(device), "products": Batch.from_data_list([b]).to(device)})
            qfeatures[q] = torch.nn.functional.normalize(result, dim=-1)[0].cpu()
    scores = {(r["query_id"], r["protein_id"]): float(qfeatures[r["query_id"]] @ pfeatures[r["protein_id"]]) for r in mapped}
    from cyp_external_common import digest
    receipt = RUN / "recovery/clipzyme/structures.json"
    finish("clipzyme_pretrained", scores, checkpoint,
           "Original mapped reactions; released or exact-sequence AlphaFold monomers with sequence-only ESMFold recovery; native ESM2+EGNN; cosine; no candidate filtering",
           structure_recovery_sha256=digest(receipt) if receipt.exists() else None,
           sequence_policy="full length, exact structure/sequence match required; native training 650 cutoff not applied")
