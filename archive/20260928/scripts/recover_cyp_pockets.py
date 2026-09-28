"""Rebuild missing native EnzymeCAGE pockets from the released Boltz complexes."""
import importlib.util
import json
from pathlib import Path

from cyp_external_common import ASSETS, RUN, digest, inputs, rows, safe_torch, source


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def recover(allow_missing=False):
    import numpy as np
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.SeqUtils import seq1
    torch = safe_torch(); upstream = source("enzymecage_pretrained")
    pocket = load_file("native_pocket", upstream / "feature/extract_pocket.py")
    gvp_module = load_file("native_gvp", upstream / "feature/gvp_torchdrug_feature.py")
    release = ASSETS / "enzymecage_data/dataset/case-study/liz/eval_rxns"
    complexes = ASSETS / "enzymecage_complexes/dataset/case-study/liz"
    if not (ASSETS / "enzymecage_complexes/complete.json").exists():
        raise RuntimeError("Complex archive extraction must finish before pocket recovery")
    mapped, proteins = inputs(); groups = {}
    for row in mapped: groups.setdefault(row["query_id"], set()).add(row["protein_id"])
    parser = MMCIFParser(QUIET=True); recovered = 0; failures = []
    for q, expected in sorted(groups.items()):
        directory = release / q; feat = directory / "feature/protein"
        gvp = torch.load(feat / "gvp_feature/gvp_protein_feature.pt", map_location="cpu", weights_only=True)
        node = torch.load(feat / "ESM-C_600M/pocket_node_feature/esm_node_feature.pt", map_location="cpu", weights_only=True)
        missing = sorted(expected - (gvp.keys() & node.keys()))
        if not missing: continue
        original = rows(complexes / "finetuning/inputs/rxn_files" / f"{q}.csv")
        positions = {}
        for i, row in enumerate(original):
            if row["protein_id"] in positions: raise ValueError(f"Ambiguous complex index: {q}")
            positions[row["protein_id"]] = i
        dest = RUN / "recovery/enzymecage" / q
        dest.mkdir(parents=True, exist_ok=True)
        additions_gvp = {}; additions_node = {}; provenance = {}
        for pid in missing:
            missing_complex = False
            try:
                index = positions[pid]
                if original[index]["sequence"] != proteins[pid]: raise ValueError("Original input sequence mismatch")
                cif = complexes / "boltz_outputs/eval" / f"{q.removeprefix('r_')}_{index}_model_0.cif"
                if not cif.exists():
                    alternatives = sorted((ASSETS / "boltzcyp_recovery").rglob(cif.name))
                    if alternatives:
                        if len({digest(p) for p in alternatives}) != 1: raise ValueError("Conflicting released complexes")
                        cif = alternatives[0]
                    else:
                        from generate_cyp_complexes import verified_complex
                        missing_complex = True
                        substrate = original[index]["reaction"].split(">>")[0].split(".")[-1]
                        cif = verified_complex(cif.name.removesuffix("_model_0.cif"), proteins[pid], q, pid, substrate)
                        missing_complex = False
                structure = parser.get_structure(pid, str(cif))
                chain = structure[0]["A"]
                aa = [r for r in chain if r.id[0] == " " and "CA" in r]
                if "".join(seq1(r.resname) for r in aa) != proteins[pid]:
                    raise ValueError("Complex chain A sequence does not match candidate")
                if [r.id[1] for r in aa] != list(range(1, len(aa)+1)):
                    raise ValueError("Complex residue numbering is not sequence-aligned")
                residues_text = pocket.get_pocket_info_boltz(str(cif), pocket_save_dir=str(dest / "pockets"))
                if not residues_text: raise ValueError("Native 8-Angstrom substrate pocket is empty")
                pdb = dest / "pockets" / cif.name.replace("_model_0.cif", ".pdb")
                ps = PDBParser(QUIET=True).get_structure(pid, str(pdb))
                residues = gvp_module.get_clean_res_list(ps.get_residues(), ensure_ca_exist=True)
                residues = [r for r in residues if all(a in r for a in ("N", "CA", "C", "O"))]
                if not residues: raise ValueError("Pocket has no complete backbone residues")
                residue_ids = [r.id[1]-1 for r in residues]
                npz = feat / "ESM-C_600M/node_level" / f"{pid}.npz"
                with np.load(npz, allow_pickle=False) as f: embedding = f["node_feature"]
                if embedding.shape not in {(len(proteins[pid]), 1152), (len(proteins[pid])+2, 1152)}:
                    raise ValueError(f"Unexpected ESM-C feature shape {embedding.shape}")
                if any(seq1(r.resname) != proteins[pid][i] for r, i in zip(residues, residue_ids)):
                    raise ValueError("Pocket residue/ESM index mismatch")
                additions_gvp[pid] = gvp_module.get_protein_feature(residues)
                additions_node[pid] = torch.from_numpy(embedding[residue_ids].copy())
                if not torch.isfinite(additions_node[pid]).all(): raise ValueError("Nonfinite ESM-C features")
                provenance[pid] = dict(complex_path=str(cif), complex_sha256=digest(cif),
                    esm_node_sha256=digest(npz), residue_ids_one_based=[i+1 for i in residue_ids],
                    native_esm_index_policy="residue_id_minus_one, preserving author preprocessing including special tokens")
                from generate_cyp_complexes import OUTPUT
                if cif.is_relative_to(OUTPUT):
                    provenance[pid]["complex_source"] = "new_boltz_0.4.1_prediction"
                    provenance[pid]["generation_receipt_sha256"] = digest(cif.parent / "complete.json")
                else:
                    provenance[pid]["complex_source"] = "released_complex"
                recovered += 1
            except (ValueError, KeyError, FileNotFoundError) as error:
                failures.append(dict(query_id=q, protein_id=pid, error=str(error), error_type=type(error).__name__, missing_complex=missing_complex))
                print(f"Cannot recover {q}/{pid}: {error}", flush=True)
        torch.save(additions_gvp, dest / "gvp.pt"); torch.save(additions_node, dest / "node.pt")
        receipt = dict(gvp_sha256=digest(dest / "gvp.pt"), node_sha256=digest(dest / "node.pt"),
                       entries=provenance, source_revision="c6c7dc64b7fffa5b6b265fb52f7b61a566123286")
        (dest / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Recovered {q}: {len(additions_gvp)}/{len(missing)}", flush=True)
    report = RUN / "recovery/enzymecage/report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(dict(recovered_pairs=recovered, failures=failures), indent=2) + "\n")
    if failures:
        if allow_missing and all(f["error_type"] == "FileNotFoundError" and f["missing_complex"] for f in failures):
            print(f"Planning only: {len(failures)} pairs need generation; evaluation remains blocked", flush=True)
            return
        raise RuntimeError(f"{len(failures)} pairs still need complex/feature generation; see {report}")
    print(f"Recovered all {recovered} missing query-candidate pockets", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing", action="store_true", help="Write generation plan, not evaluation approval")
    recover(parser.parse_args().allow_missing)
