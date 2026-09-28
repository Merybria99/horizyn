"""Recover full-pool CLIPZyme monomers without using reaction or activity labels."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cyp_external_common import ASSETS, ROOT, RUN, digest, inputs

RECOVERY = RUN / "recovery/clipzyme"


def validate_fold_sequence(seq):
    if not seq or set(seq) - set("ACDEFGHIKLMNPQRSTVWYX"):
        raise ValueError("Unsupported folding sequence; only standard residues and X are accepted")


def fold_ca_pdb(prediction, seq):
    """Export predicted backbone-frame origins (CA), retaining X as UNK.

    ESMFold's atom masks suppress UNK atoms in the default PDB export. CLIPZyme
    needs only CA coordinates, so use the predicted, Angstrom-scaled backbone
    frames for all positions instead of deleting X or guessing its identity.
    """
    import math
    from Bio.Data.IUPACData import protein_letters_1to3
    validate_fold_sequence(seq)
    coordinates = prediction["frames"][-1, 0, :, 4:].detach().float().cpu().tolist()
    if len(coordinates) != len(seq): raise ValueError("Frame/sequence length mismatch")
    names = {a: name.upper() for a, name in protein_letters_1to3.items()}
    names["X"] = "UNK"
    lines = ["REMARK   CA-only ESMFold backbone frames; unknown residues retained as UNK"]
    for i, (aa, xyz) in enumerate(zip(seq, coordinates), 1):
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise ValueError("Nonfinite or malformed backbone frame")
        x, y, z = xyz
        if any(len(f"{v:8.3f}") != 8 for v in xyz) or i > 9999:
            raise ValueError("Coordinates/residue number exceed PDB field widths")
        lines.append(f"ATOM  {i:5d}  CA  {names[aa]:3s} A{i:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}           C  ")
    return "\n".join(lines + ["TER", "END", ""])


def sequence(path):
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.SeqUtils import seq1
    parser = PDBParser(QUIET=True) if Path(path).suffix == ".pdb" else MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", str(path))
    return "".join(seq1(r.resname) for r in next(structure.get_models()).get_residues()
                   if r.id[0] == " " and "CA" in r)


def recover():
    import requests
    _, proteins = inputs()
    RECOVERY.mkdir(parents=True, exist_ok=True)
    manifest = RECOVERY / "structures.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else {}
    available = {p.name: p for p in (ASSETS / "clipzyme_structures").rglob("*.cif")}

    def resolve(item):
        pid, seq = item
        old = records.get(pid)
        if old and Path(old["path"]).is_file() and digest(old["path"]) == old["sha256"] and sequence(old["path"]) == seq:
            return pid, old
        path = available.get(f"AF-{pid}-F1-model_v4.cif")
        if path and sequence(path) == seq:
            return pid, dict(path=str(path), sha256=digest(path), source="released_alphafold_v4")
        # Download only exact full-length sequence matches. Changed accessions,
        # fragments and different isoforms are never silently substituted.
        try:
            response = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{pid}", timeout=20)
            if response.status_code == 404: return pid, None
            response.raise_for_status()
            for entry in response.json():
                if entry.get("uniprotSequence") != seq: continue
                url = entry.get("cifUrl", "")
                if not url.startswith("https://alphafold.ebi.ac.uk/files/"): continue
                data = requests.get(url, timeout=30); data.raise_for_status()
                target = RECOVERY / f"{pid}.cif"
                temporary = target.with_name(target.stem + ".partial.cif")
                temporary.write_bytes(data.content)
                if sequence(temporary) != seq: continue
                temporary.replace(target)
                return pid, dict(path=str(target), sha256=digest(target), source=url)
        except requests.RequestException as error:
            print(f"AlphaFold lookup failed for {pid}: {error}", flush=True)
        return pid, None

    missing = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, (pid, record) in enumerate(pool.map(resolve, sorted(proteins.items()))):
            if record: records[pid] = record
            else: missing.append(pid); records.pop(pid, None)
            if n % 250 == 0:
                manifest.write_text(json.dumps(records, indent=2) + "\n")
                print(f"Structure audit {n+1}/{len(proteins)}", flush=True)
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    # Exact-sequence aliases share structures; no nearest-neighbour substitution.
    by_sequence = {proteins[p]: r for p, r in sorted(records.items())}
    for pid in list(missing):
        if proteins[pid] in by_sequence:
            records[pid] = dict(by_sequence[proteins[pid]], exact_sequence_alias=True)
            missing.remove(pid)
    print(f"{len(missing)} full-length monomers require sequence-only ESMFold", flush=True)
    if missing:
        for pid in missing: validate_fold_sequence(proteins[pid])
        import torch
        from transformers import EsmForProteinFolding
        model_path = ROOT / "wet_lab/cache/esmfold/models--facebook--esmfold_v1/snapshots/75a3841ee059df2bf4d56688166c8fb459ddd97a"
        torch.manual_seed(42)
        model = EsmForProteinFolding.from_pretrained(model_path, local_files_only=True).eval()
        model.trunk.config.max_recycles = 4
        model.esm = model.esm.half(); model.trunk.set_chunk_size(32); model.cuda()
        for n, pid in enumerate(missing):
            print(f"Folding {n+1}/{len(missing)}: {pid}, {len(proteins[pid])} residues", flush=True)
            with torch.inference_mode():
                prediction = model.infer(proteins[pid])
                pdb = fold_ca_pdb(prediction, proteins[pid]) if "X" in proteins[pid] else model.output_to_pdb(prediction)[0]
            target = RECOVERY / f"{pid}.pdb"
            target.write_text(pdb)
            if sequence(target) != proteins[pid]: raise ValueError(f"Folded sequence mismatch: {pid}")
            records[pid] = dict(path=str(target), sha256=digest(target), source="sequence_only_esmfold_v1",
                               model_revision=model_path.name, num_recycles=4, seed=42,
                               unknown_residues=[i+1 for i, aa in enumerate(proteins[pid]) if aa == "X"],
                               coordinate_policy="predicted_CA_frames_UNK_preserved" if "X" in proteins[pid] else "native_all_atom_export")
            manifest.write_text(json.dumps(records, indent=2) + "\n")
            del prediction; torch.cuda.empty_cache()
    if records.keys() != proteins.keys(): raise ValueError("Incomplete structure recovery")
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Recovered full structure coverage: {len(records)} proteins", flush=True)


if __name__ == "__main__": recover()
