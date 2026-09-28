"""Generate absent CYP complexes with Boltz-1; never substitute missing candidates."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from cyp_external_common import ASSETS, ROOT, RUN, digest, inputs, rows

OUTPUT = RUN / "recovery/enzymecage_generated"
PROTOCOL = dict(boltz="0.4.1", recycling_steps=3, sampling_steps=200,
                diffusion_samples=1, seed=42, step_scale=1.638,
                msa="released when present; otherwise explicitly generated via ColabFold", chains="A:protein B:HEM C:substrate",
                weights_revision="7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1")
WEIGHTS = {"boltz1_conf.ckpt": "fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5",
           "ccd.pkl": "2d3b2f03a3c5665944adba51e33263511e51b21c9cd05d902f9c4b7c1e58d2f4"}
MSA_SERVER = "https://api.colabfold.com"


def prepare_weights(cache):
    import urllib.request
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in WEIGHTS.items():
        target = cache / name
        if not target.exists():
            print(f"Downloading pinned Boltz asset: {name}", flush=True)
            temporary = target.with_suffix(".download")
            urllib.request.urlretrieve(
                f"https://huggingface.co/boltz-community/boltz-1/resolve/{PROTOCOL['weights_revision']}/{name}", temporary)
            if digest(temporary) != expected: raise ValueError(f"Download hash mismatch: {name}")
            temporary.replace(target)
        elif digest(target) != expected:
            raise ValueError(f"Boltz asset hash mismatch: {target}")


def atomic_json(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def fasta_records(path):
    result = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            result.append([line[1:], ""])
        elif line.strip() and not line.startswith("#"):
            if not result: raise ValueError(f"Invalid FASTA: {path}")
            result[-1][1] += line.strip()
    return result


def validate_msa(path, sequence):
    # Read just the query, not all homologs in a potentially large MSA.
    query = []; started = False
    with path.open() as stream:
        for line in stream:
            if line.startswith(">"):
                if started: break
                started = True
            elif started: query.append(line.strip())
    if "".join(query) != sequence:
        raise ValueError(f"MSA query sequence mismatch: {path}")


def ligand_key(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError("Invalid substrate SMILES")
    for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def plan():
    asset = ASSETS / "boltzcyp_generation_inputs"
    if not (asset / "complete.json").is_file():
        raise RuntimeError("Finish boltzcyp_generation_inputs extraction first")
    mapped, proteins = inputs()
    pairs = {(r["query_id"], r["protein_id"]): r for r in mapped}
    report = json.loads((RUN / "recovery/enzymecage/report.json").read_text())
    files = {}
    for path in asset.rglob("*"):
        if path.suffix in {".fasta", ".a3m"}:
            files.setdefault(path.name, []).append(path)
    tables = {}; hashes = {}; checked_msas = set()
    def file_hash(path):
        if path not in hashes: hashes[path] = digest(path)
        return hashes[path]
    specs = []
    for failure in report["failures"]:
        q, pid = failure["query_id"], failure["protein_id"]
        if failure.get("error_type") != "FileNotFoundError" or not failure.get("missing_complex"):
            raise ValueError(f"Not a missing-complex failure: {failure}")
        pair = pairs[q, pid]
        if q not in tables:
            tables[q] = rows(ASSETS / "enzymecage_complexes/dataset/case-study/liz/finetuning/inputs/rxn_files" / f"{q}.csv")
        table = tables[q]
        matches = [(i, r) for i, r in enumerate(table) if r["protein_id"] == pid]
        if len(matches) != 1 or matches[0][1]["sequence"] != proteins[pid]:
            raise ValueError(f"Ambiguous original candidate: {q}/{pid}")
        index, original = matches[0]
        if original["reaction"] != pair["reaction"]:
            raise ValueError("Original reaction mismatch")
        name = f"{q.removeprefix('r_')}_{index}"
        candidates = sorted(files.get(name + ".fasta", []))
        substrate = original["reaction"].split(">>")[0].split(".")[-1]
        valid = []
        for path in candidates:
            records = fasta_records(path)
            if (len(records) == 3 and records[0][0].startswith("A|protein|")
                    and records[0][1] == proteins[pid]
                    and records[1] == ["B|ccd", "HEM"] and records[2][0] == "C|smiles"
                    and ligand_key(records[2][1]) == ligand_key(substrate)):
                valid.append(path)
        if not valid: raise ValueError(f"No matching released A/B/C input: {name}")
        substrate = fasta_records(valid[0])[2][1]
        msas = sorted(files.get(pid + ".a3m", []))
        msa_hash = None
        if msas:
            if len({file_hash(p) for p in msas}) != 1: raise ValueError(f"Conflicting MSAs for {pid}")
            msa = msas[0]
            if pid not in checked_msas:
                validate_msa(msa, proteins[pid]); checked_msas.add(pid)
            msa_hash = file_hash(msa)
        else:
            msa = OUTPUT / "msas" / pid / f"{pid}.a3m"
        specs.append(dict(name=name, query_id=q, protein_id=pid, sequence=proteins[pid],
                          substrate=substrate, msa=str(msa), msa_sha256=msa_hash,
                          msa_source="released" if msas else MSA_SERVER,
                          released_input_sha256=file_hash(valid[0]), protocol=PROTOCOL))
    if len({s["name"] for s in specs}) != len(specs): raise ValueError("Duplicate generation input")
    return sorted(specs, key=lambda s: (len(s["sequence"]), s["name"]))


def validate_complex(path, sequence):
    from Bio.PDB import MMCIFParser
    from Bio.SeqUtils import seq1
    model = MMCIFParser(QUIET=True).get_structure("complex", str(path))[0]
    residues = [r for r in model["A"] if "CA" in r]
    if "".join(seq1(r.resname) for r in residues) != sequence:
        raise ValueError("Generated chain A sequence mismatch")
    if [r.id[1] for r in residues] != list(range(1, len(sequence) + 1)):
        raise ValueError("Generated chain A numbering mismatch")
    if not any(r.resname == "HEM" for r in model["B"]) or not list(model["C"].get_atoms()):
        raise ValueError("Missing heme/substrate chains")


def verified_complex(name, sequence, query_id, protein_id, substrate):
    directory = OUTPUT / name
    receipt = json.loads((directory / "complete.json").read_text())
    spec = receipt["input"]
    if json.loads((directory / "input.json").read_text()) != spec:
        raise ValueError("Generated complex receipt/input mismatch")
    if receipt.get("checkpoint_sha256") != WEIGHTS["boltz1_conf.ckpt"] or receipt.get("ccd_sha256") != WEIGHTS["ccd.pkl"]:
        raise ValueError("Generated complex weights mismatch")
    if (spec["sequence"], spec["query_id"], spec["protein_id"], ligand_key(spec["substrate"]), spec["protocol"]) != (sequence, query_id, protein_id, ligand_key(substrate), PROTOCOL):
        raise ValueError("Generated complex input/protocol mismatch")
    path = directory / f"{name}_model_0.cif"
    if digest(path) != receipt["cif_sha256"]: raise ValueError("Generated complex hash mismatch")
    return path


def check_gpu(gpu):
    if not gpu.isdecimal(): raise ValueError("Use one physical GPU index")
    busy = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
                                    "--format=csv,noheader"], text=True).strip()
    if busy: raise RuntimeError(f"GPU {gpu} occupied; no processes stopped: {busy}")


def msa_worker(request):
    from boltz.data.msa.mmseqs2 import run_mmseqs2
    data = json.loads(request.read_text())
    directory = request.parent
    result = run_mmseqs2([data["sequence"]], str(directory / "search"),
                        use_env=True, use_pairing=False, host_url=MSA_SERVER)
    if len(result) != 1: raise ValueError("Unexpected MSA search result count")
    path = directory / f"{data['protein_id']}.a3m"
    temporary = path.with_suffix(".partial")
    temporary.write_text(result[0])
    validate_msa(temporary, data["sequence"])
    temporary.replace(path)
    from datetime import datetime, timezone
    atomic_json(directory / "complete.json", dict(input=data, sha256=digest(path),
                server=MSA_SERVER, generated_at=datetime.now(timezone.utc).isoformat(),
                database_version="server-managed; not guaranteed identical to paper"))


def ensure_msa(spec, allow_generation):
    spec = dict(spec)
    if spec["msa_source"] == "released": return spec
    path = Path(spec["msa"]); directory = path.parent
    request = dict(sequence=spec["sequence"], protein_id=spec["protein_id"])
    receipt = directory / "complete.json"
    if not receipt.exists():
        if not allow_generation:
            raise RuntimeError("Missing released MSA; pass --generate-missing-msas for public ColabFold search")
        directory.mkdir(parents=True, exist_ok=True)
        request_path = directory / "request.json"
        if request_path.exists() and json.loads(request_path.read_text()) != request:
            raise ValueError("Changed MSA request")
        atomic_json(request_path, request)
        print(f"Generating MSA for {spec['protein_id']} via {MSA_SERVER}", flush=True)
        subprocess.run([sys.executable, __file__, "--msa-worker", str(request_path)], check=True, timeout=1800)
    metadata = json.loads(receipt.read_text())
    if metadata["input"] != request or metadata["server"] != MSA_SERVER or digest(path) != metadata["sha256"]:
        raise ValueError("Generated MSA receipt mismatch")
    validate_msa(path, spec["sequence"])
    spec["msa_sha256"] = metadata["sha256"]
    spec["msa_receipt_sha256"] = digest(receipt)
    return spec


def run(specs, gpu, limit, allow_msa_generation=False):
    import shutil
    if importlib.metadata.version("boltz") != PROTOCOL["boltz"]:
        raise RuntimeError("Use the isolated Boltz 0.4.1 environment")
    if not specs: return
    check_gpu(gpu)
    cache = ROOT / ".deps/cyp-boltz-cache"
    prepare_weights(cache)
    for spec in specs[:limit] if limit else specs:
        spec = ensure_msa(spec, allow_msa_generation)
        directory = OUTPUT / spec["name"]
        directory.mkdir(parents=True, exist_ok=True)
        metadata = directory / "input.json"
        if metadata.exists() and json.loads(metadata.read_text()) != spec:
            raise ValueError(f"Changed generation inputs: {directory}")
        atomic_json(metadata, spec)
        if (directory / "complete.json").exists():
            verified_complex(spec["name"], spec["sequence"], spec["query_id"], spec["protein_id"], spec["substrate"])
            print(f"Already generated: {spec['name']}", flush=True); continue
        fasta = directory / f"{spec['name']}.fasta"
        fasta.write_text(f">A|protein|{spec['msa']}\n{spec['sequence']}\n>B|ccd\nHEM\n>C|smiles\n{spec['substrate']}\n")
        command = [str(Path(sys.executable).parent / "boltz"), "predict", str(fasta),
                   "--out_dir", str(directory / "prediction"), "--cache", str(cache),
                   "--devices", "1", "--accelerator", "gpu", "--output_format", "mmcif",
                   "--num_workers", "0", "--override"]
        for key in ("recycling_steps", "sampling_steps", "diffusion_samples", "seed", "step_scale"):
            command += ["--" + key, str(PROTOCOL[key])]
        print(f"Generating {spec['name']} ({len(spec['sequence'])} residues)", flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        subprocess.run(command, env=env, check=True)
        outputs = list((directory / "prediction").rglob(f"{spec['name']}_model_0.cif"))
        if len(outputs) != 1: raise RuntimeError(f"Boltz did not produce exactly one complex: {directory}")
        validate_complex(outputs[0], spec["sequence"])
        dest = directory / outputs[0].name
        shutil.copyfile(outputs[0], dest.with_suffix(".partial"))
        dest.with_suffix(".partial").replace(dest)
        atomic_json(directory / "complete.json", dict(input=spec, cif_sha256=digest(dest), command=command,
                    checkpoint_sha256=WEIGHTS["boltz1_conf.ckpt"], ccd_sha256=WEIGHTS["ccd.pkl"]))
        print(f"Saved {dest.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--generate-missing-msas", action="store_true", help="Send public candidate sequences with absent MSAs to ColabFold")
    parser.add_argument("--msa-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, help="Smoke test only; does not mark full recovery complete")
    args = parser.parse_args()
    if args.msa_worker:
        msa_worker(args.msa_worker); return
    if args.limit is not None and args.limit < 1: parser.error("--limit must be positive")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs = plan()
        atomic_json(OUTPUT / "plan.json", specs)
        missing_msas = {s["protein_id"] for s in specs if s["msa_source"] != "released"}
        print(f"Missing complex generation plan: {len(specs)} pairs; {len(missing_msas)} proteins require generated MSAs", flush=True)
        if not args.plan_only: run(specs, args.gpu, args.limit, args.generate_missing_msas)


if __name__ == "__main__": main()
