"""Extract only checksum-pinned external benchmark assets, without tar paths/links."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/external/cyp_specificity_2026/release"
ASSETS = ROOT / "runs/cyp_external_v1/assets"
ARCHIVES = {
    "boltzcyp_generation_inputs": ("boltzcyp_data_dir.tar.gz", "9ec82ef270d95a5e4ae2fe3a4ac7a73c3cf5b8d755c94037f5c91028c792e339"),
    "boltzcyp_recovery": ("boltzcyp_data_dir.tar.gz", "9ec82ef270d95a5e4ae2fe3a4ac7a73c3cf5b8d755c94037f5c91028c792e339"),
    "enzymecage_complexes": ("ezcage_data_dir.tar.zst", "88cdafdbabd88c43cde8271dbeb9b111dc9f1a81db63c029443d3887fe05a983"),
    "clipzyme_structures": ("clipzyme_data_dir.tar.zst", "6841e193faaec29e4f084634fe1ead8341bc1d1c4c594eee2713fb5f819056ce"),
    "clipzyme_data": ("clipzyme_data_dir.tar.zst", "6841e193faaec29e4f084634fe1ead8341bc1d1c4c594eee2713fb5f819056ce"),
    "clipzyme_models": ("clipzyme_model_ckpts.tar.zst", "78c6e095b52ac1e3a294d5573656863aaff78f5bdfe56481da0c2562daee0147"),
    "enzymecage_data": ("ezcage_data_dir.tar.zst", "88cdafdbabd88c43cde8271dbeb9b111dc9f1a81db63c029443d3887fe05a983"),
}


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def choose(kind, path):
    name = str(path)
    if kind == "boltzcyp_generation_inputs":
        return path if not path.name.startswith("._") and (path.suffix == ".a3m" or ("/eval/boltz_inputs/" in name and path.suffix == ".fasta")) else None
    if kind == "boltzcyp_recovery":
        return path if "/eval/" in name and path.suffix in {".cif", ".yaml", ".json", ".csv"} and not path.name.startswith("._") else None
    if kind == "enzymecage_complexes":
        if "/boltz_outputs/eval/" in name and path.suffix == ".cif":
            return path
        if "/finetuning/inputs/rxn_files/" in name and path.suffix == ".csv":
            return path
        return None
    if kind == "clipzyme_structures":
        return path if "/eval/" in name and path.suffix == ".cif" else None
    if kind == "clipzyme_models":
        return path if path.name in {"clipzyme_model.ckpt", "esm2_t33_650M_UR50D.pt",
                                    "esm2_t33_650M_UR50D-contact-regression.pt"} else None
    if kind == "clipzyme_data":
        if "/eval/" in name and name.endswith("_graph.pt"):
            return PurePosixPath("graphs") / path.name
        if "/eval/" in name and name.endswith(".csv"):
            return path
        return None
    # Keep evaluation features/inputs only. No training data, raw Boltz outputs,
    # adapted-model predictions or checkpoints are required for inference.
    if "eval_rxns" in path.parts and "pruned" not in path.parts:
        return path
    if path.name == "holdout_for_eval_human_dups_removed.csv":
        return path
    return None


def extract(kind):
    filename, expected = ARCHIVES[kind]
    archive, output = RELEASE / filename, ASSETS / kind
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "complete.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["archive_sha256"] != expected:
            raise ValueError("Changed archive receipt")
        print(f"Already extracted: {kind}", flush=True)
        return
    print(f"Verifying {filename}", flush=True)
    if digest(archive) != expected:
        raise ValueError(f"Archive hash mismatch: {archive}")
    files, index = {}, []
    decompressor = "gzip" if archive.suffix == ".gz" else "zstd"
    proc = subprocess.Popen([decompressor, "-dc", str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe tar path: {member.name}")
                if member.isdir():
                    continue
                index.append(dict(path=str(path), size=member.size))
                relative = choose(kind, path)
                if relative is None:
                    continue
                if not member.isfile():
                    raise ValueError(f"Not a regular file: {member.name}")
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".partial")
                if target.is_symlink() or temporary.is_symlink():
                    raise ValueError(f"Symlink target: {target}")
                with tar.extractfile(member) as source, temporary.open("wb") as dest:
                    shutil.copyfileobj(source, dest, length=8 * 1024**2)
                sha = digest(temporary)
                if str(relative) in files and files[str(relative)] != sha:
                    raise ValueError(f"Conflicting archive member: {relative}")
                temporary.replace(target)
                files[str(relative)] = sha
                if len(files) % 500 == 0:
                    print(f"{kind}: {len(files)} files extracted", flush=True)
        if proc.wait() != 0:
            raise RuntimeError("zstd decompression failed")
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.stdout.close()
    (output / "archive_index.json").write_text(json.dumps(index, indent=2) + "\n")
    temporary = receipt.with_suffix(".partial")
    temporary.write_text(json.dumps(dict(archive_sha256=expected, files=files), indent=2) + "\n")
    temporary.replace(receipt)
    print(f"Complete: {kind}, {len(files)} files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=list(ARCHIVES))
    extract(parser.parse_args().asset)
