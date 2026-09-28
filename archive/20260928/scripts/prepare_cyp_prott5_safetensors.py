#!/usr/bin/env python3
"""Copy the pinned ProtT5 weights to safetensors using a patched CPU-only loader.

Never changes the shared HF snapshot, and verifies every tensor after conversion.
Run in .deps/cyp-checkpoint-converter, not the PyTorch 2.4 inference environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cyp_specificity import digest, write_json

SOURCE_SHA256 = "7f51ba885541c7dc569d46b796af57cc7a2ba7945107dced4f19d1b5ec091157"
SOURCE = ROOT.parent / "hf_cache/hub/models--Rostlab--prot_t5_xl_half_uniref50-enc/snapshots/94a6abc029ae13029317b140b7424e012bf8dfbf"
OUTPUT = ROOT / "data/external/cyp_specificity_2026/models/prott5_safetensors"
ASSETS = ("config.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json")


def validate_conversion(directory):
    directory = Path(directory)
    receipt = json.loads((directory / "conversion.json").read_text())
    if receipt["source_sha256"] != SOURCE_SHA256 or receipt["verification"] != "all_tensors_equal":
        raise ValueError("Unverified or different ProtT5 source")
    for name, sha in receipt["outputs"].items():
        if name not in (*ASSETS, "model.safetensors") or digest(directory / name) != sha:
            raise ValueError(f"Modified converted checkpoint: {name}")
    if set(receipt["outputs"]) != {*ASSETS, "model.safetensors"}:
        raise ValueError("Incomplete converted checkpoint")
    return receipt


def convert(source=SOURCE, output=OUTPUT):
    import torch
    import safetensors
    from safetensors import safe_open
    from safetensors.torch import save_file

    if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 6):
        raise RuntimeError("Conversion requires PyTorch >=2.6; do not bypass the loading safety check")
    source, output = Path(source), Path(output)
    if output.exists():
        return validate_conversion(output)
    weights = source / "pytorch_model.bin"
    print("Checking pinned ProtT5 source SHA-256...", flush=True)
    if digest(weights) != SOURCE_SHA256:
        raise ValueError("Source weights differ from the pinned official ProtT5 checkpoint")
    for name in ASSETS:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state or any(not torch.is_tensor(v) for v in state.values()):
        raise ValueError("Expected a tensor-only model state dictionary")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prott5_conversion_", dir=output.parent) as temporary:
        staging = Path(temporary) / "model"
        staging.mkdir()
        # Keep tied weight names but give overlapping aliases separate storage.
        serializable, storages = {}, set()
        for key, tensor in state.items():
            tensor = tensor.detach().contiguous()
            pointer = tensor.untyped_storage().data_ptr()
            serializable[key] = tensor.clone() if pointer in storages else tensor
            storages.add(pointer)
        print(f"Writing {len(state)} tensors as safetensors...", flush=True)
        converted = staging / "model.safetensors"
        save_file(serializable, str(converted), metadata={"format": "pt", "source_sha256": SOURCE_SHA256})
        with safe_open(converted, framework="pt", device="cpu") as saved:
            if set(saved.keys()) != set(state):
                raise ValueError("Converted tensor keys differ")
            for key, tensor in state.items():
                if not torch.equal(tensor, saved.get_tensor(key)):
                    raise ValueError(f"Converted tensor differs: {key}")
        for name in ASSETS:
            shutil.copyfile(source / name, staging / name)
        receipt = dict(source=str(source.resolve()), source_sha256=SOURCE_SHA256,
                       torch_version=torch.__version__, safetensors_version=safetensors.__version__,
                       tensor_count=len(state), verification="all_tensors_equal",
                       outputs={name: digest(staging / name) for name in (*ASSETS, "model.safetensors")})
        write_json(staging / "conversion.json", receipt)
        staging.rename(output)
    print(f"Verified exact-weight conversion: {output}", flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    convert(args.source, args.output)
