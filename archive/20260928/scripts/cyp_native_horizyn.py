"""Official Horizyn-1 development checkpoint, forward-reaction retrieval."""
import importlib.util
import json
import re

from cyp_external_common import BENCH, DATA, ROOT, digest, finish, inputs, rows, safe_torch, source


def run(device="cuda", preflight=False):
    torch = safe_torch()
    upstream = source("horizyn1_dev")
    from horizyn.lightning_module import HorizynLitModule
    from horizyn.config import load_config
    checkpoint = DATA / "release/horizyn_v1_0_dev.ckpt.part"
    import hashlib
    with checkpoint.open("rb") as f:
        if hashlib.file_digest(f, "md5").hexdigest() != "5b1f938f8b0a82fbe91892a3b4e2bf2c":
            raise ValueError("Official development checkpoint checksum mismatch")
    model = HorizynLitModule.load_from_checkpoint(str(checkpoint), map_location="cpu", weights_only=True).eval()
    spec = importlib.util.spec_from_file_location("official_predict", upstream / "scripts/predict.py")
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    config = load_config(upstream / "configs/sota.yaml")
    mapped, proteins = inputs()
    reactions = {r["reaction_id"]: r["rxn"] for r in rows(BENCH / "queries.csv")}
    import h5py
    import numpy as np
    cache = ROOT / "runs/cyp_specificity_v1/features/proteins.h5"
    receipt = json.loads((cache.parent / "prott5.complete.json").read_text())
    if receipt["signature"]["benchmark"] != digest(BENCH / "manifest.json"):
        raise ValueError("ProtT5 cache belongs to a different benchmark")
    for record in receipt["outputs"]:
        path = type(cache)(record["path"])
        if path.stat().st_size != record["size"] or path.stat().st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"Cached feature file changed: {path}")
    means = {}
    with h5py.File(cache, "r") as f:
        ids = list(f["ids"].asstr()[:]); offsets = f["offsets"][:]
        if set(ids) != set(proteins) or f.attrs["embedding_model_type"] != "prott5":
            raise ValueError("Wrong protein feature cache")
        for i, pid in enumerate(ids):
            if len(proteins[pid]) <= 1022:
                if offsets[i + 1] - offsets[i] != len(proteins[pid]):
                    raise ValueError(f"Unexpected residue count: {pid}")
                if not preflight or i == 0:
                    means[pid] = np.asarray(f["vectors"][offsets[i]:offsets[i+1]], dtype=np.float32).mean(0)
    # Fingerprint every query during preflight; never drop an unparseable reaction.
    fps = {q: native.build_reaction_fingerprint(r, config)[0] for q, r in reactions.items()}
    if preflight:
        with torch.inference_mode():
            p = model.model.target_encoder(torch.from_numpy(next(iter(means.values()))).unsqueeze(0))
            q = model.model.query_encoder(next(iter(fps.values())))
            if not torch.isfinite(q @ p.T).all(): raise ValueError("Nonfinite native forward")
        from transformers import T5Tokenizer
        T5Tokenizer.from_pretrained(DATA / "models/prott5_safetensors", local_files_only=True)
        print("Horizyn preflight OK: checkpoint, all reactions, protein offsets, native forward, tokenizer", flush=True)
        return
    missing = sorted(set(proteins) - means.keys())
    if missing:
        from transformers import T5EncoderModel, T5Tokenizer
        path = DATA / "models/prott5_safetensors"
        tokenizer = T5Tokenizer.from_pretrained(path, local_files_only=True)
        encoder = T5EncoderModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32).to(device).eval()
        with torch.inference_mode():
            for pid in missing:
                seq = re.sub(r"[UZOB]", "X", proteins[pid])
                tokens = tokenizer(" ".join(seq), return_tensors="pt").to(device)
                hidden = encoder(**tokens).last_hidden_state[0, :len(seq)]
                if hidden.shape[0] != len(seq):
                    raise ValueError("ProtT5 sequence unexpectedly truncated")
                means[pid] = hidden.float().mean(0).cpu().numpy()
        del encoder
        if device == "cuda": torch.cuda.empty_cache()
    model.to(device)
    ids = sorted(proteins)
    with torch.inference_mode():
        p = torch.cat([model.model.target_encoder(torch.from_numpy(np.stack([means[i] for i in ids[s:s+128]])).to(device)).cpu()
                       for s in range(0, len(ids), 128)])
        q = {k: model.model.query_encoder(v.to(device)).cpu()[0] for k, v in fps.items()}
    index = {pid: i for i, pid in enumerate(ids)}
    scores = {(r["query_id"], r["protein_id"]): float(q[r["query_id"]] @ p[index[r["protein_id"]]]) for r in mapped}
    finish("horizyn1_dev", scores, checkpoint,
           "Official forward RDKitPlus+DRFP; full-sequence ProtT5 residue mean; cosine",
           reused_float16_residue_cache=str(cache), full_length_reembedded=missing,
           wrapper_sha256=digest(__file__), fingerprint_direction="forward_only")
