"""CPU inference with pinned released FusionESP weights and cached embeddings.

Run in the isolated torch>=2.6 converter environment. No arbitrary pickle loader,
remote code, model fitting, embedding extraction or CUDA allocation is used.
Architecture matches the authors' model.py at ef5a7ce3ac638c726b3144ff435d05c46de1f55d.
"""
import argparse
import io
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .cyp_baselines import DATA_SHA, MODELS_SHA, archive_members, digest, read_rows, write_json, write_rows
except ImportError:
    from cyp_baselines import DATA_SHA, MODELS_SHA, archive_members, digest, read_rows, write_json, write_rows

SOURCE = "https://gitlab.com/cyp_pred_repos/fusionesp_cyp_retrain/-/blob/ef5a7ce3ac638c726b3144ff435d05c46de1f55d/model.py"
CHECKPOINTS = {"fusionesp_pretrained": "model_ckpts/fusionesp_esm2560_best_model.pt",
               "fusionesp_cyp_adapted": "model_ckpts/fusionesp_ft_best_model.pt"}
EMBEDDINGS = {"enzyme": "data_dir/eval/inputs/embeddings/mean_sequence_embeddings.pt",
              "substrate": "data_dir/eval/inputs/embeddings/pooler_mol_embeddings_eval.pt"}


class Contrastive_learning_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enzy_refine_layer_1 = nn.Linear(2560, 2560)
        self.smiles_refine_layer_1 = nn.Linear(768, 768)
        self.enzy_refine_layer_2 = nn.Linear(2560, 128)
        self.smiles_refine_layer_2 = nn.Linear(768, 128)
        self.relu = nn.ReLU()
        self.batch_norm_enzy = nn.BatchNorm1d(2560)
        self.batch_norm_smiles = nn.BatchNorm1d(768)
        self.batch_norm_shared = nn.BatchNorm1d(128)

    def forward(self, enzyme, substrate):
        enzyme = self.enzy_refine_layer_2(self.relu(self.batch_norm_enzy(self.enzy_refine_layer_1(enzyme))))
        substrate = self.smiles_refine_layer_2(self.relu(self.batch_norm_smiles(self.smiles_refine_layer_1(substrate))))
        return (nn.functional.normalize(self.batch_norm_shared(enzyme), dim=1),
                nn.functional.normalize(self.batch_norm_shared(substrate), dim=1))


def safe_load(raw, model=False):
    # Older torch weights_only implementations had security defects. This also
    # prevents accidentally changing the training environment's torch version.
    if tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 6):
        raise RuntimeError("Use the isolated torch>=2.6 CPU environment; never weights_only=False")
    if model:
        allowed = [(Contrastive_learning_layer, "__main__.Contrastive_learning_layer"),
                   (Contrastive_learning_layer, "model.Contrastive_learning_layer"),
                   nn.Linear, nn.BatchNorm1d, nn.ReLU]
    else:
        from numpy.core.multiarray import _reconstruct
        allowed = [(_reconstruct, "numpy.core.multiarray._reconstruct"),
                   (_reconstruct, "numpy._core.multiarray._reconstruct"),
                   np.ndarray, np.dtype, type(np.dtype(np.float32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def run(data_archive, model_archive, prepared, bundle, output, threads=4):
    import hashlib
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    prepared, output = Path(prepared), Path(output)
    features_raw = dict(archive_members(data_archive, names=EMBEDDINGS.values(), expected_sha=DATA_SHA))
    features = {k: safe_load(features_raw[v]) for k, v in EMBEDDINGS.items()}
    models_raw = dict(archive_members(model_archive, names=CHECKPOINTS.values(), expected_sha=MODELS_SHA))
    queries = {r["query_id"]: r["unmapped_canonical"] for r in read_rows(prepared / "substrates.csv")}
    pairs = list(read_rows(Path(bundle) / "candidates.csv"))
    # Fail before any inference on incomplete released embeddings; never skip.
    for feature, keys, dimension in (
        (features["enzyme"], {r["protein_id"] for r in pairs}, 2560),
        (features["substrate"], set(queries.values()), 768),
    ):
        for key in keys:
            if key not in feature:
                raise ValueError(f"Released embedding missing: {key}")
            value = np.asarray(feature[key])
            if value.shape != (dimension,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid released embedding: {key}, {value.shape}, {value.dtype}")
    for method, member in CHECKPOINTS.items():
        loaded = safe_load(models_raw[member], model=True)
        if type(loaded) is not Contrastive_learning_layer:
            raise ValueError("Unexpected checkpoint architecture")
        model = Contrastive_learning_layer().eval()
        model.load_state_dict(loaded.state_dict(), strict=True)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(pairs), 32):
                batch = pairs[start:start + 32]
                enzymes = torch.from_numpy(np.stack([features["enzyme"][r["protein_id"]] for r in batch]))
                substrates = torch.from_numpy(np.stack([features["substrate"][queries[r["query_id"]]] for r in batch]))
                e, s = model(enzymes, substrates)
                scores = nn.functional.cosine_similarity(e, s, dim=1).tolist()
                predictions.extend(dict(query_id=r["query_id"], protein_id=r["protein_id"], score=value)
                                   for r, value in zip(batch, scores))
        score_file = output / f"{method}.csv"
        write_rows(score_file, predictions)
        write_json(score_file.with_suffix(".json"), dict(
            source=SOURCE, checkpoint_member=member,
            checkpoint_sha256=hashlib.sha256(models_raw[member]).hexdigest(),
            model_archive_sha256=MODELS_SHA, data_archive_sha256=DATA_SHA,
            input_manifest_sha256=digest(Path(bundle) / "manifest.json"),
            scores_sha256=digest(score_file), torch_version=torch.__version__,
            embeddings_sha256={k: hashlib.sha256(v).hexdigest() for k, v in features_raw.items()},
            input_scope="substrate-only, ESM2-3B + MolFormer released embeddings",
            upstream_exposure="UNKNOWN; no leakage-free claim",
            selection="Authors' fixed released checkpoint; not selected on CYP scores"))
        print(f"{method}: {len(predictions)} scores on CPU", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-archive", "model-archive", "prepared", "bundle", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    run(**vars(parser.parse_args()))
