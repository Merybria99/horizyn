# EnzGFM values with a ProtT5/SLEEC prior

This experiment replaces the residue representations used by the CIRCE/F3
enzyme tower with EnzGFM-650M representations. It deliberately retains the
existing frozen SLEEC checkpoint: an aligned ProtT5 HDF5 is passed only to the
SLEEC scorer, and the resulting logits bias site, mechanism, and cofactor
attention over EnzGFM residue values.

The two HDF5 files must contain the same protein IDs and exactly one vector per
residue before truncation. Training validates both embedding dimensions and
rejects length mismatches before constructing a batch.

## Prepare EnzGFM

EnzGFM is not currently published as a standard Transformers package. Clone
the authors' implementation and create their Python 3.10 environment:

```bash
git clone https://github.com/DeepBxM/EnzGFM.git third_party/EnzGFM
conda env create \
  --prefix /datastor2/deep-proteins/EnzymeDiscovery/enzgfm-env \
  -f third_party/EnzGFM/environment.yml
```

Download and unpack the EnzGFM-650M pretrained model from the authors' Zenodo
record into `checkpoints/EnzGFM-650M`. The directory must contain
`config.json` and the model weights.

## Extract aligned residue embeddings

Use the same FASTA that produced the existing ProtT5 residue HDF5:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ENZGFM_ENV_PATH=/datastor2/deep-proteins/EnzymeDiscovery/enzgfm-env \
ENZGFM_REPO="$PWD/third_party/EnzGFM" \
MODEL_LOCATION="$PWD/checkpoints/EnzGFM-650M" \
bash scripts/run_extract_enzgfm_residue_embeddings_4gpu.sh
```

The extractor removes ESM special tokens and writes only residue states. Its
default 1,022-residue truncation and `ends_center` policy match the current F3
ProtT5 pipeline. The HDF5 records model, tokenizer, and source-code provenance.

For a small CPU/debug run, call the Python adapter directly with
`--disable-fast-kernels --dtype float32 --device cpu`.

## Train the hybrid F3 model

```bash
python scripts/train_protein_pooling.py \
  --config configs/horizyn_f3_enzgfm650m_prott5_sleec.yaml
```

The supplied configuration expects 2,048-dimensional EnzGFM-650M residues and
1,024-dimensional ProtT5 scorer residues. If the downloaded checkpoint reports
a different EnzGFM hidden size, change `data.residue_dim`; extraction records
the authoritative dimension in the HDF5 metadata and training fails early on
a mismatch.

This is a backbone comparison, so reaction features, block dimensions, block
weights, loss, candidate pool, and training schedule remain unchanged from the
tracked F3 paper configuration.
