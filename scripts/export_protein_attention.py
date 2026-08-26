#!/usr/bin/env python3
"""
Export learned protein attention weights for selected protein IDs.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.utils import residue_collate_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export residue attention weights from a ProteinPooledDualModel checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Protein pooling checkpoint")
    parser.add_argument("--config", required=True, help="Training config")
    parser.add_argument("--protein-id", action="append", default=[], help="Protein ID to export")
    parser.add_argument("--protein-id-file", default=None, help="File with one protein ID per line")
    parser.add_argument("--top-n", type=int, default=25, help="Number of top residues to keep")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    args = parser.parse_args()

    protein_ids = list(args.protein_id)
    if args.protein_id_file is not None:
        with open(args.protein_id_file) as handle:
            protein_ids.extend(line.strip() for line in handle if line.strip())
    if not protein_ids:
        raise ValueError("Provide at least one --protein-id or --protein-id-file")

    config = load_config(args.config)
    module = ProteinPooledLitModule.load_from_checkpoint(args.checkpoint, map_location=args.device)
    module.eval()
    module.to(args.device)

    if module.model.pooling_name != "attention":
        raise ValueError("Checkpoint does not use learned attention pooling")

    residue_dataset = ResidueEmbedDataset(
        file_path=config.data.protein_residue_embeds_path,
        in_memory=False,
        max_tokens=config.data.get("max_protein_tokens", 1024),
        truncation=config.data.get("protein_truncation", "ends_center"),
    )

    samples = []
    for protein_id in protein_ids:
        sample = residue_dataset[protein_id]
        sample["target_id"] = protein_id
        samples.append(sample)

    batch = residue_collate_fn(samples)
    residues = batch["residue_embeddings"].to(args.device)
    padding_mask = batch["residue_padding_mask"].to(args.device)

    with torch.no_grad():
        _target_embeds, attention = module.model.encode_targets(
            residues,
            residue_padding_mask=padding_mask,
            return_attention=True,
        )

    output_records = []
    for row_idx, protein_id in enumerate(protein_ids):
        valid_len = int((~padding_mask[row_idx]).sum().item())
        weights = attention[row_idx, :valid_len].detach().cpu()
        top_n = min(args.top_n, valid_len)
        values, indices = torch.topk(weights, k=top_n)
        output_records.append(
            {
                "protein_id": protein_id,
                "length": valid_len,
                "top_residues": [
                    {
                        "position": int(index.item()),
                        "attention_weight": float(value.item()),
                    }
                    for index, value in zip(indices, values)
                ],
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(output_records, handle, indent=2)

    print(f"Wrote attention weights for {len(output_records)} proteins to {output_path}")


if __name__ == "__main__":
    main()
