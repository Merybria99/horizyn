#!/usr/bin/env python3
"""Validate every vector in a residue HDF5 store and write a bound certificate."""

from __future__ import annotations

import argparse

from horizyn.datasets.residue_hdf5 import validate_residue_hdf5_finite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", help="Residue HDF5 store to validate")
    parser.add_argument("--certificate", help="Output sidecar; defaults beside the resolved HDF5")
    parser.add_argument("--rows-per-chunk", type=int, default=8192)
    parser.add_argument("--progress-every-chunks", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = validate_residue_hdf5_finite(
        args.h5,
        certificate_path=args.certificate,
        rows_per_chunk=args.rows_per_chunk,
        progress_every_chunks=args.progress_every_chunks,
        workers=args.workers,
    )
    print(f"Wrote residue finite-value certificate: {output}", flush=True)


if __name__ == "__main__":
    main()
