#!/usr/bin/env python3
"""Train supported residue-level pipelines (V4 by default).

Usage: python train.py --config configs/pipelines/v4/reactzyme_reaction_smi.yaml
"""
from horizyn.pipelines.training import main

if __name__ == "__main__":
    main()
