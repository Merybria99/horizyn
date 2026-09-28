#!/usr/bin/env python3
"""Prepare frozen representative-own, both-entity-held-out CIRCE-v2 data.

This command performs preparation when explicitly invoked; it is not a trainer.
See --help for precomputed-cluster fixtures and resumable production controls.
"""
from pathlib import Path
import importlib.util
import sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "horizyn/datasets/horizyn1_training.py"
SPEC = importlib.util.spec_from_file_location("horizyn1_training_preparation", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load preparation implementation: {MODULE_PATH}")
implementation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = implementation
SPEC.loader.exec_module(implementation)
main = implementation.main
make_parser = implementation.make_parser
prepare = implementation.prepare


if __name__ == "__main__":
    main()
