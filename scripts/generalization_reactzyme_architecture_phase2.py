"""Compatibility imports; maintained code lives in horizyn.pipelines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from horizyn.pipelines.checkpoints import load_head
from horizyn.pipelines.scoring import canonical_dot
