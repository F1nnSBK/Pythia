"""
AlphaPit: Structural Surface Fingerprinting and Vector Indexing for AlphaFold & PDB structures.
"""

import os
import torch

# Prevent PyTorch / OpenMP thread starvation on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from alphapit.config import settings
from alphapit.pipeline import AlphaPitPipeline

__all__ = ["AlphaPitPipeline", "settings"]
