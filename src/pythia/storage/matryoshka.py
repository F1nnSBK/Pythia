"""
Matryoshka representation and multi-tier vector quantization helpers for PithosDB.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class MatryoshkaProjector:
    """
    Handles normalization, truncation, and verification for Matryoshka-structured embeddings.
    """

    def __init__(self, tiers: Sequence[int] = (64, 128, 256, 384)) -> None:
        self.tiers = sorted(list(tiers))
        self.max_dim = self.tiers[-1]

    def validate_dimension(self, dim: int) -> None:
        """Ensure input embeddings match the maximum configured tier dimension."""
        if dim != self.max_dim:
            raise ValueError(
                f"Embedding dimension {dim} does not match maximum tier dimension {self.max_dim}"
            )

    def extract_tier(
        self,
        embeddings: np.ndarray | torch.Tensor,
        tier_dim: int,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract a truncated sub-vector corresponding to a Matryoshka tier.
        """
        if tier_dim not in self.tiers:
            raise ValueError(f"Tier {tier_dim} not in configured tiers: {self.tiers}")

        if isinstance(embeddings, torch.Tensor):
            arr = embeddings.detach().cpu().numpy()
        else:
            arr = np.asarray(embeddings)

        sub = arr[..., :tier_dim].astype(np.float32)
        if normalize:
            norms = np.linalg.norm(sub, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            sub = sub / norms

        return sub

    @staticmethod
    def normalize_l2(records: np.ndarray) -> np.ndarray:
        """L2 unit-normalize continuous float embeddings."""
        records = records.astype(np.float32, copy=False)
        norms = np.linalg.norm(records, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return records / norms
