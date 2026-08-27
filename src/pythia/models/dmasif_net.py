"""
dMaSIF geometric neural network architecture for end-to-end protein surface representation,
interaction site prediction, and Matryoshka vector index embedding generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from pythia.geometry.features import SurfaceFeatureExtractor
from pythia.geometry.surface import MolecularSurface
from pythia.models.conv import QuasiGeodesicConv


@dataclass
class dMaSIFOutput:
    """Output predictions and embeddings from dMaSIFNet."""
    site_probabilities: torch.Tensor  # Shape: (P, 1), binding site probability in [0, 1]
    patch_embeddings: torch.Tensor  # Shape: (P, D_emb), normalized embedding vectors
    surface_points: torch.Tensor  # Shape: (P, 3)
    res_indices: torch.Tensor  # Shape: (P,)


class dMaSIFNet(nn.Module):
    """
    End-to-end geometric deep learning model operating directly on raw molecular surfaces.
    Processes dual-fingerprints: chemical potentials, multi-scale extrinsic curvatures,
    and intrinsic Laplace-Beltrami spectral features.
    """

    def __init__(
        self,
        in_channels: int = 21,  # 7 chemical + 10 multi-scale curvature + 4 Laplace-Beltrami HKS
        hidden_dim: int = 64,
        embedding_dim: int = 384,
        num_layers: int = 3,
        radius: float = 9.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Input projection MLP
        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
        )

        # Geometric surface convolution layers with residual connections
        self.conv_layers = nn.ModuleList([
            QuasiGeodesicConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                radius=radius,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Head 1: Interaction Site Predictor (binding hotspot probability)
        self.site_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Head 2: Matryoshka Vector Index Embedding Head (projects to 384-D normalized vector)
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim * 2, embedding_dim),
        )

    def forward(self, surface: MolecularSurface) -> dMaSIFOutput:
        """
        Forward pass from molecular surface features to predictions and embeddings.
        """
        # 1. Assemble unified input features: (P, in_channels)
        x = SurfaceFeatureExtractor.get_combined_features(surface)
        points = surface.points
        normals = surface.normals

        # 2. Input projection
        h = self.input_mlp(x)

        # 3. Geometric convolutions with residual connections
        for conv in self.conv_layers:
            h_conv = conv(h, points, normals)
            h = h + h_conv

        # 4. Heads
        site_probs = self.site_head(h)  # (P, 1)

        raw_emb = self.embedding_head(h)  # (P, embedding_dim)
        patch_emb = F.normalize(raw_emb, p=2, dim=-1)  # L2 unit normalized

        return dMaSIFOutput(
            site_probabilities=site_probs,
            patch_embeddings=patch_emb,
            surface_points=points,
            res_indices=surface.res_indices,
        )
