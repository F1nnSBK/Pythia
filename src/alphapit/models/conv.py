"""
Quasi-geodesic geometric surface convolutions for dMaSIF neural architectures.
Optimized for memory efficiency with chunked receiver point processing.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuasiGeodesicConv(nn.Module):
    """
    Memory-efficient Quasi-geodesic convolution on sampled surface point clouds.
    Extracts local tangent coordinates (radial distance rho, angular orientation theta)
    and applies angular and radial Gaussian filter banks in chunked batches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius: float = 9.0,
        num_angles: int = 8,
        num_radii: int = 4,
        dropout: float = 0.0,
        chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.num_angles = num_angles
        self.num_radii = num_radii
        self.num_kernels = num_angles * num_radii
        self.chunk_size = chunk_size

        # Learnable convolution weights: (num_kernels, in_channels, out_channels)
        self.weights = nn.Parameter(
            torch.randn(self.num_kernels, in_channels, out_channels)
            / math.sqrt(in_channels * self.num_kernels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Kernel center positions
        radii_centers = torch.linspace(0.5, radius - 0.5, num_radii)
        angle_centers = torch.linspace(0.0, 2.0 * math.pi, num_angles + 1)[:-1]

        # Meshgrid of kernel centers: (num_radii, num_angles) -> (num_kernels, 2)
        grid_r, grid_theta = torch.meshgrid(radii_centers, angle_centers, indexing="ij")
        self.register_buffer("kernel_radii", grid_r.reshape(-1))
        self.register_buffer("kernel_angles", grid_theta.reshape(-1))

        self.sigma_r = radius / (2.0 * num_radii)
        self.sigma_theta = 2.0 * math.pi / (2.0 * num_angles)

        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _compute_local_frames(self, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct orthogonal tangent basis (e1, e2) perpendicular to unit normals."""
        p = normals.shape[0]
        device = normals.device

        ref_axis = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(p, 3)
        alt_axis = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(p, 3)
        is_parallel = torch.abs(normals[:, 2]) > 0.9
        axis = torch.where(is_parallel.unsqueeze(-1), alt_axis, ref_axis)

        e1 = F.normalize(torch.cross(normals, axis, dim=-1), p=2, dim=-1)
        e2 = torch.cross(normals, e1, dim=-1)
        return e1, e2

    def forward(
        self,
        features: torch.Tensor,  # (P, in_channels)
        points: torch.Tensor,  # (P, 3)
        normals: torch.Tensor,  # (P, 3)
    ) -> torch.Tensor:
        """
        Forward convolution pass on surface points with chunked receiver evaluation.
        """
        p = features.shape[0]
        device = points.device
        e1, e2 = self._compute_local_frames(normals)

        outputs = torch.empty((p, self.out_channels), dtype=torch.float32, device=device)

        chunk_size = self.chunk_size
        for start in range(0, p, chunk_size):
            end = min(start + chunk_size, p)
            b_pts = points[start:end]  # (B, 3)
            b_nrm = normals[start:end]  # (B, 3)
            b_e1 = e1[start:end]  # (B, 3)
            b_e2 = e2[start:end]  # (B, 3)
            b_size = end - start

            # Displacement from receiver points to all points: (B, P, 3)
            dp = points.unsqueeze(0) - b_pts.unsqueeze(1)
            dist = torch.norm(dp, dim=-1)  # (B, P)
            mask = (dist <= self.radius) & (dist > 1e-4)  # (B, P)

            # Tangent projection: v = dp - (dp . n_i) * n_i
            dp_dot_n = torch.sum(dp * b_nrm.unsqueeze(1), dim=-1, keepdim=True)  # (B, P, 1)
            v = dp - dp_dot_n * b_nrm.unsqueeze(1)  # (B, P, 3)

            # Polar coordinates in local tangent frame
            x_proj = torch.sum(v * b_e1.unsqueeze(1), dim=-1)  # (B, P)
            y_proj = torch.sum(v * b_e2.unsqueeze(1), dim=-1)  # (B, P)
            theta = torch.atan2(y_proj, x_proj)
            theta = torch.remainder(theta, 2.0 * math.pi)  # (B, P)

            # Kernel evaluation: (B, P, K)
            d_r = dist.unsqueeze(-1) - self.kernel_radii.view(1, 1, -1)  # (B, P, K)
            d_theta = torch.remainder(
                theta.unsqueeze(-1) - self.kernel_angles.view(1, 1, -1) + math.pi, 2.0 * math.pi
            ) - math.pi

            gauss_r = torch.exp(- (d_r ** 2) / (2.0 * self.sigma_r ** 2))
            gauss_theta = torch.exp(- (d_theta ** 2) / (2.0 * self.sigma_theta ** 2))
            kernel_resp = (gauss_r * gauss_theta) * mask.unsqueeze(-1).float()  # (B, P, K)

            # Normalize weights per receiver point
            k_sums = torch.sum(kernel_resp, dim=1, keepdim=True).clamp(min=1e-5)  # (B, 1, K)
            kernel_weights = kernel_resp / k_sums  # (B, P, K)

            # Aggregate features for each kernel:
            # kernel_weights: (B, P, K) -> (K, B, P)
            # features: (P, in_channels)
            kw = kernel_weights.permute(2, 0, 1)  # (K, B, P)
            agg = torch.matmul(kw, features)  # (K, B, in_channels)

            # Apply weights: (K, B, in_channels) x (K, in_channels, out_channels) -> (K, B, out_channels)
            transformed = torch.bmm(agg, self.weights)  # (K, B, out_channels)

            # Sum across kernels
            out_chunk = torch.sum(transformed, dim=0) + self.bias  # (B, out_channels)
            outputs[start:end] = out_chunk

        outputs = self.norm(outputs)
        outputs = F.leaky_relu(outputs, negative_slope=0.1)
        return self.dropout(outputs)
