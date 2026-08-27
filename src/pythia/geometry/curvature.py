"""
Multi-scale surface curvature estimation for geometric deep learning.
Computes shape operators, mean curvature, Gaussian curvature, and shape index
across configurable spatial radii.
"""

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn.functional as F

from pythia.geometry.surface import MolecularSurface


class MultiScaleCurvatureEstimator:
    """
    Estimates multi-scale extrinsic curvature on sampled molecular surfaces.
    """

    def __init__(
        self,
        scales: List[float] = (1.0, 2.0, 3.0, 5.0, 10.0),
        eps: float = 1e-5,
    ) -> None:
        self.scales = list(scales)
        self.eps = eps

    def compute_curvatures_at_scale(
        self,
        points: torch.Tensor,  # (P, 3)
        normals: torch.Tensor,  # (P, 3)
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate mean curvature (H) and Gaussian curvature (K) at a given radius scale.
        """
        p = points.shape[0]
        device = points.device

        mean_curv = torch.zeros(p, dtype=torch.float32, device=device)
        gauss_curv = torch.zeros(p, dtype=torch.float32, device=device)

        # Batch vectorization over receiver points to keep memory minimal
        batch_size = 512
        for start in range(0, p, batch_size):
            end = min(start + batch_size, p)
            pts_b = points[start:end]  # (B, 3)
            nrm_b = normals[start:end]  # (B, 3)

            # Pairwise distance from batch to all points: (B, P)
            dists_b = torch.cdist(pts_b, points)
            mask_b = (dists_b <= scale) & (dists_b > 1e-4)

            # Difference vectors
            dp = points.unsqueeze(0) - pts_b.unsqueeze(1)  # (B, P, 3)
            dn = normals.unsqueeze(0) - nrm_b.unsqueeze(1)  # (B, P, 3)

            # Normal directional component: (dp . n)
            dp_dot_n = torch.sum(dp * nrm_b.unsqueeze(1), dim=-1, keepdim=True)  # (B, P, 1)
            dn_dot_n = torch.sum(dn * nrm_b.unsqueeze(1), dim=-1, keepdim=True)  # (B, P, 1)

            # Tangent projections
            v = dp - dp_dot_n * nrm_b.unsqueeze(1)  # (B, P, 3)
            w = dn - dn_dot_n * nrm_b.unsqueeze(1)  # (B, P, 3)

            # Gaussian spatial weights: w_ij = exp(-d_ij^2 / (2 * s^2))
            weights = torch.exp(- (dists_b ** 2) / (2.0 * (scale ** 2))) * mask_b.float()  # (B, P)
            w_sum = torch.sum(weights, dim=-1, keepdim=True).clamp(min=1e-5)  # (B, 1)
            weights_norm = weights / w_sum  # (B, P)

            # Covariance matrices: M_vv = sum w * (v v^T), M_wv = sum w * (w v^T)
            # v: (B, P, 3), w: (B, P, 3)
            v_weighted = v * weights_norm.unsqueeze(-1)  # (B, P, 3)
            m_vv = torch.matmul(v_weighted.transpose(1, 2), v)  # (B, 3, 3)
            m_wv = torch.matmul(v_weighted.transpose(1, 2), w)  # (B, 3, 3)

            # Fast closed-form 3x3 inversion (Cramer's rule) - 100% MPS/GPU native, zero CPU fallback
            eye = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
            a = m_vv + self.eps * eye  # (B, 3, 3)

            # Cofactor matrix for 3x3
            c00 = a[:, 1, 1] * a[:, 2, 2] - a[:, 1, 2] * a[:, 2, 1]
            c01 = -(a[:, 1, 0] * a[:, 2, 2] - a[:, 1, 2] * a[:, 2, 0])
            c02 = a[:, 1, 0] * a[:, 2, 1] - a[:, 1, 1] * a[:, 2, 0]

            c10 = -(a[:, 0, 1] * a[:, 2, 2] - a[:, 0, 2] * a[:, 2, 1])
            c11 = a[:, 0, 0] * a[:, 2, 2] - a[:, 0, 2] * a[:, 2, 0]
            c12 = -(a[:, 0, 0] * a[:, 2, 1] - a[:, 0, 1] * a[:, 2, 0])

            c20 = a[:, 0, 1] * a[:, 1, 2] - a[:, 0, 2] * a[:, 1, 1]
            c21 = -(a[:, 0, 0] * a[:, 1, 2] - a[:, 0, 2] * a[:, 1, 0])
            c22 = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]

            det = (a[:, 0, 0] * c00 + a[:, 0, 1] * c01 + a[:, 0, 2] * c02).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6)

            # Transposed cofactor matrix (adjugate)
            adj = torch.stack([
                torch.stack([c00, c10, c20], dim=-1),
                torch.stack([c01, c11, c21], dim=-1),
                torch.stack([c02, c12, c22], dim=-1)
            ], dim=-2)

            inv_vv = adj / det  # (B, 3, 3)

            # Shape operator tensor W = M_wv * inv_vv
            shape_op = torch.matmul(m_wv, inv_vv)  # (B, 3, 3)

            # Mean curvature: 0.5 * trace
            h = 0.5 * (shape_op[:, 0, 0] + shape_op[:, 1, 1] + shape_op[:, 2, 2])

            # Gaussian curvature from 2nd principal invariant
            tr = 2.0 * h
            tr2 = tr ** 2
            w_sq_tr = torch.diagonal(torch.matmul(shape_op, shape_op), dim1=-2, dim2=-1).sum(dim=-1)
            k = 0.5 * (tr2 - w_sq_tr)

            mean_curv[start:end] = torch.nan_to_num(h, nan=0.0, posinf=5.0, neginf=-5.0)
            gauss_curv[start:end] = torch.nan_to_num(k, nan=0.0, posinf=10.0, neginf=-10.0)

        return mean_curv, gauss_curv

    def compute_features(self, surface: MolecularSurface) -> torch.Tensor:
        """
        Extract multi-scale curvature tensor of shape (P, 2 * len(scales)).
        For each scale: [Mean_Curvature, Gaussian_Curvature].
        """
        features: List[torch.Tensor] = []
        for s in self.scales:
            h, k = self.compute_curvatures_at_scale(surface.points, surface.normals, scale=s)
            features.append(h.unsqueeze(-1))
            features.append(k.unsqueeze(-1))

        curv_tensor = torch.cat(features, dim=-1)  # (P, 2 * num_scales)
        # Normalize/clamp any numeric outliers
        curv_tensor = torch.nan_to_num(curv_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
        return torch.clamp(curv_tensor, min=-10.0, max=10.0)
