"""
Multi-scale surface curvature estimation for geometric deep learning.
Computes shape operators, mean curvature, Gaussian curvature, and shape index
across configurable spatial radii.
"""

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn.functional as F

from alphapit.geometry.surface import MolecularSurface


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

            # Regularized pseudo-inverse: (M_vv + eps * I)^-1
            eye = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
            m_vv_reg = m_vv + self.eps * eye
            inv_vv = torch.linalg.pinv(m_vv_reg)  # (B, 3, 3)

            # Shape operator tensor W = M_wv * inv_vv
            shape_op = torch.matmul(m_wv, inv_vv)  # (B, 3, 3)

            # Trace gives twice the mean curvature, determinant of 2D tangent space gives Gaussian curvature
            tr = torch.diagonal(shape_op, dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
            h = 0.5 * tr

            # Compute eigenvalues of symmetric part 0.5 * (W + W^T)
            shape_sym = 0.5 * (shape_op + shape_op.transpose(-1, -2))
            try:
                eigvals = torch.linalg.eigvalsh(shape_sym)  # (B, 3) sorted ascending
                # The tangent plane has 2 non-zero eigenvalues; sort by absolute magnitude
                k1 = eigvals[:, 2]
                k2 = eigvals[:, 1]
                k = k1 * k2
            except Exception:
                k = torch.zeros_like(h)

            mean_curv[start:end] = h
            gauss_curv[start:end] = k

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
