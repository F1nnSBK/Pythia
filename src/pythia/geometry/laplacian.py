"""
Discrete Laplace-Beltrami Operator (LBO) and Heat Kernel Signature (HKS)
for intrinsic spectral geometry on molecular surfaces.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from pythia.geometry.surface import MolecularSurface


class LaplaceBeltramiEstimator:
    """
    Computes discrete Laplace-Beltrami spectra and intrinsic Heat Kernel Signatures (HKS)
    on sampled molecular surface point clouds.
    """

    def __init__(
        self,
        num_eigenvalues: int = 16,
        diffusion_times: Sequence[float] = (0.1, 1.0, 5.0, 20.0),
        neighbor_radius: float = 6.0,
        sigma_lbo: float = 2.0,
    ) -> None:
        self.num_eigenvalues = num_eigenvalues
        self.diffusion_times = list(diffusion_times)
        self.neighbor_radius = neighbor_radius
        self.sigma_lbo = sigma_lbo

    def compute_hks(
        self,
        points: torch.Tensor,  # (P, 3)
        normals: Optional[torch.Tensor] = None,  # (P, 3)
    ) -> torch.Tensor:
        """
        Compute multi-scale Heat Kernel Signature of shape (P, len(diffusion_times)).
        Optimized with landmark interpolation for large point clouds to prevent O(P^3) slowdown.
        """
        p = points.shape[0]
        device = points.device

        # If too few points, return zeros
        if p < self.num_eigenvalues + 1:
            return torch.zeros((p, len(self.diffusion_times)), dtype=torch.float32, device=device)

        # For performance on very large proteins, cap maximum spectral mesh size to 1024 landmarks
        max_landmarks = 1024
        if p > max_landmarks:
            sub_idx = torch.linspace(0, p - 1, max_landmarks, dtype=torch.long, device=device)
            sub_pts = points[sub_idx]
            hks_sub = self._compute_hks_raw(sub_pts)
            # Nearest neighbor interpolation back to all P points
            dists = torch.cdist(points, sub_pts)  # (P, max_landmarks)
            nearest = torch.argmin(dists, dim=-1)  # (P,)
            return hks_sub[nearest]
        else:
            return self._compute_hks_raw(points)

    def _compute_hks_raw(self, pts: torch.Tensor) -> torch.Tensor:
        p = pts.shape[0]
        device = pts.device

        # 1. Compute pairwise distance matrix (in float32)
        dist_matrix = torch.cdist(pts, pts)  # (P, P)

        # 2. Gaussian affinity weights on local neighborhood graph
        in_range = (dist_matrix <= self.neighbor_radius)
        w = torch.exp(- (dist_matrix ** 2) / (2.0 * (self.sigma_lbo ** 2))) * in_range.float()
        w.fill_diagonal_(0.0)

        # Degree matrix D: (P,)
        degree = torch.sum(w, dim=-1).clamp(min=1e-5)
        d_inv_sqrt = torch.pow(degree, -0.5)

        # Normalized symmetric Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
        w_norm = d_inv_sqrt.unsqueeze(1) * w * d_inv_sqrt.unsqueeze(0)
        l_sym = torch.eye(p, dtype=torch.float32, device=device) - w_norm

        # 3. Spectral Decomposition
        k = min(self.num_eigenvalues, p)
        try:
            l_sym_cpu = l_sym.detach().cpu()
            eigvals, eigvecs = torch.linalg.eigh(l_sym_cpu)  # Sorted in ascending order

            evals = eigvals[:k].to(device)  # (K,)
            evecs = eigvecs[:, :k].to(device)  # (P, K)

            phi = d_inv_sqrt.unsqueeze(1) * evecs  # (P, K)
            phi_sq = phi ** 2  # (P, K)

            # 4. Multi-scale Heat Kernel Signatures
            hks_features: List[torch.Tensor] = []
            for t in self.diffusion_times:
                heat_decay = torch.exp(-torch.clamp(evals * t, max=50.0))  # (K,)
                hks_t = torch.matmul(phi_sq, heat_decay)  # (P,)

                trace = torch.sum(heat_decay).clamp(min=1e-6)
                hks_norm = hks_t / trace
                hks_features.append(hks_norm.unsqueeze(-1))

            hks_tensor = torch.cat(hks_features, dim=-1)  # (P, len(diffusion_times))
            return torch.nan_to_num(hks_tensor, nan=0.0, posinf=1.0, neginf=0.0)

        except Exception:
            return torch.zeros((p, len(self.diffusion_times)), dtype=torch.float32, device=device)

    def extract_features(self, surface: MolecularSurface) -> torch.Tensor:
        """Extract HKS tensor for a given MolecularSurface."""
        return self.compute_hks(surface.points, surface.normals)
