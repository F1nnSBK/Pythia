"""
On-the-fly molecular surface generation from atomic point clouds without precomputed triangle meshes.
Implements the smooth distance level-set formulation of dMaSIF (Sverrisson et al., 2021).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from alphapit.geometry.pointcloud import ProteinPointCloud


@dataclass
class MolecularSurface:
    """
    Sampled molecular surface representation.
    """
    structure_id: str
    points: torch.Tensor  # Shape: (P, 3), float32 (surface point coordinates)
    normals: torch.Tensor  # Shape: (P, 3), float32 (outward unit normal vectors)
    atom_indices: torch.Tensor  # Shape: (P,), int64 (nearest atom index)
    res_indices: torch.Tensor  # Shape: (P,), int64 (nearest residue index)
    chemical_features: torch.Tensor  # Shape: (P, D_chem)
    curvature_features: Optional[torch.Tensor] = None  # Shape: (P, D_curv)

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    def to(self, device: torch.device | str) -> MolecularSurface:
        """Move surface tensors to target device."""
        return MolecularSurface(
            structure_id=self.structure_id,
            points=self.points.to(device),
            normals=self.normals.to(device),
            atom_indices=self.atom_indices.to(device),
            res_indices=self.res_indices.to(device),
            chemical_features=self.chemical_features.to(device),
            curvature_features=(
                self.curvature_features.to(device)
                if self.curvature_features is not None
                else None
            ),
        )


class MolecularSurfaceGenerator:
    """
    Generates molecular surface point clouds and outward normals on-the-fly.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        distance: float = 1.05,
        smooth_sigma: float = 0.5,
        sup_sampling: int = 20,
        projection_steps: int = 3,
    ) -> None:
        self.resolution = resolution
        self.distance = distance
        self.smooth_sigma = smooth_sigma
        self.sup_sampling = sup_sampling
        self.projection_steps = projection_steps

    @staticmethod
    def sample_unit_sphere(num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Fibonacci / quasi-random uniform sampling on the unit sphere S^2.
        """
        indices = torch.arange(0, num_samples, dtype=torch.float32, device=device) + 0.5
        phi = torch.arccos(1.0 - 2.0 * indices / num_samples)
        theta = torch.pi * (1.0 + 5.0 ** 0.5) * indices
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        return torch.stack([x, y, z], dim=-1)

    def smooth_distance_and_gradient(
        self,
        query_points: torch.Tensor,  # Shape: (M, 3)
        atom_coords: torch.Tensor,  # Shape: (N, 3)
        atom_radii: torch.Tensor,  # Shape: (N,)
        chunk_size: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate smooth distance function F(x) and its analytical gradient grad F(x).
        Uses chunking to keep memory consumption minimal.
        """
        m = query_points.shape[0]
        distances = torch.empty(m, dtype=torch.float32, device=query_points.device)
        gradients = torch.empty((m, 3), dtype=torch.float32, device=query_points.device)

        sigma = self.smooth_sigma

        for start_idx in range(0, m, chunk_size):
            end_idx = min(start_idx + chunk_size, m)
            q_chunk = query_points[start_idx:end_idx]  # (B, 3)

            # Pairwise displacement: (B, 1, 3) - (1, N, 3) -> (B, N, 3)
            diff = q_chunk.unsqueeze(1) - atom_coords.unsqueeze(0)
            dist_to_atoms = torch.norm(diff, dim=-1, keepdim=False)  # (B, N)
            dist_to_atoms = torch.clamp(dist_to_atoms, min=1e-6)

            # Distance relative to atom spheres: (B, N)
            shifted_dist = (dist_to_atoms - atom_radii.unsqueeze(0)) / sigma

            # Softmin weights via softmax of negative shifted distance
            weights = F.softmax(-shifted_dist, dim=-1)  # (B, N)

            # Smooth distance: soft minimum
            logsumexp = torch.logsumexp(-shifted_dist, dim=-1)
            smooth_dist = -sigma * logsumexp

            # Analytical gradient of softmin: sum_i w_i * (q - a_i) / ||q - a_i||
            unit_dirs = diff / dist_to_atoms.unsqueeze(-1)  # (B, N, 3)
            grad = torch.sum(weights.unsqueeze(-1) * unit_dirs, dim=1)  # (B, 3)

            distances[start_idx:end_idx] = smooth_dist
            gradients[start_idx:end_idx] = grad

        return distances, gradients

    def generate_surface(
        self,
        point_cloud: ProteinPointCloud,
    ) -> MolecularSurface:
        """
        Generate on-the-fly molecular surface points and normal vectors.
        """
        device = point_cloud.coords.device
        num_atoms = point_cloud.num_atoms
        if num_atoms == 0:
            raise ValueError("Cannot generate surface from empty point cloud")

        # 1. Sample candidate points around each atom
        sphere_dirs = self.sample_unit_sphere(self.sup_sampling, device=device)  # (K, 3)
        # Expansion: (N, 1, 3) + (1, K, 3) * (N, 1, 1)
        r_expanded = (point_cloud.radii * self.distance).unsqueeze(1).unsqueeze(2)  # (N, 1, 1)
        candidates = point_cloud.coords.unsqueeze(1) + sphere_dirs.unsqueeze(0) * r_expanded  # (N, K, 3)
        candidates = candidates.reshape(-1, 3)  # (N*K, 3)

        # 2. Evaluate smooth distance and filter interior points
        dists, grads = self.smooth_distance_and_gradient(
            candidates, point_cloud.coords, point_cloud.radii
        )

        # Keep points close to the nominal zero-level envelope
        mask = (dists >= -0.5) & (dists <= 2.0)
        if not torch.any(mask):
            # Fallback if too strict
            mask = dists >= -1.0

        surface_pts = candidates[mask]
        if surface_pts.shape[0] == 0:
            surface_pts = candidates[:100]

        # 3. Refine candidate positions to zero-level set with Newton-Raphson iterations
        for _ in range(self.projection_steps):
            d, g = self.smooth_distance_and_gradient(
                surface_pts, point_cloud.coords, point_cloud.radii
            )
            g_norm_sq = torch.sum(g * g, dim=-1, keepdim=True) + 1e-8
            surface_pts = surface_pts - (d.unsqueeze(-1) / g_norm_sq) * g

        # 4. Final surface normals
        _, final_grads = self.smooth_distance_and_gradient(
            surface_pts, point_cloud.coords, point_cloud.radii
        )
        normals = F.normalize(final_grads, p=2, dim=-1)

        # 5. Spatial downsampling to target resolution
        if self.resolution > 0:
            surface_pts, normals = self._grid_subsample(surface_pts, normals, self.resolution)

        # 6. Map each surface point to its nearest atom and transfer chemical features (chunked)
        p = surface_pts.shape[0]
        n_atoms = point_cloud.coords.shape[0]
        nearest_atom_idx = torch.empty(p, dtype=torch.long, device=device)
        surface_chem = torch.empty((p, point_cloud.chem_features.shape[1]), dtype=torch.float32, device=device)

        chunk_size = 2048
        for start in range(0, p, chunk_size):
            end = min(start + chunk_size, p)
            pts_chunk = surface_pts[start:end]  # (B, 3)
            dist_chunk = torch.cdist(pts_chunk, point_cloud.coords)  # (B, N)
            nearest_atom_idx[start:end] = torch.argmin(dist_chunk, dim=-1)

            inv_dists = 1.0 / (dist_chunk + 0.5)
            weights = F.softmax(inv_dists, dim=-1)
            surface_chem[start:end] = torch.matmul(weights, point_cloud.chem_features)

        nearest_res_idx = point_cloud.res_indices[nearest_atom_idx]

        return MolecularSurface(
            structure_id=point_cloud.structure_id,
            points=surface_pts,
            normals=normals,
            atom_indices=nearest_atom_idx,
            res_indices=nearest_res_idx,
            chemical_features=surface_chem,
        )

    def _grid_subsample(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        voxel_size: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Voxel grid spatial downsampling to maintain uniform point density.
        """
        coords_min = torch.min(points, dim=0)[0]
        grid_coords = torch.floor((points - coords_min) / voxel_size).long()

        # Simple hashing for unique voxel bins
        # Multipliers based on bounding box
        grid_offset = grid_coords - torch.min(grid_coords, dim=0)[0]
        hash_keys = (
            grid_offset[:, 0]
            + grid_offset[:, 1] * 73856093
            + grid_offset[:, 2] * 19349663
        )

        unique_keys, inverse_indices = torch.unique(hash_keys, return_inverse=True)
        # Pick one representative index per unique voxel bin
        perm = torch.zeros(unique_keys.size(0), dtype=torch.long, device=points.device)
        perm.scatter_(0, inverse_indices, torch.arange(points.size(0), device=points.device))

        return points[perm], normals[perm]
