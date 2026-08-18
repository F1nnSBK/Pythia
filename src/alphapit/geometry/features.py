"""
Surface feature extraction and tensor assembly for dMaSIF neural networks.
Combines multi-scale geometric curvature with chemical atom-type potentials.
"""

from __future__ import annotations

from typing import Optional
import torch

from alphapit.config import settings
from alphapit.geometry.curvature import MultiScaleCurvatureEstimator
from alphapit.geometry.pointcloud import ProteinPointCloud
from alphapit.geometry.surface import MolecularSurface, MolecularSurfaceGenerator


class SurfaceFeatureExtractor:
    """
    Extracts unified feature vectors for each surface point:
    [Chemical_Potentials (D_chem), MultiScale_Curvatures (2 * D_scales)].
    """

    def __init__(
        self,
        surface_generator: Optional[MolecularSurfaceGenerator] = None,
        curvature_estimator: Optional[MultiScaleCurvatureEstimator] = None,
    ) -> None:
        self.surface_generator = surface_generator or MolecularSurfaceGenerator(
            resolution=settings.surface_resolution,
            distance=settings.distance_threshold,
            smooth_sigma=settings.smooth_variance,
            sup_sampling=settings.sup_sampling,
        )
        self.curvature_estimator = curvature_estimator or MultiScaleCurvatureEstimator(
            scales=settings.curvature_scales
        )

    def extract(self, point_cloud: ProteinPointCloud) -> MolecularSurface:
        """
        Full pipeline: generates surface points, normals, and computes
        chemical and geometric curvature descriptors.
        """
        # 1. Generate surface points and chemical features
        surface = self.surface_generator.generate_surface(point_cloud)

        # 2. Compute multi-scale extrinsic curvature
        curv_features = self.curvature_estimator.compute_features(surface)
        surface.curvature_features = curv_features

        return surface

    @staticmethod
    def get_combined_features(surface: MolecularSurface) -> torch.Tensor:
        """
        Concatenate chemical and curvature features into a single tensor of shape (P, D_in).
        """
        if surface.curvature_features is not None:
            return torch.cat([surface.chemical_features, surface.curvature_features], dim=-1)
        return surface.chemical_features
