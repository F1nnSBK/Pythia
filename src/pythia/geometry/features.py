"""
Surface feature extraction and tensor assembly for dMaSIF neural networks.
Combines multi-scale geometric curvature, chemical atom-type potentials,
and intrinsic Laplace-Beltrami Heat Kernel Signatures (HKS).
"""

from __future__ import annotations

from typing import Optional
import torch

from pythia.config import settings
from pythia.geometry.curvature import MultiScaleCurvatureEstimator
from pythia.geometry.laplacian import LaplaceBeltramiEstimator
from pythia.geometry.pointcloud import ProteinPointCloud
from pythia.geometry.surface import MolecularSurface, MolecularSurfaceGenerator


class SurfaceFeatureExtractor:
    """
    Extracts unified feature vectors for each surface point:
    [Chemical_Potentials (7), MultiScale_Curvatures (10), Intrinsic_HKS (4)] = 21 dimensions.
    """

    def __init__(
        self,
        surface_generator: Optional[MolecularSurfaceGenerator] = None,
        curvature_estimator: Optional[MultiScaleCurvatureEstimator] = None,
        laplacian_estimator: Optional[LaplaceBeltramiEstimator] = None,
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
        self.laplacian_estimator = laplacian_estimator or LaplaceBeltramiEstimator()

    def extract(self, point_cloud: ProteinPointCloud) -> MolecularSurface:
        """
        Full pipeline: generates surface points, normals, and computes
        chemical, geometric curvature, and intrinsic spectral LBO/HKS descriptors.
        """
        # 1. Generate surface points and chemical features
        surface = self.surface_generator.generate_surface(point_cloud)

        # 2. Compute multi-scale extrinsic curvature
        curv_features = self.curvature_estimator.compute_features(surface)
        surface.curvature_features = curv_features

        # 3. Compute intrinsic Laplace-Beltrami Heat Kernel Signatures (HKS)
        lbo_features = self.laplacian_estimator.extract_features(surface)
        surface.laplacian_features = lbo_features

        return surface

    @staticmethod
    def get_combined_features(surface: MolecularSurface) -> torch.Tensor:
        """
        Concatenate chemical, extrinsic curvature, and intrinsic HKS features into a single tensor.
        """
        feature_parts = [surface.chemical_features]
        if surface.curvature_features is not None:
            feature_parts.append(surface.curvature_features)
        if surface.laplacian_features is not None:
            feature_parts.append(surface.laplacian_features)

        return torch.cat(feature_parts, dim=-1)
