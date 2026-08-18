"""
Unit tests for atomic point clouds, surface generation, and curvature estimation.
"""

import pytest
import torch
import numpy as np

from alphapit.download.parser import ProteinStructureData
from alphapit.geometry.pointcloud import ProteinPointCloud
from alphapit.geometry.surface import MolecularSurfaceGenerator
from alphapit.geometry.curvature import MultiScaleCurvatureEstimator
from alphapit.geometry.features import SurfaceFeatureExtractor


@pytest.fixture
def sample_structure_data() -> ProteinStructureData:
    # A small synthetic tetrahedral protein atom cluster
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 1.5],
        [1.5, 1.5, 1.5],
    ], dtype=np.float32)

    radii = np.array([170.0, 155.0, 152.0, 110.0, 180.0], dtype=np.float32)  # pm
    element_indices = np.array([0, 3, 2, 1, 4], dtype=np.int64)

    return ProteinStructureData(
        structure_id="TEST",
        coords=coords,
        elements=["C", "N", "O", "H", "S"],
        element_indices=element_indices,
        radii_pm=radii,
        res_names=["ALA", "ALA", "ALA", "ALA", "GLY"],
        res_indices=np.array([1, 1, 1, 1, 2], dtype=np.int64),
        chain_ids=["A", "A", "A", "A", "A"],
        atom_names=["CA", "N", "O", "H", "SG"],
        b_factors=np.zeros(5, dtype=np.float32),
    )


def test_point_cloud_creation(sample_structure_data):
    pc = ProteinPointCloud.from_structure_data(sample_structure_data)
    assert pc.num_atoms == 5
    assert pc.coords.shape == (5, 3)
    assert pc.radii.shape == (5,)
    assert pc.radii[0] == pytest.approx(1.70, rel=1e-3)  # 170 pm -> 1.70 A
    assert pc.chem_features.shape == (5, 7)
    assert pc.chem_features[0, 0] == 1.0  # C one-hot


def test_molecular_surface_generator(sample_structure_data):
    pc = ProteinPointCloud.from_structure_data(sample_structure_data)
    gen = MolecularSurfaceGenerator(
        resolution=1.0,
        distance=1.05,
        smooth_sigma=0.5,
        sup_sampling=12,
        projection_steps=2,
    )
    surface = gen.generate_surface(pc)

    assert surface.num_points > 0
    assert surface.points.shape[1] == 3
    assert surface.normals.shape == surface.points.shape

    # Normal vectors should have unit length (magnitude ~ 1.0)
    normal_lengths = torch.norm(surface.normals, dim=-1)
    assert torch.allclose(normal_lengths, torch.ones_like(normal_lengths), atol=1e-2)

    # Check nearest atom mapping
    assert surface.atom_indices.shape == (surface.num_points,)
    assert surface.chemical_features.shape == (surface.num_points, 7)


def test_curvature_estimator(sample_structure_data):
    pc = ProteinPointCloud.from_structure_data(sample_structure_data)
    gen = MolecularSurfaceGenerator(sup_sampling=12)
    surface = gen.generate_surface(pc)

    estimator = MultiScaleCurvatureEstimator(scales=[1.5, 3.0])
    curv_features = estimator.compute_features(surface)

    # 2 scales * 2 (mean + gauss) = 4 features
    assert curv_features.shape == (surface.num_points, 4)
    assert not torch.isnan(curv_features).any()


def test_surface_feature_extractor(sample_structure_data):
    pc = ProteinPointCloud.from_structure_data(sample_structure_data)
    extractor = SurfaceFeatureExtractor()
    surface = extractor.extract(pc)

    combined = SurfaceFeatureExtractor.get_combined_features(surface)
    # 7 chemical + (5 scales * 2) = 17 features
    assert combined.shape == (surface.num_points, 17)
