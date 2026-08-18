"""
Atomic point cloud representation and tensor containers for proteins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch

from alphapit.download.parser import ProteinStructureData, VDW_RADII_PM


@dataclass
class ProteinPointCloud:
    """
    Vectorized atom point cloud containing 3D coordinates, chemical features,
    van der Waals radii, and residue indices.
    """
    structure_id: str
    coords: torch.Tensor  # Shape: (N, 3), float32
    atom_types: torch.Tensor  # Shape: (N,), int64
    radii: torch.Tensor  # Shape: (N,), float32 (in Angstroms)
    chem_features: torch.Tensor  # Shape: (N, D_chem), float32 (one-hot or continuous)
    res_indices: torch.Tensor  # Shape: (N,), int64
    b_factors: torch.Tensor  # Shape: (N,), float32
    chain_ids: List[str]
    res_names: List[str]
    atom_names: List[str]

    @property
    def num_atoms(self) -> int:
        return self.coords.shape[0]

    @property
    def center_of_mass(self) -> torch.Tensor:
        return torch.mean(self.coords, dim=0)

    def centered(self) -> ProteinPointCloud:
        """Return a copy centered at the origin."""
        com = self.center_of_mass
        return ProteinPointCloud(
            structure_id=self.structure_id,
            coords=self.coords - com,
            atom_types=self.atom_types.clone(),
            radii=self.radii.clone(),
            chem_features=self.chem_features.clone(),
            res_indices=self.res_indices.clone(),
            b_factors=self.b_factors.clone(),
            chain_ids=list(self.chain_ids),
            res_names=list(self.res_names),
            atom_names=list(self.atom_names),
        )

    def to(self, device: torch.device | str) -> ProteinPointCloud:
        """Move tensor fields to target device (CPU, MPS, CUDA)."""
        return ProteinPointCloud(
            structure_id=self.structure_id,
            coords=self.coords.to(device),
            atom_types=self.atom_types.to(device),
            radii=self.radii.to(device),
            chem_features=self.chem_features.to(device),
            res_indices=self.res_indices.to(device),
            b_factors=self.b_factors.to(device),
            chain_ids=self.chain_ids,
            res_names=self.res_names,
            atom_names=self.atom_names,
        )

    @classmethod
    def from_structure_data(
        cls,
        data: ProteinStructureData,
        num_classes: int = 7,
        device: torch.device | str = "cpu",
    ) -> ProteinPointCloud:
        """
        Build a ProteinPointCloud from parsed ProteinStructureData.
        Converts radii from picometers to Angstroms (1 Å = 100 pm).
        """
        coords_t = torch.from_numpy(data.coords).float().to(device)
        types_t = torch.from_numpy(data.element_indices).long().to(device)
        radii_t = (torch.from_numpy(data.radii_pm).float() / 100.0).to(device)  # to Angstroms
        b_factors_t = torch.from_numpy(data.b_factors).float().to(device)
        res_indices_t = torch.from_numpy(data.res_indices).long().to(device)

        # One-hot chemical feature matrix
        chem_features = torch.zeros((len(data.coords), num_classes), dtype=torch.float32, device=device)
        valid_indices = torch.clamp(types_t, 0, num_classes - 1)
        chem_features.scatter_(1, valid_indices.unsqueeze(1), 1.0)

        return cls(
            structure_id=data.structure_id,
            coords=coords_t,
            atom_types=types_t,
            radii=radii_t,
            chem_features=chem_features,
            res_indices=res_indices_t,
            b_factors=b_factors_t,
            chain_ids=data.chain_ids,
            res_names=data.res_names,
            atom_names=data.atom_names,
        )
