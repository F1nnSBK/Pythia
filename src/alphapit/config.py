"""
Configuration module for the AlphaPit library.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_default_storage_path() -> Path:
    """
    Returns the preferred NVMe SSD mount path if available,
    otherwise falls back to a local storage directory.
    """
    ssd_path = Path("/Volumes/AlphaPitData")
    if ssd_path.exists() and os.access(ssd_path, os.W_OK):
        return ssd_path
    fallback = Path("./data")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


class AlphaPitSettings(BaseSettings):
    """
    Central settings for streaming download, geometry processing,
    and PithosDB vector index storage.
    """

    model_config = SettingsConfigDict(
        env_prefix="ALPHAPIT_",
        arbitrary_types_allowed=True,
    )

    # Storage Paths
    storage_root: Path = Field(
        default_factory=get_default_storage_path,
        description="Root directory for storing raw cache, temp files, and compiled indices.",
    )
    raw_cache_dir: Path = Field(
        default=Path("raw_cache"),
        description="Subdirectory for raw streaming cache.",
    )
    index_dir: Path = Field(
        default=Path("pithos_indices"),
        description="Subdirectory for compiled PithosDB indices.",
    )
    temp_dir: Path = Field(
        default=Path("tmp"),
        description="Subdirectory for temporary computation files on SSD.",
    )

    # Remote Data Sources
    rcsb_base_url: str = "https://files.rcsb.org/download"
    alphafold_base_url: str = "https://alphafold.ebi.ac.uk/files"

    # Power & Thermal Safety Profile
    # Options: 'cool_quiet' (2 workers, high pacing), 'balanced' (4 workers, gentle pacing), 'turbo' (8 workers)
    power_profile: str = "balanced"
    max_workers: int = 4
    throttle_sleep_ms: float = 15.0  # gentle inter-item cooldown
    gc_interval_structures: int = 25  # force memory reclamation every N structures
    shard_size_structures: int = 500  # write to disk shard and free RAM every N structures

    # dMaSIF Molecular Surface & Geometry Parameters
    surface_resolution: float = 1.0  # Angstroms grid resolution
    distance_threshold: float = 1.05  # Surface iso-level distance
    smooth_variance: float = 0.1  # Gaussian smoothing factor
    sup_sampling: int = 20  # Surface super-sampling multiplier
    curvature_scales: List[float] = Field(
        default_factory=lambda: [1.0, 2.0, 3.0, 5.0, 10.0]
    )
    surface_sample_radius: float = 9.0  # Patch radius in Angstroms
    atom_types: List[str] = Field(
        default_factory=lambda: ["C", "H", "O", "N", "S", "P", "OTHER"]
    )

    # PithosDB Index Parameters
    vector_dimension: int = 384
    matryoshka_tiers: List[int] = Field(
        default_factory=lambda: [64, 128, 256, 384]
    )
    index_metric: str = "cosine"

    @property
    def full_raw_cache_path(self) -> Path:
        path = self.storage_root / self.raw_cache_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def full_index_path(self) -> Path:
        path = self.storage_root / self.index_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def full_temp_path(self) -> Path:
        path = self.storage_root / self.temp_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def configure_environment(self) -> None:
        """Point PyTorch, temp storage, and cache to the fast NVMe SSD."""
        temp_dir = str(self.full_temp_path)
        os.environ["TMPDIR"] = temp_dir
        os.environ["TORCH_HOME"] = str(self.storage_root / "cache" / "torch")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


settings = AlphaPitSettings()
settings.configure_environment()
