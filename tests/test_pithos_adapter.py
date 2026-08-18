"""
Unit and integration tests for PithosDB storage adapter, Matryoshka projector, and search.
"""

import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from alphapit.storage.adapter import (
    PithosStorageAdapter,
    SurfaceVectorRecord,
)
from alphapit.storage.matryoshka import MatryoshkaProjector


@pytest.fixture
def temp_storage_dir():
    temp_dir = tempfile.mkdtemp(prefix="alphapit_test_pithos_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_matryoshka_projector():
    projector = MatryoshkaProjector(tiers=[64, 128, 256, 384])
    vectors = np.random.randn(10, 384).astype(np.float32)

    tier64 = projector.extract_tier(vectors, 64, normalize=True)
    assert tier64.shape == (10, 64)
    norms = np.linalg.norm(tier64, axis=-1)
    assert np.allclose(norms, np.ones(10), atol=1e-3)

    tier128 = projector.extract_tier(vectors, 128, normalize=True)
    assert tier128.shape == (10, 128)

    with pytest.raises(ValueError):
        projector.extract_tier(vectors, 50)


def test_pithos_adapter_compile_and_search(temp_storage_dir):
    adapter = PithosStorageAdapter(base_storage_dir=temp_storage_dir, tiers=[64, 128, 256, 384])

    num_records = 50
    dim = 384
    records = np.random.randn(num_records, dim).astype(np.float32)

    metadata = [
        SurfaceVectorRecord(
            record_id=i,
            structure_id="1A8O",
            chain_id="A",
            res_seq=i + 1,
            atom_idx=i * 2,
            point_coords=[float(i), float(i * 0.5), float(-i)],
        )
        for i in range(num_records)
    ]

    index_name = "test_protein_patches"
    idx_path = adapter.compile_index(index_name, records, metadata=metadata)
    assert idx_path.exists()

    with adapter:
        # Search using the first record as query
        query = records[0:1]
        results = adapter.search(index_name, query, k=3)

        assert len(results) == 1
        matches = results[0]
        assert len(matches) == 3
        # First match should be the exact record (id=0) with lowest Hamming / distance score
        assert matches[0].record_id == 0
        assert matches[0].structure_id == "1A8O"
        assert matches[0].res_seq == 1
        assert matches[0].chain_id == "A"
