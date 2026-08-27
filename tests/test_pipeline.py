"""
End-to-end integration tests for PythiaPipeline.
"""

import shutil
import tempfile
from pathlib import Path
import pytest

from pythia.pipeline import PythiaPipeline
from pythia.storage.adapter import PithosStorageAdapter


@pytest.fixture
def temp_pipeline_storage():
    temp_dir = tempfile.mkdtemp(prefix="pythia_test_pipeline_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_end_to_end_pipeline(temp_pipeline_storage):
    adapter = PithosStorageAdapter(base_storage_dir=temp_pipeline_storage)
    pipeline = PythiaPipeline(storage_adapter=adapter)

    # 1. Stream and process a live PDB structure
    pdb_id = "1a8o"
    surface, output = await pipeline.stream_and_process_structure(pdb_id)

    assert surface.num_points > 50
    assert output.site_probabilities.shape == (surface.num_points, 1)
    assert output.patch_embeddings.shape == (surface.num_points, 384)

    # 2. Index the structure into PithosDB
    index_name = "test_end_to_end"
    index_path = await pipeline.index_structures(
        index_name=index_name,
        structure_ids=[pdb_id],
        show_progress=False,
    )
    assert index_path.exists()

    # 3. Search with self-query
    results = pipeline.search_similar_patches(index_name=index_name, query_surface_output=output, k=3)
    assert len(results) == surface.num_points
    # Top match for first query patch should be mapped to the structure
    assert results[0][0].structure_id == pdb_id.upper()
