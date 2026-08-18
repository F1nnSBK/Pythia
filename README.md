# AlphaPit

High-performance library for streaming protein structural bioinformatics, on-the-fly dMaSIF molecular surface generation, and model-isomorphic vector indexing with PithosDB.

---

## Architecture Overview

AlphaPit is designed for data-intensive protein structural pipelines where memory and local disk space on laptops are constrained. All heavy storage (raw PDB/AlphaFold cache, temporary memory-mapped buffers, and compiled vector indices) streams directly to external NVMe SSD storage (`/Volumes/AlphaPitData`).

```
AlphaPit/
├── .gitignore
├── pyproject.toml
├── README.md
├── src/
│   └── alphapit/
│       ├── config.py                 # NVMe SSD storage paths and hyperparameters
│       ├── download/                 # Streaming Download & Decompression
│       │   ├── client.py             # Async client for RCSB PDB & AlphaFold DB
│       │   ├── stream.py             # Memory-efficient gzip decompressor & line iterators
│       │   └── parser.py             # Fast PDB/mmCIF atomic coordinate & radius parser
│       ├── geometry/                 # dMaSIF On-The-Fly Molecular Surface Engine
│       │   ├── pointcloud.py         # Atomic point cloud & chemical one-hot encodings
│       │   ├── surface.py            # Differentiable level-set surface generator
│       │   ├── curvature.py          # Multi-scale extrinsic curvature estimator
│       │   └── features.py           # Unified geometric & chemical feature assembler
│       ├── storage/                  # PithosDB Vector Database Integration
│       │   ├── adapter.py            # Zero-copy off-heap memory-mapped index adapter
│       │   └── matryoshka.py         # Multi-tier Matryoshka dimension splitter
│       ├── models/                   # dMaSIF Neural Architecture
│       │   ├── conv.py               # Quasi-geodesic surface convolution layers
│       │   └── dmasif_net.py         # End-to-end site predictor & embedding generator
│       ├── pipeline.py               # Unified streaming-to-index orchestrator
│       └── cli.py                    # Command-line interface
└── tests/                            # Comprehensive unit & integration tests
```

---

## Key Features

1. **True Streaming Download**:
   - Streams `.pdb.gz` and `.cif.gz` directly from RCSB and AlphaFold DB.
   - Decompresses and parses records chunk-by-chunk in RAM without writing temporary uncompressed files to the internal disk.
   - Saves persistent caches and compiled indices directly onto `/Volumes/AlphaPitData`.

2. **On-the-Fly dMaSIF Molecular Surface**:
   - Computes smooth molecular surfaces directly from raw 3D atomic coordinates without precomputing heavy MSMS triangle meshes.
   - Computes multi-scale extrinsic curvatures (Mean Curvature, Gaussian Curvature) across scales (1.0, 2.0, 3.0, 5.0, 10.0 Angstroms).
   - Generates chemical features directly from atom type potentials.

3. **PithosDB Vector Indexing**:
   - Compiles Matryoshka-structured binary embeddings (64, 128, 256, 384 dimensions) into columnar off-heap binary format.
   - Fast zero-copy similarity and k-NN search across planetary-scale protein surface patches.

---

## Quickstart

### 1. Environment Setup

```bash
# Initialize virtual environment with uv
uv venv .venv
source .venv/bin/activate

# Install AlphaPit in editable mode
uv pip install -e .
```

### 2. Check System & NVMe SSD Status

```bash
alphapit status
```

### 3. Stream & Index Protein Structures

```bash
# Stream 1A8O and 6M0J directly into a PithosDB index on the SSD
alphapit index protein_index 1a8o 6m0j
```

### 4. Search Similar Surface Patches

```bash
alphapit search protein_index 1a8o --top-k 5
```

---

## Python API Example

```python
import asyncio
from alphapit import AlphaPitPipeline, PDBStreamDownloader

async def main():
    pipeline = AlphaPitPipeline()

    # Stream, compute surface, and run dMaSIF inference
    surface, output = await pipeline.stream_and_process_structure("1a8o")
    print(f"Generated {surface.num_points} surface points")
    print(f"Embedding shape: {output.patch_embeddings.shape}")

    # Index multiple proteins into PithosDB on SSD
    index_path = await pipeline.index_structures(
        index_name="my_index",
        structure_ids=["1a8o", "6m0j"]
    )
    print(f"Index compiled at: {index_path}")

asyncio.run(main())
```
