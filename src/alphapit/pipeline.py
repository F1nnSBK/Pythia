"""
End-to-end AlphaPit pipeline orchestrating streaming download, geometry generation,
dMaSIF neural inference, and PithosDB vector index compilation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set
import numpy as np
import torch
from tqdm import tqdm

from alphapit.config import settings
from alphapit.download.client import PDBStreamDownloader
from alphapit.download.parser import ProteinStructureData, StreamingPDBParser
from alphapit.geometry.features import SurfaceFeatureExtractor
from alphapit.geometry.pointcloud import ProteinPointCloud
from alphapit.geometry.surface import MolecularSurface
from alphapit.models.dmasif_net import dMaSIFNet, dMaSIFOutput
from alphapit.storage.adapter import (
    PithosStorageAdapter,
    SurfaceQueryResult,
    SurfaceVectorRecord,
)


class AlphaPitPipeline:
    """
    Unified high-level pipeline for streaming protein structures to Pithos vector database.
    """

    def __init__(
        self,
        downloader: Optional[PDBStreamDownloader] = None,
        feature_extractor: Optional[SurfaceFeatureExtractor] = None,
        model: Optional[dMaSIFNet] = None,
        storage_adapter: Optional[PithosStorageAdapter] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.downloader = downloader or PDBStreamDownloader()
        self.feature_extractor = feature_extractor or SurfaceFeatureExtractor()
        self.model = (model or dMaSIFNet()).to(self.device)
        self.model.eval()
        self.storage = storage_adapter or PithosStorageAdapter()

    async def stream_and_process_structure(
        self,
        structure_id: str,
        is_alphafold: bool = False,
        file_format: str = "pdb",
        allowed_chains: Optional[Set[str]] = None,
    ) -> Tuple[MolecularSurface, dMaSIFOutput]:
        """
        Stream a single protein structure, generate its molecular surface,
        and run dMaSIF neural network inference.
        """
        if is_alphafold:
            struct_data = await self.downloader.stream_and_parse_alphafold(
                structure_id, allowed_chains=allowed_chains
            )
        else:
            struct_data = await self.downloader.stream_and_parse_rcsb(
                structure_id, file_format=file_format, allowed_chains=allowed_chains
            )

        # 1. Convert to atomic point cloud
        pc = ProteinPointCloud.from_structure_data(struct_data, device=self.device)

        # 2. On-the-fly molecular surface and curvature extraction
        surface = self.feature_extractor.extract(pc)

        # 3. Model inference
        with torch.no_grad():
            output = self.model(surface)

        return surface, output

    async def index_structures(
        self,
        index_name: str,
        structure_ids: Sequence[str],
        is_alphafold: bool = False,
        file_format: str = "pdb",
        show_progress: bool = True,
    ) -> Path:
        """
        Stream multiple structures, extract patch embeddings, and compile into PithosDB.
        """
        all_embeddings: List[np.ndarray] = []
        all_metadata: List[SurfaceVectorRecord] = []
        global_rec_id = 0

        pbar = tqdm(structure_ids, desc="Streaming and processing", disable=not show_progress)
        for sid in pbar:
            try:
                surface, output = await self.stream_and_process_structure(
                    sid, is_alphafold=is_alphafold, file_format=file_format
                )

                embs = output.patch_embeddings.detach().cpu().numpy()  # (P, 384)
                pts = surface.points.detach().cpu().numpy()  # (P, 3)
                atoms = surface.atom_indices.detach().cpu().numpy()
                res_seqs = surface.res_indices.detach().cpu().numpy()

                p = embs.shape[0]
                for i in range(p):
                    all_metadata.append(
                        SurfaceVectorRecord(
                            record_id=global_rec_id,
                            structure_id=sid.upper(),
                            chain_id="A",
                            res_seq=int(res_seqs[i]),
                            atom_idx=int(atoms[i]),
                            point_coords=[float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])],
                        )
                    )
                    global_rec_id += 1

                all_embeddings.append(embs)
            except Exception as e:
                print(f"Warning: Failed processing structure {sid}: {e}")
                continue

        if not all_embeddings:
            raise RuntimeError("No embeddings were successfully computed from the provided structure IDs.")

        combined_records = np.vstack(all_embeddings)

        # Compile into PithosDB index
        index_path = self.storage.compile_index(
            index_name=index_name,
            embeddings=combined_records,
            metadata=all_metadata,
        )

        return index_path

    def search_similar_patches(
        self,
        index_name: str,
        query_surface_output: dMaSIFOutput,
        k: int = 5,
    ) -> List[List[SurfaceQueryResult]]:
        """
        Search for matching surface patches in PithosDB using outputs from a query protein.
        """
        queries = query_surface_output.patch_embeddings.detach().cpu().numpy()
        return self.storage.search(index_name=index_name, query_vectors=queries, k=k)
