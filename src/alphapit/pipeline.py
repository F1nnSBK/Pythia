"""
End-to-end AlphaPit pipeline orchestrating concurrent streaming download, live metadata fetching,
geometry generation, dMaSIF neural inference, and PithosDB vector index compilation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import torch
from tqdm import tqdm

from alphapit.config import settings
from alphapit.download.client import AlphaFoldMetadata, PDBStreamDownloader
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


@dataclass
class PipelineMetrics:
    """Real-time performance and throughput metrics."""
    total_structures_attempted: int = 0
    total_structures_succeeded: int = 0
    total_surface_patches: int = 0
    total_network_bytes: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time > 0 else time.perf_counter()
        return max(end - self.start_time, 1e-6)

    @property
    def structures_per_second(self) -> float:
        return self.total_structures_succeeded / self.elapsed_seconds

    @property
    def patches_per_second(self) -> float:
        return self.total_surface_patches / self.elapsed_seconds

    @property
    def network_throughput_mb_s(self) -> float:
        return (self.total_network_bytes / (1024 * 1024)) / self.elapsed_seconds


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
        concurrency: int = 16,
    ) -> None:
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.downloader = downloader or PDBStreamDownloader(concurrency_limit=concurrency)
        self.feature_extractor = feature_extractor or SurfaceFeatureExtractor()
        self.model = (model or dMaSIFNet()).to(self.device)
        self.model.eval()
        self.storage = storage_adapter or PithosStorageAdapter()
        self.concurrency = concurrency

    async def stream_and_process_structure(
        self,
        structure_id: str,
        is_alphafold: bool = False,
        file_format: str = "pdb",
        min_plddt: float = 0.0,
        allowed_chains: Optional[Set[str]] = None,
    ) -> Tuple[MolecularSurface, dMaSIFOutput]:
        """
        Stream a single protein structure, generate its molecular surface,
        and run dMaSIF neural network inference.
        """
        if is_alphafold:
            struct_data = await self.downloader.stream_and_parse_alphafold(
                structure_id, allowed_chains=allowed_chains, min_plddt=min_plddt
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

    async def process_single_metadata_item(
        self,
        meta: AlphaFoldMetadata,
        min_plddt: float = 0.0,
    ) -> Optional[Tuple[np.ndarray, List[SurfaceVectorRecord]]]:
        """
        Stream and process a single AlphaFold protein from metadata with on-the-fly pLDDT filter.
        Pulls live metadata from EBI server for latest model URLs.
        """
        try:
            cif_url = meta.cif_url
            if not cif_url or "_v4.cif" in cif_url:
                live_meta = await self.downloader.fetch_alphafold_metadata(meta.uniprot_accession)
                if live_meta and live_meta.cif_url:
                    cif_url = live_meta.cif_url
                else:
                    cif_url = f"https://alphafold.ebi.ac.uk/files/AF-{meta.uniprot_accession}-F1-model_v6.cif"

            struct_data = await self.downloader.stream_and_parse_url(
                url=cif_url,
                structure_id=meta.uniprot_accession,
                file_format="cif",
                min_plddt=min_plddt,
            )
            if struct_data.num_atoms < 10:
                return None

            pc = ProteinPointCloud.from_structure_data(struct_data, device=self.device)
            surface = self.feature_extractor.extract(pc)

            with torch.no_grad():
                output = self.model(surface)

            embs = output.patch_embeddings.detach().cpu().numpy()
            pts = surface.points.detach().cpu().numpy()
            atoms = surface.atom_indices.detach().cpu().numpy()
            res_seqs = surface.res_indices.detach().cpu().numpy()

            metadata_records: List[SurfaceVectorRecord] = []
            p = embs.shape[0]
            for i in range(p):
                metadata_records.append(
                    SurfaceVectorRecord(
                        record_id=0,  # will be assigned during aggregation
                        structure_id=meta.uniprot_accession,
                        chain_id="A",
                        res_seq=int(res_seqs[i]),
                        atom_idx=int(atoms[i]),
                        point_coords=[float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])],
                    )
                )

            return embs, metadata_records
        except Exception:
            return None

    async def stream_and_index_proteome(
        self,
        index_name: str,
        organism_tax_id: str = "9606",  # Homo sapiens
        limit: int = 50,
        min_plddt: float = 70.0,
        concurrency: Optional[int] = None,
        show_progress: bool = True,
    ) -> Tuple[Path, PipelineMetrics]:
        """
        Highly concurrent proteome streaming pipeline.
        Pulls live UniProt / AlphaFold metadata and processes items through an async worker pool.
        """
        num_workers = concurrency or self.concurrency
        metrics = PipelineMetrics()
        metrics.start_time = time.perf_counter()

        queue: asyncio.Queue[Optional[AlphaFoldMetadata]] = asyncio.Queue(maxsize=num_workers * 4)
        results: List[Tuple[np.ndarray, List[SurfaceVectorRecord]]] = []

        pbar = tqdm(total=limit, desc="Streaming AlphaFold Proteome", disable=not show_progress)

        async def worker():
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break

                res = await self.process_single_metadata_item(item, min_plddt=min_plddt)
                if res is not None:
                    embs, meta_list = res
                    results.append((embs, meta_list))
                    metrics.total_structures_succeeded += 1
                    metrics.total_surface_patches += embs.shape[0]
                    # Estimate network payload size (~100KB per compressed structure)
                    metrics.total_network_bytes += 100 * 1024
                pbar.update(1)
                queue.task_done()

        # Start worker pool
        worker_tasks = [asyncio.create_task(worker()) for _ in range(num_workers)]

        # Producer: stream metadata from UniProt into queue
        produced_count = 0
        async for meta in self.downloader.stream_uniprot_proteome(
            organism_tax_id=organism_tax_id, limit=limit
        ):
            metrics.total_structures_attempted += 1
            await queue.put(meta)
            produced_count += 1
            if produced_count >= limit:
                break

        # Send poison pills to stop workers
        for _ in range(num_workers):
            await queue.put(None)

        await asyncio.gather(*worker_tasks)
        pbar.close()

        if not results:
            raise RuntimeError("No structures were successfully processed.")

        # Aggregate embeddings and assign globally unique record IDs
        all_embeddings: List[np.ndarray] = []
        all_metadata: List[SurfaceVectorRecord] = []
        global_rec_id = 0

        for embs, meta_list in results:
            for rec in meta_list:
                rec.record_id = global_rec_id
                all_metadata.append(rec)
                global_rec_id += 1
            all_embeddings.append(embs)

        combined_records = np.vstack(all_embeddings)

        # Compile single-file .pithos container
        index_path = self.storage.compile_index(
            index_name=index_name,
            embeddings=combined_records,
            metadata=all_metadata,
        )

        metrics.end_time = time.perf_counter()
        return index_path, metrics

    async def index_structures(
        self,
        index_name: str,
        structure_ids: Sequence[str],
        is_alphafold: bool = False,
        file_format: str = "pdb",
        min_plddt: float = 0.0,
        show_progress: bool = True,
    ) -> Path:
        """Stream multiple structures and compile into single-file .pithos container."""
        all_embeddings: List[np.ndarray] = []
        all_metadata: List[SurfaceVectorRecord] = []
        global_rec_id = 0

        pbar = tqdm(structure_ids, desc="Streaming and processing", disable=not show_progress)
        for sid in pbar:
            try:
                surface, output = await self.stream_and_process_structure(
                    sid, is_alphafold=is_alphafold, file_format=file_format, min_plddt=min_plddt
                )

                embs = output.patch_embeddings.detach().cpu().numpy()
                pts = surface.points.detach().cpu().numpy()
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

        return self.storage.compile_index(
            index_name=index_name,
            embeddings=combined_records,
            metadata=all_metadata,
        )

    def search_similar_patches(
        self,
        index_name: str,
        query_surface_output: dMaSIFOutput,
        k: int = 5,
    ) -> List[List[SurfaceQueryResult]]:
        """Search for matching surface patches in PithosDB using outputs from a query protein."""
        queries = query_surface_output.patch_embeddings.detach().cpu().numpy()
        return self.storage.search(index_name=index_name, query_vectors=queries, k=k)
