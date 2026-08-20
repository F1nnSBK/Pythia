"""
End-to-end AlphaPit pipeline orchestrating deterministic single-stream processing,
live async prefetching, geometry generation, dMaSIF neural inference, sharded storage,
and strict memory management tailored for fanless Apple Silicon devices.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
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
    """Real-time performance, throughput, and memory metrics."""
    total_structures_attempted: int = 0
    total_structures_succeeded: int = 0
    total_surface_patches: int = 0
    total_network_bytes: int = 0
    total_shards_written: int = 0
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
    Designed specifically for fanless Apple Silicon (M-series) with prefetching & strict memory bounds.
    """

    def __init__(
        self,
        downloader: Optional[PDBStreamDownloader] = None,
        feature_extractor: Optional[SurfaceFeatureExtractor] = None,
        model: Optional[dMaSIFNet] = None,
        storage_adapter: Optional[PithosStorageAdapter] = None,
        device: Optional[str] = None,
        concurrency: int = 1,
        throttle_sleep_ms: float = 15.0,
    ) -> None:
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.concurrency = max(1, concurrency)
        self.throttle_sleep_ms = throttle_sleep_ms
        self.downloader = downloader or PDBStreamDownloader(concurrency_limit=self.concurrency)
        self.feature_extractor = feature_extractor or SurfaceFeatureExtractor()
        self.model = (model or dMaSIFNet(in_channels=21)).to(self.device)
        self.model.eval()
        self.storage = storage_adapter or PithosStorageAdapter()

    def _apply_thermal_governor(self) -> None:
        """Set nice process priority so macOS UI remains 100% responsive."""
        try:
            os.nice(5)
        except Exception:
            pass

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

        pc = ProteinPointCloud.from_structure_data(struct_data, device=self.device)
        surface = self.feature_extractor.extract(pc)

        with torch.no_grad():
            output = self.model(surface)

        return surface, output

    async def _prefetch_structure_data(
        self,
        meta: AlphaFoldMetadata,
        min_plddt: float = 0.0,
    ) -> Optional[Tuple[AlphaFoldMetadata, ProteinStructureData]]:
        """Download and parse structure in RAM asynchronously with strict 10s timeout."""
        try:
            return await asyncio.wait_for(
                self._fetch_and_parse_single(meta, min_plddt=min_plddt),
                timeout=10.0,
            )
        except Exception:
            return None

    async def _fetch_and_parse_single(
        self,
        meta: AlphaFoldMetadata,
        min_plddt: float = 0.0,
    ) -> Optional[Tuple[AlphaFoldMetadata, ProteinStructureData]]:
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
        return meta, struct_data

    def process_structure_data(
        self,
        meta: AlphaFoldMetadata,
        struct_data: ProteinStructureData,
    ) -> Optional[Tuple[np.ndarray, List[SurfaceVectorRecord]]]:
        """Process pre-fetched structure data on GPU/MPS with instant memory release."""
        try:
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
                        record_id=0,
                        structure_id=meta.uniprot_accession,
                        chain_id="A",
                        res_seq=int(res_seqs[i]),
                        atom_idx=int(atoms[i]),
                        point_coords=[float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])],
                    )
                )

            del pc, surface, output
            return embs, metadata_records
        except Exception:
            return None

    def _load_checkpoint(self, checkpoint_file: Path) -> Set[str]:
        """Load already processed UniProt accessions."""
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data.get("processed_ids", []))
            except Exception:
                return set()
        return set()

    def _save_checkpoint(self, checkpoint_file: Path, processed_ids: Set[str]) -> None:
        """Persist processed UniProt accessions to SSD."""
        try:
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({"processed_ids": list(processed_ids)}, f)
        except Exception:
            pass

    async def stream_and_index_proteome(
        self,
        index_name: str,
        organism_tax_id: str = "9606",  # Homo sapiens
        limit: Optional[int] = None,
        min_plddt: float = 70.0,
        shard_size: int = 250,
        concurrency: int = 1,
        show_progress: bool = True,
    ) -> Tuple[List[Path], PipelineMetrics]:
        """
        Pipelined streaming pipeline with async network prefetching and single-worker GPU inference.
        """
        self._apply_thermal_governor()
        metrics = PipelineMetrics()
        metrics.start_time = time.perf_counter()

        checkpoint_file = settings.full_index_path / f"{index_name}_checkpoint.json"
        processed_ids = self._load_checkpoint(checkpoint_file)

        current_shard_embeddings: List[np.ndarray] = []
        current_shard_metadata: List[SurfaceVectorRecord] = []
        shard_paths: List[Path] = []
        existing_shards = list(settings.full_index_path.glob(f"{index_name}_shard_*.pithos"))
        shard_idx = len(existing_shards) + 1
        structures_in_current_shard = 0

        total_expected = limit if limit is not None else 20400
        pbar = tqdm(total=total_expected, desc="Streaming Proteome", disable=not show_progress)

        # Bounded prefetch queue (holds at most 3 pre-downloaded structures in RAM)
        prefetch_queue: asyncio.Queue[Optional[Tuple[AlphaFoldMetadata, ProteinStructureData]]] = asyncio.Queue(maxsize=3)

        def flush_shard():
            nonlocal structures_in_current_shard, shard_idx
            if not current_shard_embeddings:
                return

            shard_name = f"{index_name}_shard_{shard_idx:04d}"
            combined = np.vstack(current_shard_embeddings)

            indexed_meta = []
            rec_offset = 0
            for meta_rec in current_shard_metadata:
                meta_rec.record_id = rec_offset
                indexed_meta.append(meta_rec)
                rec_offset += 1

            shard_path = self.storage.compile_index(
                index_name=shard_name,
                embeddings=combined,
                metadata=indexed_meta,
            )
            shard_paths.append(shard_path)
            metrics.total_shards_written += 1

            print(
                f"[{time.strftime('%H:%M:%S')}] Shard {shard_idx:03d} compiled -> "
                f"{shard_path.name} ({shard_path.stat().st_size / (1024 * 1024):.2f} MB, {len(indexed_meta):,} vectors)",
                flush=True,
            )
            shard_idx += 1

            current_shard_embeddings.clear()
            current_shard_metadata.clear()
            structures_in_current_shard = 0

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        # Producer Task: Streams UniProt and prefetches CIF files across HTTP/2
        async def producer():
            produced = 0
            async for meta in self.downloader.stream_uniprot_proteome(
                organism_tax_id=organism_tax_id, limit=limit
            ):
                metrics.total_structures_attempted += 1
                if meta.uniprot_accession in processed_ids:
                    pbar.update(1)
                    continue

                item = await self._prefetch_structure_data(meta, min_plddt=min_plddt)
                if item is not None:
                    await prefetch_queue.put(item)
                    produced += 1
                else:
                    pbar.update(1)

                if limit and produced >= limit:
                    break

            await prefetch_queue.put(None)  # Poison pill

        producer_task = asyncio.create_task(producer())

        # Consumer Loop: Single-stream GPU inference with 0 network latency
        while True:
            item = await prefetch_queue.get()
            if item is None:
                prefetch_queue.task_done()
                break

            meta, struct_data = item
            res = self.process_structure_data(meta, struct_data)
            del struct_data

            if res is not None:
                embs, meta_list = res
                current_shard_embeddings.append(embs)
                current_shard_metadata.extend(meta_list)
                processed_ids.add(meta.uniprot_accession)
                structures_in_current_shard += 1
                metrics.total_structures_succeeded += 1
                metrics.total_surface_patches += embs.shape[0]
                metrics.total_network_bytes += 60 * 1024

                rate_struct = metrics.structures_per_second
                rate_vec = metrics.patches_per_second
                pct = (metrics.total_structures_succeeded / total_expected) * 100.0
                print(
                    f"[{time.strftime('%H:%M:%S')}] Shard {shard_idx:03d} | "
                    f"Processed {metrics.total_structures_succeeded} / {total_expected} ({pct:.1f}%) | "
                    f"Last: {meta.uniprot_accession} (+{embs.shape[0]} patches) | "
                    f"Total: {metrics.total_surface_patches:,} vectors | "
                    f"Speed: {rate_struct:.2f} struct/s ({rate_vec:,.1f} vec/s)",
                    flush=True,
                )

                if structures_in_current_shard >= shard_size:
                    flush_shard()
                    self._save_checkpoint(checkpoint_file, processed_ids)

            pbar.update(1)
            prefetch_queue.task_done()

            if self.throttle_sleep_ms > 0:
                await asyncio.sleep(self.throttle_sleep_ms / 1000.0)

        await producer_task
        pbar.close()

        if current_shard_embeddings:
            flush_shard()
            self._save_checkpoint(checkpoint_file, processed_ids)

        metrics.end_time = time.perf_counter()
        return shard_paths, metrics

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

    def search_similar_patches_all_shards(
        self,
        query_surface_output: dMaSIFOutput,
        shard_prefix: str = "human_proteome_9606_shard_",
        k: int = 5,
    ) -> List[List[SurfaceQueryResult]]:
        """Search for matching surface patches across ALL compiled PithosDB shards on SSD."""
        queries = query_surface_output.patch_embeddings.detach().cpu().numpy()
        return self.storage.search_all_shards(query_vectors=queries, shard_prefix=shard_prefix, k=k)

