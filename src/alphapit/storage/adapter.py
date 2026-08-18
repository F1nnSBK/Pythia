"""
PithosDB storage adapter for indexing and querying high-dimensional protein surface embeddings.
Provides single-file universal container (.pithos) compilation and zero-copy off-heap vector search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pithosdb

from alphapit.config import settings
from alphapit.storage.matryoshka import MatryoshkaProjector


@dataclass
class SurfaceVectorRecord:
    """Represents a single indexed surface patch vector with its biological metadata."""
    record_id: int
    structure_id: str
    chain_id: str
    res_seq: int
    atom_idx: int
    point_coords: List[float]


@dataclass
class SurfaceQueryResult:
    """Represents the search result mapped to surface metadata."""
    record_id: int
    score: int | float
    structure_id: Optional[str] = None
    chain_id: Optional[str] = None
    res_seq: Optional[int] = None
    atom_idx: Optional[int] = None
    point_coords: Optional[List[float]] = None


class PithosStorageAdapter:
    """
    Adapter bridging AlphaPit geometric embeddings with the Pithos Single-File (.pithos) Container Engine.
    """

    def __init__(
        self,
        base_storage_dir: Optional[Path] = None,
        tiers: Optional[Sequence[int]] = None,
        metric: str = "cosine",
    ) -> None:
        self.storage_dir = base_storage_dir or settings.full_index_path
        self.tiers = list(tiers or settings.matryoshka_tiers)
        self.metric = metric
        self.projector = MatryoshkaProjector(self.tiers)
        self._db: Optional[pithosdb.VectorDb] = None
        self._loaded_indices: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, Dict[int, SurfaceVectorRecord]] = {}

    def __enter__(self) -> PithosStorageAdapter:
        self._db = pithosdb.VectorDb()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close open database handles and release memory-mapped buffers."""
        if self._db is not None:
            self._db.close()
            self._db = None
        self._loaded_indices.clear()

    def get_container_path(self, index_name: str) -> Path:
        """Return the target file path for the single .pithos container."""
        clean_name = index_name.removesuffix(".pithos")
        return self.storage_dir / f"{clean_name}.pithos"

    def get_metadata_path(self, index_name: str) -> Path:
        """Return the target file path for the metadata sidecar."""
        clean_name = index_name.removesuffix(".pithos")
        return self.storage_dir / f"{clean_name}_metadata.json"

    def compile_index(
        self,
        index_name: str,
        embeddings: np.ndarray,
        metadata: Optional[List[SurfaceVectorRecord]] = None,
        ids: Optional[Sequence[int]] = None,
    ) -> Path:
        """
        Compile continuous float surface patch embeddings into a single universal .pithos container.
        """
        records = np.asarray(embeddings, dtype=np.float32)
        if records.ndim != 2:
            raise ValueError(f"Embeddings must be a 2D array of shape (N, D), got {records.shape}")

        self.projector.validate_dimension(records.shape[1])
        records_normalized = self.projector.normalize_l2(records)

        container_file = self.get_container_path(index_name)
        container_file.parent.mkdir(parents=True, exist_ok=True)
        record_ids = list(ids) if ids is not None else list(range(len(records)))

        # Format metadata for embedded packaging & sidecar
        meta_dict = {}
        if metadata:
            meta_dict = {
                rec.record_id: {
                    "structure_id": rec.structure_id,
                    "chain_id": rec.chain_id,
                    "res_seq": rec.res_seq,
                    "atom_idx": rec.atom_idx,
                    "point_coords": rec.point_coords,
                }
                for rec in metadata
            }

        # Compile into single-file .pithos container (DIOGENES format)
        pithosdb.VectorDb.compile_container(
            path=str(container_file),
            records=records_normalized,
            ids=record_ids,
            tiers=self.tiers,
            metric=self.metric,
            user_metadata={"num_records": len(records), "dimension": records.shape[1]},
        )

        # Save metadata sidecar
        if meta_dict:
            meta_file = self.get_metadata_path(index_name)
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f)

        return container_file

    def load_index(self, index_name: str) -> Any:
        """
        Map a single-file .pithos container into off-heap memory.
        """
        if self._db is None:
            self._db = pithosdb.VectorDb()

        clean_name = index_name.removesuffix(".pithos")
        if clean_name in self._loaded_indices:
            return self._loaded_indices[clean_name]

        container_file = self.get_container_path(clean_name)
        if not container_file.exists():
            raise FileNotFoundError(f"Pithos container not found: {container_file}")

        idx_handle = self._db.load_index(clean_name, str(container_file))
        self._loaded_indices[clean_name] = idx_handle

        # Load metadata sidecar
        meta_file = self.get_metadata_path(clean_name)
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                raw_meta = json.load(f)
                self._metadata_cache[clean_name] = {
                    int(k): SurfaceVectorRecord(
                        record_id=int(k),
                        structure_id=v["structure_id"],
                        chain_id=v["chain_id"],
                        res_seq=v["res_seq"],
                        atom_idx=v["atom_idx"],
                        point_coords=v["point_coords"],
                    )
                    for k, v in raw_meta.items()
                }

        return idx_handle

    def search(
        self,
        index_name: str,
        query_vectors: np.ndarray,
        k: int = 5,
    ) -> List[List[SurfaceQueryResult]]:
        """
        Execute zero-copy batch k-NN search against the memory-mapped single-file .pithos container.
        """
        clean_name = index_name.removesuffix(".pithos")
        idx_handle = self.load_index(clean_name)

        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = np.expand_dims(queries, axis=0)

        queries_normalized = self.projector.normalize_l2(queries)
        raw_results = idx_handle.search(queries_normalized, k=k)

        meta_map = self._metadata_cache.get(clean_name, {})
        enriched_batch: List[List[SurfaceQueryResult]] = []

        for query_matches in raw_results:
            match_list: List[SurfaceQueryResult] = []
            for match in query_matches:
                rec_meta = meta_map.get(match.id)
                if rec_meta:
                    match_list.append(
                        SurfaceQueryResult(
                            record_id=match.id,
                            score=match.score,
                            structure_id=rec_meta.structure_id,
                            chain_id=rec_meta.chain_id,
                            res_seq=rec_meta.res_seq,
                            atom_idx=rec_meta.atom_idx,
                            point_coords=rec_meta.point_coords,
                        )
                    )
                else:
                    match_list.append(
                        SurfaceQueryResult(record_id=match.id, score=match.score)
                    )
            enriched_batch.append(match_list)

        return enriched_batch
