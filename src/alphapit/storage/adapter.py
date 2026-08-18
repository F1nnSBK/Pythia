"""
PithosDB storage adapter for indexing and querying high-dimensional protein surface embeddings.
Provides off-heap memory-mapped vector search, multi-tier Matryoshka indexing, and metadata association.
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
    Adapter bridging AlphaPit geometric embeddings with the Pithos Model-Isomorphic Vector Database.
    """

    def __init__(
        self,
        base_storage_dir: Optional[Path] = None,
        tiers: Optional[Sequence[int]] = None,
    ) -> None:
        self.storage_dir = base_storage_dir or settings.full_index_path
        self.tiers = list(tiers or settings.matryoshka_tiers)
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

    def get_index_path(self, index_name: str) -> Path:
        """Return the target on-disk directory path for a named index."""
        path = self.storage_dir / index_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def compile_index(
        self,
        index_name: str,
        embeddings: np.ndarray,
        metadata: Optional[List[SurfaceVectorRecord]] = None,
        ids: Optional[Sequence[int]] = None,
    ) -> Path:
        """
        Compile continuous float surface patch embeddings into a multi-tier Pithos binary index.
        """
        records = np.asarray(embeddings, dtype=np.float32)
        if records.ndim != 2:
            raise ValueError(f"Embeddings must be a 2D array of shape (N, D), got {records.shape}")

        self.projector.validate_dimension(records.shape[1])
        records_normalized = self.projector.normalize_l2(records)

        index_path = self.get_index_path(index_name)
        base_path_str = str(index_path / "index")

        record_ids = list(ids) if ids is not None else list(range(len(records)))

        # Compile Pithos binary columnar files on disk
        pithosdb.VectorDb.compile_index(
            base_path=base_path_str,
            records=records_normalized,
            ids=record_ids,
            tiers=self.tiers,
        )

        # Save metadata sidecar if provided
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
            meta_file = index_path / "metadata.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f)

        return index_path

    def load_index(self, index_name: str) -> Any:
        """
        Map a compiled Pithos index into off-heap memory.
        """
        if self._db is None:
            self._db = pithosdb.VectorDb()

        if index_name in self._loaded_indices:
            return self._loaded_indices[index_name]

        index_path = self.get_index_path(index_name)
        base_path_str = str(index_path / "index")

        idx_handle = self._db.load_index(index_name, base_path_str)
        self._loaded_indices[index_name] = idx_handle

        # Load metadata sidecar if it exists
        meta_file = index_path / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                raw_meta = json.load(f)
                self._metadata_cache[index_name] = {
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
        Execute zero-copy batch k-NN search against the memory-mapped Pithos index.
        """
        idx_handle = self.load_index(index_name)
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = np.expand_dims(queries, axis=0)

        queries_normalized = self.projector.normalize_l2(queries)
        raw_results = idx_handle.search(queries_normalized, k=k)

        meta_map = self._metadata_cache.get(index_name, {})
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
