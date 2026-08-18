"""
Asynchronous streaming HTTP client for RCSB PDB and AlphaFold Protein Structure Database.
Supports direct streaming to parser without intermediate disk I/O, as well as fast NVMe SSD caching.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, List, Optional, Set
import httpx

from alphapit.config import settings
from alphapit.download.parser import ProteinStructureData, StreamingPDBParser
from alphapit.download.stream import iter_lines_from_byte_stream


class PDBStreamDownloader:
    """
    High-performance async streaming client for fetching protein structures.
    """

    def __init__(
        self,
        rcsb_base_url: Optional[str] = None,
        alphafold_base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        concurrency_limit: int = 8,
    ) -> None:
        self.rcsb_base_url = (rcsb_base_url or settings.rcsb_base_url).rstrip("/")
        self.alphafold_base_url = (alphafold_base_url or settings.alphafold_base_url).rstrip("/")
        self.cache_dir = cache_dir or settings.full_raw_cache_path
        self.timeout = timeout or settings.download_timeout_seconds
        self.max_retries = max_retries or settings.download_max_retries
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> PDBStreamDownloader:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def build_rcsb_url(self, pdb_id: str, file_format: str = "pdb") -> str:
        """Construct standard RCSB download URL."""
        clean_id = pdb_id.strip().lower()
        fmt = file_format.lower()
        if fmt in ("pdb", "ent"):
            return f"{self.rcsb_base_url}/{clean_id}.pdb.gz"
        elif fmt in ("cif", "mmcif"):
            return f"{self.rcsb_base_url}/{clean_id}.cif.gz"
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    def build_alphafold_url(self, uniprot_id: str, version: int = 4) -> str:
        """Construct AlphaFold Database download URL for a UniProt ID."""
        clean_id = uniprot_id.strip().upper()
        return f"{self.alphafold_base_url}/AF-{clean_id}-F1-model_v{version}.cif"

    async def stream_url_bytes(self, url: str) -> AsyncIterator[bytes]:
        """
        Asynchronously stream raw bytes chunk-by-chunk from URL with retry logic.
        """
        client = self._get_client()
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.semaphore:
                    async with client.stream("GET", url) as response:
                        if response.status_code == 404:
                            raise FileNotFoundError(f"Resource not found (404) at {url}")
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes(chunk_size=settings.download_chunk_size):
                            yield chunk
                return
            except (httpx.TransportError, httpx.HTTPStatusError, asyncio.TimeoutError) as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                else:
                    raise RuntimeError(
                        f"Failed to stream from {url} after {self.max_retries} attempts: {exc}"
                    ) from last_exception

    async def stream_and_parse_rcsb(
        self,
        pdb_id: str,
        file_format: str = "pdb",
        allowed_chains: Optional[Set[str]] = None,
        parser: Optional[StreamingPDBParser] = None,
    ) -> ProteinStructureData:
        """
        Stream a PDB from RCSB directly into parser memory without disk staging.
        """
        url = self.build_rcsb_url(pdb_id, file_format)
        is_gzipped = url.endswith(".gz")
        p = parser or StreamingPDBParser(allowed_chains=allowed_chains)

        byte_stream = self.stream_url_bytes(url)
        line_stream = iter_lines_from_byte_stream(byte_stream, is_gzipped=is_gzipped)
        return await p.parse_lines_async(pdb_id.upper(), line_stream, file_format=file_format)

    async def stream_and_parse_alphafold(
        self,
        uniprot_id: str,
        allowed_chains: Optional[Set[str]] = None,
        parser: Optional[StreamingPDBParser] = None,
    ) -> ProteinStructureData:
        """
        Stream an AlphaFold structure directly into parser memory without disk staging.
        """
        url = self.build_alphafold_url(uniprot_id)
        is_gzipped = url.endswith(".gz")
        p = parser or StreamingPDBParser(allowed_chains=allowed_chains)

        byte_stream = self.stream_url_bytes(url)
        line_stream = iter_lines_from_byte_stream(byte_stream, is_gzipped=is_gzipped)
        return await p.parse_lines_async(uniprot_id.upper(), line_stream, file_format="cif")

    async def download_to_cache(
        self,
        structure_id: str,
        is_alphafold: bool = False,
        file_format: str = "pdb",
        overwrite: bool = False,
    ) -> Path:
        """
        Stream download and persist to the fast NVMe SSD storage cache.
        """
        if is_alphafold:
            url = self.build_alphafold_url(structure_id)
            ext = ".cif"
        else:
            url = self.build_rcsb_url(structure_id, file_format)
            ext = f".{file_format}.gz"

        target_file = self.cache_dir / f"{structure_id.lower()}{ext}"
        if target_file.exists() and not overwrite and target_file.stat().st_size > 0:
            return target_file

        target_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target_file.with_suffix(f"{target_file.suffix}.tmp")

        with open(temp_file, "wb") as f:
            async for chunk in self.stream_url_bytes(url):
                f.write(chunk)

        temp_file.replace(target_file)
        return target_file

    async def batch_stream_and_parse(
        self,
        structure_ids: List[str],
        file_format: str = "pdb",
        is_alphafold: bool = False,
    ) -> List[ProteinStructureData]:
        """
        Batch stream multiple structures concurrently.
        """
        tasks = []
        for sid in structure_ids:
            if is_alphafold:
                tasks.append(self.stream_and_parse_alphafold(sid))
            else:
                tasks.append(self.stream_and_parse_rcsb(sid, file_format=file_format))
        return await asyncio.gather(*tasks, return_exceptions=False)
