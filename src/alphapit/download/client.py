"""
Asynchronous streaming HTTP client for RCSB PDB, UniProtKB, and AlphaFold DB.
Fetches live metadata directly from EBI & UniProt REST servers and streams
structures with high concurrency and zero disk footprint.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Set
import httpx

from alphapit.config import settings
from alphapit.download.parser import ProteinStructureData, StreamingPDBParser
from alphapit.download.stream import iter_lines_from_byte_stream


@dataclass
class AlphaFoldMetadata:
    """Live metadata for an AlphaFold predicted protein structure."""
    uniprot_accession: str
    uniprot_id: Optional[str] = None
    gene_name: Optional[str] = None
    protein_name: Optional[str] = None
    organism_name: Optional[str] = None
    organism_tax_id: Optional[int] = None
    cif_url: Optional[str] = None
    pdb_url: Optional[str] = None
    pae_url: Optional[str] = None
    global_plddt: Optional[float] = None
    sequence_length: Optional[int] = None
    model_version: int = 4


class PDBStreamDownloader:
    """
    High-performance async streaming client for fetching protein structures and live metadata.
    """

    def __init__(
        self,
        rcsb_base_url: Optional[str] = None,
        alphafold_base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        concurrency_limit: int = 16,
    ) -> None:
        self.rcsb_base_url = (rcsb_base_url or settings.rcsb_base_url).rstrip("/")
        self.alphafold_base_url = (alphafold_base_url or settings.alphafold_base_url).rstrip("/")
        self.cache_dir = cache_dir or settings.full_raw_cache_path
        self.timeout = timeout or settings.download_timeout_seconds
        self.max_retries = max_retries or settings.download_max_retries
        self.concurrency_limit = concurrency_limit
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> PDBStreamDownloader:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(8.0, connect=5.0, read=8.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_alphafold_metadata(self, uniprot_id: str) -> Optional[AlphaFoldMetadata]:
        """
        Fetch live prediction metadata from EBI AlphaFold API.
        """
        client = self._get_client()
        clean_id = uniprot_id.strip().upper()
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{clean_id}"

        try:
            async with self.semaphore:
                resp = await client.get(url)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()

            if isinstance(data, list) and len(data) > 0:
                entry = data[0]
                return AlphaFoldMetadata(
                    uniprot_accession=clean_id,
                    uniprot_id=entry.get("uniprotId"),
                    gene_name=entry.get("gene"),
                    protein_name=entry.get("proteinDescription"),
                    organism_name=entry.get("organismScientificName"),
                    cif_url=entry.get("cifUrl"),
                    pdb_url=entry.get("pdbUrl"),
                    pae_url=entry.get("paeDocUrl"),
                    global_plddt=entry.get("globalMetricValue"),
                    sequence_length=entry.get("sequenceLength"),
                    model_version=int(entry.get("latestVersion", 4)),
                )
            return None
        except Exception:
            return None

    async def stream_uniprot_proteome(
        self,
        organism_tax_id: str = "9606",  # 9606 = Homo sapiens
        reviewed_only: bool = True,
        batch_size: int = 250,
        limit: Optional[int] = None,
    ) -> AsyncIterator[AlphaFoldMetadata]:
        """
        Stream live protein metadata from UniProtKB with automatic pagination.
        """
        client = self._get_client()
        rev_str = "true" if reviewed_only else "false"
        query = f"organism_id:{organism_tax_id}+AND+reviewed:{rev_str}"
        base_search_url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&size={batch_size}&fields=accession,id,gene_names,protein_name,organism_name,length"

        next_url: Optional[str] = base_search_url
        yielded_count = 0

        while next_url:
            async with self.semaphore:
                resp = await client.get(next_url)
                resp.raise_for_status()
                payload = resp.json()

            results = payload.get("results", [])
            if not results:
                break

            for item in results:
                acc = item.get("primaryAccession")
                if not acc:
                    continue

                genes = item.get("genes", [])
                gene_name = genes[0].get("geneName", {}).get("value") if genes else None

                meta = AlphaFoldMetadata(
                    uniprot_accession=acc,
                    uniprot_id=item.get("uniProtkbId"),
                    gene_name=gene_name,
                    protein_name=item.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value"),
                    organism_name=item.get("organism", {}).get("scientificName"),
                    organism_tax_id=int(organism_tax_id) if organism_tax_id.isdigit() else None,
                    sequence_length=item.get("sequence", {}).get("length"),
                    cif_url=f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.cif",
                )
                yield meta
                yielded_count += 1
                if limit and yielded_count >= limit:
                    return

            # Pagination via 'link' header (rel="next")
            link_header = resp.headers.get("link")
            next_url = None
            if link_header:
                match = re.search(r'<([^>]+)>;\s*rel=[\"\']?next[\"\']?', link_header)
                if match:
                    next_url = match.group(1).strip()

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
                    await asyncio.sleep(0.3 * (2 ** (attempt - 1)))
                else:
                    raise RuntimeError(
                        f"Failed to stream from {url} after {self.max_retries} attempts: {exc}"
                    ) from last_exception

    async def stream_and_parse_url(
        self,
        url: str,
        structure_id: str,
        file_format: str = "cif",
        min_plddt: float = 0.0,
        allowed_chains: Optional[Set[str]] = None,
    ) -> ProteinStructureData:
        """
        Stream any structure URL directly into parser memory.
        """
        is_gzipped = url.endswith(".gz")
        parser = StreamingPDBParser(allowed_chains=allowed_chains, min_plddt=min_plddt)
        byte_stream = self.stream_url_bytes(url)
        line_stream = iter_lines_from_byte_stream(byte_stream, is_gzipped=is_gzipped)
        return await parser.parse_lines_async(structure_id, line_stream, file_format=file_format)

    async def stream_and_parse_rcsb(
        self,
        pdb_id: str,
        file_format: str = "pdb",
        allowed_chains: Optional[Set[str]] = None,
        parser: Optional[StreamingPDBParser] = None,
    ) -> ProteinStructureData:
        """Stream a PDB from RCSB directly into parser memory."""
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
        min_plddt: float = 0.0,
    ) -> ProteinStructureData:
        """
        Stream AlphaFold structure using live metadata API lookup or direct URL.
        """
        clean_id = uniprot_id.strip().upper()
        # Fetch live metadata for accurate URL
        meta = await self.fetch_alphafold_metadata(clean_id)
        cif_url = meta.cif_url if meta and meta.cif_url else f"https://alphafold.ebi.ac.uk/files/AF-{clean_id}-F1-model_v4.cif"

        return await self.stream_and_parse_url(
            url=cif_url,
            structure_id=clean_id,
            file_format="cif",
            min_plddt=min_plddt,
            allowed_chains=allowed_chains,
        )

    async def download_to_cache(
        self,
        structure_id: str,
        is_alphafold: bool = False,
        file_format: str = "pdb",
        overwrite: bool = False,
    ) -> Path:
        """Stream download and persist to fast NVMe SSD storage cache."""
        if is_alphafold:
            clean_id = structure_id.strip().upper()
            meta = await self.fetch_alphafold_metadata(clean_id)
            url = meta.cif_url if meta and meta.cif_url else f"https://alphafold.ebi.ac.uk/files/AF-{clean_id}-F1-model_v4.cif"
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
