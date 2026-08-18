"""
Command-line interface for the AlphaPit library.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path
import torch

from alphapit.config import settings
from alphapit.download.client import PDBStreamDownloader
from alphapit.pipeline import AlphaPitPipeline


def cmd_status(args: argparse.Namespace) -> None:
    """Print hardware, NVMe storage, and system status."""
    print("=== AlphaPit System Status ===")
    ssd_path = Path("/Volumes/AlphaPitData")
    if ssd_path.exists():
        total, used, free = shutil.disk_usage(ssd_path)
        print(f"NVMe SSD Mount:     {ssd_path} (ONLINE)")
        print(f"SSD Capacity:       {total / (1024**3):.1f} GB")
        print(f"SSD Free Space:     {free / (1024**3):.1f} GB ({free / total * 100:.1f}%)")
    else:
        print(f"NVMe SSD Mount:     {ssd_path} (NOT MOUNTED, using fallback: {settings.storage_root})")

    print(f"Raw Cache Path:     {settings.full_raw_cache_path}")
    print(f"Pithos Index Path:  {settings.full_index_path}")

    # Compute Device
    if torch.backends.mps.is_available():
        device_str = "Apple Silicon MPS (Metal Performance Shaders)"
    elif torch.cuda.is_available():
        device_str = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device_str = "CPU"
    print(f"Compute Backend:    {device_str}")
    print(f"Matryoshka Tiers:   {settings.matryoshka_tiers}")


async def _run_download(args: argparse.Namespace) -> None:
    downloader = PDBStreamDownloader()
    async with downloader:
        print(f"Downloading {len(args.ids)} structures to {settings.full_raw_cache_path}...")
        for sid in args.ids:
            try:
                cached_file = await downloader.download_to_cache(
                    structure_id=sid,
                    is_alphafold=args.alphafold,
                    file_format=args.format,
                )
                print(f"  [OK] {sid.upper()} -> {cached_file.name} ({cached_file.stat().st_size / 1024:.1f} KB)")
            except Exception as e:
                print(f"  [ERROR] {sid.upper()}: {e}")


def cmd_download(args: argparse.Namespace) -> None:
    asyncio.run(_run_download(args))


async def _run_index(args: argparse.Namespace) -> None:
    pipeline = AlphaPitPipeline()
    print(f"Streaming and indexing {len(args.ids)} structures into PithosDB index '{args.name}'...")
    index_path = await pipeline.index_structures(
        index_name=args.name,
        structure_ids=args.ids,
        is_alphafold=args.alphafold,
        file_format=args.format,
    )
    print(f"Indexing complete! Multi-tier Pithos index compiled at: {index_path}")


def cmd_index(args: argparse.Namespace) -> None:
    asyncio.run(_run_index(args))


async def _run_search(args: argparse.Namespace) -> None:
    pipeline = AlphaPitPipeline()
    print(f"Streaming query structure {args.query_id}...")
    surface, output = await pipeline.stream_and_process_structure(
        args.query_id, is_alphafold=args.alphafold, file_format=args.format
    )
    print(f"Generated {surface.num_points} query surface patches. Searching index '{args.name}' (top {args.top_k})...")

    results = pipeline.search_similar_patches(args.name, output, k=args.top_k)

    print("\nTop Matches for First 3 Query Patches:")
    for q_idx, matches in enumerate(results[:3]):
        print(f"\nQuery Patch {q_idx + 1} (Residue {surface.res_indices[q_idx]}):")
        for match in matches:
            print(
                f"  Match Record {match.record_id:4d} | Score: {match.score:10.2f} | "
                f"Target: {match.structure_id} Chain {match.chain_id} Res {match.res_seq}"
            )


def cmd_search(args: argparse.Namespace) -> None:
    asyncio.run(_run_search(args))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="alphapit",
        description="AlphaPit: End-to-end geometric deep learning on protein surfaces with PithosDB vector index",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = subparsers.add_parser("status", help="Show system, NVMe SSD, and backend status")
    p_status.set_defaults(func=cmd_status)

    # download
    p_dl = subparsers.add_parser("download", help="Stream download structures to NVMe SSD cache")
    p_dl.add_argument("ids", nargs="+", help="PDB IDs (e.g. 1a8o 6m0j) or UniProt IDs")
    p_dl.add_argument("--alphafold", action="store_true", help="Download from AlphaFold Database")
    p_dl.add_argument("--format", choices=["pdb", "cif"], default="pdb", help="Structure format")
    p_dl.set_defaults(func=cmd_download)

    # index
    p_idx = subparsers.add_parser("index", help="Stream structures and compile into PithosDB index")
    p_idx.add_argument("name", help="Name of the PithosDB index")
    p_idx.add_argument("ids", nargs="+", help="PDB IDs or UniProt IDs to index")
    p_idx.add_argument("--alphafold", action="store_true", help="Download from AlphaFold Database")
    p_idx.add_argument("--format", choices=["pdb", "cif"], default="pdb", help="Structure format")
    p_idx.set_defaults(func=cmd_index)

    # search
    p_srch = subparsers.add_parser("search", help="Search compiled PithosDB index with query structure")
    p_srch.add_argument("name", help="Name of the PithosDB index to query")
    p_srch.add_argument("query_id", help="Query PDB ID or UniProt ID")
    p_srch.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbors to retrieve")
    p_srch.add_argument("--alphafold", action="store_true", help="Query is an AlphaFold ID")
    p_srch.add_argument("--format", choices=["pdb", "cif"], default="pdb", help="Structure format")
    p_srch.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
