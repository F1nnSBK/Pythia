"""
Command-line interface for the AlphaPit library.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

from alphapit.config import settings
from alphapit.download.client import PDBStreamDownloader
from alphapit.pipeline import AlphaPitPipeline
from alphapit.storage.adapter import SurfaceQueryResult


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

    print(f"Index Storage Path: {settings.full_index_path}")
    print(f"Temporary Buffer:   {settings.full_temp_path}")

    # Shard inventory
    shards = sorted(settings.full_index_path.glob("*.pithos"))
    total_shard_bytes = sum(s.stat().st_size for s in shards)
    print(f"Compiled Shards:    {len(shards)} container files ({total_shard_bytes / (1024*1024):.1f} MB)")

    # Compute Device
    if torch.backends.mps.is_available():
        device_str = "Apple Silicon MPS (Metal Performance Shaders, Single-Threaded)"
    elif torch.cuda.is_available():
        device_str = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device_str = "CPU"
    print(f"Compute Backend:    {device_str}")
    print(f"Surface Resolution: {settings.surface_resolution} A (super_sampling={settings.sup_sampling})")
    print(f"Vector Dimension:   {settings.vector_dimension} (Dual Fingerprint: 7 Chem + 10 Curv + 4 LBO)")
    print(f"Matryoshka Tiers:   {settings.matryoshka_tiers}")


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


async def _run_search_proteome(args: argparse.Namespace) -> None:
    pipeline = AlphaPitPipeline()
    t0 = time.perf_counter()

    print("=== AlphaPit Multi-Shard Proteome Pocket Search ===")
    print(f"Query Target:       {args.query_id} (format={args.format})")
    print(f"Target Storage:     {settings.full_index_path}")

    shards = pipeline.storage.list_available_shards(prefix=args.shard_prefix)
    print(f"Available Shards:   {len(shards)} (.pithos containers on SSD)")
    if not shards:
        print("Error: No compiled shards found on storage.")
        return

    print("Streaming query structure into memory & computing 21-D surface features...")
    surface, output = await pipeline.stream_and_process_structure(
        args.query_id, is_alphafold=args.alphafold, file_format=args.format
    )
    t_feat = time.perf_counter()
    print(f"Extracted {surface.num_points} query surface patches in {t_feat - t0:.2f}s.")

    print(f"Executing SIMD multi-shard search across all {len(shards)} shards (top {args.top_k} per patch)...")
    t_search_start = time.perf_counter()
    all_patch_matches = pipeline.search_similar_patches_all_shards(
        output, shard_prefix=args.shard_prefix, k=args.top_k
    )
    t_search_end = time.perf_counter()
    search_ms = (t_search_end - t_search_start) * 1000.0
    print(f"Search completed in {search_ms:.1f} ms across {len(shards)} shards!")

    # Aggregate target proteins
    target_hits: Dict[str, List[SurfaceQueryResult]] = defaultdict(list)
    for q_idx, matches in enumerate(all_patch_matches):
        for m in matches:
            if m.structure_id:
                target_hits[m.structure_id].append(m)

    # Rank targets by number of patch hits and best alignment score
    ranked_targets = sorted(
        target_hits.items(),
        key=lambda item: (len(item[1]), -np.mean([m.score for m in item[1]])),
        reverse=True,
    )

    print("\n" + "=" * 78)
    print(f"TOP HUMAN PROTEOME MATCHES FOR QUERY: {args.query_id}")
    print("=" * 78)
    print(f"{'Rank':<5} | {'UniProt ID':<12} | {'Matching Patches':<18} | {'Best Score':<12} | {'Sample Residues'}")
    print("-" * 78)

    for rank, (target_id, hit_list) in enumerate(ranked_targets[: args.max_targets], 1):
        best_score = min(m.score for m in hit_list)
        unique_res = sorted(set(m.res_seq for m in hit_list if m.res_seq is not None))[:6]
        res_str = ", ".join(f"Res {r}" for r in unique_res)
        print(f"{rank:<5} | {target_id:<12} | {len(hit_list):<18} | {best_score:<12.1f} | {res_str}")

    print("=" * 78)
    print(f"Total query execution time: {time.perf_counter() - t0:.2f} s")


def cmd_search_proteome(args: argparse.Namespace) -> None:
    asyncio.run(_run_search_proteome(args))


async def _run_proteome(args: argparse.Namespace) -> None:
    pipeline = AlphaPitPipeline(
        concurrency=1,
        throttle_sleep_ms=args.throttle_ms,
    )
    actual_limit = None if (args.limit is None or args.limit <= 0) else args.limit

    print("=== AlphaPit Proteome Streaming Pipeline ===")
    print(f"Target Organism:    Tax ID {args.organism} (9606 = Homo sapiens)")
    print(f"Target Limit:       {actual_limit if actual_limit else 'ALL (~20,400 reviewed)'}")
    print(f"Hardware Profile:   Single-Worker Gentle (<400MB RAM ceiling, {args.throttle_ms}ms throttle)")
    print(f"Shard Size:         {args.shard_size} structures / shard container")
    print(f"Quality Gate:       pLDDT >= {args.min_plddt}")
    print(f"Target Storage:     {settings.full_index_path}")
    print("--------------------------------------------")

    index_name = f"human_proteome_{args.organism}"
    shard_paths, metrics = await pipeline.stream_and_index_proteome(
        index_name=index_name,
        organism_tax_id=args.organism,
        limit=actual_limit,
        min_plddt=args.min_plddt,
        shard_size=args.shard_size,
        concurrency=1,
        show_progress=True,
    )

    print("\n=== Proteome Run Completed ===")
    print(f"Elapsed Time:           {metrics.elapsed_seconds:.2f} s")
    print(f"Structures Processed:   {metrics.total_structures_succeeded} / {metrics.total_structures_attempted}")
    print(f"Average Throughput:     {metrics.structures_per_second:.2f} structures/s ({metrics.patches_per_second:,.1f} vectors/s)")
    print(f"Total Shards Created:   {len(shard_paths)}")
    for sp in shard_paths:
        print(f"  Shard: {sp.name} ({sp.stat().st_size / (1024 * 1024):.2f} MB)")


def cmd_proteome(args: argparse.Namespace) -> None:
    asyncio.run(_run_proteome(args))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="alphapit",
        description="AlphaPit: End-to-end geometric deep learning on protein surfaces with PithosDB vector index",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = subparsers.add_parser("status", help="Show system, NVMe SSD, and backend status")
    p_status.set_defaults(func=cmd_status)

    # run-proteome
    p_prot = subparsers.add_parser("run-proteome", help="Run low-impact streaming proteome indexing")
    p_prot.add_argument("--organism", default="9606", help="NCBI Tax ID (default: 9606 for Homo sapiens)")
    p_prot.add_argument("--limit", type=int, default=0, help="Number of structures (0 or omit for all)")
    p_prot.add_argument("--shard-size", type=int, default=250, help="Structures per .pithos shard file")
    p_prot.add_argument("--min-plddt", type=float, default=70.0, help="Minimum pLDDT confidence filter")
    p_prot.add_argument("--throttle-ms", type=float, default=20.0, help="Inter-structure cooldown in milliseconds")
    p_prot.set_defaults(func=cmd_proteome)

    # search (single shard)
    p_srch = subparsers.add_parser("search", help="Search a single compiled PithosDB shard container")
    p_srch.add_argument("name", help="Name of the PithosDB index or shard")
    p_srch.add_argument("query_id", help="Query PDB ID or UniProt ID")
    p_srch.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbors to retrieve")
    p_srch.add_argument("--alphafold", action="store_true", help="Query is an AlphaFold ID")
    p_srch.add_argument("--format", choices=["pdb", "cif"], default="pdb", help="Structure format")
    p_srch.set_defaults(func=cmd_search)

    # search-proteome (all shards on SSD)
    p_all = subparsers.add_parser("search-proteome", help="Search across ALL compiled proteome shards on SSD")
    p_all.add_argument("query_id", help="Query PDB ID (e.g. 1M17, 1A8O, 6LU7) or UniProt ID")
    p_all.add_argument("--shard-prefix", default="human_proteome_9606_shard_", help="Prefix for shard containers")
    p_all.add_argument("--top-k", type=int, default=3, help="Top matches per surface patch")
    p_all.add_argument("--max-targets", type=int, default=10, help="Maximum ranked human targets to display")
    p_all.add_argument("--alphafold", action="store_true", help="Query is an AlphaFold UniProt ID")
    p_all.add_argument("--format", choices=["pdb", "cif"], default="pdb", help="Structure format")
    p_all.set_defaults(func=cmd_search_proteome)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
