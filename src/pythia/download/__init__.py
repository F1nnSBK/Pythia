"""
Download and streaming parser subsystem for Pythia.
"""

from pythia.download.client import PDBStreamDownloader
from pythia.download.parser import (
    ParsedAtomRecord,
    ProteinStructureData,
    StreamingPDBParser,
    VDW_RADII_PM,
    STANDARD_AMINO_ACIDS,
)
from pythia.download.stream import (
    StreamingGzipDecompressor,
    iter_lines_from_byte_stream,
    iter_lines_from_bytes,
)

__all__ = [
    "PDBStreamDownloader",
    "StreamingPDBParser",
    "ProteinStructureData",
    "ParsedAtomRecord",
    "VDW_RADII_PM",
    "STANDARD_AMINO_ACIDS",
    "StreamingGzipDecompressor",
    "iter_lines_from_byte_stream",
    "iter_lines_from_bytes",
]
