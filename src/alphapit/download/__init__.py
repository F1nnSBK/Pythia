"""
Download and streaming parser subsystem for AlphaPit.
"""

from alphapit.download.client import PDBStreamDownloader
from alphapit.download.parser import (
    ParsedAtomRecord,
    ProteinStructureData,
    StreamingPDBParser,
    VDW_RADII_PM,
    STANDARD_AMINO_ACIDS,
)
from alphapit.download.stream import (
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
