"""
Unit and integration tests for streaming download, decompression, and parsing.
"""

import gzip
import pytest
import numpy as np

from alphapit.download.stream import (
    StreamingGzipDecompressor,
    iter_lines_from_byte_stream,
    iter_lines_from_bytes,
)
from alphapit.download.parser import (
    StreamingPDBParser,
    ProteinStructureData,
)
from alphapit.download.client import PDBStreamDownloader


SAMPLE_PDB_CONTENT = """HEADER    PROTEIN                                 18-AUG-26   1XYZ              
ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00           N  
ATOM      2  CA  ALA A   1      12.560  13.500  10.200  1.00 20.00           C  
ATOM      3  C   ALA A   1      13.400  12.300  10.500  1.00 20.00           C  
ATOM      4  O   ALA A   1      13.000  11.150  10.400  1.00 20.00           O  
ATOM      5  CB  ALA A   1      12.800  14.600   9.150  1.00 20.00           C  
ATOM      6  N   GLY A   2      14.650  12.600  10.900  1.00 22.00           N  
ATOM      7  CA  GLY A   2      15.550  11.550  11.300  1.00 22.00           C  
ATOM      8  C   GLY A   2      16.900  12.100  11.750  1.00 22.00           C  
ATOM      9  O   GLY A   2      17.200  13.250  11.500  1.00 22.00           O  
HETATM   10  O   HOH A 101      18.000  15.000  12.000  1.00 30.00           O  
END
"""

SAMPLE_CIF_CONTENT = """data_1XYZ
# 
loop_
_atom_site.group_PDB 
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_seq_id 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z 
_atom_site.occupancy 
_atom_site.B_iso_or_equiv 
_atom_site.auth_seq_id 
_atom_site.auth_comp_id 
_atom_site.auth_asym_id 
_atom_site.auth_atom_id 
ATOM 1 N N ALA A 1 11.104 13.207 10.000 1.0 20.00 1 ALA A N 
ATOM 2 C CA ALA A 1 12.560 13.500 10.200 1.0 20.00 1 ALA A CA 
ATOM 3 C C ALA A 1 13.400 12.300 10.500 1.0 20.00 1 ALA A C 
ATOM 4 O O ALA A 1 13.000 11.150 10.400 1.0 20.00 1 ALA A O 
ATOM 5 C CB ALA A 1 12.800 14.600 9.150 1.0 20.00 1 ALA A CB 
#
"""


def test_streaming_gzip_decompressor():
    data = b"Hello, molecular streaming world with AlphaPit!"
    compressed = gzip.compress(data)

    decompressor = StreamingGzipDecompressor(buffer_size=8)
    out = bytearray()
    for i in range(0, len(compressed), 4):
        chunk = compressed[i : i + 4]
        out.extend(decompressor.decompress_chunk(chunk))
    out.extend(decompressor.flush())

    assert bytes(out) == data


@pytest.mark.asyncio
async def test_iter_lines_from_byte_stream():
    lines = [f"ATOM line {i}" for i in range(100)]
    payload = "\n".join(lines).encode("utf-8")
    compressed = gzip.compress(payload)

    async def fake_stream():
        for i in range(0, len(compressed), 16):
            yield compressed[i : i + 16]

    result_lines = []
    async for line in iter_lines_from_byte_stream(fake_stream(), is_gzipped=True):
        result_lines.append(line)

    assert result_lines == lines


def test_parser_pdb_sync():
    parser = StreamingPDBParser(ignore_waters=True)
    struct = parser.parse_lines_sync("1XYZ", SAMPLE_PDB_CONTENT.splitlines(), file_format="pdb")

    # 9 protein atoms, 1 water ignored
    assert struct.num_atoms == 9
    assert struct.coords.shape == (9, 3)
    assert len(struct.elements) == 9
    assert struct.elements[0] == "N"
    assert struct.elements[1] == "C"
    assert struct.radii_pm[0] == 155.0  # N
    assert struct.radii_pm[1] == 170.0  # C
    assert struct.chain_ids[0] == "A"
    assert struct.res_names[0] == "ALA"


def test_parser_cif_sync():
    parser = StreamingPDBParser()
    struct = parser.parse_lines_sync("1XYZ", SAMPLE_CIF_CONTENT.splitlines(), file_format="cif")

    assert struct.num_atoms == 5
    assert struct.coords.shape == (5, 3)
    assert struct.elements[0] == "N"
    assert struct.elements[1] == "C"


@pytest.mark.asyncio
async def test_live_rcsb_streaming():
    downloader = PDBStreamDownloader()
    async with downloader:
        # 1A8O is a small, stable test structure (Cytochrome C-551)
        struct = await downloader.stream_and_parse_rcsb("1a8o", file_format="pdb")
        assert struct.num_atoms > 500
        assert struct.coords.shape[1] == 3
        assert len(struct.elements) == struct.num_atoms
