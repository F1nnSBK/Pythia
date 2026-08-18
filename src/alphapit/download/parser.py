"""
Streaming PDB and mmCIF structure parser.
Extracts 3D atomic coordinates, chemical elements, van der Waals radii, and metadata
on-the-fly without requiring full file buffering or external heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np

# Standard Van der Waals radii in picometers (pm)
VDW_RADII_PM: Dict[str, float] = {
    "H": 110.0,
    "C": 170.0,
    "N": 155.0,
    "O": 152.0,
    "F": 147.0,
    "P": 180.0,
    "S": 180.0,
    "CL": 175.0,
    "SE": 190.0,
    "BR": 185.0,
    "I": 198.0,
    "OTHER": 180.0,
}

# Standard 20 amino acid residue 3-letter codes
STANDARD_AMINO_ACIDS: Set[str] = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL",  # rare selenocysteine and pyrrolysine
}

# Solvent and water identifiers to filter out by default
SOLVENT_RESIDUES: Set[str] = {"HOH", "WAT", "H2O", "DOD", "TIP3", "SOL"}


@dataclass
class ParsedAtomRecord:
    """Represents a single parsed atom record from PDB/mmCIF."""
    index: int
    name: str
    element: str
    res_name: str
    chain_id: str
    res_seq: int
    coords: Tuple[float, float, float]
    occupancy: float = 1.0
    b_factor: float = 0.0
    is_hetero: bool = False


@dataclass
class ProteinStructureData:
    """
    Structured atomic representation of a parsed protein.
    """
    structure_id: str
    coords: np.ndarray  # Shape: (N, 3), float32
    elements: List[str]  # Length: N
    element_indices: np.ndarray  # Shape: (N,), int64 mapped to standard vocab
    radii_pm: np.ndarray  # Shape: (N,), float32
    res_names: List[str]  # Length: N
    res_indices: np.ndarray  # Shape: (N,), int64
    chain_ids: List[str]  # Length: N
    atom_names: List[str]  # Length: N
    b_factors: np.ndarray  # Shape: (N,), float32 (or pLDDT confidence)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def num_atoms(self) -> int:
        return len(self.coords)

    def filter_chains(self, allowed_chains: Set[str]) -> ProteinStructureData:
        """Filter structure data down to specific chain IDs."""
        mask = np.array([c in allowed_chains for c in self.chain_ids], dtype=bool)
        if not np.any(mask):
            raise ValueError(f"No atoms remained after filtering for chains {allowed_chains}")
        return ProteinStructureData(
            structure_id=self.structure_id,
            coords=self.coords[mask],
            elements=[e for e, m in zip(self.elements, mask) if m],
            element_indices=self.element_indices[mask],
            radii_pm=self.radii_pm[mask],
            res_names=[r for r, m in zip(self.res_names, mask) if m],
            res_indices=self.res_indices[mask],
            chain_ids=[c for c, m in zip(self.chain_ids, mask) if m],
            atom_names=[a for a, m in zip(self.atom_names, mask) if m],
            b_factors=self.b_factors[mask],
            metadata=self.metadata.copy(),
        )


class StreamingPDBParser:
    """
    High-speed streaming parser for standard PDB and mmCIF lines.
    """

    def __init__(
        self,
        element_vocab: Optional[List[str]] = None,
        include_heteroatoms: bool = False,
        ignore_waters: bool = True,
        allowed_chains: Optional[Set[str]] = None,
        min_plddt: float = 0.0,
    ) -> None:
        self.element_vocab = element_vocab or ["C", "H", "O", "N", "S", "P", "OTHER"]
        self.element_to_idx = {elem: idx for idx, elem in enumerate(self.element_vocab)}
        self.include_heteroatoms = include_heteroatoms
        self.ignore_waters = ignore_waters
        self.allowed_chains = allowed_chains
        self.min_plddt = min_plddt

    def _normalize_element(self, raw_element: str, atom_name: str) -> str:
        """Clean and normalize element symbol."""
        elem = raw_element.strip().upper()
        if not elem:
            # Infer from atom name if element column is blank
            clean_name = "".join([c for c in atom_name if c.isalpha()]).upper()
            if len(clean_name) >= 2 and clean_name[:2] in VDW_RADII_PM:
                elem = clean_name[:2]
            elif len(clean_name) >= 1 and clean_name[0] in VDW_RADII_PM:
                elem = clean_name[0]
            else:
                elem = "OTHER"
        return elem if elem in VDW_RADII_PM else "OTHER"

    def parse_pdb_line(self, line: str) -> Optional[ParsedAtomRecord]:
        """Parse a single PDB ATOM / HETATM fixed-column record."""
        record_type = line[0:6].strip()
        if record_type not in ("ATOM", "HETATM"):
            return None

        is_hetero = (record_type == "HETATM")
        if is_hetero and not self.include_heteroatoms:
            return None

        res_name = line[17:20].strip().upper()
        if self.ignore_waters and res_name in SOLVENT_RESIDUES:
            return None

        chain_id = line[21:22].strip()
        if self.allowed_chains and chain_id not in self.allowed_chains:
            return None

        try:
            atom_idx = int(line[6:11].strip())
            atom_name = line[12:16].strip()
            res_seq = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            occupancy = float(line[54:60].strip()) if len(line) >= 60 and line[54:60].strip() else 1.0
            b_factor = float(line[60:66].strip()) if len(line) >= 66 and line[60:66].strip() else 0.0
            raw_element = line[76:78].strip() if len(line) >= 78 else ""
        except (ValueError, IndexError):
            return None

        element = self._normalize_element(raw_element, atom_name)

        return ParsedAtomRecord(
            index=atom_idx,
            name=atom_name,
            element=element,
            res_name=res_name,
            chain_id=chain_id or "A",
            res_seq=res_seq,
            coords=(x, y, z),
            occupancy=occupancy,
            b_factor=b_factor,
            is_hetero=is_hetero,
        )

    def parse_cif_loop_line(
        self,
        tokens: List[str],
        col_map: Dict[str, int],
    ) -> Optional[ParsedAtomRecord]:
        """Parse a single whitespace-separated row of an mmCIF atom_site loop."""
        group_col = col_map.get("_atom_site.group_PDB")
        if group_col is not None and group_col < len(tokens):
            record_type = tokens[group_col].strip().upper()
            is_hetero = (record_type == "HETATM")
            if is_hetero and not self.include_heteroatoms:
                return None
        else:
            is_hetero = False

        res_name_col = col_map.get("_atom_site.auth_comp_id") or col_map.get("_atom_site.label_comp_id")
        res_name = tokens[res_name_col].strip().upper() if res_name_col is not None and res_name_col < len(tokens) else "UNK"
        if self.ignore_waters and res_name in SOLVENT_RESIDUES:
            return None

        chain_col = col_map.get("_atom_site.auth_asym_id") or col_map.get("_atom_site.label_asym_id")
        chain_id = tokens[chain_col].strip() if chain_col is not None and chain_col < len(tokens) else "A"
        if self.allowed_chains and chain_id not in self.allowed_chains:
            return None

        try:
            idx_col = col_map.get("_atom_site.id", 0)
            atom_idx = int(tokens[idx_col]) if idx_col < len(tokens) else 0

            name_col = col_map.get("_atom_site.auth_atom_id") or col_map.get("_atom_site.label_atom_id")
            atom_name = tokens[name_col].strip() if name_col is not None and name_col < len(tokens) else "CA"

            seq_col = col_map.get("_atom_site.auth_seq_id") or col_map.get("_atom_site.label_seq_id")
            res_seq = int(tokens[seq_col]) if seq_col is not None and seq_col < len(tokens) and tokens[seq_col] != "." else 1

            x_col = col_map.get("_atom_site.Cartn_x")
            y_col = col_map.get("_atom_site.Cartn_y")
            z_col = col_map.get("_atom_site.Cartn_z")
            if x_col is None or y_col is None or z_col is None:
                return None
            x = float(tokens[x_col])
            y = float(tokens[y_col])
            z = float(tokens[z_col])

            b_col = col_map.get("_atom_site.B_iso_or_equiv")
            b_factor = float(tokens[b_col]) if b_col is not None and b_col < len(tokens) else 0.0
            if self.min_plddt > 0.0 and b_factor < self.min_plddt:
                return None

            elem_col = col_map.get("_atom_site.type_symbol")
            raw_element = tokens[elem_col].strip() if elem_col is not None and elem_col < len(tokens) else ""
        except (ValueError, IndexError):
            return None

        element = self._normalize_element(raw_element, atom_name)

        return ParsedAtomRecord(
            index=atom_idx,
            name=atom_name,
            element=element,
            res_name=res_name,
            chain_id=chain_id,
            res_seq=res_seq,
            coords=(x, y, z),
            b_factor=b_factor,
            is_hetero=is_hetero,
        )

    def assemble_structure(
        self,
        structure_id: str,
        atom_records: List[ParsedAtomRecord],
        metadata: Optional[Dict[str, str]] = None,
    ) -> ProteinStructureData:
        """Convert collected atom records into dense vectorized tensors."""
        if not atom_records:
            raise ValueError(f"No valid atom records parsed for structure {structure_id}")

        n = len(atom_records)
        coords = np.empty((n, 3), dtype=np.float32)
        element_indices = np.empty(n, dtype=np.int64)
        radii_pm = np.empty(n, dtype=np.float32)
        res_indices = np.empty(n, dtype=np.int64)
        b_factors = np.empty(n, dtype=np.float32)

        elements: List[str] = []
        res_names: List[str] = []
        chain_ids: List[str] = []
        atom_names: List[str] = []

        other_idx = self.element_to_idx.get("OTHER", len(self.element_vocab) - 1)

        for i, atom in enumerate(atom_records):
            coords[i] = atom.coords
            element_idx = self.element_to_idx.get(atom.element, other_idx)
            element_indices[i] = element_idx
            radii_pm[i] = VDW_RADII_PM.get(atom.element, 180.0)
            res_indices[i] = atom.res_seq
            b_factors[i] = atom.b_factor

            elements.append(atom.element)
            res_names.append(atom.res_name)
            chain_ids.append(atom.chain_id)
            atom_names.append(atom.name)

        return ProteinStructureData(
            structure_id=structure_id,
            coords=coords,
            elements=elements,
            element_indices=element_indices,
            radii_pm=radii_pm,
            res_names=res_names,
            res_indices=res_indices,
            chain_ids=chain_ids,
            atom_names=atom_names,
            b_factors=b_factors,
            metadata=metadata or {},
        )

    def parse_lines_sync(
        self,
        structure_id: str,
        lines: Iterable[str],
        file_format: str = "pdb",
    ) -> ProteinStructureData:
        """Synchronously parse a stream of lines (PDB or CIF)."""
        atom_records: List[ParsedAtomRecord] = []
        fmt = file_format.lower()

        if fmt in ("pdb", "ent"):
            for line in lines:
                rec = self.parse_pdb_line(line)
                if rec:
                    atom_records.append(rec)
        elif fmt in ("cif", "mmcif"):
            in_atom_site_loop = False
            col_map: Dict[str, int] = {}
            col_idx = 0

            for line in lines:
                sline = line.strip()
                if not sline or sline.startswith("#"):
                    if in_atom_site_loop and sline.startswith("#"):
                        # Loop ended
                        in_atom_site_loop = False
                        col_map = {}
                    continue

                if sline == "loop_":
                    in_atom_site_loop = False
                    col_map = {}
                    col_idx = 0
                    continue

                if sline.startswith("_atom_site."):
                    in_atom_site_loop = True
                    col_map[sline] = col_idx
                    col_idx += 1
                    continue

                if in_atom_site_loop:
                    if sline.startswith("_"):
                        in_atom_site_loop = False
                        col_map = {}
                        continue
                    tokens = sline.split()
                    rec = self.parse_cif_loop_line(tokens, col_map)
                    if rec:
                        atom_records.append(rec)

        return self.assemble_structure(structure_id, atom_records)

    async def parse_lines_async(
        self,
        structure_id: str,
        line_stream: AsyncIterator[str],
        file_format: str = "pdb",
    ) -> ProteinStructureData:
        """Asynchronously parse lines as they stream from the network."""
        atom_records: List[ParsedAtomRecord] = []
        fmt = file_format.lower()

        if fmt in ("pdb", "ent"):
            async for line in line_stream:
                rec = self.parse_pdb_line(line)
                if rec:
                    atom_records.append(rec)
        elif fmt in ("cif", "mmcif"):
            in_atom_site_loop = False
            col_map: Dict[str, int] = {}
            col_idx = 0

            async for line in line_stream:
                sline = line.strip()
                if not sline or sline.startswith("#"):
                    if in_atom_site_loop and sline.startswith("#"):
                        in_atom_site_loop = False
                        col_map = {}
                    continue

                if sline == "loop_":
                    in_atom_site_loop = False
                    col_map = {}
                    col_idx = 0
                    continue

                if sline.startswith("_atom_site."):
                    in_atom_site_loop = True
                    col_map[sline] = col_idx
                    col_idx += 1
                    continue

                if in_atom_site_loop:
                    if sline.startswith("_"):
                        in_atom_site_loop = False
                        col_map = {}
                        continue
                    tokens = sline.split()
                    rec = self.parse_cif_loop_line(tokens, col_map)
                    if rec:
                        atom_records.append(rec)

        return self.assemble_structure(structure_id, atom_records)
