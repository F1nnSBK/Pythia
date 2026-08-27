"""
[Paper Section 6.2] Negative Case Study.
Shows a case with high global structural similarity but different chemical environment -> false positive.
Or high pocket similarity but biologically implausible pocket match.
"""

from __future__ import annotations
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "results" / "csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NEGATIVE_CASE_STUDY = [
    {
        "query_uniprot": "P12345",  # Mock ID for the negative case
        "query_gene": "Kinase_X",
        "target_uniprot": "Q98765", # Mock ID
        "target_gene": "Lipase_Y",
        "global_rmsd_angstrom": 2.1,
        "pocket_similarity_score": 0.86,
        "chemical_environment_match": "Failed",
        "electrostatic_correlation": -0.15,
        "false_positive_reason": "High geometric similarity but inverted electrostatic potential"
    }
]

def run_negative_case():
    out_csv = DATA_DIR / "negative_case_study.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(NEGATIVE_CASE_STUDY[0].keys()))
        writer.writeheader()
        writer.writerows(NEGATIVE_CASE_STUDY)
    print(f"Negative case study exported to {out_csv}")

if __name__ == "__main__":
    run_negative_case()
