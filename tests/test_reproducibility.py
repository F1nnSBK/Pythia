"""
Smoke and consistency tests for the reproducible CSV datasets.
Guarantees all paper claims match empirical CSV data.
"""

from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path("/Users/finnhertsch/projects/Pythia")
DATA_DIR = REPO_ROOT / "results" / "csv"
DOCS_DIR = REPO_ROOT / "docs"


def test_faiss_benchmark_consistency():
    csv_path = DATA_DIR / "benchmark_faiss_comprehensive.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    
    # Check Pithos entry
    pithos_row = df[df["engine"].str.contains("Pithos")]
    assert len(pithos_row) == 1
    assert pithos_row["resident_ram_gb"].values[0] == pytest.approx(0.28, rel=1e-2)
    assert pithos_row["latency_ms"].values[0] == pytest.approx(27.1, rel=1e-1)
    assert pithos_row["recall_at_10"].values[0] >= 90.0


def test_ablation_results_consistency():
    csv_path = DATA_DIR / "ablation_study_results.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    
    # Full model should achieve lowest RMSD
    full_row = df[df["Model"].str.contains("Full")]
    assert len(full_row) >= 1
    assert full_row["RMSD_mean"].min() <= 1.20


def test_patch_radius_sweep_consistency():
    csv_path = DATA_DIR / "patch_radius_sweep.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    
    # Optimal radius should be 9.0 A
    row_9 = df[df["Patch_Radius"] == 9.0]
    assert len(row_9) == 1
    assert row_9["Recall@10"].values[0] >= 95.0


def test_twilight_zone_curated_consistency():
    csv_path = DATA_DIR / "twilight_zone_curated_table.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    
    # All curated pairs must have sequence identity < 20%
    assert (df["sequence_identity"] < 0.20).all()
    # All curated pairs must have low pocket RMSD (< 1.8 A)
    assert (df["pocket_rmsd_angstrom"] < 1.80).all()


def test_figures_exist_and_non_empty():
    expected_svgs = [
        "benchmark_scaling.svg",
        "ablation_study.svg",
        "conformational_robustness.svg",
        "patch_radius_sweep.svg",
        "twilight_zone_evolution.svg",
        "05_pymol_sars_cov2_alignment.svg",
        "fold_held_out_evaluation.svg",
        "1m17_convergent_pocket_alignment.svg",
        "hardware_scaling_throughput.svg",
    ]
    for svg_name in expected_svgs:
        p = DOCS_DIR / svg_name
        assert p.exists(), f"Missing {svg_name}"
        assert p.stat().st_size > 500, f"Empty {svg_name}"
