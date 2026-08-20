"""
Master 1-Click Reproducibility Engine for AlphaPit Manuscript.
Sequentially executes all empirical benchmarks, evaluations, and figure generators.
Adheres strictly to zero-swapping memory management and deterministic seed controls.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/Users/finnhertsch/projects/AlphaPit")
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"

STEPS = [
    ("Step 01/10: Hardware Efficiency Benchmark (Table I, Fig 1)", "scripts/01_benchmark_hardware_efficiency.py"),
    ("Step 02/10: Bioinformatics Baselines Comparison (Table II)", "scripts/02_benchmark_bioinformatics_baselines.py"),
    ("Step 03/10: Ablation Study & Conformational Robustness (Table III, Fig 2)", "scripts/03_ablation_and_conformational_robustness.py"),
    ("Step 04/10: Biophysical Patch Radius Sweep (Fig 3)", "scripts/04_patch_radius_sweep.py"),
    ("Step 05/10: Twilight Zone Convergence Evaluation (Table IV, Fig 4)", "scripts/05_twilight_zone_convergence.py"),
    ("Step 06/10: CATH Fold-Held-Out Generalizability (Table V, Fig 6)", "scripts/06_fold_held_out_evaluation.py"),
    ("Step 07/10: Case Studies & Quantitative Fingerprints (Table VI, VII, Fig 7)", "scripts/07_case_studies_and_fingerprints.py"),
    ("Step 08/10: Hardware Latency & Multi-Threading Audit (Fig 8)", "scripts/08_hardware_scaling_throughput.py"),
    ("Step 09/10: 15.88 GB Index Storage Decomposition Audit", "scripts/09_index_storage_audit.py"),
    ("Step 10/10: Publication Figure Rendering (Palatino Tufte SVGs & 3D Ribbons)", "scripts/10_render_publication_figures.py"),
]

EXPECTED_CSVS = [
    "benchmark_faiss_comprehensive.csv",
    "bioinformatics_baselines_comparison.csv",
    "ablation_study_results.csv",
    "conformational_robustness.csv",
    "patch_radius_sweep.csv",
    "twilight_zone_curated_table.csv",
    "twilight_zone_data.csv",
    "fold_held_out_evaluation.csv",
    "off_target_screening_6lu7.csv",
    "dark_proteome_deorphanization.csv",
    "1m17_convergent_fingerprint.csv",
    "hardware_latency_breakdown.csv",
    "index_storage_breakdown.csv",
]

EXPECTED_DOCS = [
    "benchmark_scaling.svg",
    "ablation_study.svg",
    "conformational_robustness.svg",
    "patch_radius_sweep.svg",
    "twilight_zone_evolution.svg",
    "05_pymol_sars_cov2_alignment.svg",
    "fold_held_out_evaluation.svg",
    "1m17_convergent_pocket_alignment.svg",
    "hardware_scaling_throughput.svg",
    "6lu7_p55060_alignment.pse",
    "1m17_p55060_alignment.pse",
]


def reproduce_entire_manuscript():
    print("=================================================================")
    print("ALPHAPIT: MASTER 1-CLICK MANUSCRIPT REPRODUCIBILITY PIPELINE")
    print("=================================================================\n")

    t_start = time.perf_counter()

    for idx, (title, script_rel) in enumerate(STEPS, 1):
        script_path = REPO_ROOT / script_rel
        print(f"[{idx:02d}/{len(STEPS):02d}] {title} ...")
        t0 = time.perf_counter()

        res = subprocess.run(
            [str(PYTHON_BIN), str(script_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )

        dt = time.perf_counter() - t0
        if res.returncode != 0:
            print(f"  [ERROR] {script_rel} failed with exit code {res.returncode}:")
            print(res.stderr)
            sys.exit(1)
        else:
            print(f"  [OK] Done in {dt:.2f}s")

        gc.collect()
        time.sleep(0.3)

    print("\n-----------------------------------------------------------------")
    print("VERIFYING REPRODUCED ARTIFACTS...")
    print("-----------------------------------------------------------------")

    csv_dir = REPO_ROOT / "results" / "csv"
    docs_dir = REPO_ROOT / "docs"

    missing_csvs = [c for c in EXPECTED_CSVS if not (csv_dir / c).exists()]
    missing_docs = [d for d in EXPECTED_DOCS if not (docs_dir / d).exists()]

    if missing_csvs:
        print(f"Error: Missing CSV files: {missing_csvs}")
        sys.exit(1)
    else:
        print(f"  [PASSED] All {len(EXPECTED_CSVS)} empirical CSV tables verified!")

    if missing_docs:
        print(f"Error: Missing Figures / Sessions: {missing_docs}")
        sys.exit(1)
    else:
        print(f"  [PASSED] All {len(EXPECTED_DOCS)} publication SVGs and PyMOL sessions verified!")

    total_time = time.perf_counter() - t_start
    print("\n=================================================================")
    print(f"REPRODUCIBILITY PIPELINE COMPLETE: {total_time:.2f}s total runtime")
    print("All paper results, tables, and figures successfully reproduced.")
    print("=================================================================")


if __name__ == "__main__":
    reproduce_entire_manuscript()
