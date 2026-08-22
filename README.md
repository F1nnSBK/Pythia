# Pythia: Proteome-Wide 3D Pocket Search

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://www.python.org/)
[![Hardware: Consumer Ready](https://img.shields.io/badge/RAM-0.28%20GB-orange.svg)](https://github.com/F1nnSBK/Pythia)

---

## Repository Structure

```
Pythia/
├── data/
│   └── structures/              # Validated PDB files (6LU7, 1M17, AF-P55060)
├── docs/                        # Publication figures (Tufte SVGs), PyMOL sessions (.pse) & BENCHMARK_METHODOLOGY.md
├── results/
│   └── csv/                     # Curated reproducible CSV benchmark datasets
├── scripts/                     # Numbered, modular reproduction pipeline
│   ├── 01_benchmark_hardware_efficiency.py      # Table I, Fig 1 (FAISS vs Pithos)
│   ├── 02_benchmark_bioinformatics_baselines.py # Table II (BLAST, TM-align, Foldseek)
│   ├── 03_ablation_and_conformational_robustness.py # Table III, Fig 2 (8 Models & Apo/Holo)
│   ├── 04_patch_radius_sweep.py                 # Fig 3 (Radius sweep [5, 15] A)
│   ├── 05_twilight_zone_convergence.py          # Table IV, Fig 4 (25k Proteomes)
│   ├── 06_fold_held_out_evaluation.py           # Table V, Fig 6 (CATH Holdout)
│   ├── 07_case_studies_and_fingerprints.py      # Table VI, VII, Fig 7 (Off-targets)
│   ├── 08_hardware_scaling_throughput.py        # Fig 8 (Multi-threading audit)
│   ├── 09_index_storage_audit.py                # 15.88 GB layout audit
│   ├── 10_render_publication_figures.py         # Master Tufte & 3D Ribbon renderer
│   ├── mine_global_pocket_atlas.py              # Proteome-wide 103-shard atlas miner
│   └── reproduce_all.py                         # Master 1-Click Reproducibility Runner
├── src/
│   └── alphapit/                # Core modular library (Geometry, Models, Storage, Analysis)
├── tests/                       # Automated pytest test suite
├── pyproject.toml               # Package dependencies and configuration
└── README.md                    # Project documentation
```

---

## Installation

### 1. Prerequisites
- Python 3.12+
- macOS (Apple Silicon M1/M2/M3/M4) or Linux (x86_64 / aarch64)
- (Optional) PyMOL for 3D ribbon rendering

### 2. Setup Virtual Environment
Using [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `venv`:

```bash
git clone https://github.com/F1nnSBK/Pythia.git
cd Pythia

# Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

---

## 1-Click Master Reproduction

To reproduce all 13 paper tables, run all 11 benchmarks, and regenerate all 8 Palatino/STIX Tufte SVGs, run:

```bash
python scripts/reproduce_all.py
```

### Table & Figure Mapping

| Paper Item | Script | Output Dataset in `results/csv/` | Output Figure in `docs/` |
| :--- | :--- | :--- | :--- |
| **Table I & Fig 1** | `scripts/01_benchmark_hardware_efficiency.py` | `benchmark_faiss_comprehensive.csv` | `benchmark_scaling.svg` |
| **Table II** | `scripts/02_benchmark_bioinformatics_baselines.py` | `bioinformatics_baselines_comparison.csv` | — |
| **Table III & Fig 2** | `scripts/03_ablation_and_conformational_robustness.py` | `ablation_study_results.csv`, `conformational_robustness.csv` | `ablation_study.svg`, `conformational_robustness.svg` |
| **Fig 3** | `scripts/04_patch_radius_sweep.py` | `patch_radius_sweep.csv` | `patch_radius_sweep.svg` |
| **Table IV & Fig 4** | `scripts/05_twilight_zone_convergence.py` | `twilight_zone_curated_table.csv`, `twilight_zone_data.csv` | `twilight_zone_evolution.svg` |
| **Fig 5 (3D)** | `scripts/10_render_publication_figures.py` | `off_target_screening_6lu7.csv` | `05_pymol_sars_cov2_alignment.svg` |
| **Table V & Fig 6** | `scripts/06_fold_held_out_evaluation.py` | `fold_held_out_evaluation.csv` | `fold_held_out_evaluation.svg` |
| **Table VI, VII & Fig 7** | `scripts/07_case_studies_and_fingerprints.py` | `off_target_screening_6lu7.csv`, `1m17_convergent_fingerprint.csv` | `1m17_convergent_pocket_alignment.svg` |
| **Fig 8** | `scripts/08_hardware_scaling_throughput.py` | `hardware_latency_breakdown.csv` | `hardware_scaling_throughput.svg` |

---

## 3D Structural Superposition & PyMOL Sessions

Pre-compiled interactive 3D PyMOL sessions are located in `docs/`:

* **Figure 5 (SARS-CoV-2 Mpro vs. Human CSE1L):**
  ```bash
  open -a PyMOL docs/6lu7_p55060_alignment.pse
  ```
* **Figure 7 (EGFR Kinase vs. Human CSE1L ATP Pocket):**
  ```bash
  open -a PyMOL docs/1m17_p55060_alignment.pse
  ```

---

## Testing & Quality Assurance

Run the automated test suite:

```bash
pytest tests/ -v
```

---

## License

Source code and benchmark datasets are freely available under the [MIT License](LICENSE).

---
*Submitted to Bioinformatics*
