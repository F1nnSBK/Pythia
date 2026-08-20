"""
[Paper Figures 1–8] Master Publication-Grade Figure Generator for AlphaPit.
Renders all authentic Palatino/STIX Tufte SVGs and 3D PyMOL Ribbons:
- Figure 1: Hardware Efficiency (Resident RAM vs. Latency)
- Figure 2: Ablation Study & Conformational Robustness
- Figure 3: Geodesic Patch Radius Sweep
- Figure 4: Twilight Zone Evolution Scatter
- Figure 5: SARS-CoV-2 Mpro vs. CSE1L 3D Superposition
- Figure 6: CATH Fold-Held-Out Evaluation
- Figure 7: 11-Feature Surface Fingerprint Alignment + 3D ATP-Pocket Overlay
- Figure 8: Hardware Latency & Multi-Threading Scaling
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/finnhertsch/projects/AlphaPit")
WRITING_ROOT = Path("/Users/finnhertsch/projects/writing/alphapit_bio")
DATA_DIR = REPO_ROOT / "results" / "csv"
DOCS_DIR = REPO_ROOT / "docs"
FIG_DIR = WRITING_ROOT / "figures"
PYMOL_BIN = "/Applications/PyMOL.app/Contents/MacOS/PyMOL"

# Import shared Tufte style engine
BASE_DIR = Path("/Users/finnhertsch/projects/writing")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from shared.tufte_plots.style import apply_tufte_style, align_tufte_range_spines, TUFTE_PALETTE

apply_tufte_style()


def plot_figure_01():
    df = pd.read_csv(DATA_DIR / "benchmark_faiss_comprehensive.csv")
    short_names = ["Flat (Exact)", "HNSW-32", "SQ8", "IVF,SQ8", "IVF,PQ48", "IVF,PQ64", "Pithos"]
    ram_gb = df["resident_ram_gb"].values
    latency_ms = df["latency_ms"].values
    y = np.arange(len(short_names))
    height = 0.45

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.8))

    # Panel 1: Resident RAM
    colors_ram = [TUFTE_PALETTE["hnsw"] if "Pithos" in n else TUFTE_PALETTE["pithos_tier0"] for n in short_names]
    ax1.barh(y, ram_gb, height=height, color=colors_ram, alpha=0.88, left=0.05)
    ax1.set_xscale("log")
    ax1.set_xlim(0.05, 180)
    ax1.set_xticks([0.1, 1.0, 10.0, 100.0])
    ax1.set_xticklabels(["0.1 GB", "1 GB", "10 GB", "100 GB"], fontsize=8.0)
    ax1.set_yticks(y)
    ax1.set_yticklabels(short_names, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Resident RAM Footprint (GB)", fontsize=9.0)
    align_tufte_range_spines(ax1)

    # Panel 2: Latency
    colors_lat = [TUFTE_PALETTE["hnsw"] if "Pithos" in n else TUFTE_PALETTE["ivfpq"] for n in short_names]
    ax2.barh(y, latency_ms, height=height, color=colors_lat, alpha=0.88, left=0.8)
    ax2.set_xscale("log")
    ax2.set_xlim(0.8, 450)
    ax2.set_xticks([1.0, 10.0, 100.0])
    ax2.set_xticklabels(["1 ms", "10 ms", "100 ms"], fontsize=8.0)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Query Search Latency (ms)", fontsize=9.0)
    align_tufte_range_spines(ax2)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "benchmark_scaling.svg", format="svg")
    if FIG_DIR.exists():
        fig.savefig(FIG_DIR / "01_system_ram_latency.svg", format="svg")
        fig.savefig(FIG_DIR / "01_system_ram_latency.pdf", format="pdf", dpi=300)
    plt.close(fig)


def plot_figure_02():
    df_abl = pd.read_csv(DATA_DIR / "ablation_study_results.csv")
    df_rob = pd.read_csv(DATA_DIR / "conformational_robustness.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7))

    # Panel A: Ablation RMSD
    models = ["Chem", "Curv", "HKS", "Chem+Curv", "Chem+HKS", "Curv+HKS", "21D-Base", "Pithos"]
    col = "mean_pocket_rmsd_angstrom" if "mean_pocket_rmsd_angstrom" in df_abl.columns else "mean_rmsd_angstrom"
    rmsd = df_abl[col].values
    y_pos = np.arange(len(models))

    colors_abl = [TUFTE_PALETTE["hnsw"] if "Pithos" in m else TUFTE_PALETTE["pithos_tier0"] for m in models]
    ax1.barh(y_pos, rmsd, height=0.45, color=colors_abl, alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=8.0)
    ax1.invert_yaxis()
    ax1.set_xlabel("Pocket Cavity RMSD (A)", fontsize=8.5)
    ax1.set_xlim(0.5, 2.5)
    ax1.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax1.set_xticklabels(["0.5 A", "1.0 A", "1.5 A", "2.0 A", "2.5 A"], fontsize=8.0)
    align_tufte_range_spines(ax1)

    # Panel B: Conformational Robustness
    disp = df_rob["rmsd_angstrom"].values
    ax2.plot(disp, df_rob["alphapit_full_similarity"].values, marker="o", markersize=3.5, 
             color=TUFTE_PALETTE["hnsw"], label="Pithos")
    ax2.plot(disp, df_rob["lbo_hks_intrinsic_similarity"].values, marker="s", markersize=3.5, 
             color=TUFTE_PALETTE["pithos"], linestyle="--", label="LBO-HKS")
    ax2.plot(disp, df_rob["dmasif_standard_similarity"].values, marker="^", markersize=3.5, 
             color=TUFTE_PALETTE["ivfpq"], linestyle="-.", label="dMaSIF")
    ax2.plot(disp, df_rob["euclidean_coords_similarity"].values, marker="x", markersize=3.5, 
             color=TUFTE_PALETTE["baseline_gray"], linestyle=":", label="Euclidean")

    ax2.set_xlabel("Loop Displacement (A)", fontsize=8.5)
    ax2.set_ylabel("Similarity Retention", fontsize=8.5)
    ax2.set_xlim(0.3, 4.3)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xticks([1.0, 2.0, 3.0, 4.0])
    ax2.set_xticklabels(["1.0 A", "2.0 A", "3.0 A", "4.0 A"], fontsize=8.0)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8.0)
    ax2.legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(0.98, 0.50), 
               handletextpad=0.4, labelspacing=0.35)
    align_tufte_range_spines(ax2)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "ablation_study.svg", format="svg")
    fig.savefig(DOCS_DIR / "conformational_robustness.svg", format="svg")
    if FIG_DIR.exists():
        fig.savefig(FIG_DIR / "02_ablation_and_robustness.svg", format="svg")
        fig.savefig(FIG_DIR / "02_ablation_and_robustness.pdf", format="pdf", dpi=300)
    plt.close(fig)


def plot_figure_03():
    df = pd.read_csv(DATA_DIR / "patch_radius_sweep.csv")
    r = df["patch_radius_angstrom"].values
    recall = df["recall_at_10"].values
    col = "mean_pocket_rmsd_angstrom" if "mean_pocket_rmsd_angstrom" in df.columns else "mean_rmsd_angstrom"
    rmsd = df[col].values

    fig, ax1 = plt.subplots(figsize=(5.4, 2.8))

    color_rec = TUFTE_PALETTE["hnsw"]
    color_rmsd = TUFTE_PALETTE["pithos_tier0"]

    ax1.plot(r, recall, marker="o", markersize=3.5, color=color_rec, linewidth=1.2, label="Recall@10 (%)")
    ax1.set_xlabel("Geodesic Patch Radius (A)", fontsize=9.0)
    ax1.set_ylabel("Recall@10 (%)", color=color_rec, fontsize=9.0)
    ax1.tick_params(axis="y", labelcolor=color_rec)
    ax1.set_xlim(4.5, 15.5)
    ax1.set_xticks([5, 7, 9, 11, 13, 15])
    ax1.set_xticklabels(["5 A", "7 A", "9 A", "11 A", "13 A", "15 A"], fontsize=8.0)
    ax1.set_ylim(70, 100)
    ax1.set_yticks([70, 80, 90, 100])
    ax1.set_yticklabels(["70%", "80%", "90%", "100%"], fontsize=8.0)

    ax2 = ax1.twinx()
    ax2.plot(r, rmsd, marker="s", markersize=3.5, color=color_rmsd, linewidth=1.2, linestyle="--", label="RMSD (A)")
    ax2.set_ylabel("Pocket RMSD (A)", color=color_rmsd, fontsize=9.0)
    ax2.tick_params(axis="y", labelcolor=color_rmsd)
    ax2.set_ylim(0.8, 2.6)
    ax2.set_yticks([1.0, 1.5, 2.0, 2.5])
    ax2.set_yticklabels(["1.0 A", "1.5 A", "2.0 A", "2.5 A"], fontsize=8.0)

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_color(TUFTE_PALETTE["axis_ink"])

    ax1.spines["left"].set_color(color_rec)
    ax2.spines["right"].set_color(color_rmsd)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "patch_radius_sweep.svg", format="svg")
    if FIG_DIR.exists():
        fig.savefig(FIG_DIR / "03_patch_radius_sweep.svg", format="svg")
        fig.savefig(FIG_DIR / "03_patch_radius_sweep.pdf", format="pdf", dpi=300)
    plt.close(fig)


def plot_figure_04():
    df = pd.read_csv(DATA_DIR / "twilight_zone_data.csv")
    seq_id = df["seq_identity"].values * 100.0 if df["seq_identity"].max() <= 1.0 else df["seq_identity"].values
    sim = df["pocket_similarity"].values
    regimes = df.get("regime", pd.Series(["Twilight_Zone"]*len(df))).values

    colors = []
    for r, s in zip(regimes, seq_id):
        if s < 25.0 or "Twilight" in str(r):
            colors.append(TUFTE_PALETTE["hnsw"])
        elif s < 50.0 or "Intermediate" in str(r):
            colors.append(TUFTE_PALETTE["baseline_gray"])
        else:
            colors.append(TUFTE_PALETTE["pithos"])

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.scatter(seq_id, sim, c=colors, s=24, alpha=0.85, edgecolors="none")
    ax.axvline(20.0, color=TUFTE_PALETTE["hnsw"], linestyle="--", linewidth=0.85)
    ax.axhline(0.80, color=TUFTE_PALETTE["baseline_gray"], linestyle=":", linewidth=0.8)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", label="Twilight Zone (< 20%)",
                   markerfacecolor=TUFTE_PALETTE["hnsw"], markersize=5),
        plt.Line2D([0], [0], marker="o", color="w", label="Intermediate (20–50%)",
                   markerfacecolor=TUFTE_PALETTE["baseline_gray"], markersize=5),
        plt.Line2D([0], [0], marker="o", color="w", label="Homologs (> 50%)",
                   markerfacecolor=TUFTE_PALETTE["pithos"], markersize=5),
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=7.5, loc="lower left", 
              bbox_to_anchor=(0.0, 1.02), ncol=3, columnspacing=1.0, handletextpad=0.2)

    ax.set_xlabel("Pairwise Sequence Identity (%)", fontsize=9.0)
    ax.set_ylabel("Pocket Cosine Similarity (384-D)", fontsize=9.0)
    ax.set_xlim(0, 105)
    ax.set_ylim(0.70, 1.02)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=8.0)
    ax.set_yticks([0.70, 0.80, 0.90, 1.00])
    ax.set_yticklabels(["0.70", "0.80", "0.90", "1.00"], fontsize=8.0)
    align_tufte_range_spines(ax)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "twilight_zone_evolution.svg", format="svg")
    if FIG_DIR.exists():
        fig.savefig(FIG_DIR / "04_twilight_zone_evolution.svg", format="svg")
        fig.savefig(FIG_DIR / "04_twilight_zone_evolution.pdf", format="pdf", dpi=300)
    plt.close(fig)


def plot_figure_05_3d():
    pml_path = DOCS_DIR / "align_6lu7_p55060.pml"
    pml_content = f"""
reinitialize
set bg_rgb, [1, 1, 1]
set ray_opaque_background, 0
set antialias, 2
set cartoon_transparency, 0.20
set stick_transparency, 0.0
set ray_shadows, 0
set light_count, 2
set spec_reflect, 0.15

load {REPO_ROOT}/data/structures/6lu7.pdb, viral_mpro_6lu7
load {REPO_ROOT}/data/structures/AF-P55060-F1-model_v4.pdb, human_target_p55060

hide everything
show cartoon, viral_mpro_6lu7
show cartoon, human_target_p55060

color warmpink, viral_mpro_6lu7
color deepblue, human_target_p55060

select mpro_pocket, viral_mpro_6lu7 and resi 41+143+144+145+163+164+165+166
select cse1l_pocket, human_target_p55060 and resi 231+234+238+242+245+270+274

pair_fit viral_mpro_6lu7 and resi 41+145+164+166 and name CA, human_target_p55060 and resi 231+238+245+270 and name CA

show sticks, mpro_pocket
show sticks, cse1l_pocket
color firebrick, mpro_pocket
color marine, cse1l_pocket
set stick_radius, 0.24

zoom all, 3
orient
ray 1800, 1350
png {DOCS_DIR}/temp_05_global.png, dpi=300

zoom mpro_pocket, 13
orient mpro_pocket
ray 1800, 1350
png {DOCS_DIR}/temp_05_pocket.png, dpi=300

save {DOCS_DIR}/6lu7_p55060_alignment.pse
"""
    pml_path.write_text(pml_content)
    subprocess.run([PYMOL_BIN, "-cq", str(pml_path)], check=True)

    img_global = mpimg.imread(str(DOCS_DIR / "temp_05_global.png"))
    img_pocket = mpimg.imread(str(DOCS_DIR / "temp_05_pocket.png"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), dpi=300)
    fig.patch.set_alpha(0.0)

    for ax, img, title in zip([ax1, ax2], [img_global, img_pocket], 
                             ["A  Global Architecture (Twilight Zone: 11.2% Seq Id)",
                              "B  Active-Site Pocket Cavity (RMSD: 1.55 A)"]):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=8.5, loc="left", pad=6, color=TUFTE_PALETTE["axis_ink"])

    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="w", label="Viral Target: SARS-CoV-2 Mpro (6LU7)",
                   markerfacecolor=TUFTE_PALETTE["hnsw"], markersize=7),
        plt.Line2D([0], [0], marker="s", color="w", label="Human Host Match: CSE1L Exportin (P55060)",
                   markerfacecolor=TUFTE_PALETTE["ivfpq"], markersize=7),
    ]
    fig.legend(handles=legend_elements, frameon=False, fontsize=8.0, loc="lower center", 
               bbox_to_anchor=(0.5, -0.02), ncol=2, columnspacing=2.0)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "05_pymol_sars_cov2_alignment.svg", format="svg", transparent=True, bbox_inches="tight")
    if FIG_DIR.exists():
        fig.savefig(FIG_DIR / "05_pymol_sars_cov2_alignment.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(FIG_DIR / "05_pymol_sars_cov2_alignment.pdf", format="pdf", transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)

    (DOCS_DIR / "temp_05_global.png").unlink(missing_ok=True)
    (DOCS_DIR / "temp_05_pocket.png").unlink(missing_ok=True)


def plot_figure_06():
    df = pd.read_csv(DATA_DIR / "fold_held_out_evaluation.csv")
    splits = df["split_protocol"].values
    r1 = df["recall_at_1"].values
    r5 = df["recall_at_5"].values
    r10 = df["recall_at_10"].values
    map_score = df["mAP"].values

    x = np.arange(len(splits))
    width = 0.22

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.bar(x - width, r1, width, label="Recall@1", color=TUFTE_PALETTE["pithos_tier0"], alpha=0.85)
    ax.bar(x, r5, width, label="Recall@5", color=TUFTE_PALETTE["ivfpq"], alpha=0.85)
    ax.bar(x + width, r10, width, label="Recall@10", color=TUFTE_PALETTE["hnsw"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["Random 80/20", "Pfam Family", "CATH Fold"], fontsize=8.0)
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=8.5)
    ax.set_ylim(60, 105)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_yticklabels(["60%", "70%", "80%", "90%", "100%"], fontsize=8.0)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, columnspacing=1.0)
    align_tufte_range_spines(ax)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "fold_held_out_evaluation.svg", format="svg")
    if FIG_DIR.exists():
        fig.savefig(FIG_DIR / "06_fold_held_out_evaluation.svg", format="svg")
        fig.savefig(FIG_DIR / "06_fold_held_out_evaluation.pdf", format="pdf", dpi=300)
    plt.close(fig)


def plot_figure_07():
    df = pd.read_csv(DATA_DIR / "1m17_convergent_fingerprint.csv")
    features = df["feature"].values
    query_vals = df["query_1m17_egfr"].values
    target_vals = df["target_p55060_match"].values
    y_pos = np.arange(len(features))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.scatter(query_vals, y_pos - 0.15, color=TUFTE_PALETTE["hnsw"], s=28, label="Query: 1M17 (EGFR Kinase)")
    ax.scatter(target_vals, y_pos + 0.15, color=TUFTE_PALETTE["ivfpq"], s=28, marker="s", label="Discovered Match: P55060 (CSE1L)")

    for i in range(len(features)):
        ax.plot([query_vals[i], target_vals[i]], [y_pos[i] - 0.15, y_pos[i] + 0.15], 
                color=TUFTE_PALETTE["baseline_gray"], linestyle=":", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Feature Value", fontsize=8.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(["-1.0", "-0.5", "0.0", "+0.5", "+1.0"], fontsize=8.0)
    ax.axvline(0.0, color=TUFTE_PALETTE["axis_ink"], linestyle="-", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, fontsize=8.0, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, columnspacing=1.5)
    align_tufte_range_spines(ax)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "1m17_convergent_pocket_alignment.svg", format="svg")
    if FIG_DIR.exists():
        fig.savefig(FIG_DIR / "07_fingerprint_feature_alignment.svg", format="svg")
        fig.savefig(FIG_DIR / "07_fingerprint_feature_alignment.pdf", format="pdf", dpi=300)
    plt.close(fig)


def plot_figure_08():
    df = pd.read_csv(DATA_DIR / "hardware_latency_breakdown.csv")
    df_warm = df[df["cache_state"].str.contains("Warm")].copy()

    threads = df_warm["threads"].values
    latency = df_warm["total_query_latency_ms"].values
    throughput = df_warm["throughput_million_vec_per_s"].values

    fig, ax1 = plt.subplots(figsize=(5.6, 2.8))
    color_lat = TUFTE_PALETTE["hnsw"]
    color_tp = TUFTE_PALETTE["pithos"]

    ax1.plot(threads, latency, marker="o", color=color_lat, linewidth=1.2, label="Query Latency (ms)")
    ax1.set_xlabel("CPU Core Threads", fontsize=9.0)
    ax1.set_ylabel("Query Latency (ms)", color=color_lat, fontsize=9.0)
    ax1.tick_params(axis="y", labelcolor=color_lat)
    ax1.set_xticks(threads)
    ax1.set_xlim(0.5, 17)

    ax2 = ax1.twinx()
    ax2.plot(threads, throughput, marker="s", color=color_tp, linewidth=1.2, linestyle="--", label="Throughput (M vec/s)")
    ax2.set_ylabel("Throughput (M vec/s)", color=color_tp, fontsize=9.0)
    ax2.tick_params(axis="y", labelcolor=color_tp)

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_color(TUFTE_PALETTE["axis_ink"])

    ax1.spines["left"].set_color(color_lat)
    ax2.spines["right"].set_color(color_tp)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "hardware_scaling_throughput.svg", format="svg")
    if FIG_DIR.exists():
        fig.savefig(FIG_DIR / "08_hardware_scaling_throughput.svg", format="svg")
        fig.savefig(FIG_DIR / "08_hardware_scaling_throughput.pdf", format="pdf", dpi=300)
    plt.close(fig)


def render_all_figures():
    print("Rendering Figures 1 to 8 (Tufte Palatino/STIX + 3D PyMOL Ribbons)...")
    plot_figure_01()
    plot_figure_02()
    plot_figure_03()
    plot_figure_04()
    plot_figure_05_3d()
    plot_figure_06()
    plot_figure_07()
    plot_figure_08()
    print("All Figures 1–8 successfully generated!")


if __name__ == "__main__":
    render_all_figures()
