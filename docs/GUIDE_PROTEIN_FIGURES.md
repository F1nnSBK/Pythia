# Mini-Anleitung: 3D-Protein-Abbildungen (Figure 5 & Figure 7) im Tufte-Layout erstellen

Diese Anleitung dokumentiert die automatisierte und interaktive Erstellung der publikationsreifen 3D-Strukturüberlagerungen für das Manuskript.

---

## 1. Voraussetzungen & Pfade

* **PyMOL macOS Binary:** `/Applications/PyMOL.app/Contents/MacOS/PyMOL`
* **Python Environment:** `.venv/bin/python` (in `Pythia` oder `writing`)
* **PDB-Strukturen:**
  * SARS-CoV-2 $M^{\text{pro}}$: `/Users/finnhertsch/projects/Pythia/data/structures/6lu7.pdb`
  * Human CSE1L (Host Match): `/Users/finnhertsch/projects/Pythia/data/structures/AF-P55060-F1-model_v4.pdb`
  * EGFR Kinase (ATP-Tasche): `/Users/finnhertsch/projects/Pythia/data/structures/1m17.pdb`

---

## 2. Der 1-Klick-Befehl (Vollautomatisch)

Um alle 3D-Abbildungen (Figure 5 & Figure 7) headless mit transparentem Alphakanal zu rendern und als Vektor-SVG sowie 300-DPI-PDF für LaTeX zu exportieren:

```bash
cd /Users/finnhertsch/projects/Pythia
.venv/bin/python scripts/render_publication_3d_figures.py
```

### Was das Skript im Hintergrund macht:
1. **PyMOL Headless Ray-Tracing:**
   * Setzt `ray_opaque_background, 0` für $100\,\%$ transparenten Hintergrund.
   * Berechnet das lokale `pair_fit` auf den katalytischen Taschenatomen (CA).
   * Erzeugt zwei Kamerasichten: **Global Architecture** (`zoom all, 3`) und **Pocket Detail** (`zoom mpro_pocket, 13`).
2. **Matplotlib / Tufte-Integration:**
   * Bindet die transparenten 3D-Strukturen in ein 2-Panel-Vektor-Canvas ein.
   * Verwendet die Serifenschriftart **Palatino / STIX** aus `writing/shared/tufte_plots/style.py`.
   * Exportiert direkt nach `docs/` und `writing/pythia_bio/figures/`.

---

## 3. Interaktives Anpassen in der PyMOL GUI

Falls ein Agent oder Autor den Blickwinkel, Farben oder Zoom manuell verändern möchte:

1. **PyMOL mit fertiger Session öffnen:**
   ```bash
   open -a PyMOL /Users/finnhertsch/projects/Pythia/docs/6lu7_p55060_alignment.pse
   ```
2. **Oder das Automatisierungs-Skript in der PyMOL-Konsole ausführen:**
   In die `PyMOL >`-Befehlszeile eingeben:
   ```pml
   @/Users/finnhertsch/projects/Pythia/docs/align_6lu7_p55060.pml
   ```

---

## 4. Kern-Befehle für das PyMOL-Skript (Referenz)

```pml
reinitialize
set bg_rgb, [1, 1, 1]
set ray_opaque_background, 0    # WICHTIG: Transparenter Hintergrund für SVG/PDF
set antialias, 2
set cartoon_transparency, 0.20
set ray_shadows, 0

# Strukturen laden
load /Users/finnhertsch/projects/Pythia/data/structures/6lu7.pdb, viral_mpro_6lu7
load /Users/finnhertsch/projects/Pythia/data/structures/AF-P55060-F1-model_v4.pdb, human_target_p55060

# Farbpalette (Tufte Rust & Marine)
color warmpink, viral_mpro_6lu7
color deepblue, human_target_p55060

# Katalytische Taschenatome auswählen & überlagern
select mpro_pocket, viral_mpro_6lu7 and resi 41+143+144+145+163+164+165+166
select cse1l_pocket, human_target_p55060 and resi 231+234+238+242+245+270+274
pair_fit viral_mpro_6lu7 and resi 41+145+164+166 and name CA, human_target_p55060 and resi 231+238+245+270 and name CA

# Sticks für Interaktionszentren
show sticks, mpro_pocket
show sticks, cse1l_pocket
color firebrick, mpro_pocket
color marine, cse1l_pocket
set stick_radius, 0.24

# Session speichern
save /Users/finnhertsch/projects/Pythia/docs/6lu7_p55060_alignment.pse
```

---

## 5. Einbindung in `main.tex` (LaTeX)

Die fertigen PDFs liegen im Writing-Ordner:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{figures/05_pymol_sars_cov2_alignment.pdf}
  \caption{3D structural superposition of SARS-CoV-2 main protease (6LU7 $M^{\text{pro}}$, pink) and human nuclear exporter CSE1L (P55060, blue). (A) Global architectures diverge in the Twilight Zone (11.2\% sequence identity). (B) Catalytic binding cavities align with $1.55$~\AA\ RMSD.}
  \label{fig:pymol_alignment}
\end{figure}
```
