
reinitialize
set bg_rgb, [1, 1, 1]
set ray_opaque_background, 0
set antialias, 2
set cartoon_transparency, 0.20
set stick_transparency, 0.0
set ray_shadows, 0
set light_count, 2
set spec_reflect, 0.15

# Load PDBs
load /Users/finnhertsch/projects/AlphaPit/data/structures/1m17.pdb, egfr_kinase_1m17
load /Users/finnhertsch/projects/AlphaPit/data/structures/AF-P55060-F1-model_v4.pdb, human_target_p55060

hide everything
show cartoon, egfr_kinase_1m17
show cartoon, human_target_p55060

# Palette
color warmpink, egfr_kinase_1m17
color deepblue, human_target_p55060

# Pocket residues (P-Loop / ATP Hinge)
select egfr_pocket, egfr_kinase_1m17 and resi 718+720+745+790+793
select cse1l_pocket, human_target_p55060 and resi 231+234+238+242+245

pair_fit egfr_kinase_1m17 and resi 718+720+745+790 and name CA, human_target_p55060 and resi 231+234+238+242 and name CA

show sticks, egfr_pocket
show sticks, cse1l_pocket
color firebrick, egfr_pocket
color marine, cse1l_pocket
set stick_radius, 0.24

zoom egfr_pocket, 13
orient egfr_pocket
ray 1800, 1350
png /Users/finnhertsch/projects/AlphaPit/docs/temp_07_pocket.png, dpi=300

save /Users/finnhertsch/projects/AlphaPit/docs/1m17_p55060_alignment.pse
