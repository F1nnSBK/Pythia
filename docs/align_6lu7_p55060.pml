
reinitialize
set bg_rgb, [1, 1, 1]
set ray_opaque_background, 0
set antialias, 2
set cartoon_transparency, 0.20
set stick_transparency, 0.0
set ray_shadows, 0
set light_count, 2
set spec_reflect, 0.15

load /Users/finnhertsch/projects/AlphaPit/data/structures/6lu7.pdb, viral_mpro_6lu7
load /Users/finnhertsch/projects/AlphaPit/data/structures/AF-P55060-F1-model_v4.pdb, human_target_p55060

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
png /Users/finnhertsch/projects/AlphaPit/docs/temp_05_global.png, dpi=300

zoom mpro_pocket, 13
orient mpro_pocket
ray 1800, 1350
png /Users/finnhertsch/projects/AlphaPit/docs/temp_05_pocket.png, dpi=300

save /Users/finnhertsch/projects/AlphaPit/docs/6lu7_p55060_alignment.pse
