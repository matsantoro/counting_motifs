from robust_motifs.counting import total_bcount_from_file
from pathlib import Path

for file in Path("").glob("data/worm_control_nomuscle/**/graph.pkl"):
    total_bcount_from_file(file, 10)

total_bcount_from_file(Path("data/worm/full_nomuscle/graph.pkl"), 10)
