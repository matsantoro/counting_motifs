from robust_motifs.counting import bcount_from_file
from pathlib import Path

for file in Path("").glob("data/worm_control/**/graph.pkl"):
    bcount_from_file(file, 10)
