from robust_motifs.counting import total_bcount_from_file
from pathlib import Path

for file in Path("").glob("data/bcounts/bshuffled_1/**/graph.pkl"):
    total_bcount_from_file(file, 10)

for file in Path("").glob("data/bcounts/underlying_1/**/graph.pkl"):
    total_bcount_from_file(file, 10)

total_bcount_from_file(Path("data/bcounts/column/seed_0/cons_locs_pathways_mc0_Column.pkl"), 10)
