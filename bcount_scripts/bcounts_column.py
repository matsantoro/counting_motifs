from robust_motifs.counting import bcount_from_file
from pathlib import Path

bcount_from_file(Path("data/bcounts/column/seed_0/cons_locs_pathways_mc0_Column.pkl"), 10)
