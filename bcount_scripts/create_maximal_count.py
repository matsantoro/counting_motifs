from robust_motifs.data import create_maximal_simplex_file
from pathlib import Path

create_maximal_simplex_file(Path("data/bcounts/column/seed_0/cons_locs_pathways_mc0_Column-count.h5"),Path("data/bcounts/column/seed_0/cons_locs_pathways_mc0_Column-count-maximal.h5"), True)
