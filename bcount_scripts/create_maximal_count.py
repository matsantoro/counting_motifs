from robust_motifs.data import create_maximal_simplex_file
from pathlib import Path

create_maximal_simplex_file(Path("data/bcounts/adjusted/graph-count.h5"),Path("data/bcounts/adjusted/graph-count-maximal.h5"), True)
