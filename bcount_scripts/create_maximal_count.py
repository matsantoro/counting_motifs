from robust_motifs.data import create_maximal_simplex_file
from pathlib import Path

for path in Path("").glob("data/bcounts/bshuffled_1/**/graph.pkl"):
    create_maximal_simplex_file(path.with_name("graph-count.h5"), path.with_name("graph-count-maximal.h5"), True)

for path in Path("").glob("data/bcounts/underlying_1/**/graph.pkl"):
    create_maximal_simplex_file(path.with_name("graph-count.h5"), path.with_name("graph-count-maximal.h5"), True)
