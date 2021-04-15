from robust_motifs.data import create_maximal_simplex_file
from pathlib import Path

for file in Path().glob("data/worm/worm_control/**/*-count.h5"):
    create_maximal_simplex_file(file, file.with_name("graph-count.h5"), True)
