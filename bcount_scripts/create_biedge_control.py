from robust_motifs.data import create_control_graphs_from_matrix
from pathlib import Path

create_control_graphs_from_matrix(1, Path("data/original/average/cons_locs_pathways_mc0_Column.h5"),
	Path("data/bcounts/bshuffled"),
	type = "shuffled_biedges",
	seed = 1)


