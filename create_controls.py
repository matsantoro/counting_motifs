from robust_motifs.data import create_control_graphs_from_matrix
from pathlib import Path

create_control_graphs_from_matrix(5, Path("data/original/average/cons_locs_pathways_mc0_Column.h5"),
	Path("data/ready/controls_1/shuffled"),
	type = "full",
	seed = 1)

create_control_graphs_from_matrix(5, Path("data/original/average/cons_locs_pathways_mc0_Column.h5"),
	Path("data/ready/controls_1/pathway"),
	type = "pathways",
	seed = 1)

create_control_graphs_from_matrix(5, Path("data/original/average/cons_locs_pathways_mc0_Column.h5"),
	Path("data/ready/controls_1/adjusted"),
	type = "adjusted",
	seed = 1)

