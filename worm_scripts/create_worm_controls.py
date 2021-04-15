from robust_motifs.data import create_control_graphs_from_matrix
from pathlib import Path

create_control_graphs_from_matrix(100, Path("data/worm/full/graph.pkl"),
	Path("data/worm_control/bishuffled"),
	type = "shuffled_biedges",
	seed = 1)

create_control_graphs_from_matrix(100, Path("data/worm/full/graph.pkl"),
	Path("data/worm_control/underlying"),
	type = "underlying",
	seed = 1)

create_control_graphs_from_matrix(100, Path("data/worm/full/graph.pkl"),
	Path("data/worm_control/AER"),
	type = "adjusted",
	seed = 1)

