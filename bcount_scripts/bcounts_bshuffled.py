from robust_motifs.counting import bcount_from_file
from pathlib import Path

bcount_from_file(Path("data/bcounts/underlying/seed_0/graph.pkl"), 10)
