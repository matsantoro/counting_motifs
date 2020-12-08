from robust_motifs.data import save_count_graph_from_matrix, create_digraphs
import scipy.sparse as sp
import numpy as np
from pathlib import Path
from datetime import datetime

dimension = 99
with open(Path("log.txt"), 'a+') as f:
    for p in range(70,121,10):
        f.write(str(datetime.now()) + "  Started p " + str(p) + "\n")
        path = Path("data/bcounts/test_4/dim"+str(dimension)+"/ee"+str(p))
        create_digraphs(dimension, 1500, 100, p, path)
