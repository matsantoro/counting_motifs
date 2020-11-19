from robust_motifs.data import ResultManager, load_sparse_matrix_from_pkl
from pathlib import Path
import numpy as np
import h5py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

p = Path("data/ready/controls_1/adjusted/seed_0")
r = ResultManager(p)
m = load_sparse_matrix_from_pkl("data/ready/controls_1/adjusted/seed_0/graph.pkl")
bm = m.multiply(m.T)
bidegrees = np.squeeze(np.array(bm.astype(int).sum(axis = 1)))

for dimension in range(1,7):
    vertices, pointers = r.get_ES_count(p,dimension)
    diffs = pointers[1:] - pointers[:-1]
    complex_file_path = p / ("graph-count.h5")
    complex_file = h5py.File(complex_file_path)
    simplex_list = np.array(complex_file['Cells_'+str(dimension)])

    values = np.zeros((80, dimension+1))

    processed_list = []
    for index in tqdm(range(simplex_list.shape[0])):
        values[bidegrees[simplex_list[index, dimension]],
            bidegrees[simplex_list[index, dimension]]-diffs[index]] += 1
    
    fig, axes = plt.subplots(dimension+1, 1, sharex = True)
    for i, ax in enumerate(axes):
        ax.bar(range(1,81),values[:,i])
        if dimension<4:
            ax.set_title(str(i) + " internal edges")
    axes[-1].set_xlabel("Number of bidirectional edges")
    fig.suptitle("Extended simplices of dim " + str(dimension + 1) +" per sink bidegree") 
    fig.savefig("ES_count_adj_dim" + str(dimension), facecolor = "white")
    
