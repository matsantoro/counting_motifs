from robust_motifs.data import ResultManager, load_sparse_matrix_from_pkl
from pathlib import Path
import numpy as np
import h5py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

p = Path("data/ready/average/cons_locs_pathways_mc0_Column")
r = ResultManager(p)
m = load_sparse_matrix_from_pkl("data/ready/average/cons_locs_pathways_mc0_Column/cons_locs_pathways_mc0_Column.pkl")
bm = m.multiply(m.T)
bidegrees = np.squeeze(np.array(bm.astype(int).sum(axis = 1)))
width = 0.5
for dimension in range(1,7):
    vertices, pointers = r.get_ES_count(p,dimension)
    diffs = pointers[1:] - pointers[:-1]
    complex_file_path = p / (p.parts[-1] + "-count.h5")
    complex_file = h5py.File(complex_file_path)
    simplex_list = np.array(complex_file['Cells_'+str(dimension)])

    values = np.zeros((81, dimension+1))

    processed_list = []
    for index in tqdm(range(simplex_list.shape[0])):
        values[bidegrees[simplex_list[index, dimension]],
            bidegrees[simplex_list[index, dimension]]-diffs[index]] += 1
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel("Number of bidirectional edges")
    pos = np.arange(0,81)
    swidth = width/values.shape[1]
    for i in range(values.shape[1]):
        ax.bar(pos - width + i*swidth, values[:,i], swidth, label = str(i) + " internal edges")
    ax.legend()
    fig.suptitle("Extended simplices of dim " + str(dimension + 1) +" per sink bidegree") 
    fig.savefig("ES_count_average_dim_near" + str(dimension), facecolor = "white")
    ax.set_yscale("log")
    fig.savefig("ES_count_average_dim_near_log" + str(dimension), facecolor = "white")
