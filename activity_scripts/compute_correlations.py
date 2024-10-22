from robust_motifs.counting import correlations_simplexwise
from robust_motifs.data import load_sparse_matrix_from_pkl
from pathlib import Path
import pickle
import numpy as np

data = Path("data/activity/network/")
matrix = load_sparse_matrix_from_pkl(data/ "cons_locs_pathways_mc2_Column/cons_locs_pathways_mc2_Column.pkl")
matrix = np.array(matrix.todense())
for trace in Path("data/activity/spikes/spont").glob("*"):
    gids = pickle.load(open(trace/"gids.pkl",'rb'))
    corr_matrix = pickle.load(open(trace/"pearson_correlation.pkl",'rb'))
    a = correlations_simplexwise(data/ "cons_locs_pathways_mc2_Column/cons_locs_pathways_mc2_Column-count-maximal.h5",
                             gids, 62693, 94038, corr_matrix, matrix, "all")
    pickle.dump(a, open(trace / "simplexwise_pc_all.pkl", "wb"))
