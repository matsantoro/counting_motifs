from robust_motifs.counting import correlations_maximal_simplex
from robust_motifs.data import load_sparse_matrix_from_pkl
from pathlib import Path
import pickle

data = Path("data/activity/network/")
spine_files = sorted(list(data.glob("**/*maximal*spine*.pkl")))
end_files = sorted(list(data.glob("**/*maximal*end*.pkl")))
any_files = sorted(list(data.glob("**/*maximal*any*.pkl")))
spontaneous_file_list = list(Path("data/activity/spikes/spont").glob("*"))
conn_matrix = load_sparse_matrix_from_pkl("data/activity/network/cons_locs_pathways_mc2_Column/cons_locs_pathways_mc2_Column.pkl")
for elem in spontaneous_file_list:
    print(elem)
    gids = pickle.load(open(elem/ "gids.pkl", 'rb'))
    matrix = pickle.load(open(elem/ "correlation.pkl", 'rb'))
    for result in correlations_maximal_simplex(spine_files, gids, 62693, 95038, matrix, conn_matrix,'sspine'):
        print(len(result[0]))
        print(len(result[1]))
