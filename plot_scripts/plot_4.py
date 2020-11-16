from robust_motifs.data import ResultManager, load_sparse_matrix_from_pkl
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd

r = ResultManager(Path("data/ready/average"))

for dimension in range(1,7):
    path = Path("data/ready/average/cons_locs_pathways_mc0_Column")
    ends, indices = r.get_BS_count(path, dimension)
    simplex_file = h5py.File(Path(path / "cons_locs_pathways_mc0_Column-count.h5"))
    simplex_list = np.array(simplex_file['Cells_'+str(dimension)])
    matrix = load_sparse_matrix_from_pkl(path / "cons_locs_pathways_mc0_Column.pkl")
    tmatrix = matrix.T.tocsr()
    bmatrix = matrix.multiply(matrix.T)
    selection = np.random.choice(simplex_list.shape[0], np.min([50000,simplex_list.shape[0]]), replace = False)
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    for index in selection:
        simplex = simplex_list[index, :]
        sink = simplex[-1]
        in_degree = matrix.indptr[sink+1]-matrix.indptr[sink]
        out_degree = tmatrix.indptr[sink+1] - tmatrix.indptr[sink]
        bidegree = bmatrix.indptr[sink+1] - bmatrix.indptr[sink]
        es_count = indices[index+1]-indices[index]
        a1.append((in_degree,es_count))
        a2.append((out_degree, es_count))
        a3.append((in_degree+out_degree, es_count))
        a4.append((bidegree, es_count))
    a1 = pd.DataFrame(a1, columns = ["d", "m"])
    a2 = pd.DataFrame(a2, columns = ["d", "m"])
    a3 = pd.DataFrame(a3, columns = ["d", "m"])
    a4 = pd.DataFrame(a4, columns = ["d", "m"])

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    sns.scatterplot(data = a1, x = "d", y = "m", ax = ax)
    ax.set_xlabel("In degree")
    ax.set_ylabel("Bisimplices")
    ax.set_title("Bisimplices and sink in-degree dim "+str(dimension+1))
    fig1.savefig("BS-ID"+str(dimension), facecolor="white")

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    sns.scatterplot(data = a2, x = "d", y = "m", ax = ax)
    ax.set_xlabel("Out degree")
    ax.set_ylabel("Bisimplices")
    ax.set_title("Bisimplices and sink out-degree dim "+str(dimension+1))
    fig2.savefig("BS-OD"+str(dimension), facecolor = "white")

    fig3 = plt.figure()
    ax = fig3.add_subplot(111)
    sns.scatterplot(data = a3, x = "d", y = "m", ax = ax)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Bisimplices")
    ax.set_title("Bisimplices and sink degree dim "+str(dimension+1))
    fig3.savefig("BS-D"+str(dimension), facecolor = "white")

    fig4 = plt.figure()
    ax = fig4.add_subplot(111)
    sns.scatterplot(data = a4, x = "d", y = "m", ax = ax)
    ax.set_xlabel("Bidegree")
    ax.set_ylabel("Bisimplices")
    ax.set_title("Bisimplices and sink bidegree dim "+str(dimension+1))
    fig4.savefig("BS-BD"+str(dimension), facecolor = "white")
