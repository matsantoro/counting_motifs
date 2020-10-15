import h5py
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
import networkx
from tqdm import tqdm
from typing import List, Optional, Union
import scipy.sparse as sp


def import_connectivity_matrix(path: Path = Path('data/test/cons_locs_pathways_mc0_Column.h5'),
                        zones: Optional[List[str]] = None,
                        dataframe: bool = True,
                        type: Optional[str] = None) -> Union[np.ndarray, pd.DataFrame]:
    """Imports the connectivity matrix of the BBP model.

    :argument path: (Path) Path to load the data from.
    :argument zones: (Optional[List[str]]) List of zone strings to load.
    :parameter dataframe: (bool) Whether to return the matrix in dataframe form.
    :parameter type: (Optional[str]) Sparse matrix format to return. Either 'coo',
        'csr' or 'csc'.

    :returns matix: (Union[np.ndarray, pd.DataFrame]) Connectivity matrix.
    """
    file = h5py.File(path, 'r')
    if zones is None:
        zones = list(file['populations'].keys())
    assert np.all([zone in list(file['populations'].keys()) for zone in zones])
    n_elements = []
    for element1 in tqdm(zones):
        conn_row = sp.csr_matrix(file['connectivity'][element1][zones[0]]['cMat'][:, :], dtype=bool)
        n_elements += [conn_row.shape[0]]
        for element2 in zones[1:]:
            block = sp.csr_matrix(file['connectivity'][element1][element2]['cMat'][:, :], dtype=bool)
            conn_row = sp.hstack([conn_row, block])
        try:
            matrix = sp.vstack([matrix, conn_row])
        except Exception:
            matrix = conn_row

    if not dataframe:
        if type is None or type == 'coo':
            return matrix
        elif type == 'csr':
            return matrix.tocsr()
        elif type == 'csc':
            return matrix.tocsc()
        else:
            print('Structure not recognized. coo matrix is returned.')
            return matrix
    else:
        df = pd.DataFrame.sparse.from_spmatrix(matrix)
        index1 = np.repeat(zones, n_elements)
        index2 = []
        for n in n_elements:
            index2 = index2 + list(range(n))
        df.columns = [index1, index2]
        df.index = [index1, index2]
        return df


def save_er_graph(path: Path, n_nodes: int, density: float):
    """Saves the ER graph as a flagser-readable file, and pickle-dumps
    the adjacency matrix.

    :argument path: (Path) path of flagser file to be written.
    :argument n_nodes: (int) number of nodes of the graph.
    :argument density: (float) graph edge density."""
    g = networkx.fast_gnp_random_graph(n_nodes, density, directed=True)
    a = networkx.adjacency_matrix(g)
    with open(path, "w") as f:
        f.write("dim 0\n")
        for _ in tqdm(range(n_nodes)):
            f.write("0 ")
        f.write("\n")
        f.write("dim 1\n")
        for row, col in tqdm(zip(*a.nonzero())):
            f.write(str(row) + " " + str(col) + "\n")

    with open(path.with_suffix(".pkl"), "wb") as file:
        pickle.dump(
            {'data': a.data,
             'indices': a.indices,
             'indptr': a.indptr},
            file
        )


def load_sparse_matrix(path: Path):
    with open(path, 'rb') as file:
        dictionary = pickle.load(file)
        return sp.csr_matrix((dictionary['data'],
                             dictionary['indices'],
                             dictionary['indptr']))
