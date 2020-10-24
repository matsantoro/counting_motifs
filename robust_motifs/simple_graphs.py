import networkx
from pathlib import Path
import scipy.sparse as sp
import numpy as np

from .data import save_sparse_matrix_to_pkl, flagser_count, write_flagser_file


def save_count_cyclic_graph(path: Path, n_nodes):
    a = sp.csr_matrix(
        np.diag(np.ones((n_nodes-1, )), 1)
    )
    a[-1, 0] = 1

    path.parent.mkdir(parents=True, exist_ok=True)

    write_flagser_file(path, a)
    save_sparse_matrix_to_pkl(path.with_suffix(".pkl"), a)
    flagser_count(path, path.with_name(path.stem + str("-count.h5")))

    return path, path.with_suffix(".pkl"), path.with_name(path.stem + str("-count.h5"))


def save_count_cyclic_extension_1_node(path: Path, n_nodes):
    a = sp.csr_matrix(
        np.diag(np.ones((n_nodes - 1, )), 1)
    )
    a[-1, 0] = 1

    a = sp.hstack([a, sp.csr_matrix(np.ones((a.shape[0], 1)))])
    a = sp.vstack([a, sp.csr_matrix(np.ones((1, a.shape[1])))])
    a = a.tocsr()
    a[-1, -1] = 0

    path.parent.mkdir(parents=True, exist_ok=True)

    write_flagser_file(path, a)
    save_sparse_matrix_to_pkl(path.with_suffix(".pkl"), a)
    flagser_count(path, path.with_name(path.stem + str("-count.h5")))

    return path, path.with_suffix(".pkl"), path.with_name(path.stem + str("-count.h5"))


def save_count_cyclic_extension(path: Path, n_nodes):
    a_0 = sp.csr_matrix(
        np.diag(np.ones((n_nodes - 1, )), 1)
    )

    a_0[-1, 0] = 1

    a = sp.hstack([
            a_0, sp.csr_matrix(
                np.diag(np.ones((a_0.shape[0],)))
            )
        ])

    a = sp.vstack([a,
                   sp.hstack([
                    sp.csr_matrix(np.diag(np.ones((a_0.shape[0], )))), sp.csr_matrix(np.zeros(a_0.shape))
                    ])
                   ])

    a = a.tocsr()

    path.parent.mkdir(parents=True, exist_ok=True)

    write_flagser_file(path, a)
    save_sparse_matrix_to_pkl(path.with_suffix(".pkl"), a)
    flagser_count(path, path.with_name(path.stem + str("-count.h5")))

    return path, path.with_suffix(".pkl"), path.with_name(path.stem + str("-count.h5"))


def save_count_simplex_extension(path: Path, n_nodes):
    a_0 = sp.csr_matrix(np.ones((n_nodes, n_nodes)))
    a_0 = sp.triu(a_0, 1)
    a = sp.hstack([
        a_0, sp.csr_matrix(
            np.diag(np.ones((a_0.shape[0], )))
        )
    ])

    a = sp.vstack([a,
                   sp.hstack([
                       sp.csr_matrix(np.diag(np.ones((a_0.shape[0],)))), sp.csr_matrix(np.zeros(a_0.shape))
                   ])
                   ])

    a = a.tocsr()

    path.parent.mkdir(parents=True, exist_ok=True)

    write_flagser_file(path, a)
    save_sparse_matrix_to_pkl(path.with_suffix(".pkl"), a)
    flagser_count(path, path.with_name(path.stem + str("-count.h5")))

    return path, path.with_suffix(".pkl"), path.with_name(path.stem + str("-count.h5"))


def save_count_circulant_extension_1_node(path: Path, n_nodes, degree=2):
    a = sp.csr_matrix(
        np.diag(np.ones((n_nodes - 1,)), 1) + np.diag(np.ones((n_nodes - 2,)), 2) +
        np.tril(np.ones(n_nodes), - n_nodes + degree)
    )

    a = sp.hstack([a, sp.csr_matrix(np.ones((a.shape[0], 1)))])
    a = sp.vstack([a, sp.csr_matrix(np.ones((1, a.shape[1])))])
    a = a.tocsr()
    a[-1, -1] = 0

    path.parent.mkdir(parents=True, exist_ok=True)

    write_flagser_file(path, a)
    save_sparse_matrix_to_pkl(path.with_suffix(".pkl"), a)
    flagser_count(path, path.with_name(path.stem + str("-count.h5")))

    return path, path.with_suffix(".pkl"), path.with_name(path.stem + str("-count.h5"))