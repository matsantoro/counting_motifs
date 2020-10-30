import numpy as np
from pathlib import Path
import scipy.sparse as sp
from typing import Tuple

from .data import save_sparse_matrix_to_pkl, flagser_count, write_flagser_file


def save_count_cyclic_graph(path: Path, n_nodes: int) -> Tuple[Path, Path, Path]:
    """Prepares cyclic graph for testing. In particular, prepares .pkl, .flag and .h5 files.

    :argument path: (Path) where to save the graph data.
    :argument n_nodes: (int) number of nodes of the graph.

    :returns path: (Path) path of .flag file.
    :returns pickle_path: (Path) path of .pkl file.
    :returns count_path: (Path) path of .h5 file.
    """
    a = sp.csr_matrix(
        np.diag(np.ones((n_nodes-1, )), 1)
    )
    a[-1, 0] = 1

    path.parent.mkdir(parents=True, exist_ok=True)

    write_flagser_file(path, a)
    save_sparse_matrix_to_pkl(path.with_suffix(".pkl"), a)
    flagser_count(path, path.with_name(path.stem + str("-count.h5")))

    return path, path.with_suffix(".pkl"), path.with_name(path.stem + str("-count.h5"))


def save_count_cyclic_extension_1_node(path: Path, n_nodes: int) -> Tuple[Path, Path, Path]:
    """Prepares cyclic graph with uninodal extension for testing.
    In particular, prepares .pkl, .flag and .h5 files.

    :argument path: (Path) where to save the graph data.
    :argument n_nodes: (int) number of nodes of the graph.

    :returns path: (Path) path of .flag file.
    :returns pickle_path: (Path) path of .pkl file.
    :returns count_path: (Path) path of .h5 file.
    """
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


def save_count_cyclic_extension(path: Path, n_nodes) -> Tuple[Path, Path, Path]:
    """Prepares cyclic graph with extension for testing.
    In particular, prepares .pkl, .flag and .h5 files.

    :argument path: (Path) where to save the graph data.
    :argument n_nodes: (int) number of nodes of the graph.

    :returns path: (Path) path of .flag file.
    :returns pickle_path: (Path) path of .pkl file.
    :returns count_path: (Path) path of .h5 file.
    """
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


def save_count_simplex_extension(path: Path, n_nodes: int) -> Tuple[Path, Path, Path]:
    """Prepares directed simplex graph with extension for testing.
    In particular, prepares .pkl, .flag and .h5 files.

    :argument path: (Path) where to save the graph data.
    :argument n_nodes: (int) number of nodes of the graph.

    :returns path: (Path) path of .flag file.
    :returns pickle_path: (Path) path of .pkl file.
    :returns count_path: (Path) path of .h5 file.
    """
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


def save_count_circulant_extension_1_node(path: Path, n_nodes: int, degree: int = 2) -> Tuple[Path, Path, Path]:
    """Prepares circulant graph with uninodal extension for testing.
    In particular, prepares .pkl, .flag and .h5 files.

    :argument path: (Path) where to save the graph data.
    :argument n_nodes: (int) number of nodes of the graph.
    :argument degree: (int) degree (order?) of the circulant graph.
    :returns path: (Path) path of .flag file.
    :returns pickle_path: (Path) path of .pkl file.
    :returns count_path: (Path) path of .h5 file.
    """
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