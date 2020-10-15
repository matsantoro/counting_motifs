import numpy as np
import scipy.sparse as sp


def n_edges(matrix: np.ndarray):
    """Returns the number of edges of the graph.

    :argument matrix: (np.ndarray) adjacency matrix of the directed graph.

    :returns n_edges: (int) number of edges.
    """
    return int(matrix.sum())


def graph_directionality(matrix: np.ndarray):
    """Returns the directionality of the graph.

    :argument matrix: (np.ndarray) adjacency matrix of the directed graph.

    :returns directionality: (int) directionality.
    """
    directionality = 0
    if not matrix.shape:
        return 0
    for n in range(matrix.shape[0]):
        sd = matrix[n].sum() - matrix.T[n].sum()
        directionality += sd ** 2
    return int(directionality)


def n_components(matrix: np.ndarray):
    """Returns the number of connected components of the graph.
    :argument matrix: (np.ndarray) adjacency matrix of the directed graph.

    :returns n_components: (int) n_components of the graph."""
    return sp.csgraph.connected_components(sp.csr_matrix(matrix))[0]

