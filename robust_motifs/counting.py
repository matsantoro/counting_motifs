import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple

import time


# Currently unused.
def is_acyclic_simplex(data: Tuple[List[int], Dict]) -> bool:
    """Function that returns whether a given subset of neurons gives rise to
    an acyclic subgraph.

    :argument data: (tuple[List, Dict]) data to check acyclicity on.
        The first element of the tuple contains the indices of the neurons
        in the adjacency matrix.
        The second element of the tuple contains a dictionary that specifies
        the shared memory location of the full matrix.

    :returns flag: (bool) True if complex is acyclic."""
    simplex = data[0]
    arrays = data[1]
    sm_data = SharedMemory(name=arrays['data']['name'])
    sm_indices = SharedMemory(name=arrays['indices']['name'])
    sm_indptr = SharedMemory(name=arrays['indptr']['name'])
    sdata = np.ndarray((int(arrays['data']['size'] / arrays['data']['factor']),),
                       dtype=arrays['data']['type'],
                       buffer=sm_data.buf)
    sindices = np.ndarray((int(arrays['indices']['size'] / arrays['indices']['factor']),),
                          dtype=arrays['indices']['type'],
                          buffer=sm_indices.buf)
    sindptr = np.ndarray((int(arrays['indptr']['size'] / arrays['indptr']['factor']),),
                         dtype=arrays['indptr']['type'],
                         buffer=sm_indptr.buf)
    matrix = sp.csr_matrix((sdata, sindices, sindptr))
    submatrix = matrix[simplex].T[simplex].T
    if submatrix.sum() <= len(simplex)*(len(simplex)-1)/2:
        return True
    else:
        return False


# Currently unused.
def has_dag2_extension(data: Tuple[List[int], Dict]) -> List[np.ndarray]:
    """Function that returns all dag2 subgraph extensions of a given simplex.

    :argument data: (tuple[List, Dict]) data to check acyclicity on.
        The first element of the tuple contains the indices of the neurons
        in the adjacency matrix.
        The second element of the tuple contains a dictionary that specifies
        the shared memory location of the full matrix.

    :returns dags2: (List[np.ndarray]) list of all sets of neurons whose full subgraph
        gives rise to a dag2 motif.
    """
    simplex = data[0]
    arrays = data[1]
    sm_data = SharedMemory(name=arrays['data']['name'])
    sm_indices = SharedMemory(name=arrays['indices']['name'])
    sm_indptr = SharedMemory(name=arrays['indptr']['name'])
    sdata = np.ndarray((int(arrays['data']['size'] / arrays['data']['factor']),),
                       dtype=arrays['data']['type'],
                       buffer=sm_data.buf)
    sindices = np.ndarray((int(arrays['indices']['size'] / arrays['indices']['factor']),),
                          dtype=arrays['indices']['type'],
                          buffer=sm_indices.buf)
    sindptr = np.ndarray((int(arrays['indptr']['size'] / arrays['indptr']['factor']),),
                         dtype=arrays['indptr']['type'],
                         buffer=sm_indptr.buf)
    matrix = sp.csr_matrix((sdata, sindices, sindptr))
    targets = matrix[simplex[-1]].multiply(matrix.T[simplex[-1]]).nonzero()[1]
    # it's enough to check last face.
    dags2 = []
    for target in targets:
        if matrix[target].T[simplex[:-1]].count_nonzero() == 0:
            dags2.append(np.append(simplex, target))
    return dags2


# Currently unused, does not take into account the fact that a node can be internal.
def get_dag2_signature(data: Tuple[List, Dict]):
    """Function that returns the dag2 signature of all extensions of a simplex (if any).

        :argument data: (tuple[List, Dict]) data to check acyclicity on.
            The first element of the tuple contains the indices of the neurons
            in the adjacency matrix.
            The second element of the tuple contains a dictionary that specifies
            the shared memory location of the full matrix.

        :returns dags2: (Optional[List[Tuple]]) signature of the dag2 graph.
        """
    simplex = data[0]
    arrays = data[1]
    sm_data = SharedMemory(name=arrays['data']['name'])
    sm_indices = SharedMemory(name=arrays['indices']['name'])
    sm_indptr = SharedMemory(name=arrays['indptr']['name'])
    sdata = np.ndarray((int(arrays['data']['size'] / arrays['data']['factor']),),
                       dtype=arrays['data']['type'],
                       buffer=sm_data.buf)
    sindices = np.ndarray((int(arrays['indices']['size'] / arrays['indices']['factor']),),
                          dtype=arrays['indices']['type'],
                          buffer=sm_indices.buf)
    sindptr = np.ndarray((int(arrays['indptr']['size'] / arrays['indptr']['factor']),),
                         dtype=arrays['indptr']['type'],
                         buffer=sm_indptr.buf)
    matrix = sp.csr_matrix((sdata, sindices, sindptr))

    targets = matrix[simplex[-1]].multiply(matrix.T[simplex[-1]]).nonzero()[1]
    dags2 = []
    for target in targets:
        dags2.append(
            (
                np.append(simplex, target),
                matrix[simplex[:-1]].T[target].toarray()[0]
             )
        )
    return dags2


def get_bidirectional_targets(data: Tuple[List, Dict]):
    """Function that returns the bidirectional targets of the last nodes of a simplex (if any).

        :argument data: (tuple[List, Dict]) data to check acyclicity on.
            The first element of the tuple contains the indices of the neurons
            in the adjacency matrix.
            The second element of the tuple contains a dictionary that specifies
            the shared memory location of the full matrix.

        :returns dags2: (Optional[List[Tuple]]) signature of the dag2 graph.
    """
    element = data[0]
    arrays = data[1]
    sm_data = SharedMemory(name=arrays['data']['name'])
    sm_indices = SharedMemory(name=arrays['indices']['name'])
    sm_indptr = SharedMemory(name=arrays['indptr']['name'])
    sdata = np.ndarray((int(arrays['data']['size'] / arrays['data']['factor']),),
                       dtype=arrays['data']['type'],
                       buffer=sm_data.buf)
    sindices = np.ndarray((int(arrays['indices']['size'] / arrays['indices']['factor']),),
                          dtype=arrays['indices']['type'],
                          buffer=sm_indices.buf)
    sindptr = np.ndarray((int(arrays['indptr']['size'] / arrays['indptr']['factor']),),
                         dtype=arrays['indptr']['type'],
                         buffer=sm_indptr.buf)
    matrix = sp.csr_matrix((sdata, sindices, sindptr))
    return matrix[element].multiply(matrix.T[element]).nonzero()[1]


def retrieve_sparse_shared_matrix(matrix_info):
    # Doesn't seem to work because pickling or pass by reference?
    sm_data = SharedMemory(name=matrix_info['data']['name'])
    sm_indices = SharedMemory(name=matrix_info['indices']['name'])
    sm_indptr = SharedMemory(name=matrix_info['indptr']['name'])
    sdata = np.ndarray((int(matrix_info['data']['size'] / matrix_info['data']['factor']),),
                       dtype=matrix_info['data']['type'],
                       buffer=sm_data.buf)
    sindices = np.ndarray((int(matrix_info['indices']['size'] / matrix_info['indices']['factor']),),
                          dtype=matrix_info['indices']['type'],
                          buffer=sm_indices.buf)
    sindptr = np.ndarray((int(matrix_info['indptr']['size'] / matrix_info['indptr']['factor']),),
                         dtype=matrix_info['indptr']['type'],
                         buffer=sm_indptr.buf)
    return sp.csr_matrix((sdata, sindices, sindptr))


def get_n_extended_simplices(mp_element):
    simplex = mp_element[0]
    full_matrix_info = mp_element[1]
    bidirectional_matrix_info = mp_element[2]

    sm_data = SharedMemory(name=full_matrix_info['data']['name'])
    sm_indices = SharedMemory(name=full_matrix_info['indices']['name'])
    sm_indptr = SharedMemory(name=full_matrix_info['indptr']['name'])
    sdata = np.ndarray((int(full_matrix_info['data']['size'] / full_matrix_info['data']['factor']),),
                       dtype=full_matrix_info['data']['type'],
                       buffer=sm_data.buf)
    sindices = np.ndarray((int(full_matrix_info['indices']['size'] / full_matrix_info['indices']['factor']),),
                          dtype=full_matrix_info['indices']['type'],
                          buffer=sm_indices.buf)
    sindptr = np.ndarray((int(full_matrix_info['indptr']['size'] / full_matrix_info['indptr']['factor']),),
                         dtype=full_matrix_info['indptr']['type'],
                         buffer=sm_indptr.buf)

    full_matrix = sp.csr_matrix((sdata, sindices, sindptr))

    sm_data = SharedMemory(name=bidirectional_matrix_info['data']['name'])
    sm_indices = SharedMemory(name=bidirectional_matrix_info['indices']['name'])
    sm_indptr = SharedMemory(name=bidirectional_matrix_info['indptr']['name'])
    sdata = np.ndarray((int(bidirectional_matrix_info['data']['size'] / bidirectional_matrix_info['data']['factor']),),
                       dtype=bidirectional_matrix_info['data']['type'],
                       buffer=sm_data.buf)
    sindices = np.ndarray((int(bidirectional_matrix_info['indices']['size'] / bidirectional_matrix_info['indices']['factor']),),
                          dtype=bidirectional_matrix_info['indices']['type'],
                          buffer=sm_indices.buf)
    sindptr = np.ndarray((int(bidirectional_matrix_info['indptr']['size'] / bidirectional_matrix_info['indptr']['factor']),),
                         dtype=bidirectional_matrix_info['indptr']['type'],
                         buffer=sm_indptr.buf)

    bid_matrix = sp.csr_matrix((sdata, sindices, sindptr))
    return len(set(bid_matrix[simplex[-1]].nonzero()[1]) - set(simplex[:-1])), len(bid_matrix[simplex[-1]].nonzero()[1])


def get_bisimplices(mp_element):
    simplex = mp_element[0]
    full_matrix_info = mp_element[1]
    bidirectional_matrix_info = mp_element[2]

    sm_data_1 = SharedMemory(name=full_matrix_info['data']['name'])
    sm_indices_1 = SharedMemory(name=full_matrix_info['indices']['name'])
    sm_indptr_1 = SharedMemory(name=full_matrix_info['indptr']['name'])
    sdata = np.ndarray((int(full_matrix_info['data']['size'] / full_matrix_info['data']['factor']),),
                       dtype=full_matrix_info['data']['type'],
                       buffer=sm_data_1.buf)
    sindices = np.ndarray((int(full_matrix_info['indices']['size'] / full_matrix_info['indices']['factor']),),
                          dtype=full_matrix_info['indices']['type'],
                          buffer=sm_indices_1.buf)
    sindptr = np.ndarray((int(full_matrix_info['indptr']['size'] / full_matrix_info['indptr']['factor']),),
                         dtype=full_matrix_info['indptr']['type'],
                         buffer=sm_indptr_1.buf)

    full_matrix = sp.csr_matrix((sdata, sindices, sindptr))

    sm_data_2 = SharedMemory(name=bidirectional_matrix_info['data']['name'])
    sm_indices_2 = SharedMemory(name=bidirectional_matrix_info['indices']['name'])
    sm_indptr_2 = SharedMemory(name=bidirectional_matrix_info['indptr']['name'])
    sdata = np.ndarray((int(bidirectional_matrix_info['data']['size'] / bidirectional_matrix_info['data']['factor']),),
                       dtype=bidirectional_matrix_info['data']['type'],
                       buffer=sm_data_2.buf)
    sindices = np.ndarray((int(bidirectional_matrix_info['indices']['size'] / bidirectional_matrix_info['indices']['factor']),),
                          dtype=bidirectional_matrix_info['indices']['type'],
                          buffer=sm_indices_2.buf)
    sindptr = np.ndarray((int(bidirectional_matrix_info['indptr']['size'] / bidirectional_matrix_info['indptr']['factor']),),
                         dtype=bidirectional_matrix_info['indptr']['type'],
                         buffer=sm_indptr_2.buf)

    bid_matrix = sp.csr_matrix((sdata, sindices, sindptr))
    bisimplices = []

    for elem in (set(bid_matrix[simplex[-1]].nonzero()[1]).difference(set(simplex[:-1]))):
        f = True
        signature = full_matrix[simplex[:-1]].T[elem].T
        for flag in signature:
            if not flag:
                f = False
                break
        if f:
            bisimplices.append(tuple(simplex[:-1].tolist() + sorted([simplex[-1], elem])))
    return bisimplices


