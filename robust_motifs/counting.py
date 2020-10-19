import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple


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


def get_element_targets(data: Tuple[List, Dict]):
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


def check_base(base, simplex_dictionary, targets):
    for target in targets:
        for simplex in simplex_dictionary.get(target, []):
            if np.all(np.array(base) == np.array(simplex[:-1])):
                yield simplex


def get_bisimplices(lists):
    list1 = lists[0]
    list2 = lists[1]
    for simplex1 in list1:
        for simplex2 in list2:
            if np.all(simplex1[:-1] == simplex2[:-1]):
                yield (simplex1, simplex2)
