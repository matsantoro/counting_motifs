from multiprocessing.shared_memory import SharedMemory
import numpy as np
import scipy.sparse as sp
from typing import Any, Dict, List, Tuple


def get_bidirectional_targets(data: Tuple[List, Dict]):
    """Function that returns the bidirectional targets of an index (if any).

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


def retrieve_sparse_shared_matrix(matrix_info: Dict[str, Any]):
    # Doesn't seem to work because pickling or pass by reference?
    # I reduced to rewriting this thing in each function.
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


def get_n_extended_simplices(mp_element: Tuple[List, Dict, Dict]) -> int:
    """Function that returns the number of extended simplices of a simplex.
        :argument data: (tuple[List, Dict, Dict]) data to check retrieve the extended simplices.
            The first element of the tuple contains the indices of the neurons
            in the adjacency matrix.
            The second element of the tuple contains a dictionary that specifies
            the shared memory location and info of the full matrix.
            The third element of the tuple contains a dictionary that specifies the
            shared memory location and info of the bidirectional targets matrix

        :returns es_count: (int) Number of extended simplices containing this simplex.
    """
    simplex = mp_element[0]
    full_matrix_info = mp_element[1]
    bidirectional_matrix_info = mp_element[2]

    # Retrieve first matrix location:
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

    # TODO : implement 1-matrix version
    full_matrix = sp.csr_matrix((sdata, sindices, sindptr))

    # Retrieve second matrix location:
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

    # Actual computation:
    return len(set(bid_matrix[simplex[-1]].nonzero()[1]) - set(simplex[:-1]))


def get_bisimplices(mp_element: Tuple[List, Dict, Dict]) -> List[Tuple[int]]:
    """Function that returns the list of bisimplices of a simplex.
        :argument data: (tuple[List, Dict, Dict]) data to check retrieve the extended simplices.
            The first element of the tuple contains the indices of the neurons
            in the adjacency matrix.
            The second element of the tuple contains a dictionary that specifies
            the shared memory location and info of the full matrix.
            The third element of the tuple contains a dictionary that specifies the
            shared memory location and info of the bidirectional targets matrix

        :returns bisimplices: List[Tuple[int]] List of bisimplices indices in tuple form, with
            the extra 2-clique as an ordered pair (for duplicate checking).
    """
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


def get_extended_simplices_with_signature(mp_element):
    """Function that returns the list of bisimplices of a simplex.
        :argument data: (tuple[List, Dict, Dict]) data to check retrieve the extended simplices.
            The first element of the tuple contains the indices of the neurons
            in the adjacency matrix.
            The second element of the tuple contains a dictionary that specifies
            the shared memory location and info of the full matrix.
            The third element of the tuple contains a dictionary that specifies the
            shared memory location and info of the bidirectional targets matrix

        :returns bisimplices: List[List[int]] List of extended simplices indices in list form.
    """
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
    sindices = np.ndarray(
        (int(bidirectional_matrix_info['indices']['size'] / bidirectional_matrix_info['indices']['factor']),),
        dtype=bidirectional_matrix_info['indices']['type'],
        buffer=sm_indices_2.buf)
    sindptr = np.ndarray(
        (int(bidirectional_matrix_info['indptr']['size'] / bidirectional_matrix_info['indptr']['factor']),),
        dtype=bidirectional_matrix_info['indptr']['type'],
        buffer=sm_indptr_2.buf)

    bid_matrix = sp.csr_matrix((sdata, sindices, sindptr))
    ext_simplices = []
    signatures = []
    for elem in set(bid_matrix[simplex[-1]].nonzero()[1]) - set(simplex[:-1]):
        signature = full_matrix[simplex[:-1]].T[elem].T
        ext_simplices.append(simplex.tolist() + [elem])
        signatures.append(signature)
    return ext_simplices, signatures


def get_n_extended_simplices_dense(mp_element: Tuple[List, Dict, Dict]) -> int:
    """Function that returns the number of extended simplices of a simplex.
        :argument data: (tuple[List, Dict, Dict]) data to check retrieve the extended simplices.
            The first element of the tuple contains the indices of the neurons
            in the adjacency matrix.
            The second element of the tuple contains a dictionary that specifies
            the shared memory location and info of the full matrix.
            The third element of the tuple contains a dictionary that specifies the
            shared memory location and info of the bidirectional targets matrix

        :returns es_count: (int) Number of extended simplices containing this simplex.
    """
    simplex = mp_element[0]
    full_matrix_info = mp_element[1]
    bidirectional_matrix_info = mp_element[2]

    # Retrieve first matrix location:
    full_memory_block = SharedMemory(name=full_matrix_info['name'], size=full_matrix_info['size'])
    full_matrix = np.ndarray(shape=full_matrix_info['shape'], dtype=full_matrix_info['type'], buffer=full_memory_block.buf)


    # Retrieve second matrix location:
    bid_memory_block = SharedMemory(name=bidirectional_matrix_info['name'], size=bidirectional_matrix_info['size'])
    bid_matrix = np.ndarray(shape=bidirectional_matrix_info['shape'], dtype=bidirectional_matrix_info['type'],
                             buffer=bid_memory_block.buf)

    # Actual computation:
    return len(set(bid_matrix[simplex[-1]].nonzero()[0]) - set(simplex[:-1]))


def get_bisimplices_dense(mp_element: Tuple[List, Dict, Dict]) -> List[Tuple[int]]:
    """Function that returns the list of bisimplices of a simplex.
        :argument data: (tuple[List, Dict, Dict]) data to check retrieve the extended simplices.
            The first element of the tuple contains the indices of the neurons
            in the adjacency matrix.
            The second element of the tuple contains a dictionary that specifies
            the shared memory location and info of the full matrix.
            The third element of the tuple contains a dictionary that specifies the
            shared memory location and info of the bidirectional targets matrix

        :returns bisimplices: List[Tuple[int]] List of bisimplices indices in tuple form, with
            the extra 2-clique as an ordered pair (for duplicate checking).
    """
    simplex = mp_element[0]
    full_matrix_info = mp_element[1]
    bidirectional_matrix_info = mp_element[2]

    # Retrieve first matrix location:
    full_memory_block = SharedMemory(name=full_matrix_info['name'], size=full_matrix_info['size'])
    full_matrix = np.ndarray(shape=full_matrix_info['shape'], dtype=full_matrix_info['type'],
                             buffer=full_memory_block.buf)

    # Retrieve second matrix location:
    bid_memory_block = SharedMemory(name=bidirectional_matrix_info['name'], size=bidirectional_matrix_info['size'])
    bid_matrix = np.ndarray(shape=bidirectional_matrix_info['shape'], dtype=bidirectional_matrix_info['type'],
                            buffer=bid_memory_block.buf)

    bisimplices = []
    for elem in (set(bid_matrix[simplex[-1]].nonzero()[0]).difference(set(simplex[:-1]))):
        f = True
        signature = full_matrix[simplex[:-1]].T[elem].T
        for flag in signature:
            if not flag:
                f = False
                break
        if f:
            bisimplices.append(tuple(simplex[:-1].tolist() + sorted([simplex[-1], elem])))
    return bisimplices

def count_bisimplices_dense(mp_element: Tuple[List, Dict, Dict]) -> List[Tuple[int]]:
    """Function that returns the list of bisimplices of a simplex.
        :argument data: (tuple[List, Dict, Dict]) data to check retrieve the extended simplices.
            The first element of the tuple contains the indices of the neurons
            in the adjacency matrix.
            The second element of the tuple contains a dictionary that specifies
            the shared memory location and info of the full matrix.
            The third element of the tuple contains a dictionary that specifies the
            shared memory location and info of the bidirectional targets matrix

        :returns bisimplices: List[Tuple[int]] List of bisimplices indices in tuple form, with
            the extra 2-clique as an ordered pair (for duplicate checking).
    """
    simplex = mp_element[0]
    full_matrix_info = mp_element[1]
    bidirectional_matrix_info = mp_element[2]

    # Retrieve first matrix location:
    full_memory_block = SharedMemory(name=full_matrix_info['name'], size=full_matrix_info['size'])
    full_matrix = np.ndarray(shape=full_matrix_info['shape'], dtype=full_matrix_info['type'],
                             buffer=full_memory_block.buf)

    # Retrieve second matrix location:
    bid_memory_block = SharedMemory(name=bidirectional_matrix_info['name'], size=bidirectional_matrix_info['size'])
    bid_matrix = np.ndarray(shape=bidirectional_matrix_info['shape'], dtype=bidirectional_matrix_info['type'],
                            buffer=bid_memory_block.buf)

    count = 0
    for elem in (set(bid_matrix[simplex[-1]].nonzero()[0]).difference(set(simplex[:-1]))):
        f = True
        signature = full_matrix[simplex[:-1]].T[elem].T
        for flag in signature:
            if not flag:
                f = False
                break
        if f:
            count += 1
    return count
