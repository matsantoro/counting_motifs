import datetime
import h5py
from itertools import combinations
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import os
from pathlib import Path
import pickle
import psutil
import scipy.sparse as sp
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from .data import MPDataManager, load_sparse_matrix_from_pkl, save_sparse_matrix_to_pkl


def worker_initializer(path):
    print("Initializing on PID" + str(os.getpid()))
    global global_matrix
    global global_bid_matrix
    global_matrix = load_sparse_matrix_from_pkl(path)
    global_bid_matrix = global_matrix.multiply(global_matrix.T)
    global_matrix = global_matrix.todense()
    global_bid_matrix = global_bid_matrix.todense()


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
    # full_matrix_info = mp_element[1]
    bidirectional_matrix_info = mp_element[2]

    # Retrieve first matrix location:
    # Actually causes loss of performance!
    # full_memory_block = SharedMemory(name=full_matrix_info['name'], size=full_matrix_info['size'])
    # full_matrix = np.ndarray(shape=full_matrix_info['shape'], dtype=full_matrix_info['type'], buffer=full_memory_block.buf)


    # Retrieve second matrix location:
    bid_memory_block = SharedMemory(name=bidirectional_matrix_info['name'], size=bidirectional_matrix_info['size'])
    bid_matrix = np.ndarray(shape=bidirectional_matrix_info['shape'], dtype=bidirectional_matrix_info['type'],
                             buffer=bid_memory_block.buf)

    # Actual computation:
    return len(set(bid_matrix[simplex[-1]].nonzero()[0]) - set(simplex[:-1]))


def get_bisimplices_dense(mp_element: Tuple[List, Dict, Dict]) -> np.ndarray:
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
            bisimplices.append(elem)
    return np.array(bisimplices, dtype=np.int16)

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


def get_n_extended_simplices_new(mp_element: Tuple[List, Dict, Dict]) -> int:
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
    bidirectional_matrix_info = mp_element[2]

    # Retrieve second matrix location:
    sm_indices = SharedMemory(name=bidirectional_matrix_info['indices']['name'])
    sm_indptr = SharedMemory(name=bidirectional_matrix_info['indptr']['name'])

    sindices = np.ndarray((int(bidirectional_matrix_info['indices']['size'] / bidirectional_matrix_info['indices']['factor']),),
                          dtype=bidirectional_matrix_info['indices']['type'],
                          buffer=sm_indices.buf)
    sindptr = np.ndarray((int(bidirectional_matrix_info['indptr']['size'] / bidirectional_matrix_info['indptr']['factor']),),
                         dtype=bidirectional_matrix_info['indptr']['type'],
                         buffer=sm_indptr.buf)


    # Actual computation:
    return len(set(sindices[sindptr[simplex[-1]]:sindptr[simplex[-1]+1]]) - set(simplex[:-1]))


def get_extended_simplices_dense(mp_element: Tuple[List, Dict, Dict]) -> np.ndarray:
    """Function that returns the list of extended simplices of a simplex.
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
    # full_matrix_info = mp_element[1]
    bidirectional_matrix_info = mp_element[2]

    # Retrieve first matrix location:
    # Actually causes loss of performance!
    # full_memory_block = SharedMemory(name=full_matrix_info['name'], size=full_matrix_info['size'])
    # full_matrix = np.ndarray(shape=full_matrix_info['shape'], dtype=full_matrix_info['type'], buffer=full_memory_block.buf)


    # Retrieve second matrix location:
    bid_memory_block = SharedMemory(name=bidirectional_matrix_info['name'], size=bidirectional_matrix_info['size'])
    bid_matrix = np.ndarray(shape=bidirectional_matrix_info['shape'], dtype=bidirectional_matrix_info['type'],
                             buffer=bid_memory_block.buf)

    # Actual computation:
    return np.array(list(set(bid_matrix[simplex[-1]].nonzero()[0]) - set(simplex[:-1])))


def get_bisimplices_clean(mp_element: List) -> np.ndarray:
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

    global global_bid_matrix
    global global_matrix

    simplex = mp_element

    bisimplices = []
    for elem in (set(global_bid_matrix[simplex[-1]].nonzero()[0]).difference(set(simplex[:-1]))):
        f = True
        signature = global_matrix[simplex[:-1]].T[elem].T
        for flag in signature:
            if not flag:
                f = False
                break
        if f:
            bisimplices.append(elem)
    return np.array(bisimplices, dtype=np.int16)


def get_extended_simplices_clean(mp_element: List) -> np.ndarray:
    """Function that returns the list of extended simplices of a simplex.
        :argument data: (tuple[List, Dict, Dict]) data to check retrieve the extended simplices.
            The first element of the tuple contains the indices of the neurons
            in the adjacency matrix.
            The second element of the tuple contains a dictionary that specifies
            the shared memory location and info of the full matrix.
            The third element of the tuple contains a dictionary that specifies the
            shared memory location and info of the bidirectional targets matrix

        :returns es_count: (int) Number of extended simplices containing this simplex.
    """
    simplex = mp_element
    global global_bid_matrix
    # Actual computation:
    return np.array(list(set(global_bid_matrix[simplex[-1]].nonzero()[0]) - set(simplex[:-1])))


class Processor:
    """Class to manage pipeline execution on files in a path.

    :argument in_path: (Path) path to execute pipeline on.
    """
    def __init__(self, in_path: Path):
        self.in_path = in_path
        self.file_list = list(self.in_path.glob("**/*.pkl"))

    def list_extended_simplices(self):
        """Produces es files for file in path."""
        pool = mp.Pool()
        print("Found " + str(len(self.file_list)) + " .pkl files.\n")
        for elem in tqdm(self.file_list, ):
            with open(elem.with_name('log.txt'),'a+') as log:
                if not elem.with_name("ES_count.npz").exists():
                    log.write(str(datetime.datetime.now()) + " Starting ES count\n")
                    motif_counts = np.zeros((7, 2), dtype=int)
                    manager = MPDataManager(elem, None)
                    log.write(str(datetime.datetime.now()) + " Instantiated manager\n")
                    dimensions = range(1, len(manager._count_file.keys())+1)
                    for dimension in dimensions:
                        log.write(str(datetime.datetime.now()) + " Started dim " + str(dimension) + "\n")
                        chunked_iterator = manager.mp_chunks(dimension=dimension)
                        array = np.empty(shape=(manager._count_file['Cells_' + str(dimension)].shape[0] * 15,),
                                         dtype=np.int16, )
                        indptr = np.empty(shape=(manager._count_file['Cells_' + str(dimension)].shape[0] + 1,),
                                          dtype=np.int32, )
                        indptr[0] = 0
                        count = 0
                        for iterator in chunked_iterator:
                            r = pool.imap(get_extended_simplices_dense, iterator, chunksize=5000)
                            for element in r:
                                count += 1
                                indptr[count] = indptr[count - 1] + len(element)
                                try:
                                    array[indptr[count - 1]:indptr[count]] = element
                                except ValueError:
                                    log.write(str(datetime.datetime.now()) + " Array too small. Extending by 20%.")
                                    log.write(str(datetime.datetime.now()) + " Before: " + str(array.nbytes))
                                    array = np.concatenate(
                                        [array, np.empty(shape=(int(array.shape[0]*0.20) + 1000,), dtype=np.int16, )]
                                    )
                                    log.write(str(datetime.datetime.now()) + " After: " + str(array.nbytes))
                                    array[indptr[count - 1]:indptr[count]] = element

                        try:
                            log.write(str(datetime.datetime.now()) + " Counted dim " + str(dimension) + "\n")
                            path1 = elem.with_name("ES_D" + str(dimension) + ".npz")
                            np.savez_compressed(open(path1, 'wb'), array[:indptr[-1]])
                            path2 = path1.with_name(path1.stem + "indptr.npz")
                            np.savez_compressed(open(path2, 'wb'), indptr)
                            log.write(str(datetime.datetime.now()) + " Saved dim " + str(dimension)+ "\n")
                        except Exception as e:
                            log.write(str(datetime.datetime.now()) + " " + str(e))
                        motif_counts[dimension - 1, 0] = len(indptr) - 1
                        motif_counts[dimension - 1, 1] = indptr[-1]
                        del array
                        del indptr
                    count_path = elem.with_name("ES_count.npz")
                    np.savez_compressed(open(count_path, 'wb'), motif_counts)
                    log.write(str(datetime.datetime.now()) + " Saved whole count.\n")
                    log.write(str(datetime.datetime.now()) + " Available memory: " +
                              str(psutil.virtual_memory().available) +
                              "  Percent used:" + str(psutil.virtual_memory().percent) + "\n")
                    manager._shut_shared_memory()
                    del manager
                    log.write(str(datetime.datetime.now()) + " Shut down memory.\n")
                    log.write(str(datetime.datetime.now()) + " Available memory: " +
                              str(psutil.virtual_memory().available) +
                              "  Percent used:" + str(psutil.virtual_memory().percent) + "\n")
                else:
                    log.write(str(datetime.datetime.now()) + " Found existing ES count. Skipping...\n")
                    pass

    def list_bisimplices(self):
        """Produces bs files for file in path."""
        pool = mp.Pool()
        print("Found " + str(len(self.file_list)) + " .pkl files.")
        for elem in tqdm(self.file_list, ):
            with open(elem.with_name('log.txt'), 'a+') as log:
                if not elem.with_name("BS_count.npz").exists():
                    log.write(str(datetime.datetime.now()) + " Starting BS count\n")
                    motif_counts = np.zeros((7, 2), dtype=int)
                    manager = MPDataManager(elem, None)
                    log.write(str(datetime.datetime.now()) + " Instantiated manager\n")
                    dimensions = range(1, len(manager._count_file.keys())+1)
                    for dimension in dimensions:
                        log.write(str(datetime.datetime.now()) + " Started dim " + str(dimension) + "\n")
                        chunked_iterator = manager.mp_chunks(dimension=dimension)
                        array = np.empty(shape=(manager._count_file['Cells_' + str(dimension)].shape[0],),
                                         dtype=np.int16, )
                        indptr = np.empty(shape=(manager._count_file['Cells_' + str(dimension)].shape[0] + 1,),
                                          dtype=np.int32, )
                        indptr[0] = 0
                        count = 0
                        for iterator in chunked_iterator:

                            r = pool.imap(get_bisimplices_dense, iterator, chunksize=5000)
                            for element in r:
                                count += 1
                                indptr[count] = indptr[count - 1] + len(element)
                                try:
                                    array[indptr[count - 1]:indptr[count]] = element
                                except ValueError:
                                    log.write(str(datetime.datetime.now()) + " Array too small. Extending by 20%.")
                                    log.write(str(datetime.datetime.now()) + " Before: " + str(array.nbytes))
                                    array = np.concatenate(
                                        [array, np.empty(shape=(int(array.shape[0] * 0.20) + 1000,), dtype=np.int16, )]
                                    )
                                    log.write(str(datetime.datetime.now()) + " After: " + str(array.nbytes))
                                    array[indptr[count - 1]:indptr[count]] = element
                        try:
                            log.write(str(datetime.datetime.now()) + " Counted dim " + str(dimension) + "\n")
                            path1 = elem.with_name("BS_D" + str(dimension) + ".npz")
                            np.savez_compressed(open(path1, 'wb'), array[:indptr[-1]])
                            path2 = path1.with_name(path1.stem + "indptr.npz")
                            np.savez_compressed(open(path2, 'wb'), indptr)
                            log.write(str(datetime.datetime.now()) + " Saved dim " + str(dimension) + "\n")
                        except Exception as e:
                            log.write(str(datetime.datetime.now()) + " " + str(e))
                        motif_counts[dimension-1, 0] = len(indptr)-1
                        motif_counts[dimension-1, 1] = indptr[-1]
                        del array
                        del indptr
                    count_path = elem.with_name("BS_count.npz")
                    np.savez_compressed(open(count_path, 'wb'), motif_counts)
                    log.write(str(datetime.datetime.now()) + " Saved whole count.\n")
                    log.write(str(datetime.datetime.now()) + " Available memory: " +
                              str(psutil.virtual_memory().available) +
                              "  Percent used:" + str(psutil.virtual_memory().percent) + "\n")
                    manager._shut_shared_memory()
                    del manager
                    log.write(str(datetime.datetime.now()) + " Shut down memory.\n")
                    log.write(str(datetime.datetime.now()) + " Available memory: " +
                              str(psutil.virtual_memory().available) +
                              "  Percent used:" + str(psutil.virtual_memory().percent) + "\n")
                else:
                    log.write(str(datetime.datetime.now()) + " Found existing BS count. Skipping...\n")
                    pass


def count_bidirectional_edges(matrix: np.ndarray, count_file: h5py.File, dimension: int):
    counts_per_dimension = {}
    for i in range(1, dimension + 1):
        counts_per_dimension.update({i: np.zeros((i + 1, i + 1))})
    for i in range(1, dimension + 1):
        try:
            simplices = count_file['Cells_' + str(i)]
            for simplex in tqdm(simplices):
                counts_per_dimension[i] += matrix[simplex].T[simplex].T
        except KeyError:
            pass
    return counts_per_dimension


def bcount_from_file(path: Path, dimension: int):
    matrix = load_sparse_matrix_from_pkl(path)
    matrix = np.array(matrix.todense())
    count_file = h5py.File(path.with_name(path.stem + "-count.h5"))
    counts_per_dimension = count_bidirectional_edges(matrix, count_file, dimension)
    with open(path.with_name("bcounts.pkl"), 'wb') as file:
        pickle.dump(counts_per_dimension, file)

def maximal_matrices_from_file(path: Path):
    matrix = load_sparse_matrix_from_pkl(path)
    mcount_file = h5py.File(path.with_name(path.stem + "-count-maximal.h5"))
    for i in range(1,len(mcount_file.keys())):
        simplices = np.array(mcount_file['Cells_' + str(i+1)])
        edges = np.unique(np.vstack(
                    [np.unique(simplices[:, x], axis = 0) for x in
                     combinations(range(simplices.shape[1]),2)]
                ), axis = 0)
                save_sparse_matrix_to_pkl(path.with_name(f'maximal_dim{i + 1}_any.pkl'),
                                sp.csr_matrix((np.ones((edges.shape[0],), dtype = bool), (edges[:,0],edges[:,1])),
                                              shape = matrix.shape)
                                  )
        edges = np.unique(np.vstack([simplices[:, [x, x+1]] for x in range(simplices.shape[1]-1)]), axis = 0)
        save_sparse_matrix_to_pkl(path.with_name(f'maximal_dim{i + 1}_spine.pkl'),
                                  sp.csr_matrix((np.ones((edges.shape[0],), dtype = bool), (edges[:,0],edges[:,1])),
                                              shape = matrix.shape)
                                  )
        edges = np.unique(simplices[:, [-2, -1]], axis = 0)
        save_sparse_matrix_to_pkl(path.with_name(f'maximal_dim{i + 1}_end.pkl'),
                                  sp.csr_matrix((np.ones((edges.shape[0],), dtype = bool), (edges[:,0],edges[:,1])),
                                              shape = matrix.shape)
                                  )
