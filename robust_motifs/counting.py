import datetime
import h5py
from itertools import combinations
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
from pathlib import Path
import pickle
import psutil
import scipy.sparse as sp
from tqdm import tqdm
from typing import Dict, List, Tuple

from .data import MPDataManager, load_sparse_matrix_from_pkl


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
    bidirectional_matrix_info = mp_element[2]

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

        :returns bisimplices: (np.ndarray) array of extra nodes that turn the simplex into a bisimplex.
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
    """Function that returns the counts of bisimplices of a simplex.
        :argument data: (tuple[List, Dict, Dict]) data to check retrieve the extended simplices.
            The first element of the tuple contains the indices of the neurons
            in the adjacency matrix.
            The second element of the tuple contains a dictionary that specifies
            the shared memory location and info of the full matrix.
            The third element of the tuple contains a dictionary that specifies the
            shared memory location and info of the bidirectional targets matrix

        :returns bisimplex_count: (int) number of bisimplices present
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


def get_extended_simplices_dense(mp_element: Tuple[List, Dict, Dict]) -> np.ndarray:
    """Function that returns the list of extended simplices of a simplex.
        :argument data: (tuple[List, Dict, Dict]) data to check retrieve the extended simplices.
            The first element of the tuple contains the indices of the neurons
            in the adjacency matrix.
            The second element of the tuple contains a dictionary that specifies
            the shared memory location and info of the full matrix.
            The third element of the tuple contains a dictionary that specifies the
            shared memory location and info of the bidirectional targets matrix

        :returns ess: (np.ndarray) Extended simplices containing the original simplex.
    """
    simplex = mp_element[0]
    # full_matrix_info = mp_element[1]
    bidirectional_matrix_info = mp_element[2]

    # Retrieve second matrix location:
    bid_memory_block = SharedMemory(name=bidirectional_matrix_info['name'], size=bidirectional_matrix_info['size'])
    bid_matrix = np.ndarray(shape=bidirectional_matrix_info['shape'], dtype=bidirectional_matrix_info['type'],
                             buffer=bid_memory_block.buf)

    # Actual computation:
    return np.array(list(set(bid_matrix[simplex[-1]].nonzero()[0]) - set(simplex[:-1])))


class Processor:
    """Class to manage pipeline execution on files in a path.

    :argument in_path: (Path) path to execute pipeline on. Files that end with .pkl are considered to be
        the adjacency matrices of the things to count. Automatically discards bcounts.pkl files
    """
    def __init__(self, in_path: Path):
        self.in_path = in_path
        self.file_list = list(self.in_path.glob("**/*.pkl"))
        self.file_list = [elem for elem in self.file_list if not elem.name.endswith("bcounts.pkl")]
        self.file_list = [elem for elem in self.file_list if not elem.name.endswith("neurons.pkl")]

    def list_extended_simplices(self):
        """Produces es files for file in path."""
        pool = mp.Pool()
        print("Found " + str(len(self.file_list)) + " .pkl files.\n")
        for elem in tqdm(self.file_list, ):
            print("File: " + str(elem))
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


def count_bidirectional_edges(matrix: np.ndarray, count_file: h5py.File, dimension: int) -> Dict[int, np.ndarray]:
    """Function that returns bidirectional edge counts in simplices per dimension.

    :argument matrix: (np.ndarray) connectivity matrix of graph.
    :argument count_file: (h5py.File) the flagser output file.
    :argument dimension: (int) maximum dimension to consider

    :returns count_dictionary: (Dict[int, np.ndarray]) dictionary containing
        the bidirectional edge counts in simplices per dimension
    """
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


def count_total_bidirectional_edges(matrix: np.ndarray, count_file: h5py.File, dimension: int) -> Dict:
    """Function that returns bidirectional edge counts in simplices per dimension.

    :argument matrix: (np.ndarray) connectivity matrix of graph.
    :argument count_file: (h5py.File) the flagser output file.
    :argument dimension: (int) maximum dimension to consider

    :returns count_dictionary: (Dict[int, Tuple(int, int)]) dictionary containing
        dimension and relative biedge and simplex count
    """
    counts_per_dimension = {}
    for i in range(1, dimension + 1):
        counts_per_dimension.update({i: 0})
    for i in range(1, dimension + 1):
        try:
            simplices = count_file['Cells_' + str(i)]
            for simplex in tqdm(simplices):
                simplex_matrix = matrix[simplex].T[simplex].T
                counts_per_dimension[i] += np.count_nonzero(np.multiply(simplex_matrix, simplex_matrix.T))
            counts_per_dimension[i] = (int(counts_per_dimension[i]), len(simplices))
        except KeyError:
            pass
    return counts_per_dimension


def count_bidirectional_edges_from_binary(matrix: np.ndarray, count_files: List[Path]):
    counts_per_dimension = {}
    for file in count_files:
        simplices = binary2simplex(file)
        for simplex in tqdm(simplices):
            counts_per_dimension[len(simplex)] = counts_per_dimension.setdefault(
                                                    len(simplex),
                                                    np.zeros((len(simplex), len(simplex)))
                                                ) + matrix[simplex].T[simplex].T
    return counts_per_dimension


def bcount_from_file(path: Path, dimension: int, binary: bool = False):
    """
    Function that generates bidirectional edge count files from a pickle/flagser output
        combination.

    :argument path: (Path) path to the .pkl file containing the sparse matrix
    :argument dimension: (int) maximum dimension to consider.
    """
    matrix = load_sparse_matrix_from_pkl(path)
    matrix = np.array(matrix.todense()).astype(bool)
    if binary:
        count_files = [p for p in path.parent.glob("*.binary")]
        counts_per_dimension = count_bidirectional_edges_from_binary(matrix, count_files)
        name = "bbcounts.pkl"
    else:
        count_file = h5py.File(path.with_name(path.stem + "-count.h5"))
        counts_per_dimension = count_bidirectional_edges(matrix, count_file, dimension)
        name = "bcounts.pkl"
    with open(path.with_name(name), 'wb') as file:
        pickle.dump(counts_per_dimension, file)


def total_bcount_from_file(path: Path, dimension: int):
    """
    Function that generates bidirectional edge count files from a pickle/flagser output
        combination.

    :argument path: (Path) path to the .pkl file containing the sparse matrix
    :argument dimension: (int) maximum dimension to consider.
    """
    matrix = load_sparse_matrix_from_pkl(path)
    matrix = np.array(matrix.todense()).astype(bool)
    count_file = h5py.File(path.with_name(path.stem + "-count-representative.h5"))
    counts_per_dimension = count_total_bidirectional_edges(matrix, count_file, dimension)
    name = "total_rbcounts.pkl"
    with open(path.with_name(name), 'wb') as file:
        pickle.dump(counts_per_dimension, file)


def binary2simplex(address: Path) -> List[List[int]]:
    """Converts from binary format given by flagser when using --binary to a list of simplices
    Taken from https://github.com/JasonPSmith/flagser-count/, courtesy of Jason Smith.

    :argument address: (Path) path to binary file output of flagser count.

    :returns S: (List(List)) list of simplices in list format."""
    X = np.fromfile(address, dtype='uint64')                         #Load binary file
    S=[]                                                             #Initialise empty list for simplices

    i=0
    pbar = tqdm()
    while i < len(X):
        b = format(X[i], '064b')                                     #Load the 64bit integer as a binary string
        if b[0] == '0':                                              #If the first bit is 0 this is the start of a new simplex
            S.append([])
        t=[int(b[-21:],2), int(b[-42:-21],2), int(b[-63:-42],2)]     #Compute the 21bit ints stored in this 64bit int
        for j in t:
            if j != 2097151:                                         #If an int is 2^21 this means we have reached the end of the simplex, so don't add it
                S[-1].append(j)
        i+=1
        pbar.update(1)
    return S


def correlations_simplexwise(maximal_count_path: Path, gids: np.ndarray,
                             gid_start: int, gid_end: int, corr_matrix: np.ndarray,
                             conn_matrix: np.ndarray, type: str,
                               bs: bool=False):
    """
    Function to compute simplexwise correlation in edges. All edges in simplices are considered with repetitions.

    :argument maximal_count_path: (Path) path to h5 file with maximal simplices only.
    :argument gids: (np.ndarray) array containing the gids of the neurons in the correlation matrix.
    :argument gid_start: (int) first GID.
    :argument gid_end: (int) last GID.
    :argument corr_matrix: (np.ndarray) matrix of correlations to consider.
    :argument conn_matrix: (np.ndarray) connectivity matrix of the graph.
    :argument type: (str) type of edges to consider. Either 'all', 'spine', or 'end'.
    :argument bs: (bool) whether to do the analysis in bisimplices instead of simplices.
    """
    mcount_file = h5py.File(maximal_count_path)
    dvalues = []
    bvalues = []
    dvariances = []
    bvariances = []
    values = []
    variances = []
    btot = []
    dtot = []
    tot = []
    for i in tqdm(range(0, len(mcount_file.keys()))):
        dimension = i+1
        simplices = np.array(mcount_file['Cells_' + str(i + 1)])
        if bs:
            bisimplices = simplices[conn_matrix[simplices[:,-1], simplices[:,-2]], :]
            simplices = bisimplices
        if type == 'end':
            edges = simplices[:, [-2, -1]]
        elif type == 'spine':
            edges = np.vstack([simplices[:, [x, x + 1]] for x in range(simplices.shape[1] - 1)])
        elif type == 'all':
            edges = np.vstack(
                        [simplices[:, x] for x in
                            combinations(range(simplices.shape[1]), 2)]
            )
        posarray = np.empty((gid_end - gid_start + 1,))
        posarray[:] = np.nan
        for j, element in enumerate(gids - gid_start):
            posarray[element] = j

        dcorrelations = np.empty((edges.shape[0],))
        dcorrelations[:] = np.nan
        bcorrelations = np.empty((edges.shape[0],))
        bcorrelations[:] = np.nan
        correlations = np.empty((edges.shape[0],))
        correlations[:] = np.nan
        extra_count = 0
        for j, (row, col) in enumerate(edges):
            if np.isnan(posarray[row]) or np.isnan(posarray[col]):
                value = 0
                extra_count += 1
            else:
                value = corr_matrix[int(posarray[row])][int(posarray[col])]
            if conn_matrix[col, row]:
                bcorrelations[j] = value
            else:
                dcorrelations[j] = value
            correlations[j] = value
        bvalues.append(np.nanmean(bcorrelations))
        bvariances.append(np.nanvar(bcorrelations))
        btot.append(np.sum(np.logical_not(np.isnan(bcorrelations))))
        dvalues.append(np.nanmean(dcorrelations))
        dvariances.append(np.nanvar(dcorrelations))
        dtot.append(np.sum(np.logical_not(np.isnan(dcorrelations))))
        values.append(np.nanmean(correlations))
        variances.append(np.nanvar(correlations))
        tot.append(len(correlations))

    return ((bvalues, bvariances, btot),
            (dvalues, dvariances, dtot),
            (values, variances, tot),
            extra_count)
