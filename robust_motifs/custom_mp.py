from multiprocessing.shared_memory import SharedMemory
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple


def prepare_shared_memory(matrix: sp.csr_matrix, prefix: str) -> Tuple[Dict, List]:
    """Function that prepares shared memory for sparse matrix.

    :argument matrix: (sp.csr_matrix) matrix to add in shared memory.
    :argument prefix: (str) prefix for SM usage.

    :returns arrays: (Dict) dictionary with all the shared memory
        info for the matrix. Argument of all sm functions."""
    array = {}
    data = SharedMemory(size=matrix.data.nbytes,
                                         create=True, name=prefix + 'data')
    shared_data = np.ndarray(
        matrix.data.shape,
        dtype=matrix.data.dtype,
        buffer=data.buf)
    shared_data[:] = matrix.data[:]
    array['data'] = {'name': prefix + 'data',
                     'size': matrix.data.nbytes,
                     'type': matrix.data.dtype,
                     'factor': matrix.data.itemsize}
    indices = SharedMemory(size=matrix.indices.nbytes,
                                            create=True, name=prefix + 'indices')
    shared_indices = np.ndarray(matrix.indices.shape, dtype=matrix.indices.dtype, buffer=indices.buf)
    shared_indices[:] = matrix.indices[:]
    array['indices'] = {'name': prefix + 'indices',
                     'size': matrix.indices.nbytes,
                     'type': matrix.indices.dtype,
                     'factor': matrix.indices.itemsize}
    indptr = SharedMemory(size=matrix.indptr.nbytes, create=True, name=prefix + 'indptr')
    shared_indptr = np.ndarray(matrix.indptr.shape, dtype=matrix.indptr.dtype, buffer=indptr.buf)
    shared_indptr[:] = matrix.indptr[:]
    array['indptr'] = {'name': prefix + 'indptr',
                        'size': matrix.indptr.nbytes,
                        'type': matrix.indptr.dtype,
                        'factor': matrix.indptr.itemsize}
    return array, [data, indices, indptr]


def share_dense_matrix(matrix: np.ndarray) -> Tuple[Dict, List]:
    """Add a dense matrix in a shared memory block.

    :argument matrix: (np.ndarray) dense matrix to be shared.

    :returns array: (dictionary) array_info to access shared memory.
    :returns links: (List) shared memory objects for unlinking.
    """
    memory_block = SharedMemory(create=True, size=matrix.nbytes)
    array_pointer = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=memory_block.buf)
    array_pointer[:] = matrix
    array = {'name': memory_block.name,
                'size': matrix.nbytes,
                'type': matrix.dtype,
                'factor': matrix.itemsize,
                'shape': matrix.shape
            }
    return array, [memory_block]
