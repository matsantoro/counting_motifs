import h5py
from itertools import product
import networkx
import multiprocessing as mp
import numpy as np
from pathlib import Path
import sys
from typing import Callable, Optional
import time
from tqdm import tqdm

from .data import (
    load_sparse_matrix_from_pkl, write_flagser_file,
    flagser_count
)
from .custom_mp import prepare_shared_memory
from .counting import get_n_extended_simplices, get_bisimplices


class Timer:
    def __init__(self,
                 matrix_path: Path,
                 n_nodes: Optional[int] = None, density: Optional[float] = None,
                 overwrite: bool = True):

        self._flagser_path = matrix_path.with_suffix(".flag")
        self._count_path = matrix_path.with_name(matrix_path.stem + "-count.h5")

        self._flagser_path.parent.mkdir(exist_ok=True, parents=True)
        # Prepare matrix
        if matrix_path.exists():
            self._matrix = load_sparse_matrix_from_pkl(matrix_path)
            self._n_nodes = self._matrix.shape[0]
            self._density = self._matrix.count_nonzero()/(self._n_nodes**2)
        else:
            if n_nodes is not None and density is not None:
                self._n_nodes = n_nodes
                self._density = density
                g = networkx.fast_gnp_random_graph(n_nodes, density, directed=True)
                self._matrix = networkx.adjacency_matrix(g)
            else:
                raise ValueError
        # Prepare flagser file
        if not self._flagser_path.exists() or overwrite:
            write_flagser_file(self._flagser_path, self._matrix)
        # Flagser count
        flagser_count(self._flagser_path, self._count_path, overwrite=overwrite)

        self._count_file = h5py.File(self._count_path, 'r')

    def _prepare_multiprocessing(self, cores_count: Optional[int] = None):
        if cores_count:
            self.pool = mp.Pool(processes=cores_count)
        else:
            self.pool = mp.Pool()

        self._bid_matrix = self._matrix.multiply(self._matrix.T)

        self._matrix_info, self._matrix_links = prepare_shared_memory(self._matrix, 'full')
        self._bid_matrix_info, self._bid_matrix_link = prepare_shared_memory(self._bid_matrix, 'bid')

    def _shutdown(self):
        for link in self._matrix_links:
            link.unlink()
        for link in self._bid_matrix_link:
            link.unlink()

    def _prepare_random_selection(self, n_simplices: int, dimension: int):
        self._random_selection = np.random.choice(self._count_file["Cells_" + str(dimension)].shape[0],
                                            min(
                                                n_simplices,
                                                self._count_file["Cells_" + str(dimension)].shape[0]
                                            ),
                                            replace=False)
        self._random_selection.sort()

    def time_mp(self, cores_count: Optional[int] = None,
             n_simplices: int = 10000,
             n_iterations: int = 10,
             dimension: int = 1,
             random: bool = True):
        self._prepare_multiprocessing(cores_count)
        timings = []
        prep_timings = []
        mem_usages = []
        for _ in range(n_iterations):

            pre_start = time.time()

            if random:
                self._prepare_random_selection(n_simplices, dimension)

                simplex_iterator = self._count_file['Cells_'+str(dimension)][self._random_selection]
            else:
                simplex_iterator = self._count_file['Cells_'+str(dimension)][
                    np.arange(min(n_simplices, self._count_file['Cells_'+str(dimension)].shape[0]))
                ]

            mp_iterator = product(simplex_iterator, [self._matrix_info], [self._bid_matrix_info])

            start = time.time()
            # Unfortunately, you can't pass the function to be timed as an argument. You need to
            # rewrite this part.
            r = self.pool.imap(get_bisimplices, mp_iterator)  # call to have things run
            l1 = set()
            for elem in r:
                l1 = l1.union(set(elem))
            # stop rewriting here.
            print(l1)
            end = time.time()

            timings.append(end-start)
            prep_timings.append(start - pre_start)
            # manually compute memory usage here
            mem_usages.append(sys.getsizeof(l1))
        self._shutdown()

        return timings, prep_timings, mem_usages
