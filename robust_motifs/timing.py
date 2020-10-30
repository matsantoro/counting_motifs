import h5py
from itertools import product
import multiprocessing as mp
import networkx
import numpy as np
from pathlib import Path
import time
import sys
from tqdm import tqdm
from typing import List, Optional, Tuple

from .data import (
    load_sparse_matrix_from_pkl, write_flagser_file,
    flagser_count
)
from .custom_mp import prepare_shared_memory
from .counting import get_n_extended_simplices, get_bisimplices, get_extended_simplices_with_signature


class Timer:
    """Timer class for timing multiprocessing function calls.

    :argument matrix_path: (Path) path to the matrix, if exists, or path to save ER graph.
    :argument n_nodes: (int) number of nodes to create ER graph with.
    :argument density: (float) edge density to create ER graph with.
    :parameter overwrite: (bool) whether to override preexisting .flag, .h5 data.
    """
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
        if not self._count_path.exists() or overwrite:
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
             random: bool = True) -> Tuple[List[float], List[float], List[int]]:
        """Core timing function for mp over simplices.

        :argument n_simplices: (int) number of simplices per iteration.
        :argument n_iterations: (int) number of iterations.
        :argument dimension: (int) dimension of simplices to use.
        :parameter random: (bool) whether to use random simplices. Takes more time!
        """
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
            ########## bisimplices ######################
            # r = self.pool.imap(get_bisimplices, mp_iterator)  # call to have things run
            # l1 = set()
            # for elem in r:
            #    l1 = l1.union(set(elem))
            ######### extended simplices ##################
            # r = self.pool.imap(get_n_extended_simplices, mp_iterator)
            # [_ for _ in r]
            ######### extended simplices + signatures ####
            r = self.pool.imap(get_extended_simplices_with_signature, mp_iterator)
            motifs = []
            signatures = []
            for elem in r:
                motifs.extend(elem[0])
                signatures.extend(elem[1])
            # stop rewriting here.
            end = time.time()

            timings.append(end-start)
            prep_timings.append(start - pre_start)
            # manually compute memory usage here
            ########## bisimplices #######################
            #mem_usages.append(sys.getsizeof(l1))
            ########## extended simplices ################
            #mem_usages.append(0)
            ########## extended simplices + signatures ###
            mem_usages.append(sys.getsizeof(motifs) + sys.getsizeof(signatures))
        self._shutdown()

        return timings, prep_timings, mem_usages
