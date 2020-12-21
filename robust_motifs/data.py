import h5py
from itertools import product
import os
import numpy as np
import networkx
from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm
from typing import List, Optional, Union
import scipy.sparse as sp

from .custom_mp import prepare_shared_memory, share_dense_matrix
from .utilities import build_triu_matrix

def import_connectivity_matrix(path: Path = Path('data/test/cons_locs_pathways_mc0_Column.h5'),
                        zones: Optional[List[str]] = None,
                        dataframe: bool = True,
                        type: Optional[str] = None,
                        pathway_shuffle: bool = False) -> Union[np.ndarray, pd.DataFrame, sp.csr_matrix]:
    """Imports the connectivity matrix of the BBP model (h5 format)

    :argument path: (Path) Path to load the data from.
    :argument zones: (Optional[List[str]]) List of zone strings to load.
    :parameter dataframe: (bool) Whether to return the matrix in dataframe form.
    :parameter type: (Optional[str]) Sparse matrix format to return. Either 'coo',
        'csr' or 'csc'.
    :parameter pathway_shuffle: (bool) whether to shuffle connections pathway-wise.

    :returns matrix: (Union[np.ndarray, pd.DataFrame]) Connectivity matrix.
    """
    file = h5py.File(path, 'r')
    if zones is None:
        zones = list(file['populations'].keys())
    assert np.all([zone in list(file['populations'].keys()) for zone in zones])
    n_elements = []
    for element1 in tqdm(zones):
        conn_row = sp.csr_matrix(file['connectivity'][element1][zones[0]]['cMat'][:, :], dtype=bool)

        if pathway_shuffle:
            conn_row = matrix_shuffle(conn_row, exclude_diagonal=(element1 is zones[0]))

        n_elements += [conn_row.shape[0]]
        for element2 in zones[1:]:
            block = sp.csr_matrix(file['connectivity'][element1][element2]['cMat'][:, :], dtype=bool)

            if pathway_shuffle:
                block = matrix_shuffle(block, exclude_diagonal=(element1 is element2))

            conn_row = sp.hstack([conn_row, block])
        try:
            matrix = sp.vstack([matrix, conn_row])
        except Exception:
            matrix = conn_row

    if not dataframe:
        if type is None or type == 'coo':
            return matrix
        elif type == 'csr':
            return matrix.tocsr()
        elif type == 'csc':
            return matrix.tocsc()
        else:
            print('Structure not recognized. coo matrix is returned.')
            return matrix
    else:
        df = pd.DataFrame.sparse.from_spmatrix(matrix)
        index1 = np.repeat(zones, n_elements)
        index2 = []
        for n in n_elements:
            index2 = index2 + list(range(n))
        df.columns = [index1, index2]
        df.index = [index1, index2]
        return df


def matrix_shuffle(matrix: Union[sp.csr_matrix, np.ndarray], exclude_diagonal: bool = False, sparse: bool = True):
    """Returns shuffled version of a matrix, with care not to put elements on the diagonal.

        :argument matrix: (Union[sp.csr_matrix, np.ndarray]) matrix to shuffle. Both sparse and dense formats
            acceptable.
        :parameter exclude_diagonal: (bool) whether to exclude the diagonal of the matrix from the shuffling.
            Only for square matrices.
        :parameter sparse: (bool) whether the matrix is sparse.

        :returns shuffled_matrix: (Union[sp.csr_matrix, np.ndarray]) shuffled matrix.
    """
    if sparse:
        if exclude_diagonal:
            assert matrix.shape[0] == matrix.shape[1]
            # squash matrix before shuffling
            upper_matrix = sp.triu(matrix, 1, 'csc')
            lower_matrix = sp.tril(matrix, -1, 'csr')
            upper_matrix.indices += 1
            upper_matrix = upper_matrix.tocsr()
            _matrix = (lower_matrix + upper_matrix)[1:]
        else:
            _matrix = matrix
        buffer = np.array(_matrix.todense()).flatten()
        np.random.shuffle(buffer)
        _matrix = sp.csr_matrix(buffer.reshape(_matrix.shape))
        if exclude_diagonal:
            lower_matrix = sp.tril(_matrix, 0, 'csr')
            aux = sp.csr_matrix(np.zeros((1, lower_matrix.shape[1])).astype(bool))
            lower_matrix = sp.vstack([aux, lower_matrix])
            lower_matrix = lower_matrix.tocsr()
            upper_matrix = sp.triu(_matrix, 1, 'csr')
            upper_matrix = sp.vstack([upper_matrix, aux])
            _matrix = lower_matrix + upper_matrix
        return _matrix

    else:
        if exclude_diagonal:
            assert matrix.shape[0] == matrix.shape[1]
            # squash matrix before shuffling
            upper_matrix = np.triu(matrix)
            lower_matrix = np.tril(matrix)
            upper_matrix = np.vstack([np.zeros((1, matrix.shape[0]), dtype=bool), upper_matrix])[:-1]
            _matrix = (lower_matrix + upper_matrix)[1:]
        else:
            _matrix = matrix
        buffer = _matrix.flatten()
        np.random.shuffle(buffer)
        _matrix = buffer.reshape(_matrix.shape)
        if exclude_diagonal:
            lower_matrix = np.tril(_matrix, 0)
            upper_matrix = np.triu(_matrix, 1)
            aux = np.zeros((1, _matrix.shape[1]), dtype=bool)
            lower_matrix = np.vstack([aux, lower_matrix])
            upper_matrix = np.vstack([upper_matrix, aux])
            _matrix = lower_matrix + upper_matrix
        return _matrix


def write_flagser_file(path: Path, matrix: sp.csr_matrix, verbose = True):
    """Writes a matrix as a flagser file.
        :argument path: flagser file path.
        :argument matrix: sparse matrix.
        :argument verbose: whether to see tqdm pbar.
    """
    n_nodes = matrix.shape[0]
    with open(path, "w") as f:
        f.write("dim 0\n")
        if verbose:
            iterator = tqdm(range(n_nodes))
        else:
            iterator = range(n_nodes)
        for _ in iterator:
            f.write("0 ")
        f.write("\n")
        f.write("dim 1\n")
        if verbose:
            iterator = tqdm(zip(*matrix.nonzero()))
        else:
            iterator = zip(*matrix.nonzero())
        for row, col in iterator:
            f.write(str(row) + " " + str(col) + "\n")


def save_er_graph(path: Path, n_nodes: int, density: float):
    """Saves the ER graph as a flagser-readable file, and pickle-dumps
        the adjacency matrix.

        :argument path: (Path) path of flagser file to be written.
        :argument n_nodes: (int) number of nodes of the graph.
        :argument density: (float) graph edge density.

        :returns flag_path: (Path) flagser file path.
        :returns pickle_path: (Path) pickle file path.
    """
    g = networkx.fast_gnp_random_graph(n_nodes, density, directed=True)
    a = networkx.adjacency_matrix(g)

    write_flagser_file(path, a)

    save_sparse_matrix_to_pkl(path.with_suffix(".pkl"), a)

    return path, path.with_suffix(".pkl")


def save_count_er_graph(path: Path, n_nodes: int, density: float):
    """Saves the ER graph as a flagser-readable file, and pickle-dumps
    the adjacency matrix. Creates the h5 count file with flagser.

        :argument path: (Path) path of flagser file to be written.
        :argument n_nodes: (int) number of nodes of the graph.
        :argument density: (float) graph edge density.

        :returns flag_path: (Path) flagser file path.
        :returns pickle_path: (Path) pickle file path.
        :returns count_path: (Path) flagser-count h5 file path.
    """
    path, pickle_path = save_er_graph(path, n_nodes, density)
    count_path = path.parent / Path(path.stem + "-count.h5")
    flagser_count(path, count_path)

    return path, pickle_path, count_path


def save_count_graph_from_matrix(path: Path, matrix: sp.csr_matrix, verbose: bool = True, maximal = False):
    """Saves the graph from the adjacency matrix as a flagser-readable file.
    Pickle-dumps the adjacency matrix. Creates the h5 count file with flagser.

        :argument path: (Path) path of flagser file to be written.
        :argument matrix: (sp.csr_matrix) matrix to be dumped.

        :returns flag_path: (Path) flagser file path.
        :returns pickle_path: (Path) pickle file path.
        :returns count_path: (Path) flagser-count h5 file path.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    pickle_path = path.with_suffix(".pkl")
    count_path = path.parent / Path(path.stem + "-count.h5")
    write_flagser_file(path, matrix, verbose)
    save_sparse_matrix_to_pkl(pickle_path, matrix)
    flagser_count(path, count_path, maximal=maximal)
    return path, pickle_path, count_path


def load_sparse_matrix_from_pkl(path: Path):
    """Loads a sparse matrix from a pickle file.

    :argument path: (Path) path of the pickle file to be loaded.

    :returns matrix: (sp.csr_matrix) Loaded sparse matrix."""
    with open(path, 'rb') as file:
        dictionary = pickle.load(file)
        return sp.csr_matrix((dictionary['data'],
                             dictionary['indices'],
                             dictionary['indptr']))


def save_sparse_matrix_to_pkl(path: Path, matrix: sp.csr_matrix):
    """Saves a sparse matrix to a pickle file.

    :argument path: (Path) path of the pickle file to be created.
    :argument matrix: (sp.csr_matrix) matrix to be saved.
    """
    with open(path, "wb") as file:
        pickle.dump(
            {'data': matrix.data,
             'indices': matrix.indices,
             'indptr': matrix.indptr},
            file
        )


def unique_identifier(mat):
    dt = np.dtype([('raw', np.void, mat.shape[1] * mat.dtype.itemsize)])
    return mat.reshape(np.prod(mat.shape)).view(dt)


def find_maximal_simplices_from_all(simplices, simplices_higher_dim):
    one_index = unique_identifier(simplices)
    simplices_higher_dim_stacked = unique_identifier(np.vstack(
        [simplices_higher_dim[:, np.delete(np.arange(simplices_higher_dim.shape[1]), x)] for x in
         range(simplices_higher_dim.shape[1])]))
    return simplices[np.logical_not(np.isin(one_index, simplices_higher_dim_stacked)), :]


def create_maximal_simplex_file(simplex_file_path, maximal_file_path, overwrite):
    cfile = h5py.File(simplex_file_path)

    if overwrite:
        maximal_file_path.unlink(missing_ok=True)
    maximal_cfile = h5py.File(maximal_file_path, 'w')
    for i in tqdm(range(len(cfile.keys()) - 1)):
        simplices1 = cfile['Cells_' + (str(i + 1))][:]
        simplices2 = cfile['Cells_' + (str(i + 2))][:]
        new_simplices = find_maximal_simplices_from_all(simplices1, simplices2)
        maximal_cfile.create_dataset('Cells_' + (str(i + 1)), data=new_simplices)


def flagser_count(in_path: Path, out_path: Path, overwrite: bool = True, maximal = False):
    """Uses flagser to build flagser count file.

    :argument in_path: (Path) flagser file to read and count.
    :argument out_path: (Path) file to write as output.
    :parameter overwrite: (bool) whether to overwrite if it exists.
    """
    if overwrite:
        out_path.unlink(missing_ok=True)
    os.system("flagser-count " + str(in_path) + " --out " + str(out_path))

    if maximal:
        maximal_path = out_path.with_name(out_path.stem + "-maximal.h5")
        create_maximal_simplex_file(out_path, maximal_path, overwrite)


def adjust_bidirectional_edges(matrix: sp.csr_matrix, target: int):
    """Function to adjust the number of bidirectional edges of a matrix.

        :argument matrix: (sp.csr_matrix) matrix to adjust the edges of.
        :argument target: (int) number of bidirectional edges to reach.

        :returns adjusted_matrix: (sp.csr_matrix) adjusted matrix.
    """
    b_matrix = matrix.multiply(matrix.T)
    b_edges = int(b_matrix.count_nonzero()/2)
    d_matrix = matrix - b_matrix
    _matrix = matrix.copy()

    selection = np.random.choice(d_matrix.count_nonzero(), target - b_edges, replace=False)
    selection = np.array(list(zip(*d_matrix.nonzero())))[selection]

    for elem in tqdm(selection, desc='Removing edges...'):
        _matrix[elem[0], elem[1]] = False
        d_matrix[elem[0], elem[1]] = False

    d_matrix.eliminate_zeros()

    selection = np.random.choice(d_matrix.count_nonzero(), target - b_edges, replace=False)
    selection = np.array(list(zip(*d_matrix.nonzero())))[selection]

    _matrix = _matrix.tolil()

    for elem in tqdm(selection, desc='Adding bidirectional edges...'):
        _matrix[elem[1], elem[0]] = True

    _matrix = _matrix.tocsr()
    _matrix.eliminate_zeros()
    return _matrix


def load_bbp_matrix_format(path: Path, shuffle_type: str = None):
    """load data in the bbp format.

    :argument path: (Path) path of the npy file.
    :parameter shuffle type: (str) type of shuffling. Can be 'all', 'pathway', or None.

    :returns matrix: (sp.csr_matrix) sparse adjacency matrix.
    """
    mat = np.load(str(path))
    neuron_data = pd.read_pickle(path.with_name(path.stem.replace("matrix", "neuron_data")))

    if shuffle_type == "all":
        mat = matrix_shuffle(mat, exclude_diagonal=True, sparse=False)

    elif shuffle_type == "pathway":
        available_pops = neuron_data.mtype.unique()
        d = {}
        for pop in available_pops:
            d[pop] = retrieve_indices(pop, neuron_data)
        for pop1 in tqdm(available_pops):
            for pop2 in available_pops:
                temp_mat = matrix_shuffle(
                    mat[d[pop1]][:, d[pop2]],
                    exclude_diagonal=(pop1 == pop2),
                    sparse=False
                )
                for i1, elem1 in enumerate(d[pop1]):
                    for i2, elem2 in enumerate(d[pop2]):
                        mat[elem1, elem2] = temp_mat[i1, i2]

    elif shuffle_type is None:
        pass
    else:
        raise ValueError
    return sp.csr_matrix(mat)


def retrieve_indices(pop: str, neuron_data: pd.DataFrame) -> np.array:
    """Retrieves indices of neurons in a given population.

    :argument pop: (str) string that indicates population to be considered.
    :argument neuron_data: (pd.DataFrame) dataframe containing neuron data.

    :returns indices: (np.array) list of indices of neurons.
    """

    shift = neuron_data.index.values[0]  # this is the gid of the first neuron we should shift by them
    return neuron_data.query("mtype=='"+pop+"'").index.values - shift


class MPDataManager:
    """Class to produce iterators ready for multiprocessing from data.

    :argument path: (pathlib.Path) reference path for the manager. Data will be laoded from here if
        it exists, or saved here if not. Each folder will be populated with the adjacency matrix in both
        .pkl and .flag format, and the h5 count file.
    :argument matrix: (sp.csr_matrix) adjacency matrix in sparse format."""
    def __init__(self, path: Path, matrix: Optional[sp.csr_matrix]):
        if path.exists():
            if path.suffix == ".flag":
                try:
                    self._matrix = load_sparse_matrix_from_pkl(path.with_suffix(".pkl"))
                    flagser_count(path, path.with_name(path.stem + "-count.h5"))

                    self._flagser_path = path
                    self._pickle_path = path.with_suffix(".pkl")
                    self._count_path = path.with_name(path.stem + "-count.h5")

                except Exception as message:
                    print("Flagser file found, but matrix not found in pickle format.")
                    print(message)

            elif path.suffix == ".pkl":
                try:
                    self._matrix = load_sparse_matrix_from_pkl(path)
                    self._flagser_path = path.with_suffix(".flag")
                    self._pickle_path = path
                    self._count_path = path.with_name(path.stem + "-count.h5")
                    if self._flagser_path.exists() and self._count_path.exists():
                        print("Using preexisting flagser files at "+str(self._count_path))
                    else:
                        write_flagser_file(self._flagser_path)
                        flagser_count(self._flagser_path, self._count_path)
                except Exception as message:
                    print("Couldn't load matrix from pickle.")
                    print(message)
        else:
            path.parent.mkdir(exist_ok=True, parents=True)
            if matrix is not None:
                self._matrix = matrix
                self._flagser_path, self._pickle_path, self._count_path = save_count_graph_from_matrix(path, matrix)
            else:
                print("If matrix is None, path should exist.")

        try:
            self._count_file = h5py.File(self._count_path, 'r')

        except Exception as message:
            print("Could not open count file " + str(self._count_path))
            print(message)

        self._bid_matrix = self._matrix.multiply(self._matrix.T)

    def __del__(self):
        self._shut_shared_memory()

    def _shut_shared_memory(self):
        try:
            for link in self._full_matrix_link + self._bid_matrix_link:
                link.unlink()
        except:
            pass

    def _prepare_shared_memory(self):
        """Prepare shared memory with necessary sparse matrices."""
        try:
            self._full_matrix_info, self._full_matrix_link = prepare_shared_memory(self._matrix, 'full')
            self._bid_matrix_info, self._bid_matrix_link = prepare_shared_memory(self._bid_matrix, 'bid')
        except FileExistsError:
            pass

    def _prepare_shared_memory_dense(self):
        """Prepare shared memory with necessary dense matrices."""
        try:
            self._full_matrix_info, self._full_matrix_link = share_dense_matrix(np.array(self._matrix.todense()))
            self._bid_matrix_info, self._bid_matrix_link = share_dense_matrix(np.array(self._bid_matrix.todense()))
        except FileExistsError:
            pass
        
    def _prepare_random_selection(self, n_simplices: int, dimension: int):
        self._random_selection = np.random.choice(self._count_file["Cells_" + str(dimension)].shape[0],
                                            min(
                                                n_simplices,
                                                self._count_file["Cells_" + str(dimension)].shape[0]
                                            ),
                                            replace=False)
        self._random_selection.sort()

    def mp_simplex_iterator(self, n: Optional[int] = None, dimension: int = 1, random: bool = False):
        """Simplex iterator with sparse matrix references.

        :argument n: (Optional[int]) number of simplices to build the iterator with.
        :argument dimension: (int) dimension of simplices to prepare.
        :parameter random: (bool) whether to randomize simplices selection. Only works with specified n.

        :returns iterator: (iterable) the requested iterator.
        """
        self._prepare_shared_memory()
        if random:
            self._prepare_random_selection(n, dimension)
            simplex_iterator = self._count_file['Cells_' + str(dimension)][self._random_selection]
        else:
            if n:
                simplex_iterator = self._count_file['Cells_' + str(dimension)][:n]
            else:
                simplex_iterator = self._count_file['Cells_' + str(dimension)]

        return product(simplex_iterator, [self._full_matrix_info], [self._bid_matrix_info])

    def mp_np_simplex_iterator(self, n: Optional[int] = None, dimension: int = 1, random: False = bool):
        """Simplex iterator with sparse matrix references and numpy-imported simplex list.

        :argument n: (Optional[int]) number of simplices to build the iterator with.
        :argument dimension: (int) dimension of simplices to prepare.
        :parameter random: (bool) whether to randomize simplices selection. Only works with specified n.

        :returns iterator: (iterable) the requested iterator.
        """
        self._prepare_shared_memory()
        if random:
            self._prepare_random_selection(n, dimension)
            simplex_iterator = np.array(self._count_file['Cells_' + str(dimension)][self._random_selection])
        else:
            if n:
                simplex_iterator = np.array(self._count_file['Cells_' + str(dimension)][:n])
            else:
                simplex_iterator = np.array(self._count_file['Cells_' + str(dimension)])

        return product(simplex_iterator, [self._full_matrix_info], [self._bid_matrix_info])

    def mp_np_simplex_iterator_dense(self, n: Optional[int] = None, dimension: int = 1, random: bool = False,
                                     part: slice = None):
        """Simplex iterator with dense matrix references and numpy-imported simplex list.

        :argument n: (Optional[int]) number of simplices to build the iterator with.
        :argument dimension: (int) dimension of simplices to prepare.
        :parameter random: (bool) whether to randomize simplices selection. Only works with specified n.

        :returns iterator: (iterable) the requested iterator.
        """
        self._prepare_shared_memory_dense()
        if random:
            self._prepare_random_selection(n, dimension)
            simplex_iterator = np.array(self._count_file['Cells_' + str(dimension)][self._random_selection])
        else:
            if n:
                simplex_iterator = np.array(self._count_file['Cells_' + str(dimension)][:n])
            elif part:
                simplex_iterator = np.array(self._count_file['Cells_' + str(dimension)])[part]
            else:
                simplex_iterator = np.array(self._count_file['Cells_' + str(dimension)])

        return product(simplex_iterator, [self._full_matrix_info], [self._bid_matrix_info])

    def mp_chunks(self, dimension: int = 1, chunk_dimension: int = 10000000):
        """Iterator over big chunks for memory issues.

        :argument dimension: (int) dimension of simplices to load.
        :argument chunk_dimension: (int) number of simplices to load at once.

        :returns iterator: (iterable) returns simplex iterators of specified dimension."""
        total_length = self._count_file['Cells_'+str(dimension)].shape[0]
        for i in range(np.ceil(total_length/chunk_dimension).astype(int)):
            self._shut_shared_memory()
            yield self.mp_np_simplex_iterator_dense(
                dimension=dimension,
                part=slice(i*chunk_dimension, (i+1)*chunk_dimension)
            )

    def worker_init_dense(self):
        def initializer():
            print("Initializing on PID " + str(os.getppid()))
            global global_bid_matrix
            global_bid_matrix = self._bid_matrix.todense().copy()
            global global_matrix
            global_matrix = self._matrix.todense().copy()

        return initializer

    def worker_init_sparse(self):
        def initializer():
            print("Initializing on PID " + str(os.getppid()))
            global global_bid_matrix
            global_bid_matrix = self._bid_matrix.copy()
            global global_matrix
            global_matrix = self._matrix.copy()

        return initializer

    def mp_np_clean_simplex_iterator(self, n: Optional[int] = None, dimension: int = 1, random: bool = False,
                                     part: slice = None):
        if random:
            self._prepare_random_selection(n, dimension)
            simplex_iterator = np.array(self._count_file['Cells_' + str(dimension)][self._random_selection])
        else:
            if n:
                simplex_iterator = np.array(self._count_file['Cells_' + str(dimension)][:n])
            elif part:
                simplex_iterator = np.array(self._count_file['Cells_' + str(dimension)])[part]
            else:
                simplex_iterator = np.array(self._count_file['Cells_' + str(dimension)])

        return simplex_iterator

    def mp_clean_chunks(self, dimension: int = 1, chunk_dimension: int = 10000000):
        """Iterator over big chunks for memory issues.

        :argument dimension: (int) dimension of simplices to load.
        :argument chunk_dimension: (int) number of simplices to load at once.

        :returns iterator: (iterable) returns simplex iterators of specified dimension."""
        total_length = self._count_file['Cells_'+str(dimension)].shape[0]
        for i in range(np.ceil(total_length/chunk_dimension).astype(int)):
            yield self.mp_np_clean_simplex_iterator(
                dimension=dimension,
                part=slice(i*chunk_dimension, (i+1)*chunk_dimension)
            )


class Pickleizer:
    def __init__(self, in_path: Path):
        self.in_path = in_path
        self.file_list = list(self.in_path.glob("**/*Column.h5"))
        print("Found " + str(len(self.file_list)) + " files.")

    def pickle_it(self, destination: Path = None):
        if destination is None:
            for file_path in self.file_list:
                m = import_connectivity_matrix(file_path, dataframe=False, type='csr')
                save_count_graph_from_matrix(file_path.with_suffix(".flag"), m)
        else:
            destination.mkdir(exist_ok=True, parents=True)
            for file_path in self.file_list:
                m = import_connectivity_matrix(file_path, dataframe=False, type='csr')
                save_count_graph_from_matrix(destination / file_path.parent.stem / file_path.stem / file_path.with_suffix('.flag').name, m)


def create_test_graphs(n_instances: int, n_nodes: int, density: float, path: Path):
    for i in tqdm(range(n_instances)):
        save_path = path / ("seed_" + str(i))
        save_path.mkdir(exist_ok=True, parents=True)
        g = networkx.fast_gnp_random_graph(n_nodes, density, directed=True, seed = i)
        a = networkx.adjacency_matrix(g)
        b = save_path / "graph.flag"
        print(b)
        save_count_graph_from_matrix(b, a)


def create_control_graphs_from_matrix(n_instances: int, matrix_path: Path, path: Path, type: str, seed: int = 1,
                                      maximal=True):
    np.random.seed(seed)
    if type == 'full':
        for n in tqdm(range(n_instances)):
            m = import_connectivity_matrix(matrix_path, dataframe=False, type = 'csr')
            m = matrix_shuffle(m, exclude_diagonal=True)
            save_path = path / ("seed_"+str(n)) / "graph.flag"
            save_count_graph_from_matrix(save_path, m, maximal=maximal)
    if type == 'pathways':
        for n in tqdm(range(n_instances)):
            m = import_connectivity_matrix(matrix_path, dataframe=False, type='csr', pathway_shuffle=True)
            save_path = path / ("seed_"+str(n)) / "graph.flag"
            save_count_graph_from_matrix(save_path, m, maximal=maximal)
    if type == 'adjusted':
        for n in tqdm(range(n_instances)):
            m = import_connectivity_matrix(matrix_path, dataframe=False, type='csr')
            bm = m.multiply(m.T)
            m = matrix_shuffle(m, exclude_diagonal=True)
            m = adjust_bidirectional_edges(m, int(bm.count_nonzero()/2))
            save_path = path / ("seed_"+str(n)) / "graph.flag"
            save_count_graph_from_matrix(save_path, m, maximal=maximal)
    if type == 'shuffled_biedges':
        for n in tqdm(range(n_instances)):
            m = import_connectivity_matrix(matrix_path, dataframe=False, type='csr')
            m = m.tolil()
            bm = sp.triu(m.multiply(m.T))
            n_bidirectional_edges = bm.count_nonzero()
            bm = bm.tocoo()
            for row, col in zip(bm.row, bm.col):
                if np.random.binomial(1,0.5,1):
                    m[row, col] = False
                else:
                    m[col, row] = False
            n_edges = m.count_nonzero()
            m1 = m.tocoo()
            for index in np.random.choice(np.arange(n_edges), size=n_bidirectional_edges, replace=False):
                m[m1.col[index], m1.row[index]] = True
            save_path = path / ("seed_"+str(n)) / "graph.flag"
            m = m.tocsr()
            save_count_graph_from_matrix(save_path, m, maximal=maximal)
    if type == 'underlying':
        for n in tqdm(range(n_instances)):
            m = import_connectivity_matrix(matrix_path, dataframe=False, type='csr')
            mdag = sp.triu(m + m.T).tolil()
            bm = sp.triu(m.multiply(m.T))
            n_bidirectional_edges = bm.count_nonzero()
            n_edges = mdag.count_nonzero()
            m1 = mdag.tocoo()
            for index in np.random.choice(np.arange(n_edges), size=n_bidirectional_edges, replace=False):
                mdag[m1.col[index], m1.row[index]] = True
            save_path = path / ("seed_" + str(n)) / "graph.flag"
            m = mdag.tocsr()
            save_count_graph_from_matrix(save_path, m, maximal=maximal)



class ResultManager:
    def __init__(self, path: Path):
        self.processed_file_list = []
        for file in sorted(path.glob("**/ES_count.npz")):
            self.processed_file_list.append(file.parent)

    @property
    def counts(self):
        es_counts = []
        bs_counts = []
        for file in self.processed_file_list:
            es_counts.append(np.load(file / "ES_count.npz")['arr_0'])
            bs_counts.append(np.load(file / "BS_count.npz")['arr_0'])
        return es_counts, bs_counts

    def get_counts_dataframe(self, group: str):
        a = []
        for file in self.processed_file_list:
            try:
                matrix_path = file / (file.parts[-1] + ".pkl")
                m = load_sparse_matrix_from_pkl(matrix_path)
            except:
                matrix_path = file / "graph.pkl"
                m = load_sparse_matrix_from_pkl(matrix_path)
            bm = m.multiply(m.T)
            es_count = np.load(file / "ES_count.npz")['arr_0']
            a.append([m.shape[0], 0, group, "ES", str(file)])
            a.append([m.shape[0], 0, group, "BS", str(file)])
            a.append([m.shape[0], 0, group, "S", str(file)])
            a.append([bm.count_nonzero(), 1, group, "ES", str(file)])
            a.append([bm.count_nonzero()/2, 1, group, "BS", str(file)])
            for dim, elem in enumerate(es_count[:, 1].tolist()):
                a.append([elem, int(dim+2), group, "ES",str(file)])
            for dim, elem in enumerate(es_count[:, 0].tolist()):
                a.append([elem, int(dim+1), group, "S",str(file)])
            for dim, elem in enumerate(np.nan_to_num(es_count[:, 1]/es_count[:, 0]).tolist()):
                a.append([elem, int(dim+2), group, "RES",str(file)])
            bs_count = np.load(file / "BS_count.npz")['arr_0']
            for dim, elem in enumerate(bs_count[:, 1].tolist()):
                a.append([elem/2, int(dim+2), group, "BS", str(file)])
            for dim, elem in enumerate(np.nan_to_num(bs_count[:, 1] / bs_count[:, 0]).tolist()):
                a.append([elem/2, int(dim+2), group, "RBS", str(file)])

            for dim, elem in enumerate((
                    np.concatenate([np.array([bm.count_nonzero()]),bs_count[:, 1]])[:-1] / bs_count[:,0]
                    ).tolist()):
                a.append([elem/2, int(dim+1), group, "RBS+", str(file)])
            for dim, elem in enumerate((
                    np.concatenate([np.array([bm.count_nonzero()]),es_count[:, 1]])[:-1] / es_count[:,0]
                    ).tolist()):
                a.append([elem, int(dim+1), group, "RES+", str(file)])

        return pd.DataFrame(a, columns=["count", "dim", "group", "motif", "filename"])

    def get_ES_count(self, file: Path, dimension: int):
        p1 = file / ("ES_D" + str(dimension) + ".npz")
        return np.load(p1)['arr_0'], np.load(p1.with_name(p1.stem + "indptr.npz"))['arr_0']

    def get_BS_count(self, file: Path, dimension: int):
        p1 = file / ("BS_D" + str(dimension) + ".npz")
        return np.load(p1)['arr_0'], np.load(p1.with_name(p1.stem + "indptr.npz"))['arr_0']

    def get_vertex_es_count(self, file: Path, dimension: int):
        p1 = file / ("ES_D" + str(dimension) + "indptr.npz")
        ends = np.load(p1)['arr_0']

        try:
            matrix_path = file / (file.parts[-1] + ".pkl")
            m = load_sparse_matrix_from_pkl(matrix_path)

        except:
            matrix_path = file / "graph.pkl"
            m = load_sparse_matrix_from_pkl(matrix_path)

        complex_file_path = matrix_path.with_name(matrix_path.stem + "-count.h5")
        complex_file = h5py.File(complex_file_path)

        simplex_list = np.array(complex_file['Cells_' + str(dimension)])

        diffs = ends[1:] - ends[:-1]
        vertex_es_count = np.empty((m.shape[0],))

        for i, simplex in tqdm(enumerate(simplex_list)):
            vertex_es_count[simplex[-1]] += diffs[i]

        return vertex_es_count

    def get_vertex_bs_count(self, file: Path, dimension: int):
        p1 = file / ("BS_D" + str(dimension) + "indptr.npz")
        ends = np.load(p1)['arr_0']

        try:
            matrix_path = file / (file.parts[-1] + ".pkl")
            m = load_sparse_matrix_from_pkl(matrix_path)

        except:
            matrix_path = file / "graph.pkl"
            m = load_sparse_matrix_from_pkl(matrix_path)

        complex_file_path = matrix_path.with_name(matrix_path.stem + "-count.h5")
        complex_file = h5py.File(complex_file_path)

        simplex_list = np.array(complex_file['Cells_' + str(dimension)])

        diffs = ends[1:] - ends[:-1]
        vertex_bs_count = np.zeros((m.shape[0],))

        for i, simplex in tqdm(enumerate(simplex_list)):
            vertex_bs_count[simplex[-1]] += diffs[i]

        return vertex_bs_count

    def get_file_matrix(self, file: Path):

        try:
            matrix_path = file / (file.parts[-1] + ".pkl")
            m = load_sparse_matrix_from_pkl(matrix_path)

        except:
            matrix_path = file / "graph.pkl"
            m = load_sparse_matrix_from_pkl(matrix_path)

        return m

    def get_matrix_properties(self, original_file_path, processed_h5_path, dimension):
        matrix = import_connectivity_matrix(original_file_path, dataframe=True)
        m = import_connectivity_matrix(original_file_path, dataframe = False, type = 'csr')
        complex_file_path = processed_h5_path
        complex_file = h5py.File(complex_file_path)


        zones_array = np.array([elem[0] for elem in matrix.index.values])

        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + "indptr.npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + "indptr.npz")
        bspointers = np.load(pbs)['arr_0']
        espointers = np.load(pes)['arr_0']
        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + ".npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + ".npz")
        bsends = np.load(pbs)['arr_0']
        esends = np.load(pes)['arr_0']

        bspairlist = []
        espairlist = []
        simplex_list = np.array(complex_file['Cells_' + str(dimension)])
        for i, simplex in tqdm(enumerate(simplex_list)):
            for element in bsends[bspointers[i]:bspointers[i+1]]:
                bspairlist.append([
                    zones_array[simplex[-1]][3:].replace("_", ""), zones_array[element][3:].replace("_", "")
                ])
            for element in esends[espointers[i]:espointers[i + 1]]:
                espairlist.append([
                    zones_array[simplex[-1]][3:].replace("_", ""), zones_array[element][3:].replace("_", "")
                ])
        bsdf = pd.DataFrame(bspairlist, columns = ['mtype1', 'mtype2'])
        esdf = pd.DataFrame(espairlist, columns = ['mtypesink', 'mtypeextra'])

        return esdf, bsdf

    def get_motif_GID(self, original_file_path, processed_h5_path, dimension):
        matrix = import_connectivity_matrix(original_file_path, dataframe=True)
        complex_file_path = processed_h5_path
        complex_file = h5py.File(complex_file_path)

        zones_array = np.array([elem[0] for elem in matrix.index.values])

        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + "indptr.npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + "indptr.npz")
        bspointers = np.load(pbs)['arr_0']
        espointers = np.load(pes)['arr_0']
        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + ".npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + ".npz")
        bsends = np.load(pbs)['arr_0']
        esends = np.load(pes)['arr_0']

        bslist = []
        eslist = []
        simplex_list = np.array(complex_file['Cells_' + str(dimension)])
        for i, simplex in tqdm(enumerate(simplex_list)):
            for element in bsends[bspointers[i]:bspointers[i+1]]:
                bslist.append(
                    simplex[-1]
                )
            for element in esends[espointers[i]:espointers[i + 1]]:
                eslist.append(
                    simplex[-1]
                )
        bsdf = pd.DataFrame(bslist, columns = ['GID'])
        esdf = pd.DataFrame(eslist, columns = ['GIDsink'])

        return esdf, bsdf

    def get_motif_mtype(self, original_file_path, processed_h5_path, dimension):
        matrix = import_connectivity_matrix(original_file_path, dataframe=True)
        complex_file_path = processed_h5_path
        complex_file = h5py.File(complex_file_path)

        zones_array = np.array([elem[0] for elem in matrix.index.values])

        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + "indptr.npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + "indptr.npz")
        bspointers = np.load(pbs)['arr_0']
        espointers = np.load(pes)['arr_0']
        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + ".npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + ".npz")
        bsends = np.load(pbs)['arr_0']
        esends = np.load(pes)['arr_0']

        bslist = []
        eslist = []
        simplex_list = np.array(complex_file['Cells_' + str(dimension)])
        for i, simplex in tqdm(enumerate(simplex_list)):
            zone = zones_array[simplex[-1]][3:].replace("_", "")
            for element in bsends[bspointers[i]:bspointers[i+1]]:
                bslist.append(
                    zone
                )
            for element in esends[espointers[i]:espointers[i + 1]]:
                eslist.append(
                    zone
                )
        bsdf = pd.DataFrame(bslist, columns = ['mtype'])
        esdf = pd.DataFrame(eslist, columns = ['mtypesink'])

        return esdf, bsdf

    def get_motif_mtype(self, original_file_path, processed_h5_path, dimension):
        matrix = import_connectivity_matrix(original_file_path, dataframe=True)
        complex_file_path = processed_h5_path
        complex_file = h5py.File(complex_file_path)

        zones_array = np.array([elem[0] for elem in matrix.index.values])

        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + "indptr.npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + "indptr.npz")
        bspointers = np.load(pbs)['arr_0']
        espointers = np.load(pes)['arr_0']
        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + ".npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + ".npz")
        bsends = np.load(pbs)['arr_0']
        esends = np.load(pes)['arr_0']

        bslist = []
        eslist = []
        simplex_list = np.array(complex_file['Cells_' + str(dimension)])
        for i, simplex in tqdm(enumerate(simplex_list)):
            zone = zones_array[simplex[-1]][:2]
            for element in bsends[bspointers[i]:bspointers[i+1]]:
                bslist.append(
                    zone
                )
            for element in esends[espointers[i]:espointers[i + 1]]:
                eslist.append(
                    zone
                )
        bsdf = pd.DataFrame(bslist, columns = ['layer'])
        esdf = pd.DataFrame(eslist, columns = ['layersink'])

        return esdf, bsdf

    def get_GID_counts(self, original_file_path, processed_h5_path, dimension):
        matrix = import_connectivity_matrix(original_file_path, dataframe=True)
        complex_file_path = processed_h5_path
        complex_file = h5py.File(complex_file_path)

        zones_array = np.array([elem[0] for elem in matrix.index.values])

        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + "indptr.npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + "indptr.npz")
        bspointers = np.load(pbs)['arr_0']
        espointers = np.load(pes)['arr_0']
        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + ".npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + ".npz")
        bsends = np.load(pbs)['arr_0']
        esends = np.load(pes)['arr_0']
        bs_count = np.zeros((matrix.shape[0],))
        es_count = np.zeros((matrix.shape[0],))
        simplex_list = np.array(complex_file['Cells_' + str(dimension)])
        for i, simplex in tqdm(enumerate(simplex_list)):
            bs_count[simplex[-1]] += bspointers[i+1]-bspointers[i]
            es_count[simplex[-1]] += espointers[i+1]-espointers[i]
        return es_count, bs_count

    def get_morph_counts(self, original_file_path, processed_h5_path, dimension):
        matrix = import_connectivity_matrix(original_file_path, dataframe=True)
        complex_file_path = processed_h5_path
        complex_file = h5py.File(complex_file_path)

        def get_morph(denom):
            return denom[3:].replace("_", "")

        zones_array = np.array([elem[0] for elem in matrix.index.values])
        morph_list = list(set([get_morph(elem[0]) for elem in matrix.index.unique().values]))
        morph_dict = {}
        for i, element in enumerate(morph_list):
            morph_dict.update({element: i})

        bs_morph_counts = np.zeros((len(morph_list),))
        es_morph_counts = np.zeros((len(morph_list),))

        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + "indptr.npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + "indptr.npz")
        bspointers = np.load(pbs)['arr_0']
        espointers = np.load(pes)['arr_0']
        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + ".npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + ".npz")
        bsends = np.load(pbs)['arr_0']
        esends = np.load(pes)['arr_0']

        simplex_list = np.array(complex_file['Cells_' + str(dimension)])
        for i, simplex in tqdm(enumerate(simplex_list)):
            bs_morph_counts[morph_dict[get_morph(zones_array[simplex[-1]])]] += bspointers[i+1]-bspointers[i]
            es_morph_counts[morph_dict[get_morph(zones_array[simplex[-1]])]] += espointers[i + 1] - espointers[i]
        return es_morph_counts, bs_morph_counts, morph_list

    def get_layer_counts(self, original_file_path, processed_h5_path, dimension):
        matrix = import_connectivity_matrix(original_file_path, dataframe=True)
        complex_file_path = processed_h5_path
        complex_file = h5py.File(complex_file_path)

        def get_morph(denom):
            return denom[:2]

        zones_array = np.array([elem[0] for elem in matrix.index.values])
        morph_list = list(set([get_morph(elem[0]) for elem in matrix.index.unique().values]))
        morph_dict = {}
        for i, element in enumerate(morph_list):
            morph_dict.update({element: i})

        bs_morph_counts = np.zeros((len(morph_list),))
        es_morph_counts = np.zeros((len(morph_list),))

        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + "indptr.npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + "indptr.npz")
        bspointers = np.load(pbs)['arr_0']
        espointers = np.load(pes)['arr_0']
        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + ".npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + ".npz")
        bsends = np.load(pbs)['arr_0']
        esends = np.load(pes)['arr_0']

        simplex_list = np.array(complex_file['Cells_' + str(dimension)])
        for i, simplex in tqdm(enumerate(simplex_list)):
            bs_morph_counts[morph_dict[get_morph(zones_array[simplex[-1]])]] += bspointers[i + 1] - bspointers[i]
            es_morph_counts[morph_dict[get_morph(zones_array[simplex[-1]])]] += espointers[i + 1] - espointers[i]
        return es_morph_counts, bs_morph_counts, morph_list

    def get_2d_es_layer_hist_count(self, original_file_path, processed_h5_path, dimension):
        matrix = import_connectivity_matrix(original_file_path, dataframe=True)
        msparse = import_connectivity_matrix(original_file_path, dataframe=False, type='csr')
        complex_file_path = processed_h5_path
        complex_file = h5py.File(complex_file_path)

        def get_morph(denom):
            return denom[:2]

        zones_array = np.array([elem[0] for elem in matrix.index.values])
        morph_list = sorted(list(set([get_morph(elem[0]) for elem in matrix.index.unique().values])))
        morph_dict = {}
        for i, element in enumerate(morph_list):
            morph_dict.update({element: i})

        bs_morph_counts = np.zeros((len(morph_list),))
        es_morph_counts = np.zeros((len(morph_list),))

        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + "indptr.npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + "indptr.npz")
        bspointers = np.load(pbs)['arr_0']
        espointers = np.load(pes)['arr_0']
        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + ".npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + ".npz")
        bsends = np.load(pbs)['arr_0']
        esends = np.load(pes)['arr_0']

        es_morph_matrix = np.zeros((len(morph_list), len(morph_list)))
        bs_morph_matrix = np.zeros((len(morph_list), len(morph_list)))
        biedge_morph_matrix = np.zeros((len(morph_list), len(morph_list)))
        simplex_list = np.array(complex_file['Cells_' + str(dimension)])
        for i, simplex in tqdm(enumerate(simplex_list)):
            sink_morph = morph_dict[get_morph(zones_array[simplex[-1]])]
            for element in bsends[bspointers[i]:bspointers[i+1]]:
                bs_morph_matrix[sink_morph, morph_dict[get_morph(zones_array[element])]] += 1
            for element in esends[espointers[i]:espointers[i+1]]:
                es_morph_matrix[sink_morph, morph_dict[get_morph(zones_array[element])]] += 1
        bmsparse = msparse.multiply(msparse.T).tocsr()
        for i in range(len(bmsparse.indptr)-1):
            row_morph = morph_dict[get_morph(zones_array[i])]
            for col in bmsparse.indices[bmsparse.indptr[i]:bmsparse.indptr[i+1]]:
                biedge_morph_matrix[row_morph, morph_dict[get_morph(zones_array[col])]] += 1
        return es_morph_matrix, bs_morph_matrix, biedge_morph_matrix, morph_list

    def get_2d_es_morph_hist_count(self, original_file_path, processed_h5_path, dimension):
        matrix = import_connectivity_matrix(original_file_path, dataframe=True)
        msparse = import_connectivity_matrix(original_file_path, dataframe=False, type = 'csr')
        complex_file_path = processed_h5_path
        complex_file = h5py.File(complex_file_path)

        def get_morph(denom):
            return denom[3:].lstrip('_')

        zones_array = np.array([elem[0] for elem in matrix.index.values])
        morph_list = sorted(list(set([get_morph(elem[0]) for elem in matrix.index.unique().values])))
        morph_dict = {}
        for i, element in enumerate(morph_list):
            morph_dict.update({element: i})

        bs_morph_counts = np.zeros((len(morph_list),))
        es_morph_counts = np.zeros((len(morph_list),))

        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + "indptr.npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + "indptr.npz")
        bspointers = np.load(pbs)['arr_0']
        espointers = np.load(pes)['arr_0']
        pbs = processed_h5_path.parent / ("BS_D" + str(dimension) + ".npz")
        pes = processed_h5_path.parent / ("ES_D" + str(dimension) + ".npz")
        bsends = np.load(pbs)['arr_0']
        esends = np.load(pes)['arr_0']

        es_morph_matrix = np.zeros((len(morph_list), len(morph_list)))
        bs_morph_matrix = np.zeros((len(morph_list), len(morph_list)))
        biedge_morph_matrix = np.zeros((len(morph_list), len(morph_list)))
        simplex_list = np.array(complex_file['Cells_' + str(dimension)])
        for i, simplex in tqdm(enumerate(simplex_list)):
            sink_morph = morph_dict[get_morph(zones_array[simplex[-1]])]
            for element in bsends[bspointers[i]:bspointers[i+1]]:
                bs_morph_matrix[sink_morph, morph_dict[get_morph(zones_array[element])]] += 1
            for element in esends[espointers[i]:espointers[i+1]]:
                es_morph_matrix[sink_morph, morph_dict[get_morph(zones_array[element])]] += 1
        bmsparse = msparse.multiply(msparse.T).tocsr()
        for i in range(len(bmsparse.indptr)-1):
            row_morph = morph_dict[get_morph(zones_array[i])]
            for col in bmsparse.indices[bmsparse.indptr[i]:bmsparse.indptr[i+1]]:
                biedge_morph_matrix[row_morph, morph_dict[get_morph(zones_array[col])]] += 1
        return es_morph_matrix, bs_morph_matrix, biedge_morph_matrix, morph_list


def create_simplices(dimension: int, instances: int, extra_edges: int, path: Path = None,
                     verbose = True, seed: int = 0, in_place = True):
    """Function that creates simplices instances with some extra edges at a given path.

    :argument dimension: (int) dimension of the simplex to create.
    :argument instances: (int) number of intances to create.
    :argument extra_edges: (int) number of extra edges to add to the simplex.
    :argument path: (pathlib.Path) path to create instances at."""
    np.random.seed(seed)
    if verbose:
        iterator = tqdm(range(instances))
    else:
        iterator = range(instances)

    counts_per_dimension = {}
    for i in range(1, dimension+1):
        counts_per_dimension.update({i:np.zeros((i+1, i+1))})

    for i in iterator:
        matrix = np.triu(np.ones((dimension+1, dimension+1), dtype=bool), 1)
        n = matrix.shape[0]
        v = np.concatenate([np.ones((extra_edges,), dtype=bool), np.zeros((int(n*(n-1)/2-extra_edges),), dtype=bool)])
        np.random.shuffle(v)
        extra = build_triu_matrix(v).T.astype(bool)
        matrix += extra
        matrix1 = sp.csr_matrix(matrix)
        if path is None:
            if in_place:
                path = Path("data/bcounts/dim" + str(dimension) + "/instance" + str(0) + "/graph.flag")
            else:
                path = Path("data/bcounts/dim" + str(dimension) + "/instance" + str(i) + "/graph.flag")
        if in_place:
            f, p, c = save_count_graph_from_matrix(path / ("seed" + str(0) + "/graph.flag"), matrix1, verbose=False)
        else:
            f, p, c = save_count_graph_from_matrix(path / ("seed" + str(i) + "/graph.flag"), matrix1, verbose=False)
        count_file = h5py.File(c)
        for i in range(1, dimension+1):
            simplices = count_file['Cells_'+str(i)]
            for simplex in simplices:
                counts_per_dimension[i] += matrix[simplex].T[simplex].T
    with open(path / "bcount.pkl", 'wb') as f:
        pickle.dump(counts_per_dimension, f)

def create_dags(dimension: int, n_edges: int, instances: int, extra_edges: int, path: Path = None,
                     verbose = True, seed: int = 0, in_place = True):
    """Function that creates DAG instances with some extra edges at a given path.

    :argument dimension: (int) dimension of the simplex to create.
    :argument instances: (int) number of intances to create.
    :argument extra_edges: (int) number of extra edges to add to the simplex.
    :argument path: (pathlib.Path) path to create instances at."""
    np.random.seed(seed)
    if verbose:
        iterator = tqdm(range(instances))
    else:
        iterator = range(instances)

    counts_per_dimension = {}
    for i in range(1, 15):
        counts_per_dimension.update({i:np.zeros((i+1, i+1))})

    for i in iterator:
        n = dimension + 1
        total_possible_edges = int(n * (n - 1) / 2)
        n_unidirectional_edges = (n_edges - extra_edges)
        v = np.concatenate([
                np.ones((n_unidirectional_edges,), dtype=int),
                2*np.ones((extra_edges,), dtype=int),
                np.zeros((total_possible_edges - n_edges,), dtype=int)
        ])
        np.random.shuffle(v)
        extra = np.clip((build_triu_matrix(v) - np.ones((dimension+1, dimension+1))),0,1).T.astype(bool)
        matrix = build_triu_matrix(v).astype(bool)
        matrix += extra
        matrix1 = sp.csr_matrix(matrix)
        if path is None:
            if in_place:
                path = Path("data/bcounts/dim" + str(dimension) + "/instance" + str(0) + "/graph.flag")
            else:
                path = Path("data/bcounts/dim" + str(dimension) + "/instance" + str(i) + "/graph.flag")
        if in_place:
            f, p, c = save_count_graph_from_matrix(path / ("seed" + str(0) + "/graph.flag"), matrix1, verbose=False)
        else:
            f, p, c = save_count_graph_from_matrix(path / ("seed" + str(i) + "/graph.flag"), matrix1, verbose=False)
        count_file = h5py.File(c)
        for i in range(1, len(count_file.keys())+1):
            simplices = count_file['Cells_'+str(i)]
            for simplex in simplices:
                counts_per_dimension[i] += matrix[simplex].T[simplex].T
    with open(path / "bcount.pkl", 'wb') as f:
        pickle.dump(counts_per_dimension, f)


def create_digraphs(dimension: int, n_edges: int, instances: int, extra_edges: int, path: Path = None,
                     verbose = True, seed: int = 0, in_place = True):
    """Function that creates digraph instances with some extra edges at a given path.

    :argument dimension: (int) dimension of the simplex to create.
    :argument instances: (int) number of intances to create.
    :argument extra_edges: (int) number of extra edges to add to the simplex.
    :argument path: (pathlib.Path) path to create instances at."""
    np.random.seed(seed)
    if verbose:
        iterator = tqdm(range(instances))
    else:
        iterator = range(instances)

    counts_per_dimension = {}
    for i in range(1, 15):
        counts_per_dimension.update({i:np.zeros((i+1, i+1))})

    for i in iterator:
        n = dimension + 1
        total_possible_edges = int(n * (n - 1) / 2)
        n_unidirectional_edges = (n_edges - extra_edges)
        v = np.concatenate([
                np.ones((n_unidirectional_edges,), dtype=int),
                2*np.ones((extra_edges,), dtype=int),
                np.zeros((total_possible_edges - n_edges,), dtype=int)
        ])
        np.random.shuffle(v)
        extra = np.clip((build_triu_matrix(v) - np.ones((dimension+1, dimension+1))), 0, 1).T.astype(bool)
        matrix = build_triu_matrix(v).astype(bool)
        matrix += extra
        for i in range(dimension+1):
            for j in range(i, dimension+1):
                if matrix[i,j] and np.random.binomial(1,0.5,1):
                    matrix[i,j] = matrix[j,i]
                    matrix[j,i] = True
        matrix1 = sp.csr_matrix(matrix)
        if path is None:
            if in_place:
                path = Path("data/bcounts/dim" + str(dimension) + "/instance" + str(0) + "/graph.flag")
            else:
                path = Path("data/bcounts/dim" + str(dimension) + "/instance" + str(i) + "/graph.flag")
        if in_place:
            f, p, c = save_count_graph_from_matrix(path / ("seed" + str(0) + "/graph.flag"), matrix1, verbose=False)
        else:
            f, p, c = save_count_graph_from_matrix(path / ("seed" + str(i) + "/graph.flag"), matrix1, verbose=False)
        count_file = h5py.File(c)
        for i in range(1, len(count_file.keys())+1):
            simplices = count_file['Cells_'+str(i)]
            for simplex in simplices:
                counts_per_dimension[i] += matrix[simplex].T[simplex].T
    with open(path / "bcount.pkl", 'wb') as f:
        pickle.dump(counts_per_dimension, f)


class BcountResultManager:
    def __init__(self, path: Path):
        self.processed_file_list = []
        for file in sorted(path.glob("**/ES_count.npz")):
            self.processed_file_list.append(file.parent)

    def get_counts_dataframe(self, group: str):
        a = []
        for file in self.processed_file_list:
            try:
                matrix_path = file / (file.parts[-1] + ".pkl")
                m = load_sparse_matrix_from_pkl(matrix_path)
            except:
                matrix_path = file / "graph.pkl"
                m = load_sparse_matrix_from_pkl(matrix_path)
            bm = m.multiply(m.T)
            es_count = np.load(file / "ES_count.npz")['arr_0']
            a.append([m.shape[0], 0, group, "ES", str(file)])
            a.append([m.shape[0], 0, group, "BS", str(file)])
            a.append([m.shape[0], 0, group, "S", str(file)])
            a.append([bm.count_nonzero(), 1, group, "ES", str(file)])
            a.append([bm.count_nonzero()/2, 1, group, "BS", str(file)])
            for dim, elem in enumerate(es_count[:, 1].tolist()):
                a.append([elem, int(dim+2), group, "ES",str(file)])
            for dim, elem in enumerate(es_count[:, 0].tolist()):
                a.append([elem, int(dim+1), group, "S",str(file)])
            for dim, elem in enumerate(np.nan_to_num(es_count[:, 1]/es_count[:, 0]).tolist()):
                a.append([elem, int(dim+2), group, "RES",str(file)])
            bcounts = pickle.load(open(file/"bcounts.pkl",'rb'))
            bs_count = np.array([elem[-1][-2]/2 for elem in bcounts.values()])
            bs_count = bs_count[1:(len(es_count)+1)]
            for dim, elem in enumerate(bs_count):
                a.append([elem, int(dim+2), group, "BS", str(file)])
            for dim, elem in enumerate(np.nan_to_num(bs_count / es_count[:, 0]).tolist()):
                a.append([elem, int(dim+2), group, "RBS", str(file)])

            for dim, elem in enumerate((
                    np.concatenate([np.array([bm.count_nonzero()]),bs_count])[:-1] / es_count[:,0]
                    ).tolist()):
                a.append([elem, int(dim+1), group, "RBS+", str(file)])
            for dim, elem in enumerate((
                    np.concatenate([np.array([bm.count_nonzero()]),es_count[:, 1]])[:-1] / es_count[:,0]
                    ).tolist()):
                a.append([elem, int(dim+1), group, "RES+", str(file)])

        return pd.DataFrame(a, columns=["count", "dim", "group", "motif", "filename"])