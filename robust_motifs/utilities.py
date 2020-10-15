import itertools as it
import numpy as np
from tqdm import tqdm
from typing import Callable, List, Optional

from .invariants import n_components, n_edges, graph_directionality


def get_gen(x: int):
    """ Given x, return the number n such that n*(n-1)/2 = x

    :argument x: (int) Parameter of the equation.

    :returns n: (int) Solution of the equation.
    """
    return int(1/2 + np.sqrt(2*x + 1/4))


def build_triu_matrix(vector: np.ndarray):
    """Returns the upper-triangular matrix with 0 diagonal
    such that components are filled up from left to right
    and top to bottom.

    :argument vector: (numpy.ndarray) Vector with arguments to be distributed
        in the matrix.

    :returns matrix: (numpy.ndarray) Upper-triangular matrix. components of the
        vector are placed from left to right and from top to bottom.
    """
    n = get_gen(len(vector))
    matrix = np.zeros((n, n))
    counter = 0
    for row in range(n-1):
        matrix[row][(row+1):] = vector[counter:(counter + n-row-1)]
        counter += n-row-1
    return matrix


def get_pos(n: int, jitter: bool = True):
    """Get lattice positions of nodes to display them.

    :argument n: (int) number of positions to get.
    :parameter jitter: (bool) whether nodes should be jittered
        for ease of visualization.
    """
    point = [0, 0]
    for _ in range(n):
        c = point.copy()
        if jitter:
            c[0] += np.random.normal(scale=0.1)
            c[1] += np.random.normal(scale=0.1)
        yield c
        if point[1] == 0:
            point[1] = point[0] + 1
            point[0] = 0
        else:
            point[1] -= 1
            point[0] += 1


def are_psimilar(m1: np.ndarray, m2: np.ndarray,
                 invariants: Optional[List[Callable]] = [
                     n_components, n_edges, graph_directionality
                 ]):
    """Check whether two matrices are permutation - similar.

    :argument m1: (numpy.ndarray) First matrix.
    :argument m2: (numpy.ndarray) Second matrix.
    :argument invariants: (Optional[List[Callable]]) List of invariants to check before
    checking similarity.

    :returns boolean: (bool) Boolean that is True when the matrices are found similar.
    """
    for invariant in invariants:
        if not invariant(m1) == invariant(m2):
            return False
    for permutation in it.permutations(np.arange(m1.shape[0])):
        permuted_m1 = m1[permutation, :].T[permutation, :].T
        if not (permuted_m1 - m2).any():
            return True
    return False


def _forbidden_list(matrices: List[np.ndarray]):
    """Returns list of indices of redundant matrices from matrices.

    :argument matrices: (List[np.ndarray]) List of square matrices with the same size.

    :returns forbidden: (List[int]) List of indices of forbidden matrices.
    """
    forbidden = []
    for index_pair in tqdm(list(it.combinations(range(len(matrices)), 2))):
        if not index_pair[0] in forbidden:
            if are_psimilar(matrices[index_pair[0]], matrices[index_pair[1]]):
                forbidden.append(index_pair[1])
    return forbidden


def purged_list(matrices):
    """Returns the list of class representatives of matrices.

    :argument matrices: (List[np.ndarray]) List of square matrices with the same size.

    :returns representatives: (List[np.ndarray]) List of representatives of the isomorphism
        classes of original list.
    """
    forbidden = _forbidden_list(matrices)
    return [matrices[index] for index in range(len(matrices)) if index not in forbidden]
