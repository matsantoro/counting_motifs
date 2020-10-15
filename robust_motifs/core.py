import numpy as np
from typing import Callable, List

from .utilities import purged_list, build_triu_matrix


def get_dag_matrices(n: int):
    """Gets all adjacency matrices of DAGs with n nodes.

    :argument n: (int) number of nodes.

    :returns matrices: (List[np.ndarray]) DAG matrices.
    """
    matrices = []
    for i in range(2 ** (int(n * (n - 1) / 2))):
        vector = np.array([int(elem) for elem in np.binary_repr(i, (int(n * (n - 1) / 2)))])
        matrices.append(build_triu_matrix(vector))
    return matrices


def get_representative_dag_matrices(n: int):
    """Gets all adjacency matrices of isomorphism classes of
    DAGs with n nodes.

    :argument n: (int) Number of nodes.

    :returns matrices: (List[np.ndarray]) representative matrices.
    """
    return purged_list(get_dag_matrices(n))


def dag1(matrix: np.ndarray):
    """Returns robust motif DAG1 built from graph expressed in adj matrix.

    :argument matrix: (numpy.ndarray) matrix to build DAG1 upon.

    :returns m2: (numpy.ndarray) DAG1 adjacency matrix.
    """
    m1 = np.hstack([matrix, np.ones((matrix.shape[0],1))])
    m2 = np.vstack([m1, np.zeros((1,m1.shape[0]+1))])
    return m2


def dag2(matrix):
    """Returns robust motif DAG2 built from graph expressed in adj matrix.

    :argument matrix: (numpy.ndarray) matrix to build DAG1 upon.

    :returns m2: (numpy.ndarray) DAG1 adjacency matrix.
    """
    m1 = dag1(matrix)
    for i in range(2**matrix.shape[0]):
        vector = np.array([int(elem) for elem in np.binary_repr(i, matrix.shape[0])])
        vector = np.expand_dims(np.append(vector, 1),1)
        m2 = np.hstack([m1, vector])
        m3 = np.vstack([m2, np.zeros((1,m2.shape[1]))])
        m3[-1][-2] = 1
        yield m3


def get_representative_dag1_matrices(n: int):
    """Gets adjacency matrices of DAG1 family with n nodes.

    :argument n: (int) Number of nodes.

    :returns matrices: (List[np.ndarray]) representative matrices.
    """
    if n == 1:
        print("n=1 manually determined")
        return []
    elif n == 2:
        print("n=2 manually determined")
        return [np.array([[0, 1], [0, 0]])]
    matrices = get_representative_dag_matrices(n-1)
    return [dag1(matrix) for matrix in matrices]


def get_representative_dag2_matrices(n: int):
    """Gets adjacency matrices of DAG2 family with n nodes.

    :argument n: (int) Number of nodes.

    :returns matrices: (List[np.ndarray]) Representative matrices.
    """
    if n == 1:
        print("n=1 manually determined")
        return []
    elif n == 2:
        print("n=2 manually determined")
        return [np.array([[0, 1], [1, 0]])]
    elif n == 3:
        print("n=3 manually determined")
        return [
            np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]]),
            np.array([[0, 1, 1], [0, 0, 1], [0, 1, 0]])
        ]
    else:
        matrices = get_representative_dag_matrices(n-2)
        dag2_matrices = []
        for matrix in matrices:
            dag2_matrices.extend(list(dag2(matrix)))
        return purged_list(dag2_matrices)


def get_subgroups(matrices: List[np.ndarray], functions: List[Callable], get_indices: bool = False):
    """
    Organizes matrices in subgroups according to values in functions.

    :argument matrices: (List[np.ndarray]) List of matrices to divide in subgroups.
    :parameter functions: (List[Callable]) List of functions to divide the matrices with.
    :parameter get_indices: (Bool) Whether to get indices or the whole matrices.

    :returns subgroup_dict: (Dict) List of subgroups with associated indices.
    """
    subgroup_dict = {}
    for index, matrix in enumerate(matrices):
        signature = tuple([function(matrix) for function in functions])
        temp_list = subgroup_dict.get(signature, [])
        if get_indices:
            temp_list.append(index)
        else:
            temp_list.append(matrix)
        subgroup_dict[signature]=temp_list
    return subgroup_dict
