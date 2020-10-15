import matplotlib.pyplot as plt
import networkx
from networkx.drawing.nx_pylab import draw_networkx
import numpy as np
from typing import Dict, List


def plot_matrices(matrices: List[np.ndarray], draw_args: Dict = {}):
    """Matplotlib plot of graph expressed by matrices."""
    n_rows = int(len(matrices)/7)+1
    fig = plt.figure(figsize=(14, 2.0 * n_rows))
    for index, matrix in enumerate(matrices):
        ax = fig.add_subplot(n_rows, 8, index+1)
        graph = networkx.DiGraph(matrix)
        draw_networkx(graph, ax=ax, **draw_args)
    plt.show()
