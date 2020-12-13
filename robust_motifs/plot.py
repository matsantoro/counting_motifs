import matplotlib.pyplot as plt
from matplotlib import cm
import networkx
from networkx.drawing.nx_pylab import draw_networkx
import numpy as np
import seaborn as sns
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

def compare_graphs(dictionary_list, n_instances, name,
                   title_list = ['Simplices', 'Column', 'Adjusted ER', 'Shuffled biedges', 'Underlying']):
    dictionary_value_list = [dictionary.values() for dictionary in dictionary_list]
    for i,elem in enumerate(zip(*dictionary_value_list)):
        fig, axs = plt.subplots(1,len(elem)+1, figsize = [30,10])
        simplices = [m[0,-1] for m in elem]
        axs[0].bar(range(len(elem)),simplices)
        d = 0
        for matrix in elem:
            matrix_max = np.max(np.tril(matrix))
            if matrix_max > d:
                d = matrix_max
        for matrix,ax in zip(elem,axs[1:]):
            sns.heatmap(np.tril(matrix)/ n_instances,ax = ax, annot = True, cmap = 'Reds',
                        cbar = (ax==axs[-1]), vmin = 0, vmax = d)
        for title,ax in zip(title_list, axs):
            ax.set_title(title)
        fig.savefig(name + str(i+1), facecolor = "white")


def plot_simplex_counts(dictionary_list, dim, dim_annot, titles, name):
    dictionary_value_list = [list(dictionary.values()) for dictionary in dictionary_list]
    colormap = cm.get_cmap('Set1')
    dimensions = list(dictionary_list[0].keys())
    fig = plt.figure(figsize=[10, 6])
    ax = fig.add_subplot()
    annotation_counter = 0
    for elem, title in zip(dictionary_value_list, titles):
        annotation_counter += 1
        counts = [value[0][-1] for value in elem[:dim + 1]]
        ax.plot(dimensions[:dim + 1], counts, label=title, color=colormap(annotation_counter / len(titles)))
        for j in range(dim_annot, dim + 1):
            ax.annotate(f"{counts[j]:.2E}", (j, counts[j]), (j, ax.get_ylim()[1] / 10 * annotation_counter),
                        backgroundcolor=colormap(annotation_counter / len(titles)))
    ax.legend()
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Simplices")
    fig.savefig(name, facecolor='white')


def plot_biedge_counts(dictionary_list, dim, dim_annot, titles, name):
    dictionary_value_list = [list(dictionary.values()) for dictionary in dictionary_list]
    colormap = cm.get_cmap('Set1')
    dimensions = list(dictionary_list[0].keys())
    fig = plt.figure(figsize=[10, 6])
    ax = fig.add_subplot()
    annotation_counter = 0
    for elem, title in zip(dictionary_value_list, titles):
        annotation_counter += 1
        counts = [np.sum(np.tril(value)) for value in elem[:dim + 1]]
        ax.plot(dimensions[:dim + 1], counts, label=title, color=colormap(annotation_counter / len(titles)))
        for j in range(dim_annot, dim + 1):
            ax.annotate(f"{counts[j]:.2E}", (j, counts[j]), (j, ax.get_ylim()[1] / 10 * annotation_counter),
                        backgroundcolor=colormap(annotation_counter / len(titles)))
    ax.legend()
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Bidirectional edges")
    fig.savefig(name, facecolor='white')


def plot_biedge_cumulative(dictionary_list, dim, dim_annot, titles, name):
    dictionary_value_list = [list(dictionary.values()) for dictionary in dictionary_list]
    colormap = cm.get_cmap('Set1')
    dimensions = list(dictionary_list[0].keys())
    fig = plt.figure(figsize = [10,6])
    ax = fig.add_subplot()
    annotation_counter = 0
    for elem, title in zip(dictionary_value_list, titles):
        annotation_counter += 1
        counts = [np.sum(np.tril(value)) for value in elem[:dim+1]]
        counts = np.cumsum(counts)
        ax.plot(dimensions[:dim+1], counts, label = title, color = colormap(annotation_counter/len(titles)))
        for j in range(dim_annot, dim+1):
            ax.annotate(f"{counts[j]:.2E}", (j, counts[j]), (j, ax.get_ylim()[1]/10*annotation_counter), backgroundcolor = colormap(annotation_counter/len(titles)))
    ax.legend()
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Bidirectional edges")
    fig.savefig(name, facecolor = 'white')


def compare_graphs_percent(dictionary_list, n_instances, name,
                           title_list = ['Simplices', 'Column', 'Adjusted ER', 'Shuffled biedges', 'Underlying']):
    dictionary_value_list = [dictionary.values() for dictionary in dictionary_list]
    for i, elem in enumerate(zip(*dictionary_value_list)):
        fig, axs = plt.subplots(1, len(elem) + 1, figsize=[30, 10])
        simplices = [m[0, -1] for m in elem]
        axs[0].bar(range(len(elem)), simplices)
        for matrix, ax in zip(elem, axs[1:]):
            hmap = np.tril(matrix)
            hmap[hmap == 0] = np.nan
            if np.sum(np.tril(matrix)):
                sns.heatmap(hmap / np.sum(np.tril(matrix)) / n_instances, ax=ax, annot=True, cmap='Reds',
                            cbar=(ax == axs[-1]), vmin=0, vmax=1)
            else:
                sns.heatmap(hmap / n_instances, ax=ax, annot=True, cmap='Reds',
                            cbar=(ax == axs[-1]), vmin=0, vmax=1)

        for title, ax in zip(title_list, axs):
            ax.set_title(title)
        fig.savefig(name + str(i + 1), facecolor="white")


def compare_graphs_normalized(dictionary_list, n_instances, name,
                              title_list = ['Simplices', 'Column', 'Adjusted ER', 'Shuffled biedges', 'Underlying']):
    dictionary_value_list = [dictionary.values() for dictionary in dictionary_list]
    title_list = ['Simplices', 'Column', 'Adjusted ER', 'Shuffled biedges', 'Underlying']
    for i, elem in enumerate(zip(*dictionary_value_list)):
        fig, axs = plt.subplots(1, len(elem) + 1, figsize=[30, 10])
        simplices = [m[0, -1] for m in elem]
        axs[0].bar(range(len(elem)), simplices)
        for matrix, ax in zip(elem, axs[1:]):
            if matrix[0][-1]:
                sns.heatmap(np.tril(matrix) / matrix[0][-1] / n_instances, ax=ax, annot=True, cmap='Reds',
                            cbar=(ax == axs[-1]), vmin=0, vmax=0.2)
            else:
                sns.heatmap(np.tril(matrix) / n_instances, ax=ax, annot=True, cmap='Reds',
                            cbar=(ax == axs[-1]), vmin=0, vmax=0.2)

        for title, ax in zip(title_list, axs):
            ax.set_title(title)
        fig.savefig(name + str(i + 1), facecolor="white")
