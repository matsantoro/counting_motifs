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
        axs[0].set_xticks(range(len(title_list) - 1))
        axs[0].set_xticklabels(title_list[1:])
        plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
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
        fig.suptitle(f"Bidirectional edge counts per position for various models in dim {i+1}")
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


def plot_biedge_counts(dictionary_list, dim, dim_annot, titles, name, plot_table = True,
                       ylabel = "Bidirectional edges",
                       figtitle = None):
    dictionary_value_list = [list(dictionary.values()) for dictionary in dictionary_list]
    colormap = cm.get_cmap('Set1')
    dimensions = list(dictionary_list[0].keys())
    fig = plt.figure(figsize=[20, 12])
    ax = fig.add_subplot()
    annotation_counter = 0
    table = []
    for elem, title in zip(dictionary_value_list, titles):
        annotation_counter += 1
        counts = [np.sum(np.tril(value)) for value in elem[:dim + 1]]
        ax.plot(dimensions[:dim + 1], counts, label=title, color=colormap(annotation_counter / len(titles)),
               marker = '.', linewidth = 3)
        table.append([f"{count:.2E}" for count in counts])
    table = [list(elem) for elem in zip(*table)]
    ax.legend(loc = 'upper left')
    ax.set_xlabel("Dimension")
    ax.set_ylabel(ylabel)
    if figtitle:
        ax.set_title(figtitle)
    if plot_table:
        ax.table(table, colLabels = titles, rowLabels = [str(elem) for elem in dimensions[:dim + 1]],
             bbox = [1,0,0.30,1])
    fig.savefig(name, facecolor='white', bbox_inches = 'tight')


def plot_biedge_cumulative(dictionary_list, dim, dim_annot, titles, name, plot_table = True,
                           ylabel = "Bidirectional edges cumulative",
                           figtitle = None):
    dictionary_value_list = [list(dictionary.values()) for dictionary in dictionary_list]
    colormap = cm.get_cmap('Set1')
    dimensions = list(dictionary_list[0].keys())
    fig = plt.figure(figsize = [20,12])
    ax = fig.add_subplot()
    annotation_counter = 0
    table = []
    for elem, title in zip(dictionary_value_list, titles):
        annotation_counter += 1
        counts = [np.sum(np.tril(value)) for value in elem[:dim+1]]
        counts = np.cumsum(counts)
        ax.plot(dimensions[:dim+1], counts, label = title, color = colormap(annotation_counter/len(titles)),
                marker = '.', linewidth = 3)
        table.append([f"{count:.2E}" for count in counts])
    table = [list(elem) for elem in zip(*table)]
    ax.legend()
    ax.set_xlabel("Dimension")
    ax.set_ylabel(ylabel)
    if figtitle:
        ax.set_title(figtitle)
    if plot_table:
        ax.table(table, colLabels = titles, rowLabels = [str(elem) for elem in dimensions[:dim + 1]],
             bbox = [1,0,0.30,1])
    fig.savefig(name, facecolor = 'white', bbox_inches='tight')


def plot_bisimplex_counts(dictionary_list, dim, dim_annot, titles, name, plot_table = True,
                       ylabel = "Bisimplices",
                       figtitle = None):
    dictionary_value_list = [list(dictionary.values()) for dictionary in dictionary_list]
    colormap = cm.get_cmap('Set1')
    dimensions = list(dictionary_list[0].keys())
    fig = plt.figure(figsize=[20, 12])
    ax = fig.add_subplot()
    annotation_counter = 0
    table = []
    for elem, title in zip(dictionary_value_list, titles):
        annotation_counter += 1
        counts = [value[-1, -2] for value in elem[:dim + 1]]
        ax.plot(dimensions[:dim + 1], counts, label=title, color=colormap(annotation_counter / len(titles)),
               marker = '.', linewidth = 3)
        table.append([f"{count:.2E}" for count in counts])
    table = [list(elem) for elem in zip(*table)]
    ax.legend(loc = 'upper left')
    ax.set_xlabel("Dimension")
    ax.set_ylabel(ylabel)
    if figtitle:
        ax.set_title(figtitle)
    if plot_table:
        ax.table(table, colLabels = titles, rowLabels = [str(elem) for elem in dimensions[:dim + 1]],
             bbox = [1,0,0.30,1])
    fig.savefig(name, facecolor='white', bbox_inches = 'tight')


def plot_bisimplex_ratio(dictionary_list, dim, dim_annot, titles, name, plot_table=True,
                          ylabel="Bisimplices over simplices",
                          figtitle=None):
    dictionary_value_list = [list(dictionary.values()) for dictionary in dictionary_list]
    colormap = cm.get_cmap('Set1')
    dimensions = list(dictionary_list[0].keys())
    fig = plt.figure(figsize=[20, 12])
    ax = fig.add_subplot()
    annotation_counter = 0
    table = []
    for elem, title in zip(dictionary_value_list, titles):
        annotation_counter += 1
        counts = [value[-1, -2]/value[0, -1] for value in elem[:dim + 1]]
        ax.plot(dimensions[:dim + 1], counts, label=title, color=colormap(annotation_counter / len(titles)),
                marker='.', linewidth=3)
        table.append([f"{count:.2E}" for count in counts])
    table = [list(elem) for elem in zip(*table)]
    ax.legend(loc='upper left')
    ax.set_xlabel("Dimension")
    ax.set_ylabel(ylabel)
    if figtitle:
        ax.set_title(figtitle)
    if plot_table:
        ax.table(table, colLabels=titles, rowLabels=[str(elem) for elem in dimensions[:dim + 1]],
                 bbox=[1, 0, 0.30, 1])
    fig.savefig(name, facecolor='white', bbox_inches='tight')


def compare_graphs_percent(dictionary_list, n_instances, name,
                           title_list = ['Simplices', 'Column', 'Adjusted ER', 'Shuffled biedges', 'Underlying']):
    dictionary_value_list = [dictionary.values() for dictionary in dictionary_list]
    for i, elem in enumerate(zip(*dictionary_value_list)):
        fig, axs = plt.subplots(1, len(elem) + 1, figsize=[30, 10])
        simplices = [m[0, -1] for m in elem]
        axs[0].bar(range(len(elem)), simplices)
        axs[0].set_xticks(range(len(title_list) - 1))
        axs[0].set_xticklabels(title_list[1:])
        plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        d = 0
        for matrix in elem:
            matrix_max = np.max(np.tril(matrix)/ np.sum(np.tril(matrix)))
            if matrix_max > d:
                d = matrix_max
        if d is np.nan:
            d = 0.12
        for matrix, ax in zip(elem, axs[1:]):
            hmap = np.tril(matrix)
            hmap[hmap == 0] = np.nan
            if np.sum(np.tril(matrix)):
                sns.heatmap(hmap / np.sum(np.tril(matrix)) / n_instances, ax=ax, annot=True, cmap='Reds',
                            cbar=(ax == axs[-1]), vmin=0, vmax=d)
            else:
                sns.heatmap(hmap / n_instances, ax=ax, annot=True, cmap='Reds',
                            cbar=(ax == axs[-1]), vmin=0, vmax=0.12)

        for title, ax in zip(title_list, axs):
            ax.set_title(title)
        fig.suptitle(f"Bidirectional edge percentages per position for various models in dim {i+1}")
        fig.savefig(name + str(i + 1), facecolor="white")


def compare_graphs_normalized(dictionary_list, n_instances, name,
                              title_list = ['Simplices', 'Column', 'Adjusted ER', 'Shuffled biedges', 'Underlying']):
    dictionary_value_list = [dictionary.values() for dictionary in dictionary_list]
    for i, elem in enumerate(zip(*dictionary_value_list)):
        fig, axs = plt.subplots(1, len(elem) + 1, figsize=[30, 10])
        simplices = [m[0, -1] for m in elem]
        axs[0].bar(range(len(elem)), simplices)
        axs[0].set_xticks(range(len(title_list)-1))
        axs[0].set_xticklabels(title_list[1:])
        plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        d = 0
        for matrix in elem:
            matrix_max = np.max(np.tril(matrix) / matrix[0,-1])
            if matrix_max > d:
                d = matrix_max
        if d is np.nan:
            d = 0.2
        for matrix, ax in zip(elem, axs[1:]):
            if matrix[0][-1]:
                sns.heatmap(np.tril(matrix) / matrix[0][-1] / n_instances, ax=ax, annot=True, cmap='Reds',
                            cbar=(ax == axs[-1]), vmin=0, vmax=d)
            else:
                sns.heatmap(np.tril(matrix) / n_instances, ax=ax, annot=True, cmap='Reds',
                            cbar=(ax == axs[-1]), vmin=0, vmax=0.2)

        for title, ax in zip(title_list, axs):
            ax.set_title(title)
        fig.suptitle(f"Bidirectional edge counts over simplex count per position for various models in dim {i+1}")
        fig.savefig(name + str(i + 1), facecolor="white")


def compare_graphs_diff(dictionary_list, n_instances, name,
                              title_list = ['Simplices', 'Column', 'Adjusted ER', 'Shuffled biedges', 'Underlying']):
    dictionary_value_list = [dictionary.values() for dictionary in dictionary_list]
    for i, elem in enumerate(zip(*dictionary_value_list)):
        fig, axs = plt.subplots(2, len(elem), figsize=[30, 21])
        simplices = [m[0, -1] for m in elem]
        axs[0][0].bar(range(len(elem)), simplices)
        axs[0][0].set_xticks(range(len(title_list)-1))
        axs[0][0].set_xticklabels(title_list[1:])
        plt.setp(axs[0][0].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        d = 0
        for matrix in elem:
            matrix_max = np.max(np.tril(matrix))
            if matrix_max > d:
                d = matrix_max
        diffs = [elem[0] - comparison for comparison in elem[1:]]
        vmin = 0
        vmax = 0
        for matrix in diffs:
            matrix_max = np.max(np.tril(matrix))
            if matrix_max > vmax:
                vmax = matrix_max
            matrix_min = np.min(np.tril(matrix))
            if matrix_min < vmin:
                vmin = matrix_min
        cap = np.max([np.abs(vmin), np.abs(vmax)])
        for matrix, ax in zip(elem[1:], axs[0][1:]):
            matrix = np.tril(matrix)
            matrix[matrix == 0] = np.nan
            sns.heatmap(matrix / n_instances, ax=ax, annot=True, cmap='Reds',
                        cbar=(ax == axs[0][-1]), vmin=0, vmax=d)
        for matrix, ax in zip(diffs, axs[1][1:]):
            matrix = np.tril(matrix)
            matrix[matrix == 0] = np.nan
            sns.heatmap(matrix / n_instances, ax=ax, annot=True, cmap='bwr',
                        cbar=(ax == axs[1][-1]), vmin=-cap, vmax=cap)
        for title, ax in zip(title_list[2:], axs[0][1:]):
            ax.set_title(title)
        axs[0][0].set_title(title_list[0])
        axs[1][0].set_title(title_list[1])

        sns.heatmap(np.tril(elem[0]) / n_instances, ax=axs[1][0], annot=True, cmap='Reds',
                    cbar=False, vmin=0, vmax=d)
        fig.suptitle("Difference of bidirectional edges absolute counts with other models")
        fig.savefig(name + str(i + 1), facecolor="white")


def compare_graphs_diff_normalized(dictionary_list, n_instances, name,
                              title_list = ['Simplices', 'Column', 'Adjusted ER', 'Shuffled biedges', 'Underlying']):
    dictionary_value_list = [dictionary.values() for dictionary in dictionary_list]
    for i, elem in enumerate(zip(*dictionary_value_list)):
        simplices = [m[0, -1] for m in elem]
        elem = [matrix/matrix[0][-1] if matrix[0][-1] else matrix for matrix in elem]
        fig, axs = plt.subplots(2, len(elem), figsize=[30, 21])
        axs[0][0].bar(range(len(elem)), simplices)
        axs[0][0].set_xticks(range(len(title_list)-1))
        axs[0][0].set_xticklabels(title_list[1:])
        plt.setp(axs[0][0].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        d = 0
        for matrix in elem:
            matrix_max = np.max(np.tril(matrix))
            if matrix_max > d:
                d = matrix_max
        diffs = [elem[0] - comparison for comparison in elem[1:]]
        vmin = 0
        vmax = 0
        for matrix in diffs:
            matrix_max = np.max(np.tril(matrix))
            if matrix_max > vmax:
                vmax = matrix_max
            matrix_min = np.min(np.tril(matrix))
            if matrix_min < vmin:
                vmin = matrix_min
        cap = np.max([np.abs(vmin), np.abs(vmax)])
        for matrix, ax in zip(elem[1:], axs[0][1:]):
            matrix = np.tril(matrix)
            matrix[matrix == 0] = np.nan
            sns.heatmap(matrix / n_instances, ax=ax, annot=True, cmap='Reds',
                        cbar=(ax == axs[0][-1]), vmin=0, vmax=d)
        for matrix, ax in zip(diffs, axs[1][1:]):
            matrix = np.tril(matrix)
            matrix[matrix == 0] = np.nan
            sns.heatmap(matrix / n_instances, ax=ax, annot=True, cmap='bwr',
                        cbar=(ax == axs[1][-1]), vmin=-cap, vmax=cap)
        for title, ax in zip(title_list[2:], axs[0][1:]):
            ax.set_title(title)
        axs[0][0].set_title(title_list[0])
        axs[1][0].set_title(title_list[1])

        sns.heatmap(np.tril(elem[0]) / n_instances, ax=axs[1][0], annot=True, cmap='Reds',
                    cbar=False, vmin=0, vmax=d)

        fig.suptitle("Difference of bidirectional edges over simplices with other models")
        fig.savefig(name + str(i + 1), facecolor="white")


def compare_graphs_diff_percent(dictionary_list, n_instances, name,
                              title_list = ['Simplices', 'Column', 'Adjusted ER', 'Shuffled biedges', 'Underlying']):
    dictionary_value_list = [dictionary.values() for dictionary in dictionary_list]
    for i, elem in enumerate(zip(*dictionary_value_list)):
        simplices = [m[0, -1] for m in elem]
        elem = [matrix/np.sum(np.tril(matrix)) if matrix[0][-1] else matrix for matrix in elem]
        fig, axs = plt.subplots(2, len(elem), figsize=[30, 21])

        axs[0][0].bar(range(len(elem)), simplices)
        axs[0][0].set_xticks(range(len(title_list)-1))
        axs[0][0].set_xticklabels(title_list[1:])
        plt.setp(axs[0][0].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        d = 0
        for matrix in elem:
            matrix_max = np.max(np.tril(matrix))
            if matrix_max > d:
                d = matrix_max
        diffs = [elem[0] - comparison for comparison in elem[1:]]
        vmin = 0
        vmax = 0
        for matrix in diffs:
            matrix_max = np.max(np.tril(matrix))
            if matrix_max > vmax:
                vmax = matrix_max
            matrix_min = np.min(np.tril(matrix))
            if matrix_min < vmin:
                vmin = matrix_min

        cap = np.max([np.abs(vmin), np.abs(vmax)])
        for matrix, ax in zip(elem[1:], axs[0][1:]):
            matrix = np.tril(matrix)
            matrix[matrix == 0] = np.nan
            sns.heatmap(matrix / n_instances, ax=ax, annot=True, cmap='Reds',
                        cbar=(ax == axs[0][-1]), vmin=0, vmax=d)
        for matrix, ax in zip(diffs, axs[1][1:]):
            matrix = np.tril(matrix)
            matrix[matrix == 0] = np.nan
            sns.heatmap(matrix / n_instances, ax=ax, annot=True, cmap='bwr',
                        cbar=(ax == axs[1][-1]), vmin=-cap, vmax=cap)
        for title, ax in zip(title_list[2:], axs[0][1:]):
            ax.set_title(title)
        axs[0][0].set_title(title_list[0])
        axs[1][0].set_title(title_list[1])

        sns.heatmap(np.tril(elem[0]) / n_instances, ax=axs[1][0], annot=True, cmap='Reds',
                    cbar=False, vmin=0, vmax=d)

        fig.suptitle("Difference of bidirectional edges percentages with other models")
        fig.savefig(name + str(i + 1), facecolor="white")
