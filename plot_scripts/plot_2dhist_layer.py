from robust_motifs.data import ResultManager
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

p = Path("data/ready/average/cons_locs_pathways_mc2_Column")
p1 = Path("data/original/average/cons_locs_pathways_mc2_Column.h5")

r = ResultManager(p)
for dimension in range(1,7):
    esdf, bsdf, edgedf, morphs = r.get_2d_es_layer_hist_count(p1, p / (p.parts[-1] + "-count.h5"), dimension)
    esdf[esdf == 0] = np.nan
    bsdf[bsdf == 0] = np.nan
    edgedf[edgedf == 0] = np.nan
    bsdf -= np.diag(np.diag(bsdf))/2
    edgedf -= np.diag(np.diag(edgedf))/2
    morphs = [elem if elem != "L2" else "L23" for elem in morphs]
    fig = plt.figure(figsize = [7, 7])
    ax = fig.add_subplot()
    sns.heatmap(esdf, xticklabels = morphs, yticklabels = morphs, cmap = 'viridis')
    ax.set_title("Extended simplices count per layer on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("ESlayer2dhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [7, 7])
    ax = fig.add_subplot()
    sns.heatmap(bsdf, xticklabels = morphs, yticklabels = morphs,cmap= 'viridis')
    ax.set_title("Bisimplices count per layer on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("BSlayer2dhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [7, 7])
    ax = fig.add_subplot()
    sns.heatmap(edgedf, xticklabels = morphs, yticklabels = morphs,cmap= 'viridis')
    ax.set_title("Bidirectional edge count per layer on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("BElayer2dhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [7, 7])
    ax = fig.add_subplot()
    sns.heatmap(np.divide(bsdf,edgedf), xticklabels = morphs, yticklabels = morphs,cmap= 'viridis')
    ax.set_title("Bisimplex count over bidirectional edge count per layer on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("BSBElayer2dhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [7, 7])
    ax = fig.add_subplot()
    sns.heatmap(np.divide(esdf,edgedf), xticklabels = morphs, yticklabels = morphs,cmap= 'viridis')
    ax.set_title("E. simplex count over bidirectional edge count per layer on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("ESBElayer2dhistD" + str(dimension+1), facecolor = "white")

    esdf[np.isnan(esdf)] = 0
    bsdf[np.isnan(bsdf)] = 0
    edgedf[np.isnan(edgedf)] = 0

    fig = plt.figure(figsize = [8, 6])
    ax = fig.add_subplot()
    ax.bar(range(len(morphs)), np.sum(esdf, axis = 1), tick_label = morphs)
    ax.set_title("E. simplex sink count per layer on dim " + str(dimension +1))
    ax.set_yscale("log")
    fig.savefig("ESsinklayerhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [8, 6])
    ax = fig.add_subplot() # sink
    ax.bar(range(len(morphs)), np.sum(edgedf, axis = 1), tick_label = morphs)
    ax.set_title("Bidirectional edge count per layer on dim " + str(dimension +1))
    ax.set_yscale("log")
    fig.savefig("BElayerhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [8, 6])
    ax = fig.add_subplot() # end
    ax.bar(range(len(morphs)), np.sum(esdf, axis = 0), tick_label = morphs)
    ax.set_title("E. simplex end count per layer on dim " + str(dimension +1))
    ax.set_yscale("log")
    fig.savefig("ESendlayerhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [8, 6])
    ax = fig.add_subplot() # end
    ax.bar(range(len(morphs)), np.sum(bsdf, axis = 0), tick_label = morphs)
    ax.set_title("Bisimplex sink count per layer on dim " + str(dimension +1))
    ax.set_yscale("log")
    fig.savefig("BSsinklayerhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [8, 6])
    ax = fig.add_subplot() # end
    ax.bar(range(len(morphs)), np.divide(np.sum(bsdf, axis = 0), np.sum(edgedf, axis = 0)), tick_label = morphs)
    ax.set_title("Bisimplex sink over bidegree counts per layer on dim " + str(dimension +1))
    fig.savefig("BSsinkBElayerhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [8, 6])
    ax = fig.add_subplot() # end
    ax.bar(range(len(morphs)), np.divide(np.sum(esdf, axis = 1), np.sum(edgedf, axis = 0)), tick_label = morphs)
    ax.set_title("Extended simplex sink over bidegree counts per layer on dim " + str(dimension +1))
    fig.savefig("ESsinkBElayerhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [8, 6])
    ax = fig.add_subplot() # end
    ax.bar(range(len(morphs)), np.divide(np.sum(esdf, axis = 0), np.sum(edgedf, axis = 0)), tick_label = morphs)
    ax.set_title("Extended simplex end over bidegree counts per layer on dim " + str(dimension +1))
    fig.savefig("ESendBElayerhistD" + str(dimension+1), facecolor = "white")


