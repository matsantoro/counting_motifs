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
    esdf, bsdf, edgedf, morphs = r.get_2d_es_morph_hist_count(p1, p / (p.parts[-1] + "-count.h5"), dimension)
    esdf[esdf == 0] = np.nan
    bsdf[bsdf == 0] = np.nan
    edgedf[edgedf == 0] = np.nan
    edgedf -= np.diag(np.diag(edgedf))/2
    bsdf -= np.diag(np.diag(bsdf))/2
    fig = plt.figure(figsize = [13, 13])
    ax = fig.add_subplot()
    sns.heatmap(esdf, xticklabels = morphs, yticklabels = morphs, cmap = 'viridis')
    ax.set_title("Extended simplices count per mtype on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("ESmtype2dhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [13, 13])
    ax = fig.add_subplot()
    sns.heatmap(bsdf, xticklabels = morphs, yticklabels = morphs,cmap= 'viridis')
    ax.set_title("Bisimplices count per mtype on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("BSmtype2dhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [13, 13])
    ax = fig.add_subplot()
    sns.heatmap(edgedf, xticklabels = morphs, yticklabels = morphs,cmap= 'viridis')
    ax.set_title("Bidirectional edge count per mtype on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("BEmtype2dhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [13, 13])
    ax = fig.add_subplot()
    sns.heatmap(np.divide(bsdf,edgedf), xticklabels = morphs, yticklabels = morphs,cmap= 'viridis')
    ax.set_title("Bisimplex count over bidirectional edge count per mtype on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("BSBEmtype2dhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [13, 13])
    ax = fig.add_subplot()
    sns.heatmap(np.divide(esdf,edgedf), xticklabels = morphs, yticklabels = morphs,cmap= 'viridis')
    ax.set_title("E. simplex count over bidirectional edge count per mtype on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("ESBEmtype2dhistD" + str(dimension+1), facecolor = "white")

    esdf[np.isnan(esdf)] = 0
    bsdf[np.isnan(bsdf)] = 0
    edgedf[np.isnan(edgedf)] = 0

    fig = plt.figure(figsize = [13, 7])
    ax = fig.add_subplot()
    ax.bar(range(len(morphs)), np.sum(esdf, axis = 1), tick_label = morphs)
    ax.set_title("E. simplex sink count per mtype on dim " + str(dimension +1))
    fig.savefig("ESsinkmtypehistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [13, 7])
    ax = fig.add_subplot()
    ax.bar(range(len(morphs)), np.sum(esdf, axis = 1), tick_label = morphs)
    ax.set_title("Bidirectional edge count per mtype on dim " + str(dimension +1))
    fig.savefig("BEmtypehistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [13, 7])
    ax = fig.add_subplot()
    ax.bar(range(len(morphs)), np.sum(esdf, axis = 0), tick_label = morphs)
    ax.set_title("E. simplex end count per mtype on dim " + str(dimension +1))
    fig.savefig("ESendmtypehistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [13, 7])
    ax = fig.add_subplot()
    ax.bar(range(len(morphs)), np.sum(bsdf, axis = 0), tick_label = morphs)
    ax.set_title("Bisimplex sink count per mtype on dim " + str(dimension +1))
    fig.savefig("BSsinkmtypehistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [13, 7])
    ax = fig.add_subplot()
    ax.bar(range(len(morphs)), np.divide(np.sum(bsdf, axis = 0), np.sum(edgedf, axis = 0)), tick_label = morphs)
    ax.set_title("Bisimplex sink over bidegree counts per mtype on dim " + str(dimension +1))
    fig.savefig("BSsinkBEmtypehistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [13, 7])
    ax = fig.add_subplot()
    ax.bar(range(len(morphs)), np.divide(np.sum(esdf, axis = 1), np.sum(bsdf, axis = 0)), tick_label = morphs)
    ax.set_title("Extended simplex sink over bidegree counts per mtype on dim " + str(dimension +1))
    fig.savefig("ESsinkBEmtypehistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [13, 7])
    ax = fig.add_subplot()
    ax.bar(range(len(morphs)), np.divide(np.sum(esdf, axis = 0), np.sum(bsdf, axis = 0)), tick_label = morphs)
    ax.set_title("Extended simplex end over bidegree counts per mtype on dim " + str(dimension +1))
    fig.savefig("ESendBEmtypehistD" + str(dimension+1), facecolor = "white")


