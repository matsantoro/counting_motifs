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
    esdf, bsdf, morphs = r.get_2d_es_morph_hist_count(p1, p / (p.parts[-1] + "-count.h5"), dimension)
    esdf[esdf == 0] = np.nan
    bsdf[bsdf == 0] = np.nan
    fig = plt.figure(figsize = [13, 13])
    ax = fig.add_subplot()
    sns.heatmap(esdf, xticklabels = morphs, yticklabels = morphs, cmap = 'viridis')
    ax.set_title("Extended simplices count per layer on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("ESlayer2dhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize = [13, 13])
    ax = fig.add_subplot()
    sns.heatmap(bsdf, xticklabels = morphs, yticklabels = morphs,cmap= 'viridis')
    ax.set_title("Bisimplices count per layer on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("BSlayer2dhistD" + str(dimension+1), facecolor = "white")
