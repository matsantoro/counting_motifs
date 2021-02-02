from robust_motifs.data import ResultManager
from pathlib import Path
import matplotlib.pyplot as plt


# plot 1d layer histograms of ES/BS sink.

p = Path("data/ready/average/cons_locs_pathways_mc2_Column")
p1 = Path("data/original/average/cons_locs_pathways_mc2_Column.h5")

r = ResultManager(p)
for dimension in range(1,7):
    esdf, bsdf, morphs = r.get_layer_counts(p1, p / (p.parts[-1] + "-count.h5"), dimension)

    fig = plt.figure(figsize = [10, 6])
    ax = fig.add_subplot()
    ax.bar(range(len(morphs)), esdf, tick_label = morphs)
    ax.set_ylabel("Count")
    ax.set_title("Extended simplices count per layer on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("ESlayerhistD" + str(dimension+1), facecolor = "white")

    fig = plt.figure(figsize=[10, 6])
    ax = fig.add_subplot()
    ax.bar(range(len(morphs)), bsdf, tick_label = morphs)
    ax.set_ylabel("Count")
    ax.set_title("Bisimplices count per layer on dim "+ str(dimension + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    fig.savefig("BSlayerhistD" + str(dimension+1), facecolor = "white")
