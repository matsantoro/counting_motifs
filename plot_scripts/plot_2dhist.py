from robust_motifs.data import ResultManager
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

p = Path("data/ready/average/cons_locs_pathways_mc2_Column")
p1 = Path("data/original/average/cons_locs_pathways_mc2_Column.h5")

r = ResultManager(p)
dimension = 1
esdf, bsdf = r.get_matrix_properties(p1, p / (p.parts[-1] + "-count.h5"), dimension)

fig = plt.figure(figsize = [10, 10])
ax = fig.add_subplot()
sns.jointplot("mtypeextra", "mtypesink", data = esdf, kind = "hist", ax = ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha ='right')
fig.savefig("ES2dhistmorphD" + str(dimension+1), facecolor = "white")

fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot()
sns.jointplot("mtype1", "mtype2", data = bsdf, kind = "hist", ax= ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha ='right')
fig.savefig("BS2dhistmorphD" + str(dimension+1), facecolor = "white")

