from robust_motifs.data import ResultManager
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Plot outdegree distribution of control models.
# This script was modified to plot indegree distribution as well.

p_average = Path("data/ready/average/cons_locs_pathways_mc0_Column")
p_adjusted = Path("data/bcounts/bshuffled_1/seed_0")
p_pathways = Path("data/bcounts/underlying_1/seed_0")

r_average = ResultManager(p_average)
r_adjusted = ResultManager(p_adjusted)
r_pathways = ResultManager(p_pathways)

m_average = r_average.get_file_matrix(p_average)
bm_average = m_average.multiply(m_average.T)
m_adjusted = r_adjusted.get_file_matrix(p_adjusted)
bm_adjusted = m_adjusted.multiply(m_adjusted.T)
m_pathways = r_pathways.get_file_matrix(p_pathways)
bm_pathways = m_pathways.multiply(m_pathways.T)


count_a = np.squeeze(np.array(m_average.sum(axis=1)))
count_b = np.squeeze(np.array(m_adjusted.sum(axis=1)))
count_c = np.squeeze(np.array(m_pathways.sum(axis=1)))

dfa = pd.DataFrame(count_a, columns = ["Outdegree"])
dfb = pd.DataFrame(count_b, columns = ["Outdegree"])
dfc = pd.DataFrame(count_c, columns = ["Outdegree"])

fig, axes = plt.subplots(3,1, sharex = True, figsize = [8.4, 6.4])
sns.histplot(x = "Outdegree", data = dfa, ax = axes[0], bins = np.arange(300))
sns.histplot(x = "Outdegree", data = dfb, ax = axes[1], bins = np.arange(300))
sns.histplot(x = "Outdegree", data = dfc, ax = axes[2], bins = np.arange(300))
axes[0].set_title("Average")
axes[1].set_title("Bishuffled")
axes[2].set_title("Underlying")
axes[0].set_xlabel("")
axes[1].set_xlabel("")
# bidirectional edge only
# axes[0].annotate("BE count: " + str(bm_average.count_nonzero()), (40,2000))
# axes[1].annotate("BE count: " + str(bm_adjusted.count_nonzero()), (40,2000))
# axes[2].annotate("BE count: " + str(bm_pathways.count_nonzero()), (40,2000))
# axes[3].annotate("BE count: " + str(bm_pathways.count_nonzero()), (40,3000))
fig.subplots_adjust(hspace = 0.4)
fig.savefig("outdegree_distr_bcount", facecolor = "white")
