from robust_motifs.data import ResultManager
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Plot bidegree distributions of control models.
# This script was modified to plot indegree and outdegree distributions.

p_average = Path("data/ready/average/cons_locs_pathways_mc0_Column")
p_adjusted = Path("data/ready/controls_1/adjusted/seed_0")
p_pathways = Path("data/ready/controls_1/pathway/seed_0")
p_er = Path("data/ready/controls_1/shuffled/seed_0")

r_average = ResultManager(p_average)
r_adjusted = ResultManager(p_adjusted)
r_pathways = ResultManager(p_pathways)
r_er = ResultManager(p_er)

m_average = r_average.get_file_matrix(p_average)
bm_average = m_average.multiply(m_average.T)
m_adjusted = r_adjusted.get_file_matrix(p_adjusted)
bm_adjusted = m_adjusted.multiply(m_adjusted.T)
m_pathways = r_pathways.get_file_matrix(p_pathways)
bm_pathways = m_pathways.multiply(m_pathways.T)
m_er = r_er.get_file_matrix(p_er)
bm_er = m_er.multiply(m_er.T)


count_a = np.squeeze(np.array(bm_average.sum(axis=0)))
count_b = np.squeeze(np.array(bm_adjusted.sum(axis=0)))
count_c = np.squeeze(np.array(bm_pathways.sum(axis=0)))
count_d = np.squeeze(np.array(bm_er.sum(axis=0)))

dfa = pd.DataFrame(count_a, columns = ["Bidegree"])
dfb = pd.DataFrame(count_b, columns = ["Bidegree"])
dfc = pd.DataFrame(count_c, columns = ["Bidegree"])
dfd = pd.DataFrame(count_d, columns = ["Bidegree"])

fig, axes = plt.subplots(4,1, sharex = True, figsize = [8.4, 6.4])
sns.histplot(x = "Bidegree", data = dfa, ax = axes[0], bins = np.arange(50))
sns.histplot(x = "Bidegree", data = dfb, ax = axes[2], bins = np.arange(50))
sns.histplot(x = "Bidegree", data = dfc, ax = axes[3], bins = np.arange(50))
sns.histplot(x = "Bidegree", data = dfd, ax = axes[1], bins = np.arange(50))
axes[0].set_title("Average")
axes[1].set_title("ER")
axes[2].set_title("Adjusted ER")
axes[3].set_title("Pathways")
axes[0].set_xlabel("")
axes[1].set_xlabel("")
axes[2].set_xlabel("")
# bidirectional edge only
axes[0].annotate("BE count: " + str(int(bm_average.count_nonzero()/2)), (40,2000))
axes[1].annotate("BE count: " + str(int(bm_er.count_nonzero()/2)), (40,5000))
axes[2].annotate("BE count: " + str(int(bm_adjusted.count_nonzero()/2)), (40,3000))
axes[3].annotate("BE count: " + str(int(bm_pathways.count_nonzero()/2)), (40,3000))
fig.subplots_adjust(hspace = 0.4)
fig.savefig("bidegree_distr", facecolor = "white")
