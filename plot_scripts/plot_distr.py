from robust_motifs.data import ResultManager
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

p_average = Path("data/ready/average/cons_locs_pathways_mc0_Column")
p_adjusted = Path("data/ready/controls_1/adjusted/seed_0")
p_pathways = Path("data/ready/controls_1/pathway/seed_0")

r_average = ResultManager(p_average)
r_adjusted = ResultManager(p_adjusted)
r_pathways = ResultManager(p_pathways)

m_average = r_average.get_file_matrix(p_average)
bm_average = m_average.multiply(m_average.T)
m_adjusted = r_adjusted.get_file_matrix(p_adjusted)
bm_adjusted = m_adjusted.multiply(m_adjusted.T)
m_pathways = r_pathways.get_file_matrix(p_pathways)
bm_pathways = m_pathways.multiply(m_pathways.T)

count_a = np.squeeze(np.array(m_average.sum(axis=0)))
count_b = np.squeeze(np.array(m_adjusted.sum(axis=0)))
count_c = np.squeeze(np.array(m_pathways.sum(axis=0)))

dfa = pd.DataFrame(count_a, columns = ["In degree"])
dfb = pd.DataFrame(count_b, columns = ["In degree"])
dfc = pd.DataFrame(count_c, columns = ["In degree"])

fig, axes = plt.subplots(3,1, sharex = False, figsize = [8.4, 6.4])
sns.histplot(x = "In degree", data = dfa, ax = axes[0])
sns.histplot(x = "In degree", data = dfb, ax = axes[1], bins = np.arange(320))
sns.histplot(x = "In degree", data = dfc, ax = axes[2])
axes[0].set_title("Average")
axes[1].set_title("Adjusted")
axes[2].set_title("Pathways")

fig.savefig("indegree_distr", facecolor = "white")