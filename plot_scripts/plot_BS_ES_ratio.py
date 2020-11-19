from robust_motifs.data import ResultManager
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

p_average = Path("data/ready/average/cons_locs_pathways_mc2_Column")
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

count_a = np.squeeze(np.array(bm_average.sum(axis=1)))
count_b = np.squeeze(np.array(bm_adjusted.sum(axis=1)))
count_c = np.squeeze(np.array(bm_pathways.sum(axis=1)))

for dimension in range(1,7):
    es_a = r_average.get_vertex_es_count(p_average, dimension)
    bs_a = r_average.get_vertex_bs_count(p_average, dimension)
    es_b = r_adjusted.get_vertex_es_count(p_adjusted, dimension)
    bs_b = r_adjusted.get_vertex_bs_count(p_adjusted, dimension)
    es_c = r_pathways.get_vertex_es_count(p_pathways, dimension)
    bs_c = r_pathways.get_vertex_bs_count(p_pathways, dimension)

    dflist = []
    for bidegree, count1, count2 in zip(count_a, bs_a, es_a):
        dflist.append([bidegree,count1/2/count2,"average"])
    for bidegree, count1, count2 in zip(count_b, bs_b, es_b):
        dflist.append([bidegree, count1/2/count2, "adjusted"])
    for bidegree, count1, count2 in zip(count_c, bs_c, es_c):
        dflist.append([bidegree, count1/2/count2, "pathways"])

    df = pd.DataFrame(dflist, columns = ["Bidegree","BS/ES ratio", "group"])
    
    fig, axes = plt.subplots(3, 1, figsize = [12.8, 10.8], sharex = False)
    fig.suptitle("Bisimplices per sink bidegree, dim " + str(dimension+1))
    sns.barplot(x = "Bidegree", y = "BS/ES ratio", data = df[df["group"] == "average"], ax = axes[0])
    axes[0].set_title("Average rat")
    axes[0].set_xlabel("")
    sns.barplot(x = "Bidegree", y = "BS/ES ratio", data = df[df["group"] == "adjusted"], ax = axes[1])
    axes[1].set_title("ER adjusted")
    axes[1].set_xlabel("")
    sns.barplot(x = "Bidegree", y = "BS/ES ratio", data = df[df["group"] == "pathways"], ax = axes[2])
    axes[2].set_title("Pathway-shuffled")
    axes[2].set_xlabel("")
    fig.savefig("BS_ratio_barplot_split_D" + str(dimension+1))

    fig = plt.figure()
    ax = fig.add_subplot()
    sns.barplot(x = "Bidegree", y = "BS/ES ratio", data = df, hue = "group", ax = ax, ci = None)
    ax.set_xlim([0,20])
    ax.set_title("ES/BS ratio per sink bidegree")
    ax.set_yscale("log")
    fig.savefig("BS_ratio_barplot_log_D" + str(dimension+1))
