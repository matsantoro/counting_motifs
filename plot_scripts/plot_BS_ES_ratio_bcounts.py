from robust_motifs.data import ResultManager
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

p_average = Path("data/ready/average/cons_locs_pathways_mc2_Column")
p_adjusted = Path("data/ready/controls_1/adjusted/seed_0")
p_pathways = Path("data/ready/controls_1/pathway/seed_0")
p_bshuffled = Path("data/bcounts/bshuffled_1/seed_0")
p_underlying = Path("data/bcounts/underlying/seed_0")

r_average = ResultManager(p_average)
r_adjusted = ResultManager(p_adjusted)
r_pathways = ResultManager(p_pathways)
r_bshuffled = ResultManager(p_bshuffled)
r_underlying = ResultManager(p_underlying)

m_average = r_average.get_file_matrix(p_average)
bm_average = m_average.multiply(m_average.T)
m_adjusted = r_adjusted.get_file_matrix(p_adjusted)
bm_adjusted = m_adjusted.multiply(m_adjusted.T)
m_pathways = r_pathways.get_file_matrix(p_pathways)
bm_pathways = m_pathways.multiply(m_pathways.T)
m_bshuffled = r_bshuffled.get_file_matrix(p_bshuffled)
bm_bshuffled = m_bshuffled.multiply(m_bshuffled.T)
m_underlying = r_underlying.get_file_matrix(p_underlying)
bm_underlying = m_underlying.multiply(m_underlying.T)

count_a = np.squeeze(np.array(bm_average.sum(axis=1)))
count_b = np.squeeze(np.array(bm_adjusted.sum(axis=1)))
count_c = np.squeeze(np.array(bm_pathways.sum(axis=1)))
count_d = np.squeeze(np.array(bm_bshuffled.sum(axis=1)))
count_e = np.squeeze(np.array(bm_underlying.sum(axis=1)))

for dimension in range(1,7):
    es_a = r_average.get_vertex_es_count(p_average, dimension)
    bs_a = r_average.get_vertex_bs_count(p_average, dimension)
    es_b = r_adjusted.get_vertex_es_count(p_adjusted, dimension)
    bs_b = r_adjusted.get_vertex_bs_count(p_adjusted, dimension)
    es_c = r_pathways.get_vertex_es_count(p_pathways, dimension)
    bs_c = r_pathways.get_vertex_bs_count(p_pathways, dimension)
    es_d = r_bshuffled.get_vertex_es_count(p_bshuffled, dimension)
    bs_d = r_bshuffled.get_vertex_bs_count(p_bshuffled, dimension)
    es_e = r_underlying.get_vertex_es_count(p_underlying, dimension)
    bs_e = r_underlying.get_vertex_bs_count(p_underlying, dimension)

    dflist = []
    for bidegree, count1, count2 in zip(count_a, bs_a, es_a):
        dflist.append([bidegree,count1/2/count2,"average"])
    for bidegree, count1, count2 in zip(count_b, bs_b, es_b):
        dflist.append([bidegree, count1/2/count2, "adjusted"])
    for bidegree, count1, count2 in zip(count_c, bs_c, es_c):
        dflist.append([bidegree, count1/2/count2, "pathways"])
    for bidegree, count1, count2 in zip(count_d, bs_d, es_d):
        dflist.append([bidegree, count1/2/count2, "bshuffled"])
    for bidegree, count1, count2 in zip(count_e, bs_e, es_e):
        dflist.append([bidegree, count1/2/count2, "underlying"])

    df = pd.DataFrame(dflist, columns = ["Bidegree","BS/ES ratio", "group"])

    fig = plt.figure()
    ax = fig.add_subplot()
    sns.barplot(x = "Bidegree", y = "BS/ES ratio", data = df, hue = "group", ax = ax, ci = None)
    ax.set_xlim([0,20])
    ax.set_title("ES/BS ratio per sink bidegree")
    fig.savefig("BS_ratio_barplot_log_D" + str(dimension+1))
