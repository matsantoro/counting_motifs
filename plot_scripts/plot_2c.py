from robust_motifs.data import ResultManager
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

r_average = ResultManager(Path("data/ready/average"))
r_shuffled = ResultManager(Path("data/ready/controls/shuffled"))
r_pathways = ResultManager(Path("data/ready/controls_1/pathway"))
r_adjusted = ResultManager(Path("data/ready/controls_1/adjusted"))

df_average = r_average.get_counts_dataframe("average")
df_shuffled = r_shuffled.get_counts_dataframe("shuffled")
df_pathways = r_pathways.get_counts_dataframe("pathways")
df_adjusted = r_adjusted.get_counts_dataframe("adjusted")

for df in [df_shuffled, df_pathways, df_adjusted]:
    df_average = df_average.append(df, ignore_index = True)

df1 = df_average[df_average['motif'] == 'RES+']
df1 = df1[df1["dim"]<6]
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df1, x = 'dim', y = 'count', hue = 'group', ax = ax)
ax.set_ylabel("Extended simplices / simplex")
ax.set_yscale("log")
fig.savefig("res1shiftlog", facecolor = "white")

df3 = df_average[df_average['motif'] == 'RBS+']
df3 = df3[df3['dim'] < 6]
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df3, x = 'dim', y = 'count', hue = 'group', ax = ax)
ax.set_ylabel("Bisimplices / simplex")
ax.set_yscale("log")
fig.savefig("rbs1shiftlog", facecolor = "white")