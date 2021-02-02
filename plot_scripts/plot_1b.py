from robust_motifs.data import ResultManager
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Plots of motif counts where individual lines are plotted instead of CIs.

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

df1 = df_average[df_average['motif'] == 'ES']
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df1, x = 'dim', y = 'count', hue = 'group', ax = ax, units='filename', estimator = None, lw=1)
ax.set_ylabel("Extended simplices")
fig.savefig("es1-lines", facecolor = "white")

df2 = df_average[df_average['motif'] == 'S']
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df2, x = 'dim', y = 'count', hue = 'group', ax = ax, units='filename', estimator = None, lw=1)
ax.set_ylabel("Simplices")
fig.savefig("s1-lines", facecolor = "white")

df3 = df_average[df_average['motif'] == 'BS']
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df3, x = 'dim', y = 'count', hue = 'group', ax = ax, units='filename', estimator = None, lw=1)
ax.set_ylabel("Bisimplices")
fig.savefig("bs1-lines", facecolor = "white")

