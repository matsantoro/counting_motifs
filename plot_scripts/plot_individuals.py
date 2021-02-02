from robust_motifs.data import ResultManager
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Plot absolute motif counts for indivudal rats.

r_average = ResultManager(Path("data/ready/average"))
r = []
for pathway in range(13,18):
    r.append(ResultManager(Path("data/ready/individuals_1/pathways_P14-"+str(pathway))))

df_average = r_average.get_counts_dataframe("average")
dfs = []
for i, result in enumerate(r):
    dfs.append(result.get_counts_dataframe("P"+str(i+13)))

for df in dfs:
    df_average = df_average.append(df, ignore_index = True)

df1 = df_average[df_average['motif'] == 'ES']
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df1, x = 'dim', y = 'count', hue = 'group', ax = ax)
ax.set_ylabel("Extended simplices")
ax.set_xlabel("Dimension")
fig.savefig("es_dimension_individuals", facecolor = "white")

df2 = df_average[df_average['motif'] == 'S']
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df2, x = 'dim', y = 'count', hue = 'group', ax = ax)
ax.set_ylabel("Simplices")
ax.set_xlabel("Dimension")
fig.savefig("s_dimension_individuals", facecolor = "white")
 
df3 = df_average[df_average['motif'] == 'BS']
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df3, x = 'dim', y = 'count', hue = 'group', ax = ax)
ax.set_ylabel("Bisimplices")
ax.set_xlabel("Dimension")
fig.savefig("bs_dimension_individuals", facecolor = "white")

df4 = df_average[df_average['motif'] == 'ES']
df4 = df4[df4['dim']>3]
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df4, x = 'dim', y = 'count', hue = 'group', ax = ax)
ax.set_ylabel("Extended simplices")
ax.set_xlabel("Dimension")
ax.set_ylim([-2,10])
fig.savefig("es_dimension_high_individuals", facecolor = "white")

df5 = df_average[df_average['motif'] == 'BS']
df5 = df5[df5['dim']>2]
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df5, x = 'dim', y = 'count', hue = 'group', ax = ax)
ax.set_ylabel("Bisimplices")
ax.set_xlabel("Dimension")
ax.set_ylim([-2,10])
fig.savefig("bs_dimension_high_individuals", facecolor = "white")
