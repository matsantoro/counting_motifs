from robust_motifs.data import ResultManager, BcountResultManager
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Plots the ratios of bisimplices/simplices for all rats, compares to new control models.

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

r_underlying = BcountResultManager(Path("data/bcounts/underlying"))
r_bshuffled = BcountResultManager(Path("data/bcounts/bshuffled_1"))

df_underlying = r_underlying.get_counts_dataframe('underlying')
df_bshuffled = r_bshuffled.get_counts_dataframe('bshuffled')

df_average['control'] = False
df_underlying['control'] = True
df_bshuffled['control'] = True

df_average = df_average.append(df_underlying, ignore_index = True)
df_average = df_average.append(df_bshuffled, ignore_index = True)

df1 = df_average[df_average['motif'] == 'RES+']
df1 = df1[df1['dim']<7]
df1['dim'] = df1['dim'].apply(lambda x: int(x))
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df1, x = 'dim', y = 'count', hue = 'group', ax = ax, style = 'control')
ax.set_ylabel("Extended simplices/simplex")
ax.set_xlabel("Dimension")
ax.set_yscale("log")
fig.savefig("es_dimension_individuals_ratio_bcounts", facecolor = "white")

df3 = df_average[df_average['motif'] == 'RBS+']
df3 = df3[df3['dim']<7]
df3['dim'] = df3['dim'].apply(lambda x: int(x))
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df3, x = 'dim', y = 'count', hue = 'group', ax = ax, style = 'control')
ax.set_ylabel("Bisimplices/simplex")
ax.set_xlabel("Dimension")
fig.savefig("bs_dimension_individuals_ratio_bcounts", facecolor = "white")
