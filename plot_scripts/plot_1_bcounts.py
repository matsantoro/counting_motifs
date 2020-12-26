from robust_motifs.data import ResultManager, BcountResultManager
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

r_average = ResultManager(Path("data/ready/average"))
r_shuffled = ResultManager(Path("data/ready/controls/shuffled"))
r_pathways = ResultManager(Path("data/ready/controls/pathway"))
r_adjusted = ResultManager(Path("data/ready/controls/adjusted"))
r_underlying = BcountResultManager(Path("data/bcounts/underlying"))
r_bshuffled = BcountResultManager(Path("data/bcounts/bshuffled_1"))

df_average = r_average.get_counts_dataframe("average")
df_shuffled = r_shuffled.get_counts_dataframe("shuffled")
df_pathways = r_pathways.get_counts_dataframe("pathways")
df_adjusted = r_adjusted.get_counts_dataframe("adjusted")
df_underlying = r_underlying.get_counts_dataframe("underlying")
df_bshuffled = r_bshuffled.get_counts_dataframe("bshuffled")

for df in [df_shuffled, df_pathways, df_adjusted, df_underlying, df_bshuffled]:
    df_average = df_average.append(df, ignore_index = True)

df1 = df_average[df_average['motif'] == 'ES']
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df1, x = 'dim', y = 'count', hue = 'group', ax = ax)
ax.set_ylabel("Extended simplices")
fig.savefig("es1bcounts", facecolor = "white")

df2 = df_average[df_average['motif'] == 'S']
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df2, x = 'dim', y = 'count', hue = 'group', ax = ax)
ax.set_ylabel("Simplices")
fig.savefig("s1bcounts", facecolor = "white")

df3 = df_average[df_average['motif'] == 'BS']
fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(data = df3, x = 'dim', y = 'count', hue = 'group', ax = ax)
ax.set_ylabel("Bisimplices")
fig.savefig("bs1bcounts", facecolor = "white")

