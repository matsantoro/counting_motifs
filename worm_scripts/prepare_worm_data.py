from robust_motifs.data import prepare_nemanode_data
from pathlib import Path

for csvfile in Path("data/worm").glob("**/*.csv"):
    prepare_nemanode_data(csvfile)
