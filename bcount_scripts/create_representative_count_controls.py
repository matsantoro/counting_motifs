from robust_motifs.data import create_representative_simplex_file
from pathlib import Path
from tqdm import tqdm

for file in tqdm(Path("").glob("data/bcounts/bshuffled_1/**/graph-count-maximal.h5")):
    create_representative_simplex_file(file, file.with_name("graph-count-maximal-representative.h5"), True)

for file in tqdm(Path("").glob("data/bcounts/underlying_1/**/graph-count-maximal.h5")):
    create_representative_simplex_file(file, file.with_name("graph-count-maximal-representative.h5"), True)
