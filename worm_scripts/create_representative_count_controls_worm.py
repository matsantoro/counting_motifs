from robust_motifs.data import create_representative_simplex_file
from pathlib import Path
from tqdm import tqdm

for file in tqdm(Path("").glob("data/worm_control_nomuscle/**/graph-count-maximal.h5")):
    create_representative_simplex_file(file, file.with_name("graph-count-maximal-representative.h5"), True)

for file in tqdm(Path("").glob("data/worm_control_nomuscle/**/graph-count.h5")):
    create_representative_simplex_file(file, file.with_name("graph-count-representative.h5"), True)

create_representative_simplex_file(Path("data/worm/full_nomuscle/graph-count.h5"),Path("data/worm/full_nomuscle/graph-count-representative.h5"), True )
create_representative_simplex_file(Path("data/worm/full_nomuscle/graph-count-maximal.h5"),Path("data/worm/full_nomuscle/graph-count-maximal-representative.h5"), True )
