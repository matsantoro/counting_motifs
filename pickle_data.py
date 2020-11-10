from robust_motifs.data import Pickleizer
from pathlib import Path

p = Pickleizer(Path("data/original/individuals"))
p.pickle_it(Path("data/ready/individuals"))
