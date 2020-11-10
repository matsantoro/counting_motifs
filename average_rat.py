from robust_motifs.counting import Processor
from pathlib import Path

p = Processor(Path("data/ready/average"))
p.list_extended_simplices()
p.list_bisimplices()
