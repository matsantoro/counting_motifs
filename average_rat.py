from robust_motifs.counting import Processor
from pathlib import Path
import datetime

try:
    p = Processor(Path("data/ready/individuals/pathways_P14-16"))
    p.list_extended_simplices()
    p.list_bisimplices()
except Exception as e:
    log = open(Path("log.txt"), 'a+')
    log.write(str(datetime.datetime.now())+" Exception:\n" + str(e))
