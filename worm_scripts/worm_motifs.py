from robust_motifs.counting import Processor
from pathlib import Path
import datetime
import traceback

try:
    p = Processor(Path("data/worm"))
    p.list_extended_simplices()
    p.list_bisimplices()
except Exception as e:
    log = open(Path("log.txt"), 'a+')
    log.write(str(datetime.datetime.now())+" Exception:\n" + str(e))
    log.write(str(datetime.datetime.now())+" Traceback:\n" + traceback.format_exc())
