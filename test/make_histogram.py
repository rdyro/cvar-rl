from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as pl
import numpy as np

with open("../data/standard.txt", "r") as fp:
  R = [float(line.strip("\n")) for line in fp.readlines()]
with open("../data/cvar.txt", "r") as fp:
  R2 = [float(line.strip("\n")) for line in fp.readlines()]

print(np.mean(R))
print(np.mean(R2))

pl.figure()
pl.hist([R, R2], label=["Unmodified Policy", "Robust Policy"], rwidth=0.9, bins=20)
pl.xlabel("Total episode reward")
pl.legend()
pl.tight_layout()
pl.savefig("../fig/hist.png", dpi=200)
pl.show()

