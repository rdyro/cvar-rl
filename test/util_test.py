from __future__ import division
from __future__ import print_function
import sys, os
sys.path.append(os.path.dirname(__file__) + "../src")
from util import *

import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm
import time

S = np.array([[30, 0.5],
              [0.5, 1]])
#S = np.eye(7)
Sinv = np.linalg.inv(S)

t1 = time.time()
(d, p) = normal_dist(S)
t2 = time.time()

print(d)
print(p)
print(t2 - t1)

(d, p) = normal_dist(S, 30, Sinv=Sinv)

xx, yy = np.meshgrid(d[:, 0], d[:, 1])
zz = griddata((d[:, 0], d[:, 1]), p, (xx, yy), method="linear")

fig = pl.figure()
ax = fig.gca(projection="3d")
ax.plot_surface(xx, yy, zz)
pl.show()
