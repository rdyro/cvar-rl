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

# test is_in_discrete
print("Testing is_in_discrete")
points = np.array([[0, 0], [3, 3]])
smin = [0.0, 0.0]
smax = [1.0, 1.0]

N = 10
X, Y = np.meshgrid(np.linspace(smin[0], smax[0], N), np.linspace(smin[1],
  smax[1], N))

s1 = np.hstack([X.reshape((-1, 1)), Y.reshape((-1, 1))])
s2 = np.copy(s1)
s2[:, :] = 0.0
s3 = np.copy(s1)
s3[:, :] = 0.5

s = np.dstack([s1, s2, s3])
mask = is_in_discrete(s, points, 4, smin, smax)
print("mask = ")
print(mask)
print(mask.shape)

for i in range(s.shape[2]):
  pl.figure(16 - i)
  pl.imshow(mask[:, :, i].reshape((N, N)))
  print(mask[:, :, i].reshape((N, N)))

pl.show()
