from __future__ import division
from __future__ import print_function
import sys, os
sys.path.append(os.path.dirname(__file__) + "../src")
from util import *

import env
import matplotlib.pyplot as pl

environment = env.Mars()

(X, Y) = np.meshgrid(range(9), range(9))
points = list(zip(X.reshape(-1), Y.reshape(-1)))
Z = np.array([environment.is_terminal(np.array(points[i])) for i in
  range(len(points))]).reshape((9, 9))
print(Z)
Z = Z.astype(np.float64) * -1
Z[0, 0] = 1
pl.imshow(Z, origin="lower")
pl.tight_layout()
pl.savefig("../fig/mars.png", dpi=200)
pl.show()

# Environment Tests ###########################################################
e = env.Environment()

fl = env.FrozenLake()
dfl = env.DiscreteEnvironment(fl, 20)
assert np.all(np.equal(dfl.all_states().shape, (20**fl.sdim, fl.sdim, 1)))
n = 20
pos = np.hstack([c.reshape((-1, 1)) for c in np.meshgrid(*[np.linspace(0, 1, n)
  for i in range(2)])])
ns = np.hstack([pos[:, 0:1], np.zeros((n**2, 1)), pos[:, 1:2],
  np.zeros((n**2, 1))])
s = np.hstack([np.ones((n**2, 1)) for i in range(fl.sdim)])

ns = make3D(ns, fl.sdim)
s = make3D(s, fl.sdim)

holes = -1 * is_in_discrete(ns[:, [0, 2]], fl.holes, fl.gridn, fl.smin[:, [0, 2]],
    fl.smax[:, [0, 2]]).reshape((n, n))
goals = 1 * is_in_discrete(ns[:, [0, 2]], fl.goals, fl.gridn, fl.smin[:, [0, 2]],
    fl.smax[:, [0, 2]]).reshape((n, n))
pl.figure(7)
pl.imshow(holes + goals)

mars = env.Mars()

print(mars.sample_states(5).shape)
print(mars.sample_states(5).reshape(-1, 2))

pl.show()
