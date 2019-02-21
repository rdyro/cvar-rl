from __future__ import division
from __future__ import print_function
import sys, os
sys.path.append(os.path.dirname(__file__) + "../src")
from util import *
import sol
import env
import pol
import val

import matplotlib.pyplot as pl

# Solver Tests ################################################################
mars = env.Mars()

policy = pol.OptimalDiscretePolicy(mars.sdim, mars.amin, mars.amax, 4)
solver = sol.TabularDiscreteSolver(mars, policy, 9)
for i in range(100):
  print(solver.iterate(), end=" ")
print()

all_s = solver.environment.all_states()
v = solver.value_function.value(all_s)
all_a = solver.policy.choose_action(all_s)

#for i in range(all_a.shape[0]):
#  print("[%d, %d] -> %d" % (all_s[i, 0, 0], all_s[i, 1, 0], all_a[i, 0, 0]))

(ns, p) = solver.environment.next_state_full(all_s, all_a)
(all_s_exp, all_a_exp, ns) = match_03(all_s, all_a, ns)
r = np.sum(p * solver.environment.reward(all_s_exp, all_a_exp, ns), axis=2)

idxs = solver.value_function._sidx(make3D([2, 3], 2)).reshape(-1)[0]
print(ns[idxs, :, :])
print(ns[idxs, :, :].transpose((1, 0)))

UV = np.array([[1, 0] if a == 0 else [0, 1] if a == 1 else [-1, 0] if a == 2
  else [0, -1] for a in all_a])

U = UV[:, 0]
V = UV[:, 1]
pl.figure(17)
pl.imshow(v.reshape((9, 9)), origin="lower")
pl.colorbar()
pl.quiver(all_s[:, 0].reshape((9, 9)), all_s[:, 1].reshape((9, 9)), 
          U.reshape((9, 9)), V.reshape((9, 9)))
print(val.TabularValueFunction(mars.smin, mars.smax, 9).qvalue(mars, [2, 3],
  2))

pl.show()
