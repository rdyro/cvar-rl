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

def all_states(environment):
  n = [9, 9]
  slin = [np.linspace(environment.smin.reshape(-1)[i],
    environment.smax.reshape(-1)[i], n[i]) for i in range(environment.sdim)]
  sspc = np.hstack([sarr.reshape((-1, 1)) for sarr in np.meshgrid(*slin)])
  sspc = make3D(sspc, environment.sdim)
  return sspc

def visualize_mars_policy(solver):
  all_s = all_states(solver.environment)
  all_a = solver.policy.choose_action(all_s)

  UV = np.array([[1, 0] if a == 0 else [0, 1] if a == 1 else [-1, 0] if a == 2
    else [0, -1] for a in all_a])

  U = UV[:, 0]
  V = UV[:, 1]
  pl.figure()
  pl.quiver(all_s[:, 0].reshape((9, 9)), all_s[:, 1].reshape((9, 9)), 
            U.reshape((9, 9)), V.reshape((9, 9)))

# Solver Tests ################################################################
mars = env.Mars()

policy = pol.OptimalDiscretePolicy(mars.sdim, mars.amin, mars.amax, 4)
#solver = sol.TabularDiscreteSolver(mars, policy, 9)
solver = sol.ModelDiscreteSolver(mars, policy, 9, "nn")
#solver = sol.PolicyGradientDiscreteSolver(mars, 4)

# testing state sampling ------------------------------------------------------
#all_s = solver.environment.all_states()
all_s = mars.sample_states(100)
(ns_sample, p) = mars.next_state_sample(all_s, [0])
for i in range(all_s.shape[0]):
  pass
  #print(all_s[i, :, :].reshape((1, -1)), end=" -> ")
  #print(ns_sample[i, :, :].reshape((1, -1)), end=" -> ")
  #print(p[i, :, :].reshape(-1))
# end of state samples test ---------------------------------------------------

for i in range(10):
  print(solver.iterate())
print()
visualize_mars_policy(solver)

pl.show()
  

"""
all_s = solver.environment.all_states()
v = solver.value_function.value(all_s)
all_a = solver.policy.choose_action(all_s)

(ns, p) = solver.environment.next_state_full(all_s, all_a)
(all_s_exp, all_a_exp, ns) = match_03(all_s, all_a, ns)
r = np.sum(p * solver.environment.reward(all_s_exp, all_a_exp, ns), axis=2)

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
"""