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
  
  # value function overlay
  has_baseline = False
  if hasattr(solver, "params") and "baseline" in solver.params:
    has_baseline = solver.params["baseline"]
  has_value_function = hasattr(solver, "value_function")

  if has_baseline:
    (all_s, layer_nb) = unstack2D(all_s)
    v = solver.sess.run(solver.b_, feed_dict={solver.s_: all_s})
    all_s = stack2D(all_s, layer_nb)
  elif has_value_function:
    (all_s, layer_nb) = unstack2D(all_s)
    v = solver.value_function.value(all_s)
    all_s = stack2D(all_s, layer_nb)
  pl.imshow(v.reshape((9, 9)), origin="lower")
  """
  (X, Y) = np.meshgrid(range(9), range(9))
  points = list(zip(X.reshape(-1), Y.reshape(-1)))
  Z = np.array([solver.environment.is_terminal(np.array(points[i])) for i in
    range(len(points))]).reshape((9, 9))
  print(Z)
  pl.imshow(Z.astype(np.float64), origin="lower")
  """
  pl.colorbar()

def evaluate_policy(environment, policy, N=100):
  R = []
  max_ep_len = 1000
  for i in range(N):
    done = False
    s = environment.sample_states(1)
    r_total = 0
    j = 0
    print(i)
    while done == False and j < max_ep_len:
      a = policy.choose_action(s)
      (ns, _) = environment.next_state_sample(s, a)
      r = environment.reward(s, a, ns)
      r_total += environment.gamma**j * r.reshape(-1)[0]
      done = environment.is_terminal(s)
      s = ns
      
      j += 1
    R.append(r_total)
  return np.array(R)

# Solver Tests ################################################################
mars = env.Mars()

policy = pol.OptimalDiscretePolicy(mars.sdim, mars.amin, mars.amax, 4)
#solver = sol.TabularDiscreteSolver(mars, policy, 9)
solver = sol.ModelDiscreteSolver(mars, policy, 9, "nn", sample=False)
#solver = sol.PolicyGradientDiscreteSolver(mars, 4, episodes_nb=100,
#    episode_len=100, normalize_adv=True, baseline=True, h=3e-2)

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

it_n = 100
for i in range(it_n):
  print(solver.iterate())
print()
visualize_mars_policy(solver)

mars_cvar = env.MarsCVaR(solver.value_function)
policy2 = pol.OptimalDiscretePolicy(mars.sdim, mars.amin, mars.amax, 4)
solver2 = sol.ModelDiscreteSolver(mars_cvar, policy2, 9, "nn", sample=False)
for i in range(it_n):
  print(solver2.iterate())
print()
visualize_mars_policy(solver2)

pl.show()


ep_n = int(1e4)
R2 = evaluate_policy(mars, solver2.policy, ep_n)
print("Average reward for normal = ", np.mean(R2))

R = evaluate_policy(mars, solver.policy, ep_n)
print("Average reward for normal = ", np.mean(R))

with open("../data/standard.txt", "w") as fp:
  for r in R:
    fp.write("%f\n" % r)

with open("../data/cvar.txt", "w") as fp:
  for r in R2:
    fp.write("%f\n" % r)

pl.figure()
pl.hist([R, R2], bins=20, rwidth=0.7)

"""
mars_aug = env.MarsAugmentedReward(solver.value_function)
solver3 = sol.PolicyGradientDiscreteSolver(mars_aug, 4, episodes_nb=20,
    episode_len=20, normalize_adv=True, baseline=True, h=3e-2)

for i in range(30):
  print(solver3.iterate())
print()
visualize_mars_policy(solver3)
"""

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
