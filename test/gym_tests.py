from __future__ import division
from __future__ import print_function
import sys, os
sys.path.append(os.path.dirname(__file__) + "../src")
from util import *

import env
import sol
import pol
import matplotlib.pyplot as pl
import gym
import time


"""
cp1 = gym.make("CartPole-v1").env
cp2 = env.GymWrapper("CartPole-v1")

assert np.all(cp1.observation_space.low == cp2.smin.reshape(-1))
assert np.all(cp1.observation_space.high == cp2.smax.reshape(-1))
assert cp2.adim == 1
assert cp2.sdim == 4
assert cp1.action_space.n == cp2.amax[0] + 1
assert cp2.amin[0] == 0

for i in range(200):
  cp1.reset()
  a = np.random.randint(2)
  s = cp1.state
  (ns1, r1, done1, _) = cp1.step(a)
  (ns2, _) = cp2.next_state_sample(s, [a])
  r2 = cp2.reward(s, [a], ns2)
  done2 = cp2.is_terminal(ns2)

  assert np.all(ns1.reshape(-1) == ns2.reshape(-1))
  assert np.all([r1] == r2.reshape(-1))
  assert np.all([done1] == done2.reshape(-1))
"""

"""
print("ns1 =   ", ns1)
print("r1 =    ", r1)
print("done1 = ", done1)
print("")
print("ns2 =   ", ns2)
print("r2 =    ", r2)
print("done2 = ", done2)
"""

"""
N = 5
M = 100
S = cp2.sample_states(N)
A = np.zeros((N, cp2.adim, 0))
R = np.zeros((N, 1, 0))
DONE = np.zeros((N, 1, 0), dtype=np.bool)

t1 = time.time()
for i in range(M):
  s = make3D(S[:, :, -1], cp2.sdim)
  a = make3D(np.random.randint(cp2.amax + 1, size=(N, cp2.adim)), cp2.adim)
  (ns, _) = cp2.next_state_sample(s, a)
  r = cp2.reward(s, a, ns)
  done = cp2.is_terminal(ns)

  S = np.dstack([S, ns])
  A = np.dstack([A, a])
  R = np.dstack([R, r])
  DONE = np.dstack([DONE, done])
S = S[:, :, :-1]
t2 = time.time()
print(t2 - t1)

for i in range(N):
  time.sleep(2)
  for j in range(M):
    cp2.render(S[i, :, j])
"""

cp2 = env.GymWrapper("CartPole-v1")
cp2.smin = np.clip(cp2.smin, -5.0, 5.0)
cp2.smax = np.clip(cp2.smax, -5.0, 5.0)

print(cp2.amax + 1)
print()
policy = pol.OptimalDiscretePolicy(cp2.sdim, cp2.amin, cp2.amax, cp2.amax + 1)
solver = sol.ModelSampleSolver(cp2, policy, 5000, "nn")
for i in range(40):
  print(solver.iterate())

while True:
  s = cp2.sample_states(1, True)
  for i in range(100):
    a = policy.choose_action(s)
    (ns, _) = cp2.next_state_sample(s, a)
    s = ns
    cp2.render(s)
    time.sleep(1 / 60)
