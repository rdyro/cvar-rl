from env import *
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import sys


# Policy Tests ################################################################
N = int(1e5)

p = Policy(5, 2)

amin = [0, 0]
amax = [5, 3]
sdim = 5
adim = 2
up = UnifRandPolicy(sdim, amin, amax)
a_up = up.choose_action(np.ones((N, sdim)))
assert len(a_up.shape) == 3
assert a_up.shape[2] == 1
assert np.all(np.logical_and(a_up >= make3D(amin, adim), a_up <= make3D(amax,
  adim)))
print("UnifRandomPolicy")
print("  mu ~= [2.5, 1.5]")
print(np.mean(a_up, axis=0))
print("  mu  = [%f, %f]" % tuple(np.mean(a_up, axis=0).reshape(-1)))

mu = [-1, 2]
sig = [1e-1, 10]
ip = IndepNormRandPolicy(sdim, mu, sig)
a_ip = ip.choose_action(np.ones((N, sdim)))
assert len(a_ip.shape) == 3
print(ip.choose_action(np.ones((N, sdim, 5))).shape)
assert np.all(np.equal(ip.choose_action(np.ones((N, sdim, 5))).shape, [N, adim,
  5]))
assert a_ip.shape[2] == 1
print("IndepRandomPolicy")
print("  mu ~= [-1, 2]")
print("  mu  = [%f, %f]" % tuple(np.mean(a_ip, axis=0).reshape(-1)))
print("  std ~= [1e-1, 10]")
print("  std  = [%f, %f]" % tuple(np.std(a_ip, axis=0).reshape(-1)))

mu = [-1, 2]
sig = [1e-1, 10]
ip = IndepNormRandPolicy(sdim, mu, sig)
a_ip = ip.choose_action(np.ones((N, sdim)))
print("IndepNormRandPolicy")
print("  mu ~= [-1, 2]")
print("  mu  = [%f, %f]" % tuple(np.mean(a_ip, axis=0).reshape(-1)))

pl.figure(0)
pl.title("IndepNormRandPolicy")
pl.hist2d(a_ip[:, 0, 0], a_ip[:, 1, 0], bins=100)

mu = [-1, 2]
S = np.array([[2, -0.99], [-0.99, 1]])
mp = MultNormRandPolicy(sdim, mu, S)
a_mp = mp.choose_action(np.ones((N, sdim)))
print("MultNormRandPolicy")
print("  mu ~= [-1, 2]")
print("  mu  = [%f, %f]" % tuple(np.mean(a_ip, axis=0).reshape(-1)))

pl.figure(1)
pl.title("MultNormRandPolicy")
pl.hist2d(a_mp[:, 0, 0], a_mp[:, 1, 0], bins=100)

print("DiscretePolicy")
dp = DiscretePolicy(sdim, amin, amax, 5)
a_dp = dp.choose_action(np.ones((N, sdim)))
assert np.all(np.equal(dp.choose_action(np.ones((N, sdim)), 0), make3D(amin,
  adim))) 
assert np.all(np.equal(dp.choose_action(np.ones((N, sdim)), -1), make3D(amax,
  adim)))
print("  middle action = ", a_dp[1, :, 0])
print("  first action = ", dp.choose_action(np.ones((1, sdim, 1)),
  0).reshape(-1))
print("  last action = ", dp.choose_action(np.ones((1, sdim, 1)),
  -1).reshape(-1))

# Environment Tests ###########################################################
e = Environment()

fl = FrozenLake()
dfl = DiscreteEnvironment(fl, 20)
assert np.all(np.equal(dfl.all_states().shape, (20**fl.sdim, fl.sdim, 1)))
n = 20
pos = np.hstack([c.reshape((-1, 1)) for c in np.meshgrid(*[np.linspace(0, 1, n)
  for i in range(2)])])
ns = np.hstack([pos[:, 0:1], np.zeros((n**2, 1)), pos[:, 1:2],
  np.zeros((n**2, 1))])
s = np.hstack([np.ones((n**2, 1)) for i in range(fl.sdim)])

ns = make3D(ns, fl.sdim)
s = make3D(s, fl.sdim)

holes = -1 * fl._is_in_discrete(ns, fl.holes).reshape((n, n))
goals = 1.0 * fl._is_in_discrete(ns, fl.goals).reshape((n, n))
pl.figure(7)
pl.imshow(holes + goals)

# Solver Tests ################################################################
n = 10
pol = OptimalDiscretePolicy(fl.sdim, fl.amin, fl.amax, n)
sol = TabularDiscreteSolver(fl, pol, 10)
print(sol.iterate())


# RK4 test ####################################################################
N = 50
K = 20
x = np.random.rand(N, 2, K)
a = np.ones((N, 1, 1))

def f(x, t, p):
  return np.hstack([x[:, 1:2], p - x[:, 0:1]])

M = int(1e2)
X = np.zeros((N, 2, K, M))
X[:, :, :, 0] = x[:, :, :]
t = np.linspace(0, 10, M)
h = 1e-2

for i in range(1, M):
  x = rk4_fn(f, x, t[i-1], t[i], h, a)
  X[:, :, :, i] = x[:, :, :]

pl.figure(10)
pl.clf()
for i in range(N):
  r = np.random.randint(K)
  pl.plot(t, X[i, 0, r, :].reshape(-1))


pl.show()
