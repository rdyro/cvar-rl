from env import *
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D


# Policy Tests ################################################################
N = int(1e5)

p = Policy(5, 2)

amin = [0, 0]
amax = [5, 3]
sdim = 5
up = UnifRandPolicy(sdim, amin, amax)
a_up = up.choose_action(np.ones((N, sdim)))
assert np.all(np.logical_and(a_up >= amin, a_up <= amax))
print("UnifRandomPolicy")
print("  mu ~= [2.5, 1.5]")
print(np.mean(a_up, axis=0))
print("  mu  = [%f, %f]" % tuple(np.mean(a_up, axis=0)))

mu = [-1, 2]
sig = [1e-1, 10]
ip = IndepNormRandPolicy(sdim, mu, sig)
a_ip = ip.choose_action(np.ones((N, sdim)))
print("IndepRandomPolicy")
print("  mu ~= [-1, 2]")
print("  mu  = [%f, %f]" % tuple(np.mean(a_ip, axis=0).reshape(-1)))


mu = [-1, 2]
sig = [1e-1, 10]
ip = IndepNormRandPolicy(sdim, mu, sig)
a_ip = ip.choose_action(np.ones((N, sdim)))
print("IndepNormRandPolicy")
print("  mu ~= [-1, 2]")
print("  mu  = [%f, %f]" % tuple(np.mean(a_ip, axis=0).reshape(-1)))

pl.figure(0)
pl.title("IndepNormRandPolicy")
pl.hist2d(a_ip[:, 0], a_ip[:, 1], bins=100)

mu = [-1, 2]
S = np.array([[2, -0.99], [-0.99, 1]])
mp = MultNormRandPolicy(sdim, mu, S)
a_mp = mp.choose_action(np.ones((N, sdim)))
print("MultNormRandPolicy")
print("  mu ~= [-1, 2]")
print("  mu  = [%f, %f]" % tuple(np.mean(a_ip, axis=0).reshape(-1)))

pl.figure(1)
pl.title("MultNormRandPolicy")
pl.hist2d(a_mp[:, 0], a_mp[:, 1], bins=100)

dp = DiscretePolicy(sdim, amin, amax, 5)
a_dp = dp.choose_action(np.ones((N, sdim)))
print(a_dp)
assert np.all(dp.choose_action(np.ones((N, sdim)), 0) == amin)
assert np.all(dp.choose_action(np.ones((N, sdim)), -1) == amax)

# Environment Tests ###########################################################
e = Environment()

fl = FrozenLake()

# RK4 test ####################################################################
N = 50
x = np.random.rand(N, 2)
a = np.ones(N).reshape((-1, 1))

def f(x, t, p):
  return np.hstack([x[:, 1:2], p - x[:, 0:1]])

M = int(1e1)
X = np.zeros((N, 2, M))
X[:, :, 0] = x[:, :]
t = np.linspace(0, 10, M)
h = 1e-2

for i in range(1, M):
  x = rk4_fn(f, x, t[i-1], t[i], h, a)
  X[:, :, i] = x[:, :]

pl.figure(10)
pl.clf()
for i in range(N):
  pl.plot(t, X[i, 0, :].reshape(-1))
pl.show()
