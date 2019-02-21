from __future__ import division
from __future__ import print_function
import sys, os
sys.path.append(os.path.dirname(__file__) + "../src")
from util import *

import matplotlib.pyplot as pl
import pol

# Policy Tests ################################################################
N = int(1e5)

p = pol.Policy(5, 2)

amin = [0, 0]
amax = [5, 3]
sdim = 5
adim = 2
up = pol.UnifRandPolicy(sdim, amin, amax)
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
ip = pol.IndepNormRandPolicy(sdim, mu, sig)
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
ip = pol.IndepNormRandPolicy(sdim, mu, sig)
a_ip = ip.choose_action(np.ones((N, sdim)))
print("IndepNormRandPolicy")
print("  mu ~= [-1, 2]")
print("  mu  = [%f, %f]" % tuple(np.mean(a_ip, axis=0).reshape(-1)))

pl.figure(0)
pl.title("IndepNormRandPolicy")
pl.hist2d(a_ip[:, 0, 0], a_ip[:, 1, 0], bins=100)

mu = [-1, 2]
S = np.array([[2, -0.99], [-0.99, 1]])
mp = pol.MultNormRandPolicy(sdim, mu, S)
a_mp = mp.choose_action(np.ones((N, sdim)))
print("MultNormRandPolicy")
print("  mu ~= [-1, 2]")
print("  mu  = [%f, %f]" % tuple(np.mean(a_ip, axis=0).reshape(-1)))

pl.figure(1)
pl.title("MultNormRandPolicy")
pl.hist2d(a_mp[:, 0, 0], a_mp[:, 1, 0], bins=100)

print("DiscretePolicy")
dp = pol.DiscretePolicy(sdim, amin, amax, 5)
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
